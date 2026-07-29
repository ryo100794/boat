from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .edge_conditional_probability_lcb_v13 import (
    _market_probabilities,
    probability_lower_bound_details,
    t300_odds,
)
from .strict_prior_t300_divergence_passthrough_v16 import (
    REGISTERED_DIVERGENCE_LOWER as BAND_LOW,
    REGISTERED_DIVERGENCE_UPPER as BAND_HIGH,
)

MODEL_NAME = "v16_fixed_band_ranking_diagnostics"
STAKE_YEN = 100
MAX_DAILY_TICKETS = 100
BAND_CENTER = (BAND_LOW + BAND_HIGH) / 2.0
RULES = (
    "safe_ev_desc",
    "raw_probability_desc",
    "estimated_closing_odds_asc",
    "divergence_center_distance",
    "per_race_round_robin_diversified",
)
_CLOSING_FIELDS = (
    "point_final_odds", "predicted_closing_odds",
    "closing_forecasts", "forecast_closing_odds",
)
_RESULT_FIELDS = {
    "actual_combination", "actual_payout_yen", "hit", "return_yen",
}


@dataclass(frozen=True)
class FixedBandDiagnosticInputs:
    decision_candidates: tuple[Mapping[str, Any], ...]
    settlements: Mapping[tuple[str, str], int]
    evaluated_races_by_day: Mapping[str, int]
    rejected_races_by_reason: Mapping[str, int]


def _positive(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and result > 0.0 else None


def _closing(
    race: Mapping[str, Any],
    forecasts: Mapping[str, Mapping[str, float]] | None,
) -> dict[str, float]:
    if forecasts is not None:
        sources = (forecasts.get(str(race.get("race_id") or "")),)
    else:
        sources = tuple(race.get(field) for field in _CLOSING_FIELDS if field in race)
    for source in sources:
        if isinstance(source, Mapping):
            result = {
                str(key): value
                for key, raw in source.items()
                if (value := _positive(raw)) is not None
            }
            return result if len(result) == 120 else {}
    return {}


def _identity(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return str(row["race_date"]), str(row["race_id"]), str(row["combination"])


def _hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def candidate_population_fingerprint(
    candidates: Iterable[Mapping[str, Any]],
) -> str:
    return _hash(sorted(_identity(row) for row in candidates))


def decision_information_fingerprint(
    candidates: Iterable[Mapping[str, Any]],
) -> str:
    rows = []
    for row in sorted(candidates, key=_identity):
        if _RESULT_FIELDS.intersection(row):
            raise ValueError("result or payout field reached decision information")
        rows.append({
            key: row[key] for key in (
                "race_date", "race_id", "jcd", "rno", "combination",
                "raw_probability", "t300_odds", "t300_market_probability",
                "log_model_market_divergence", "estimated_closing_odds", "safe_ev",
            )
        })
    return _hash(rows)


def _probability_and_divergence(
    race: Mapping[str, Any],
    combination: str,
    probabilities: Mapping[str, float],
    market: Mapping[str, float],
    artifact: Mapping[str, Any] | None,
) -> tuple[float, float]:
    if artifact is not None:
        detail = probability_lower_bound_details(
            dict(race), combination, dict(artifact)
        )
        return (
            float(detail["probability"]),
            float(detail.get("log_model_market_divergence", float("-inf"))),
        )
    probability = float(probabilities[combination])
    divergence = math.log(probability / float(market[combination]))
    return probability, divergence


def build_fixed_band_diagnostic_inputs(
    races: Iterable[Mapping[str, Any]],
    *,
    closing_forecasts: Mapping[str, Mapping[str, float]] | None = None,
    probability_lcb: Mapping[str, Any] | None = None,
) -> FixedBandDiagnosticInputs:
    """Build all fixed-band decisions and a physically separate settlement map."""
    candidates: list[Mapping[str, Any]] = []
    settlements: dict[tuple[str, str], int] = {}
    evaluated: dict[str, int] = defaultdict(int)
    rejected: dict[str, int] = defaultdict(int)
    for race in sorted(
        races,
        key=lambda row: (
            str(row.get("race_date") or ""), str(row.get("race_id") or "")
        ),
    ):
        date = str(race.get("race_date") or "")
        race_id = str(race.get("race_id") or "")
        raw_probabilities = race.get("model_probabilities")
        if not date or not race_id:
            rejected["missing_race_identity"] += 1
            continue
        if not isinstance(raw_probabilities, Mapping) or len(raw_probabilities) != 120:
            rejected["incomplete_model_probabilities"] += 1
            continue
        probabilities = {
            str(key): _positive(raw) for key, raw in raw_probabilities.items()
        }
        if any(value is None or value > 1.0 for value in probabilities.values()):
            rejected["invalid_model_probability"] += 1
            continue
        t300 = t300_odds(race)
        market = _market_probabilities(t300)
        closing = _closing(race, closing_forecasts)
        keys = set(probabilities)
        if len(t300) != 120 or set(t300) != keys:
            rejected["incomplete_t300"] += 1
            continue
        if len(closing) != 120 or set(closing) != keys:
            rejected["incomplete_closing_forecast"] += 1
            continue
        if probability_lcb is not None and not probability_lcb.get("ready"):
            rejected["probability_lcb_not_ready"] += 1
            continue

        evaluated[date] += 1
        actual = str(race.get("actual_combination") or "")
        payout = race.get("actual_payout_yen")
        if actual in keys and payout is not None:
            payout = int(payout)
            if payout < 0:
                raise ValueError("actual_payout_yen must be non-negative")
            key = race_id, actual
            if key in settlements and settlements[key] != payout:
                raise ValueError("conflicting settlement")
            settlements[key] = payout

        for combination in sorted(keys):
            probability, divergence = _probability_and_divergence(
                race, combination, probabilities, market, probability_lcb
            )
            if probability <= 0.0 or not BAND_LOW <= divergence < BAND_HIGH:
                continue
            odds = float(closing[combination])
            candidates.append(MappingProxyType({
                "race_date": date, "race_id": race_id,
                "jcd": race.get("jcd"), "rno": int(race.get("rno") or 0),
                "combination": combination,
                "raw_probability": probability, "probability": probability,
                "t300_odds": float(t300[combination]),
                "t300_market_probability": float(market[combination]),
                "log_model_market_divergence": divergence,
                "divergence_center_distance": abs(divergence - BAND_CENTER),
                "estimated_closing_odds": odds, "estimated_odds": odds,
                "safe_ev": probability * odds,
            }))

    candidates.sort(key=_identity)
    return FixedBandDiagnosticInputs(
        tuple(candidates),
        MappingProxyType(dict(sorted(settlements.items()))),
        MappingProxyType(dict(sorted(evaluated.items()))),
        MappingProxyType(dict(sorted(rejected.items()))),
    )


def _rank_key(rule: str, row: Mapping[str, Any]) -> tuple[Any, ...]:
    keys = {
        "safe_ev_desc": (-float(row["safe_ev"]),),
        "raw_probability_desc": (-float(row["raw_probability"]),),
        "estimated_closing_odds_asc": (float(row["estimated_closing_odds"]),),
        "divergence_center_distance": (float(row["divergence_center_distance"]),),
    }
    if rule not in keys:
        raise ValueError(f"unsupported ranking rule: {rule}")
    return keys[rule] + _identity(row)


def _round_robin(rows: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["race_id"])].append(row)
    queues = {
        key: sorted(values, key=lambda row: _rank_key("safe_ev_desc", row))
        for key, values in grouped.items()
    }
    result, depth = [], 0
    while True:
        appended = False
        for race_id in sorted(queues):
            if depth < len(queues[race_id]):
                result.append(queues[race_id][depth])
                appended = True
        if not appended:
            return result
        depth += 1


def select_diagnostic_portfolio(
    candidates: Iterable[Mapping[str, Any]],
    *, rule: str, daily_budget_yen: int = 10_000,
) -> tuple[Mapping[str, Any], ...]:
    if rule not in RULES:
        raise ValueError(f"unsupported ranking rule: {rule}")
    if daily_budget_yen < 0 or daily_budget_yen % STAKE_YEN:
        raise ValueError("daily_budget_yen must be a non-negative 100-yen multiple")
    candidates = list(candidates)
    if any(_RESULT_FIELDS.intersection(row) for row in candidates):
        raise ValueError("result or payout field reached portfolio selection")
    limit = min(MAX_DAILY_TICKETS, daily_budget_yen // STAKE_YEN)
    if not candidates or not limit:
        return ()
    ranked = (
        _round_robin(candidates)
        if rule == "per_race_round_robin_diversified"
        else sorted(candidates, key=lambda row: _rank_key(rule, row))
    )
    return tuple(ranked[:limit])


def _settle(
    selected: Iterable[Mapping[str, Any]],
    settlements: Mapping[tuple[str, str], int],
) -> dict[str, Any]:
    selected = list(selected)
    returns = [
        int(settlements.get((str(row["race_id"]), str(row["combination"])), 0))
        for row in selected
    ]
    stake, returned = len(selected) * STAKE_YEN, sum(returns)
    largest = max(returns, default=0)
    return {
        "tickets": len(selected), "hits": sum(value > 0 for value in returns),
        "stake_yen": stake, "return_yen": returned,
        "profit_yen": returned - stake,
        "roi": returned / stake if stake else None,
        "largest_hit_return_yen": largest,
        "return_excluding_largest_hit_yen": returned - largest,
        "roi_excluding_largest_hit": (returned - largest) / stake if stake else None,
    }


def compare_v16_fixed_band_ranking_rules(
    races: Iterable[Mapping[str, Any]],
    *,
    closing_forecasts: Mapping[str, Mapping[str, float]] | None = None,
    probability_lcb: Mapping[str, Any] | None = None,
    daily_budget_yen: int = 10_000,
) -> dict[str, Any]:
    """Research comparison; post-hoc winners are never promotion evidence."""
    inputs = build_fixed_band_diagnostic_inputs(
        races, closing_forecasts=closing_forecasts, probability_lcb=probability_lcb
    )
    by_day: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in inputs.decision_candidates:
        by_day[str(row["race_date"])].append(row)
    outputs = {}
    for rule in RULES:
        daily, all_selected = [], []
        for date, race_count in inputs.evaluated_races_by_day.items():
            source = by_day.get(date, [])
            selected = select_diagnostic_portfolio(
                source, rule=rule, daily_budget_yen=daily_budget_yen
            )
            all_selected.extend(selected)
            daily.append({
                "race_date": date, "evaluated_races": race_count,
                "candidate_tickets": len(source),
                "candidate_population_fingerprint":
                    candidate_population_fingerprint(source),
                "decision_information_fingerprint":
                    decision_information_fingerprint(source),
                "selected_portfolio_fingerprint":
                    candidate_population_fingerprint(selected),
                **_settle(selected, inputs.settlements),
            })
        outputs[rule] = {
            "aggregate": {
                **_settle(all_selected, inputs.settlements),
                "evaluation_days": len(daily),
                "evaluated_races": sum(row["evaluated_races"] for row in daily),
                "candidate_tickets": len(inputs.decision_candidates),
                "selected_portfolio_fingerprint":
                    candidate_population_fingerprint(all_selected),
            },
            "daily": daily,
        }
    return {
        "model_name": MODEL_NAME,
        "status": "research_diagnostic_real_betting_disabled",
        "real_betting_enabled": False,
        "daily_budget_yen": daily_budget_yen,
        "stake_unit_yen": STAKE_YEN,
        "maximum_tickets_per_day": min(
            MAX_DAILY_TICKETS, daily_budget_yen // STAKE_YEN
        ),
        "zero_purchase_allowed": True,
        "closing_forecast_source":
            "callback_mapping" if closing_forecasts is not None else "race_field_fallback",
        "probability_source":
            "probability_lcb_artifact_callback" if probability_lcb is not None
            else "direct_fixed_band_fallback",
        "fixed_divergence_band": {
            "lower_inclusive": BAND_LOW, "upper_exclusive": BAND_HIGH,
            "center": BAND_CENTER,
        },
        "candidate_population_tickets": len(inputs.decision_candidates),
        "candidate_population_fingerprint":
            candidate_population_fingerprint(inputs.decision_candidates),
        "decision_information_fingerprint":
            decision_information_fingerprint(inputs.decision_candidates),
        "settlement_separation": {
            "decision_fields_contain_result_or_payout": False,
            "settlements_joined_after_portfolio_selection": True,
        },
        "rejected_races_by_reason": dict(inputs.rejected_races_by_reason),
        "rules": outputs,
        "research_warning": (
            "Retrospective ranking diagnostic only. A best rule chosen after "
            "observing returns is not promotion evidence; pre-register it and "
            "evaluate it on untouched prospective dates."
        ),
        "post_hoc_best_rule_is_promotion_evidence": False,
    }
