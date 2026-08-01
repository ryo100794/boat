from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..bankroll_bootstrap import bootstrap_daily_roi
from ..listwise.market_calibration import (
    normalized_market_probabilities,
    write_json_atomic,
)
from ..listwise.paired_bootstrap import paired_mean_bootstrap


JST = timezone(timedelta(hours=9))
V21_STRATEGY_NAME = "v21_triple_head_t300"
COMBINATIONS = tuple(
    f"{a}-{b}-{c}"
    for a in range(1, 7)
    for b in range(1, 7)
    if b != a
    for c in range(1, 7)
    if c not in (a, b)
)
COMBINATION_SET = frozenset(COMBINATIONS)
EPSILON = 1e-15


@dataclass(frozen=True)
class V21ProspectiveEvidenceConfig:
    start_date: str | date
    model_key: str
    through_date: str | date | None = None
    expected_model_hash: str | None = None
    expected_strategy_name: str = V21_STRATEGY_NAME
    diagnostic_key: str = "v21_triple_head"
    evidence_kind: str = "v21_frozen_identity_fully_unseen_prospective"
    max_decision_delay_seconds: float = 90.0
    bootstrap_samples: int = 20_000
    bootstrap_seed: int = 20260731
    minimum_clean_days: int = 30
    minimum_races: int = 1_000
    minimum_tickets: int = 200
    minimum_effective_hits: float = 20.0
    minimum_profitable_day_fraction: float = 0.60
    minimum_market_confidence: float = 0.95
    minimum_selected_probability_calibration_pvalue: float = 0.05

    def __post_init__(self) -> None:
        start = _date_text(self.start_date, "start_date")
        if self.through_date is not None and _date_text(
            self.through_date, "through_date"
        ) < start:
            raise ValueError("through_date must not precede start_date")
        if not str(self.model_key).strip():
            raise ValueError("model_key must not be empty")
        if not str(self.expected_strategy_name).strip():
            raise ValueError("expected_strategy_name must not be empty")
        if not str(self.diagnostic_key).strip():
            raise ValueError("diagnostic_key must not be empty")
        if not str(self.evidence_kind).strip():
            raise ValueError("evidence_kind must not be empty")
        if self.expected_model_hash is not None and not str(
            self.expected_model_hash
        ).strip():
            raise ValueError("expected_model_hash must not be empty")
        if not math.isfinite(self.max_decision_delay_seconds) or (
            self.max_decision_delay_seconds <= 0
        ):
            raise ValueError("max_decision_delay_seconds must be positive")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        for value, name in (
            (self.minimum_clean_days, "minimum_clean_days"),
            (self.minimum_races, "minimum_races"),
            (self.minimum_tickets, "minimum_tickets"),
        ):
            if isinstance(value, bool) or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.minimum_effective_hits <= 0:
            raise ValueError("minimum_effective_hits must be positive")
        for value, name in (
            (self.minimum_profitable_day_fraction, "minimum_profitable_day_fraction"),
            (self.minimum_market_confidence, "minimum_market_confidence"),
            (
                self.minimum_selected_probability_calibration_pvalue,
                "minimum_selected_probability_calibration_pvalue",
            ),
        ):
            if not 0 <= float(value) <= 1:
                raise ValueError(f"{name} must be between zero and one")


def _date_text(value: object, field: str = "race_date") -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    try:
        return date.fromisoformat(str(value or "").strip()).isoformat()
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO date") from exc


def _datetime(value: object, field: str) -> datetime:
    if isinstance(value, datetime):
        result = value
    else:
        try:
            result = datetime.fromisoformat(
                str(value or "").strip().replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ValueError(f"{field} must be an ISO datetime") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return result


def _mapping(value: object, field: str) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field} must be valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _sequence(value: object, field: str) -> list[Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field} must be valid JSON") from exc
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field} must be a sequence")
    return list(value)


def _number(value: object, field: str, *, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{field} must be finite{' and positive' if positive else ''}")
    return result


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if result < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return result


def _combination(value: object, field: str) -> str:
    result = str(value or "").strip().replace("=", "-").replace(" ", "")
    if result not in COMBINATION_SET:
        raise ValueError(f"{field} must be a valid trifecta combination")
    return result


def _probabilities(value: object, field: str) -> dict[str, float]:
    raw = _mapping(value, field)
    if set(raw) != COMBINATION_SET:
        raise ValueError(f"{field} must contain exactly 120 combinations")
    result = {key: _number(item, f"{field}.{key}") for key, item in raw.items()}
    if any(item < 0 for item in result.values()) or not math.isclose(
        sum(result.values()), 1.0, rel_tol=0, abs_tol=1e-8
    ):
        raise ValueError(f"{field} must be a normalized probability vector")
    return result


def _rows(values: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in values:
        if not isinstance(row, Mapping):
            raise ValueError("input rows must be mappings")
        result.append(dict(row))
    return result


def _source_index(rows: Iterable[Mapping[str, Any]]) -> tuple[dict[int, Any], set[int]]:
    grouped: dict[int, dict[str, Any]] = {}
    duplicates: set[int] = set()
    for row in _rows(rows):
        snapshot_id = _integer(row.get("snapshot_id"), "snapshot_id", minimum=1)
        item = grouped.setdefault(
            snapshot_id, {"race_ids": set(), "captured_at": set(), "odds": {}}
        )
        item["race_ids"].add(str(row.get("race_id") or ""))
        item["captured_at"].add(str(row.get("captured_at") or ""))
        combination = str(row.get("combination") or "").strip()
        if combination in item["odds"]:
            duplicates.add(snapshot_id)
        item["odds"][combination] = row.get("odds")
    return grouped, duplicates


def _payout_index(rows: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], set[str]]:
    result: dict[str, tuple[str, int]] = {}
    duplicates: set[str] = set()
    for row in _rows(rows):
        race_id = str(row.get("race_id") or "").strip()
        try:
            value = (
                _combination(row.get("combination"), "payout.combination"),
                _integer(row.get("payout_yen"), "payout.payout_yen"),
            )
        except ValueError:
            continue
        if race_id in result:
            duplicates.add(race_id)
        result[race_id] = value
    return result, duplicates


def _frozen_identity(
    decisions: Sequence[Mapping[str, Any]], config: V21ProspectiveEvidenceConfig
) -> tuple[tuple[str, str] | None, set[tuple[str, str]], bool]:
    ordered = sorted(
        decisions,
        key=lambda row: (
            str(row.get("race_date") or ""),
            str(row.get("target_t300_at") or ""),
            str(row.get("race_id") or ""),
        ),
    )
    observed = {
        (str(row.get("model_hash") or ""), str(row.get("strategy_name") or ""))
        for row in ordered
    }
    if config.expected_model_hash is not None:
        expected = (str(config.expected_model_hash), config.expected_strategy_name)
    elif ordered:
        expected = (
            str(ordered[0].get("model_hash") or ""),
            config.expected_strategy_name,
        )
    else:
        expected = None
    return expected, observed, bool(expected) and observed == {expected}


def _audit_decision(
    row: Mapping[str, Any],
    *,
    expected_identity: tuple[str, str] | None,
    sources: Mapping[int, Mapping[str, Any]],
    duplicate_sources: set[int],
    max_delay: float,
    diagnostic_key: str,
) -> dict[str, Any]:
    decision_id = _integer(row.get("decision_id"), "decision_id", minimum=1)
    race_id = str(row.get("race_id") or "").strip()
    identity = (str(row.get("model_hash") or ""), str(row.get("strategy_name") or ""))
    if not race_id or expected_identity is None or identity != expected_identity:
        raise ValueError("model identity differs from frozen identity")
    target = _datetime(row.get("target_t300_at"), "target_t300_at")
    completed = _datetime(
        row.get("decision_completed_at") or row.get("created_at"), "decision_completed_at"
    )
    delay = (completed - target).total_seconds()
    if not 0 <= delay < max_delay:
        raise ValueError("decision delay is outside [0, max_delay_seconds)")
    snapshot_id = _integer(row.get("source_snapshot_id"), "source_snapshot_id", minimum=1)
    source = sources.get(snapshot_id)
    if source is None or snapshot_id in duplicate_sources:
        raise ValueError("source snapshot is missing or duplicated")
    if source["race_ids"] != {race_id} or len(source["captured_at"]) != 1:
        raise ValueError("source snapshot identity is inconsistent")
    captured = _datetime(next(iter(source["captured_at"])), "source captured_at")
    if captured != _datetime(row.get("source_captured_at"), "source_captured_at"):
        raise ValueError("decision and source captured_at differ")
    if captured > target:
        raise ValueError("source snapshot was captured after T300")
    odds = {
        _combination(key, "source combination"): _number(value, "source odds", positive=True)
        for key, value in source["odds"].items()
    }
    if set(odds) != COMBINATION_SET:
        raise ValueError("source snapshot must contain exactly 120 odds")
    probabilities = _probabilities(row.get("probabilities"), "probabilities")
    diagnostics = _mapping(row.get("diagnostics"), "diagnostics")
    model_diagnostic = _mapping(diagnostics.get(diagnostic_key), diagnostic_key)
    ranking = _probabilities(
        model_diagnostic.get("ranking_probabilities"), "ranking_probabilities"
    )
    if (
        model_diagnostic.get("decision_features") != "t300_or_earlier"
        or model_diagnostic.get("outer_result_used") is not False
        or model_diagnostic.get("outer_payout_used") is not False
        or model_diagnostic.get("real_betting_enabled") is not False
        or _integer(
            model_diagnostic.get("source_snapshot_id"),
            "diagnostic source_snapshot_id",
            minimum=1,
        )
        != snapshot_id
    ):
        raise ValueError("model information boundary or shadow-only flag is invalid")
    selected = []
    for candidate in _sequence(row.get("selected_candidates"), "selected_candidates"):
        candidate = _mapping(candidate, "selected_candidate")
        stake = _integer(candidate.get("stake_yen"), "candidate.stake_yen")
        if stake <= 0:
            raise ValueError("selected candidate stake must be positive")
        selected.append(
            {
                "combination": _combination(candidate.get("combination"), "candidate.combination"),
                "stake_yen": stake,
            }
        )
    total_stake = _integer(row.get("total_stake_yen"), "total_stake_yen")
    if sum(item["stake_yen"] for item in selected) != total_stake:
        raise ValueError("selected stakes do not equal decision total stake")
    return {
        "decision_id": decision_id,
        "race_id": race_id,
        "probabilities": probabilities,
        "ranking": ranking,
        "market": normalized_market_probabilities(odds),
        "selected": selected,
        "stake_yen": total_stake,
        "delay_seconds": delay,
    }


def _evaluate_settlement(
    decision: Mapping[str, Any],
    settlement: Mapping[str, Any],
    official: tuple[str, int],
) -> dict[str, Any]:
    actual = _combination(settlement.get("actual_combination"), "actual_combination")
    payout = _integer(settlement.get("payout_yen_per_100"), "payout_yen_per_100")
    if official != (actual, payout):
        raise ValueError("settlement differs from official payout")
    stake = _integer(settlement.get("stake_yen"), "stake_yen")
    returned = _integer(settlement.get("return_yen"), "return_yen")
    profit = _integer(settlement.get("profit_yen"), "profit_yen", minimum=-10**12)
    expected = sum(
        item["stake_yen"] * payout // 100
        for item in decision["selected"]
        if item["combination"] == actual
    )
    if stake != decision["stake_yen"] or returned != expected or profit != returned - stake:
        raise ValueError("settlement ledger is inconsistent")
    return {"actual": actual, "stake_yen": stake, "return_yen": returned}


def _poisson_binomial_cdf(probabilities: Sequence[float], observed: int) -> float:
    """Return P(X <= observed) for independent race-level selected outcomes."""
    if observed < 0:
        return 0.0
    if not probabilities or observed >= len(probabilities):
        return 1.0
    distribution = [1.0] + [0.0] * observed
    for probability in probabilities:
        for hits in range(observed, 0, -1):
            distribution[hits] = (
                distribution[hits] * (1.0 - probability)
                + distribution[hits - 1] * probability
            )
        distribution[0] *= 1.0 - probability
    return min(max(sum(distribution), 0.0), 1.0)


def aggregate_v21_prospective_evidence(
    *,
    config: V21ProspectiveEvidenceConfig,
    races: Iterable[Mapping[str, Any]],
    decisions: Iterable[Mapping[str, Any]],
    settlements: Iterable[Mapping[str, Any]],
    source_odds: Iterable[Mapping[str, Any]],
    payouts: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Purely aggregate frozen V21 evidence; outcomes never enter decision inputs."""
    start = _date_text(config.start_date, "start_date")
    through = _date_text(config.through_date, "through_date") if config.through_date else None

    def in_period(row: Mapping[str, Any]) -> bool:
        try:
            day = _date_text(row.get("race_date"))
        except ValueError:
            return False
        return day >= start and (through is None or day <= through)

    race_rows = [row for row in _rows(races) if in_period(row)]
    decision_rows = [
        row
        for row in _rows(decisions)
        if in_period(row) and str(row.get("model_key") or "") == config.model_key
    ]
    race_days: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in race_rows:
        if _integer(row.get("lane_count", 6), "lane_count") == 6:
            race_days[_date_text(row.get("race_date"))].append(row)
    decision_days: dict[str, list[dict[str, Any]]] = defaultdict(list)
    decision_ids: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in decision_rows:
        decision_days[_date_text(row.get("race_date"))].append(row)
        try:
            decision_ids[_integer(row.get("decision_id"), "decision_id", minimum=1)].append(row)
        except ValueError:
            pass
    settlement_index: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in _rows(settlements):
        try:
            settlement_index[_integer(row.get("decision_id"), "decision_id", minimum=1)].append(row)
        except ValueError:
            pass
    sources, duplicate_sources = _source_index(source_odds)
    official, duplicate_payouts = _payout_index(payouts)
    expected_identity, observed_identities, identity_fixed = _frozen_identity(
        decision_rows, config
    )

    clean: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    observations: list[dict[str, float | int]] = []
    hit_returns: list[int] = []
    selected_event_probabilities: list[float] = []
    for day in sorted(race_days):
        race_ids = [str(row.get("race_id") or "") for row in race_days[day]]
        day_decisions = decision_days.get(day, [])
        decision_races = [str(row.get("race_id") or "") for row in day_decisions]
        reasons: set[str] = set()
        details: dict[str, Any] = {}
        if len(race_ids) != len(set(race_ids)):
            reasons.add("duplicate_race_rows")
        if len(day_decisions) != len(race_ids) or set(decision_races) != set(race_ids):
            reasons.add("decision_coverage_mismatch")
            details["missing_decision_race_ids"] = sorted(set(race_ids) - set(decision_races))
            details["unexpected_decision_race_ids"] = sorted(set(decision_races) - set(race_ids))
        if len(decision_races) != len(set(decision_races)):
            reasons.add("duplicate_decisions")
        audited = []
        for row in day_decisions:
            race_id = str(row.get("race_id") or "")
            try:
                item = _audit_decision(
                    row,
                    expected_identity=expected_identity,
                    sources=sources,
                    duplicate_sources=duplicate_sources,
                    max_delay=config.max_decision_delay_seconds,
                    diagnostic_key=config.diagnostic_key,
                )
                if len(decision_ids[item["decision_id"]]) != 1:
                    raise ValueError("decision_id is not unique")
                audited.append(item)
            except (KeyError, TypeError, ValueError) as exc:
                reasons.add("decision_boundary_invalid")
                details.setdefault("decision_errors", {})[race_id] = str(exc)
        if len(audited) != len(race_ids):
            reasons.add("decision_boundary_coverage_mismatch")
        evaluated = []
        for item in audited:
            race_id = str(item["race_id"])
            rows = settlement_index.get(int(item["decision_id"]), [])
            if len(rows) != 1:
                reasons.add("settlement_coverage_mismatch")
                details.setdefault("settlement_count", {})[race_id] = len(rows)
                continue
            try:
                if race_id in duplicate_payouts or race_id not in official:
                    raise ValueError("official payout is missing or ambiguous")
                evaluated.append((item, _evaluate_settlement(item, rows[0], official[race_id])))
            except (TypeError, ValueError) as exc:
                reasons.add("settlement_or_payout_invalid")
                details.setdefault("settlement_errors", {})[race_id] = str(exc)
        if len(evaluated) != len(race_ids):
            reasons.add("settlement_coverage_mismatch")
        coverage = {
            "six_boat_races": len(race_ids),
            "model_decisions": len(day_decisions),
            "valid_decision_boundaries": len(audited),
            "valid_settlements": len(evaluated),
        }
        if reasons:
            excluded.append(
                {"race_date": day, "reasons": sorted(reasons), "coverage": coverage, "details": details}
            )
            continue
        stake = sum(result["stake_yen"] for _, result in evaluated)
        returned = sum(result["return_yen"] for _, result in evaluated)
        day_hits = [result["return_yen"] for _, result in evaluated if result["return_yen"] > 0]
        hit_returns.extend(day_hits)
        day_selected_probabilities = [
            sum(item["probabilities"][ticket["combination"]] for ticket in item["selected"])
            for item, _ in evaluated
            if item["selected"]
        ]
        selected_event_probabilities.extend(day_selected_probabilities)
        clean.append(
            {
                "race_date": day,
                "races": len(race_ids),
                "tickets": sum(len(item["selected"]) for item, _ in evaluated),
                "hit_tickets": sum(
                    sum(ticket["combination"] == result["actual"] for ticket in item["selected"])
                    for item, result in evaluated
                ),
                "expected_hit_tickets": sum(day_selected_probabilities),
                "expected_no_hit_probability": math.prod(
                    1.0 - probability for probability in day_selected_probabilities
                ),
                "stake_yen": stake,
                "return_yen": returned,
                "profit_yen": returned - stake,
                "roi": returned / stake if stake else None,
                "coverage": coverage,
            }
        )
        for item, result in evaluated:
            actual = result["actual"]
            model_loss = -math.log(max(EPSILON, item["probabilities"][actual]))
            market_loss = -math.log(max(EPSILON, item["market"][actual]))
            model_top5 = sorted(item["ranking"], key=lambda key: (-item["ranking"][key], key))[:5]
            market_top5 = sorted(item["market"], key=lambda key: (-item["market"][key], key))[:5]
            observations.append(
                {
                    "model_loss": model_loss,
                    "market_loss": market_loss,
                    "model_top5": int(actual in model_top5),
                    "market_top5": int(actual in market_top5),
                }
            )

    stake = sum(row["stake_yen"] for row in clean)
    returned = sum(row["return_yen"] for row in clean)
    races_count = sum(row["races"] for row in clean)
    tickets = sum(row["tickets"] for row in clean)
    largest_hit = max(hit_returns, default=0)
    hit_square_sum = sum(value * value for value in hit_returns)
    effective_hits = returned * returned / hit_square_sum if returned and hit_square_sum else 0.0
    bootstrap = (
        bootstrap_daily_roi(clean, samples=config.bootstrap_samples, seed=config.bootstrap_seed)
        if clean
        else None
    )
    profitable_fraction = (
        sum(row["profit_yen"] > 0 for row in clean) / len(clean) if clean else None
    )
    bankroll = {
        "clean_days": len(clean),
        "races": races_count,
        "tickets": tickets,
        "hit_tickets": len(hit_returns),
        "stake_yen": stake,
        "return_yen": returned,
        "profit_yen": returned - stake,
        "roi": returned / stake if stake else None,
        "largest_hit_return_yen": largest_hit if hit_returns else None,
        "return_without_largest_hit_yen": returned - largest_hit,
        "roi_without_largest_hit": (returned - largest_hit) / stake if stake else None,
        "profitable_day_fraction": profitable_fraction,
        "daily_cluster_bootstrap_roi_lower_95": bootstrap["roi_ci95_lower"] if bootstrap else None,
        "effective_hit_count": effective_hits,
        "bootstrap": bootstrap,
    }
    observed_selected_hits = len(hit_returns)
    expected_selected_hits = sum(selected_event_probabilities)
    selected_variance = sum(
        probability * (1.0 - probability)
        for probability in selected_event_probabilities
    )
    purchase_probability_calibration = {
        "selected_races": len(selected_event_probabilities),
        "observed_hits": observed_selected_hits,
        "expected_hits": expected_selected_hits,
        "observed_to_expected_hit_ratio": (
            observed_selected_hits / expected_selected_hits
            if expected_selected_hits > 0
            else None
        ),
        "expected_no_hit_probability": math.prod(
            1.0 - probability for probability in selected_event_probabilities
        ),
        "standardized_hit_residual": (
            (observed_selected_hits - expected_selected_hits)
            / math.sqrt(selected_variance)
            if selected_variance > 0
            else None
        ),
        "probability_at_most_observed_hits": _poisson_binomial_cdf(
            selected_event_probabilities, observed_selected_hits
        ),
        "method": "exact_poisson_binomial_lower_tail_over_disjoint_race_selections",
    }
    if observations:
        loss_diff = [row["model_loss"] - row["market_loss"] for row in observations]
        top5_diff = [row["model_top5"] - row["market_top5"] for row in observations]
        loss_boot = paired_mean_bootstrap(
            loss_diff, samples=config.bootstrap_samples, seed=config.bootstrap_seed + 1
        )
        top5_boot = paired_mean_bootstrap(
            top5_diff, samples=config.bootstrap_samples, seed=config.bootstrap_seed + 2
        )
        count = len(observations)
        market = {
            "races": count,
            "model_trifecta_log_loss": sum(row["model_loss"] for row in observations) / count,
            "market_trifecta_log_loss": sum(row["market_loss"] for row in observations) / count,
            "model_trifecta_top5": sum(row["model_top5"] for row in observations) / count,
            "market_trifecta_top5": sum(row["market_top5"] for row in observations) / count,
            "log_loss_difference_model_minus_market": sum(loss_diff) / count,
            "top5_difference_model_minus_market": sum(top5_diff) / count,
            "log_loss_race_bootstrap": loss_boot,
            "top5_race_bootstrap": top5_boot,
            "market_noninferiority_confidence": loss_boot["probability_less_than_zero"],
            "top5_improvement_confidence": top5_boot["probability_greater_than_zero"],
        }
    else:
        market = {
            "races": 0,
            "model_trifecta_log_loss": None,
            "market_trifecta_log_loss": None,
            "model_trifecta_top5": None,
            "market_trifecta_top5": None,
            "log_loss_difference_model_minus_market": None,
            "top5_difference_model_minus_market": None,
            "log_loss_race_bootstrap": None,
            "top5_race_bootstrap": None,
            "market_noninferiority_confidence": None,
            "top5_improvement_confidence": None,
        }
    checks = {
        "identity_fixed": identity_fixed,
        "minimum_clean_days": len(clean) >= config.minimum_clean_days,
        "minimum_races": races_count >= config.minimum_races,
        "minimum_tickets": tickets >= config.minimum_tickets,
        "minimum_effective_hits": effective_hits >= config.minimum_effective_hits,
        "roi_above_one": bankroll["roi"] is not None and bankroll["roi"] > 1,
        "largest_hit_excluded_roi_above_one": (
            bankroll["roi_without_largest_hit"] is not None and bankroll["roi_without_largest_hit"] > 1
        ),
        "daily_cluster_bootstrap_lower_95_above_one": (
            bankroll["daily_cluster_bootstrap_roi_lower_95"] is not None
            and bankroll["daily_cluster_bootstrap_roi_lower_95"] > 1
        ),
        "profitable_day_fraction": (
            profitable_fraction is not None
            and profitable_fraction >= config.minimum_profitable_day_fraction
        ),
        "market_log_loss_noninferiority_confidence": (
            market["market_noninferiority_confidence"] is not None
            and market["market_noninferiority_confidence"] >= config.minimum_market_confidence
        ),
        "market_top5_improvement_confidence": (
            market["top5_improvement_confidence"] is not None
            and market["top5_improvement_confidence"] >= config.minimum_market_confidence
        ),
        "selected_probability_not_overconfident": (
            purchase_probability_calibration[
                "probability_at_most_observed_hits"
            ]
            >= config.minimum_selected_probability_calibration_pvalue
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": 2,
        "evidence_kind": config.evidence_kind,
        "model_key": config.model_key,
        "start_date": start,
        "through_date": through,
        "model_identity": {
            "expected_model_hash": expected_identity[0] if expected_identity else None,
            "expected_strategy_name": expected_identity[1] if expected_identity else None,
            "observed": [
                {"model_hash": model_hash, "strategy_name": strategy}
                for model_hash, strategy in sorted(observed_identities)
            ],
            "fixed": identity_fixed,
        },
        "information_boundary": {
            "decision_inputs": "stored_decision_and_source_snapshot_at_t300_or_earlier",
            "outcomes_used_only_for": "settlement_and_evaluation",
            "outer_result_used_as_decision_feature": False,
            "outer_payout_used_as_decision_feature": False,
            "real_betting_enabled": False,
        },
        "daily": clean,
        "excluded_days": excluded,
        "bankroll": bankroll,
        "purchase_probability_calibration": purchase_probability_calibration,
        "market": market,
        "promotion_gate": {
            "thresholds": {
                "clean_days": config.minimum_clean_days,
                "races": config.minimum_races,
                "tickets": config.minimum_tickets,
                "effective_hits": config.minimum_effective_hits,
                "roi_strictly_above": 1.0,
                "largest_hit_excluded_roi_strictly_above": 1.0,
                "daily_cluster_bootstrap_lower_95_strictly_above": 1.0,
                "profitable_day_fraction": config.minimum_profitable_day_fraction,
                "market_confidence": config.minimum_market_confidence,
                "selected_probability_calibration_pvalue": (
                    config.minimum_selected_probability_calibration_pvalue
                ),
            },
            "checks": checks,
            "failed_checks": failed,
            "pass": not failed,
        },
    }


build_v21_prospective_evidence = aggregate_v21_prospective_evidence


def _fetch(cursor: Any) -> list[dict[str, Any]]:
    return [dict(row) for row in cursor.fetchall()]


def collect_v21_prospective_evidence(
    conn: Any,
    *,
    config: V21ProspectiveEvidenceConfig,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Read append-only PostgreSQL records and optionally publish atomic JSON."""
    if getattr(conn, "dialect", None) != "postgresql":
        raise ValueError("V21 prospective evidence requires PostgreSQL")
    through = (
        _date_text(config.through_date, "through_date")
        if config.through_date
        else datetime.now(JST).date().isoformat()
    )
    bounded = replace(config, through_date=through)
    period = (_date_text(config.start_date, "start_date"), through)
    race_rows = _fetch(
        conn.execute(
            """SELECT r.race_id, r.race_date, COUNT(DISTINCT e.lane) AS lane_count
               FROM races r LEFT JOIN entries e ON e.race_id = r.race_id
               WHERE r.race_date >= ? AND r.race_date <= ?
               GROUP BY r.race_id, r.race_date ORDER BY r.race_date, r.race_id""",
            period,
        )
    )
    params = (*period, config.model_key)
    decision_rows = _fetch(
        conn.execute(
            """SELECT decision_id, race_date, race_id, model_key, model_hash,
                      strategy_name, decision_at, decision_completed_at, created_at,
                      target_t300_at, source_snapshot_id, source_captured_at,
                      probabilities, selected_candidates,
                      diagnostics, total_stake_yen
               FROM intraday_t300_shadow_decisions
               WHERE race_date >= ? AND race_date <= ? AND model_key = ?
               ORDER BY race_date, target_t300_at, race_id, decision_id""",
            params,
        )
    )
    settlement_rows = _fetch(
        conn.execute(
            """SELECT s.decision_id, s.result_status, s.actual_combination,
                      s.payout_yen_per_100, s.stake_yen, s.return_yen, s.profit_yen
               FROM intraday_t300_shadow_settlements s
               JOIN intraday_t300_shadow_decisions d ON d.decision_id = s.decision_id
               WHERE d.race_date >= ? AND d.race_date <= ? AND d.model_key = ?
               ORDER BY s.decision_id""",
            params,
        )
    )
    odds_rows = _fetch(
        conn.execute(
            """SELECT os.snapshot_id, os.race_id, os.captured_at,
                      ot.combination, ot.odds
               FROM intraday_t300_shadow_decisions d
               JOIN odds_snapshots os ON os.snapshot_id = d.source_snapshot_id
               JOIN odds_trifecta ot ON ot.snapshot_id = os.snapshot_id
               WHERE d.race_date >= ? AND d.race_date <= ? AND d.model_key = ?
               ORDER BY os.snapshot_id, ot.combination""",
            params,
        )
    )
    payout_rows = _fetch(
        conn.execute(
            """SELECT p.race_id, p.combination, p.payout_yen
               FROM payouts p
               JOIN intraday_t300_shadow_decisions d ON d.race_id = p.race_id
               WHERE d.race_date >= ? AND d.race_date <= ? AND d.model_key = ?
                 AND p.bet_type = '3連単'
               ORDER BY p.race_id, p.combination""",
            params,
        )
    )
    result = aggregate_v21_prospective_evidence(
        config=bounded,
        races=race_rows,
        decisions=decision_rows,
        settlements=settlement_rows,
        source_odds=odds_rows,
        payouts=payout_rows,
    )
    if output_path is not None:
        write_v21_prospective_evidence_atomic(output_path, result)
    return result


load_v21_prospective_evidence = collect_v21_prospective_evidence


def write_v21_prospective_evidence_atomic(
    path: str | Path, payload: Mapping[str, Any]
) -> None:
    write_json_atomic(Path(path), dict(payload))


__all__ = [
    "V21ProspectiveEvidenceConfig",
    "aggregate_v21_prospective_evidence",
    "build_v21_prospective_evidence",
    "collect_v21_prospective_evidence",
    "load_v21_prospective_evidence",
    "write_v21_prospective_evidence_atomic",
]
