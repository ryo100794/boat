from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping

from .edge_conditional_probability_lcb_v13 import (
    LOG_DIVERGENCE_BANDS,
    _band,
    _market_probabilities,
    t300_odds,
)


STAKE_PER_TICKET_YEN = 100


def strict_prior_divergence_band_metrics(
    races: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Settlement-only diagnostics for strict-prior model/T300 divergence."""
    aggregates: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "race_ids": set(),
            "tickets": 0,
            "sum_predicted_probability": 0.0,
            "hits": 0,
            "stake_yen": 0,
            "return_yen": 0,
        }
    )
    evaluated_races = 0
    missing_t300_races = 0
    for race in races:
        probabilities = race.get("model_probabilities") or {}
        odds = t300_odds(race)
        market = _market_probabilities(odds)
        actual = str(race.get("actual_combination") or "")
        if (
            len(probabilities) != 120
            or len(market) != 120
            or set(probabilities) != set(market)
            or actual not in probabilities
        ):
            missing_t300_races += 1
            continue
        evaluated_races += 1
        payout = int(race.get("actual_payout_yen") or 0)
        race_id = str(race["race_id"])
        for combination, raw_probability in probabilities.items():
            probability = float(raw_probability)
            market_probability = float(market[combination])
            if (
                not math.isfinite(probability)
                or probability < 0.0
                or market_probability <= 0.0
            ):
                continue
            divergence = math.log(
                max(probability, float.fromhex("0x1.0p-1022"))
                / market_probability
            )
            label = _band(divergence, LOG_DIVERGENCE_BANDS)
            row = aggregates[label]
            row["race_ids"].add(race_id)
            row["tickets"] += 1
            row["sum_predicted_probability"] += probability
            row["stake_yen"] += STAKE_PER_TICKET_YEN
            if str(combination) == actual:
                row["hits"] += 1
                row["return_yen"] += payout
    bands = []
    for _upper, label in LOG_DIVERGENCE_BANDS:
        source = aggregates[label]
        expected = float(source["sum_predicted_probability"])
        hits = int(source["hits"])
        stake = int(source["stake_yen"])
        returned = int(source["return_yen"])
        bands.append({
            "divergence_band": label,
            "unique_races_in_band": len(source["race_ids"]),
            "race_count_semantics": "unique_races_with_at_least_one_ticket_in_this_band",
            "tickets": int(source["tickets"]),
            "sum_predicted_probability": expected,
            "hits": hits,
            "observed_hits_to_predicted_hits_ratio": (
                hits / expected if expected > 0.0 else None
            ),
            "stake_yen": stake,
            "return_yen": returned,
            "actual_payout_roi": returned / stake if stake > 0 else None,
        })
    return {
        "definition": (
            "log(model_probability / normalized_T300_market_probability)"
        ),
        "prediction_boundary": "strict_prior_whole_day_T300",
        "closing_odds_used_as_feature": False,
        "result_and_payout_usage": "settlement_diagnostic_only",
        "stake_per_ticket_yen": STAKE_PER_TICKET_YEN,
        "evaluated_races": evaluated_races,
        "missing_t300_races": missing_t300_races,
        "bands": bands,
    }


def aggregate_strict_prior_divergence_band_metrics(
    metrics: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [dict(item) for item in metrics if isinstance(item, Mapping)]
    totals: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "race_band_memberships": 0,
            "tickets": 0,
            "sum_predicted_probability": 0.0,
            "hits": 0,
            "stake_yen": 0,
            "return_yen": 0,
        }
    )
    for metric in rows:
        for band in metric.get("bands") or []:
            label = str(band["divergence_band"])
            target = totals[label]
            target["race_band_memberships"] += int(
                band.get("unique_races_in_band") or 0
            )
            for key in (
                "tickets", "sum_predicted_probability", "hits", "stake_yen",
                "return_yen",
            ):
                target[key] += band.get(key) or 0
    bands = []
    for _upper, label in LOG_DIVERGENCE_BANDS:
        source = totals[label]
        expected = float(source["sum_predicted_probability"])
        hits = int(source["hits"])
        stake = int(source["stake_yen"])
        returned = int(source["return_yen"])
        bands.append({
            "divergence_band": label,
            "race_band_memberships": int(source["race_band_memberships"]),
            "race_count_semantics": (
                "sum_of_per_day_unique_races_in_band_not_additive_across_bands"
            ),
            "tickets": int(source["tickets"]),
            "sum_predicted_probability": expected,
            "hits": hits,
            "observed_hits_to_predicted_hits_ratio": (
                hits / expected if expected > 0.0 else None
            ),
            "stake_yen": stake,
            "return_yen": returned,
            "actual_payout_roi": returned / stake if stake > 0 else None,
        })
    return {
        "definition": (
            "log(model_probability / normalized_T300_market_probability)"
        ),
        "prediction_boundary": "strict_prior_whole_day_T300",
        "closing_odds_used_as_feature": False,
        "result_and_payout_usage": "settlement_diagnostic_only",
        "stake_per_ticket_yen": STAKE_PER_TICKET_YEN,
        "evaluation_days": len(rows),
        "evaluated_races": sum(int(row.get("evaluated_races") or 0) for row in rows),
        "missing_t300_races": sum(
            int(row.get("missing_t300_races") or 0) for row in rows
        ),
        "bands": bands,
    }
