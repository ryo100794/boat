from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any, Iterable, Mapping

import numpy as np

from .closing_odds_multihorizon_v11 import normalize_labeled_checkpoints
from .odds_path_conservative_v7 import (
    MAX_TICKETS_PER_RACE,
    SAFE_EV_THRESHOLD,
    _rank_groups,
)


MODEL_NAME = "edge_conditional_probability_lcb_v13"
METHOD = "strict_prior_daily_cluster_hierarchical_probability_lcb_v13"
STATUS = "research_invalid_deprecated"
T300_LABEL = "t300"
TARGET_LOWER_QUANTILE = 0.05
DEFAULT_BOOTSTRAP_SAMPLES = 2_000
DEFAULT_SEED = 13_031
MIN_GLOBAL_DAYS = 3
GLOBAL_PRIOR_EXPECTED = 20.0
RANK_PRIOR_EXPECTED = 12.0
CELL_PRIOR_EXPECTED = 8.0
RANK_FULL_WEIGHT_EXPECTED = 20.0
CELL_FULL_WEIGHT_EXPECTED = 10.0
RANK_FULL_WEIGHT_DAYS = 8
CELL_FULL_WEIGHT_DAYS = 6

RANK_GROUPS = ("top2", "top5", "top20", "rest")
PROBABILITY_BANDS = (
    (0.005, "p_lt_005"),
    (0.010, "p_005_010"),
    (0.020, "p_010_020"),
    (0.050, "p_020_050"),
    (math.inf, "p_ge_050"),
)
LOG_DIVERGENCE_BANDS = (
    (-0.50, "d_lt_m050"),
    (0.00, "d_m050_000"),
    (0.50, "d_000_050"),
    (1.00, "d_050_100"),
    (math.inf, "d_ge_100"),
)


def _finite_positive(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and result > 0.0 else None


def t300_odds(race: Mapping[str, object]) -> dict[str, float]:
    checkpoints = normalize_labeled_checkpoints(
        race,
        as_of_offset_seconds=300,
    )
    point = checkpoints.get(T300_LABEL) or {}
    odds = point.get("odds") if isinstance(point, Mapping) else None
    if not isinstance(odds, Mapping):
        return {}
    result = {
        str(combination): value
        for combination, raw in odds.items()
        if (value := _finite_positive(raw)) is not None
    }
    return result if len(result) == 120 else {}


def _market_probabilities(odds: Mapping[str, float]) -> dict[str, float]:
    inverse = {
        str(combination): 1.0 / float(value)
        for combination, value in odds.items()
        if _finite_positive(value) is not None
    }
    total = sum(inverse.values())
    if len(inverse) != 120 or not math.isfinite(total) or total <= 0.0:
        return {}
    return {key: value / total for key, value in inverse.items()}


def _band(value: float, bands: tuple[tuple[float, str], ...]) -> str:
    for upper, label in bands:
        if value < upper:
            return label
    raise AssertionError("band definition must end in infinity")


def _condition(
    probabilities: Mapping[str, float],
    market: Mapping[str, float],
    combination: str,
) -> dict[str, Any]:
    probability = max(float(probabilities[combination]), np.finfo(float).tiny)
    market_probability = max(float(market[combination]), np.finfo(float).tiny)
    rank = _rank_groups(dict(probabilities))[combination]
    probability_band = _band(probability, PROBABILITY_BANDS)
    log_divergence = math.log(probability / market_probability)
    divergence_band = _band(log_divergence, LOG_DIVERGENCE_BANDS)
    return {
        "rank_group": rank,
        "probability_band": probability_band,
        "divergence_band": divergence_band,
        "log_model_market_divergence": log_divergence,
        "market_implied_probability": market_probability,
        "cell_key": "|".join((rank, probability_band, divergence_band)),
    }


def _node_daily_totals(
    races: Iterable[dict[str, Any]],
) -> tuple[
    list[str],
    dict[str, dict[str, list[float]]],
    dict[str, int],
]:
    totals: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(lambda: [0.0, 0.0])
    )
    race_counts: dict[str, int] = defaultdict(int)
    dates: set[str] = set()
    for race in races:
        date = str(race["race_date"])
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
            continue
        dates.add(date)
        race_counts[date] += 1
        for combination, raw_probability in probabilities.items():
            probability = float(raw_probability)
            if not math.isfinite(probability) or probability < 0.0:
                continue
            condition = _condition(probabilities, market, str(combination))
            hit = float(str(combination) == actual)
            keys = (
                "global",
                f"rank:{condition['rank_group']}",
                f"cell:{condition['cell_key']}",
            )
            for key in keys:
                totals[key][date][0] += probability
                totals[key][date][1] += hit
    return sorted(dates), totals, dict(race_counts)


def _bootstrap_lower_ratio(
    daily: Mapping[str, list[float]],
    *,
    dates: list[str],
    indices: np.ndarray,
    parent_factor: float,
    prior_expected: float,
) -> tuple[float, float, float, int]:
    expected = np.asarray(
        [float((daily.get(date) or [0.0, 0.0])[0]) for date in dates],
        dtype=np.float64,
    )
    hits = np.asarray(
        [float((daily.get(date) or [0.0, 0.0])[1]) for date in dates],
        dtype=np.float64,
    )
    sampled_expected = expected[indices].sum(axis=1)
    sampled_hits = hits[indices].sum(axis=1)
    ratios = (sampled_hits + prior_expected * parent_factor) / (
        sampled_expected + prior_expected
    )
    lower = float(np.quantile(ratios, TARGET_LOWER_QUANTILE))
    active_days = int(np.count_nonzero((expected > 0.0) | (hits > 0.0)))
    return (
        float(np.clip(lower, 0.0, 1.0)),
        float(expected.sum()),
        float(hits.sum()),
        active_days,
    )


def _shrink_weight(
    *, expected: float, days: int, full_expected: float, full_days: int
) -> float:
    return float(min(1.0, expected / full_expected, days / full_days))


def fit_edge_conditional_probability_lcb(
    crossfit_races: list[dict[str, Any]],
    *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Fit a whole-day clustered LCB from strict-prior crossfit predictions."""
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    dates, totals, race_counts = _node_daily_totals(crossfit_races)
    base = {
        "model_name": MODEL_NAME,
        "method": METHOD,
        "status": STATUS,
        "research_invalid": True,
        "deprecated": True,
        "promotion_eligible": False,
        "invalid_reason": (
            "optimistic_pseudo_counts_and_double_parent_shrinkage_do_not_form_"
            "a_valid_probability_lower_confidence_bound"
        ),
        "target_lower_quantile": TARGET_LOWER_QUANTILE,
        "bootstrap_unit": "whole_race_day",
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "training_days": len(dates),
        "training_races": sum(race_counts.values()),
        "training_dates": dates,
        "trained_through_date": dates[-1] if dates else None,
        "decision_checkpoint": T300_LABEL,
        "uses_result_for_fit_only": True,
        "uses_payout": False,
        "probability_bands": [label for _upper, label in PROBABILITY_BANDS],
        "divergence_bands": [label for _upper, label in LOG_DIVERGENCE_BANDS],
    }
    if len(dates) < MIN_GLOBAL_DAYS:
        return {
            **base,
            "ready": False,
            "reason": "insufficient_strict_prior_crossfit_days",
            "global_factor": 0.0,
            "factors": {group: 0.0 for group in RANK_GROUPS},
            "rank_nodes": {},
            "conditional_cells": {},
        }
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(dates), size=(bootstrap_samples, len(dates)))
    global_raw, global_expected, global_hits, global_days = (
        _bootstrap_lower_ratio(
            totals["global"],
            dates=dates,
            indices=indices,
            parent_factor=1.0,
            prior_expected=GLOBAL_PRIOR_EXPECTED,
        )
    )
    global_factor = min(1.0, global_raw)
    rank_nodes: dict[str, dict[str, Any]] = {}
    factors: dict[str, float] = {}
    for rank in RANK_GROUPS:
        raw, expected, hits, active_days = _bootstrap_lower_ratio(
            totals[f"rank:{rank}"],
            dates=dates,
            indices=indices,
            parent_factor=global_factor,
            prior_expected=RANK_PRIOR_EXPECTED,
        )
        weight = _shrink_weight(
            expected=expected,
            days=active_days,
            full_expected=RANK_FULL_WEIGHT_EXPECTED,
            full_days=RANK_FULL_WEIGHT_DAYS,
        )
        factor = min(1.0, weight * raw + (1.0 - weight) * global_factor)
        factors[rank] = factor
        rank_nodes[rank] = {
            "raw_lower_factor": raw,
            "factor": factor,
            "parent_factor": global_factor,
            "shrinkage_weight": weight,
            "expected_hits": expected,
            "observed_hits": hits,
            "active_days": active_days,
        }
    conditional_cells: dict[str, dict[str, Any]] = {}
    for node_key in sorted(key for key in totals if key.startswith("cell:")):
        cell_key = node_key.removeprefix("cell:")
        rank = cell_key.split("|", 1)[0]
        parent_factor = factors.get(rank, global_factor)
        raw, expected, hits, active_days = _bootstrap_lower_ratio(
            totals[node_key],
            dates=dates,
            indices=indices,
            parent_factor=parent_factor,
            prior_expected=CELL_PRIOR_EXPECTED,
        )
        weight = _shrink_weight(
            expected=expected,
            days=active_days,
            full_expected=CELL_FULL_WEIGHT_EXPECTED,
            full_days=CELL_FULL_WEIGHT_DAYS,
        )
        factor = min(1.0, weight * raw + (1.0 - weight) * parent_factor)
        conditional_cells[cell_key] = {
            "raw_lower_factor": raw,
            "factor": factor,
            "parent_rank": rank,
            "parent_factor": parent_factor,
            "shrinkage_weight": weight,
            "expected_hits": expected,
            "observed_hits": hits,
            "active_days": active_days,
            "resolution": (
                "conditional_cell"
                if weight >= 1.0
                else "conditional_cell_shrunk_to_rank"
                if weight > 0.0
                else "rank_fallback"
            ),
        }
    return {
        **base,
        "ready": True,
        "reason": None,
        "global_factor": global_factor,
        "global_node": {
            "raw_lower_factor": global_raw,
            "factor": global_factor,
            "expected_hits": global_expected,
            "observed_hits": global_hits,
            "active_days": global_days,
        },
        "factors": factors,
        "rank_nodes": rank_nodes,
        "conditional_cells": conditional_cells,
    }


def probability_lower_bound_details(
    race: Mapping[str, Any],
    combination: str,
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    probabilities = race.get("model_probabilities") or {}
    raw_probability = float(probabilities.get(combination, 0.0) or 0.0)
    if not artifact.get("ready") or raw_probability <= 0.0:
        return {
            "probability": 0.0,
            "raw_probability": max(0.0, raw_probability),
            "factor": 0.0,
            "resolution": "not_ready",
        }
    if artifact.get("method") == "strict_prior_daily_cluster_probability_ratio_lcb_v14":
        from .edge_conditional_probability_lcb_v14 import (
            probability_lower_bound_details_v14,
        )

        return probability_lower_bound_details_v14(
            race, combination, artifact
        )
    if artifact.get("method") != METHOD:
        rank = _rank_groups(dict(probabilities)).get(combination)
        factor = float((artifact.get("factors") or {}).get(rank, 0.0))
        return {
            "probability": raw_probability * factor,
            "raw_probability": raw_probability,
            "factor": factor,
            "rank_group": rank,
            "resolution": "legacy_rank",
        }
    odds = t300_odds(race)
    market = _market_probabilities(odds)
    if combination not in market or set(probabilities) != set(market):
        return {
            "probability": 0.0,
            "raw_probability": raw_probability,
            "factor": 0.0,
            "resolution": "missing_complete_t300_market",
        }
    condition = _condition(probabilities, market, combination)
    cell = (artifact.get("conditional_cells") or {}).get(condition["cell_key"])
    rank_node = (artifact.get("rank_nodes") or {}).get(condition["rank_group"])
    if isinstance(cell, Mapping):
        factor = float(cell.get("factor", 0.0))
        resolution = str(cell.get("resolution") or "conditional_cell")
    elif isinstance(rank_node, Mapping):
        factor = float(rank_node.get("factor", 0.0))
        resolution = "rank_fallback"
    else:
        factor = float(artifact.get("global_factor", 0.0))
        resolution = "global_fallback"
    factor = float(np.clip(factor, 0.0, 1.0))
    return {
        **condition,
        "probability": raw_probability * factor,
        "raw_probability": raw_probability,
        "factor": factor,
        "resolution": resolution,
    }


def conditional_calibration_metrics(
    races: list[dict[str, Any]],
    *,
    closing_forecasts: Mapping[str, Mapping[str, float]],
    probability_lcb: Mapping[str, Any],
) -> dict[str, Any]:
    """Score raw-high-EV candidates after the purchase boundary is frozen."""
    candidates: list[dict[str, Any]] = []
    missing_t300_races = 0
    for race in races:
        race_id = str(race["race_id"])
        probabilities = race.get("model_probabilities") or {}
        closing = closing_forecasts.get(race_id) or {}
        if len(t300_odds(race)) != 120:
            missing_t300_races += 1
            continue
        if len(probabilities) != 120 or len(closing) != 120:
            continue
        race_candidates = []
        for combination, closing_lower in closing.items():
            raw_probability = float(probabilities[combination])
            raw_ev = raw_probability * float(closing_lower)
            if raw_ev < SAFE_EV_THRESHOLD:
                continue
            detail = probability_lower_bound_details(
                race, str(combination), probability_lcb
            )
            race_candidates.append({
                **detail,
                "race_id": race_id,
                "race_date": str(race["race_date"]),
                "combination": str(combination),
                "raw_estimated_ev": raw_ev,
                "adjusted_estimated_ev": (
                    float(detail["probability"]) * float(closing_lower)
                ),
                "hit": int(str(combination) == str(race["actual_combination"])),
            })
        race_candidates.sort(
            key=lambda row: (
                -float(row["raw_estimated_ev"]),
                -float(row["raw_probability"]),
                str(row["combination"]),
            )
        )
        candidates.extend(race_candidates[:MAX_TICKETS_PER_RACE])
    by_cell: dict[str, dict[str, float]] = defaultdict(
        lambda: {"candidate_count": 0.0, "raw_expected_hits": 0.0,
                 "adjusted_expected_hits": 0.0, "observed_hits": 0.0}
    )
    for row in candidates:
        cell_key = str(row.get("cell_key") or row.get("rank_group") or "unknown")
        cell = by_cell[cell_key]
        cell["candidate_count"] += 1
        cell["raw_expected_hits"] += float(row["raw_probability"])
        cell["adjusted_expected_hits"] += float(row["probability"])
        cell["observed_hits"] += int(row["hit"])
    raw_expected = sum(float(row["raw_probability"]) for row in candidates)
    adjusted_expected = sum(float(row["probability"]) for row in candidates)
    observed = sum(int(row["hit"]) for row in candidates)
    raw_over = max(0.0, raw_expected - observed)
    adjusted_over = max(0.0, adjusted_expected - observed)
    condition_rows = []
    for cell_key, values in sorted(by_cell.items()):
        row = {"cell_key": cell_key, **values}
        row["candidate_count"] = int(row["candidate_count"])
        row["observed_hits_to_adjusted_predicted_hits_ratio"] = (
            float(row["observed_hits"]) / float(row["adjusted_expected_hits"])
            if float(row["adjusted_expected_hits"]) > 0.0
            else None
        )
        condition_rows.append(row)
    return {
        "status": STATUS,
        "research_invalid": True,
        "deprecated": True,
        "promotion_metric_valid": False,
        "evaluation_days": len({str(race["race_date"]) for race in races}),
        "candidate_count": len(candidates),
        "raw_expected_hits": raw_expected,
        "adjusted_expected_hits": adjusted_expected,
        "observed_hits": observed,
        "raw_overprediction_hits": raw_over,
        "adjusted_overprediction_hits": adjusted_over,
        "overprediction_reduction_hits": raw_over - adjusted_over,
        "observed_hits_to_adjusted_predicted_hits_ratio": (
            observed / adjusted_expected if adjusted_expected > 0.0 else None
        ),
        "daily_lower_bound_covered": (
            observed + 1e-12 >= adjusted_expected if candidates else None
        ),
        "missing_t300_races": missing_t300_races,
        "conditions": condition_rows,
    }


def aggregate_conditional_calibration_metrics(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    values = [dict(row) for row in rows if isinstance(row, Mapping)]
    candidates = sum(int(row.get("candidate_count") or 0) for row in values)
    raw_expected = sum(float(row.get("raw_expected_hits") or 0.0) for row in values)
    adjusted_expected = sum(
        float(row.get("adjusted_expected_hits") or 0.0) for row in values
    )
    observed = sum(float(row.get("observed_hits") or 0.0) for row in values)
    eligible_coverage = [
        bool(row["daily_lower_bound_covered"])
        for row in values
        if row.get("daily_lower_bound_covered") is not None
    ]
    raw_over = max(0.0, raw_expected - observed)
    adjusted_over = max(0.0, adjusted_expected - observed)
    return {
        "status": STATUS,
        "research_invalid": True,
        "deprecated": True,
        "promotion_metric_valid": False,
        "evaluation_days": len(values),
        "coverage_days": len(eligible_coverage),
        "daily_lower_bound_coverage": (
            sum(eligible_coverage) / len(eligible_coverage)
            if eligible_coverage
            else None
        ),
        "candidate_count": candidates,
        "raw_expected_hits": raw_expected,
        "adjusted_expected_hits": adjusted_expected,
        "observed_hits": observed,
        "raw_overprediction_hits": raw_over,
        "adjusted_overprediction_hits": adjusted_over,
        "overprediction_reduction_hits": raw_over - adjusted_over,
        "relative_overprediction_reduction": (
            (raw_over - adjusted_over) / raw_over if raw_over > 0.0 else None
        ),
        "observed_hits_to_adjusted_predicted_hits_ratio": (
            observed / adjusted_expected if adjusted_expected > 0.0 else None
        ),
        "missing_t300_races": sum(
            int(row.get("missing_t300_races") or 0) for row in values
        ),
    }


def artifact_fingerprint(artifact: Mapping[str, Any]) -> str:
    payload = repr(sorted((artifact.get("conditional_cells") or {}).items()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
