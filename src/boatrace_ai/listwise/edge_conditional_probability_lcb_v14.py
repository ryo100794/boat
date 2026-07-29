from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any, Iterable, Mapping

import numpy as np

from .closing_odds_multihorizon_v11 import normalize_labeled_checkpoints
from .edge_conditional_probability_lcb_v13 import (
    PROBABILITY_BANDS,
    RANK_GROUPS,
    _band,
    _finite_positive,
    _market_probabilities,
)
from .odds_path_conservative_v7 import _rank_groups


MODEL_NAME = "edge_conditional_probability_lcb_v14"
METHOD = "strict_prior_daily_cluster_probability_ratio_lcb_v14"
T300_LABEL = "t300"
TARGET_LOWER_QUANTILE = 0.05
DEFAULT_BOOTSTRAP_SAMPLES = 2_000
DEFAULT_SEED = 14_031
MIN_TRAINING_DAYS = 3
MIN_ACTIVE_NODE_DAYS = 3
REGISTERED_DIVERGENCE_LOWER = 0.5
REGISTERED_DIVERGENCE_UPPER = 1.0
REGISTERED_DIVERGENCE_LABEL = "d_050_100"


def _normalized_market(values: Mapping[str, object]) -> dict[str, float]:
    odds = {
        str(key): value
        for key, raw in values.items()
        if (value := _finite_positive(raw)) is not None
    }
    return _market_probabilities(odds) if len(odds) == 120 else {}


def t300_snapshot_consistency(race: Mapping[str, object]) -> dict[str, Any]:
    """Verify that V8 market offsets and V12 features use the same T300 row."""
    checkpoints = normalize_labeled_checkpoints(
        race,
        as_of_offset_seconds=300,
    )
    point = checkpoints.get(T300_LABEL) or {}
    point_odds = point.get("odds") if isinstance(point, Mapping) else None
    if not isinstance(point_odds, Mapping):
        return {"consistent": False, "reason": "missing_complete_t300_checkpoint"}
    checkpoint_market = _normalized_market(point_odds)
    source_market = race.get("market_probabilities")
    if not isinstance(source_market, Mapping) or len(source_market) != 120:
        return {"consistent": False, "reason": "missing_v8_market_probabilities"}
    try:
        source = {
            str(key): float(value) for key, value in source_market.items()
        }
    except (TypeError, ValueError, OverflowError):
        return {"consistent": False, "reason": "invalid_v8_market_probabilities"}
    if (
        len(checkpoint_market) != 120
        or set(source) != set(checkpoint_market)
        or not all(math.isfinite(value) and value > 0.0 for value in source.values())
    ):
        return {"consistent": False, "reason": "incomplete_t300_market_mapping"}
    source_total = sum(source.values())
    if not math.isfinite(source_total) or source_total <= 0.0:
        return {"consistent": False, "reason": "invalid_v8_market_probability_mass"}
    source = {key: value / source_total for key, value in source.items()}

    root_snapshot_id = race.get("snapshot_id")
    point_snapshot_id = point.get("snapshot_id")
    if (
        root_snapshot_id not in (None, "")
        and point_snapshot_id not in (None, "")
        and str(root_snapshot_id) != str(point_snapshot_id)
    ):
        return {
            "consistent": False,
            "reason": "t300_snapshot_id_mismatch",
            "v8_snapshot_id": root_snapshot_id,
            "checkpoint_snapshot_id": point_snapshot_id,
        }
    maximum_probability_difference = max(
        abs(source[key] - checkpoint_market[key]) for key in source
    )
    if maximum_probability_difference > 1e-10:
        return {
            "consistent": False,
            "reason": "t300_market_probability_mismatch",
            "maximum_probability_difference": maximum_probability_difference,
        }
    root_odds = race.get("odds")
    if isinstance(root_odds, Mapping) and len(root_odds) == 120:
        try:
            maximum_odds_difference = max(
                abs(float(root_odds[key]) - float(point_odds[key]))
                for key in checkpoint_market
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            return {"consistent": False, "reason": "t300_odds_mapping_mismatch"}
        if maximum_odds_difference > 1e-9:
            return {
                "consistent": False,
                "reason": "t300_odds_value_mismatch",
                "maximum_odds_difference": maximum_odds_difference,
            }
    return {
        "consistent": True,
        "reason": None,
        "market_probabilities": checkpoint_market,
        "snapshot_id": point_snapshot_id,
        "maximum_probability_difference": maximum_probability_difference,
    }


def _condition(
    probabilities: Mapping[str, float],
    market: Mapping[str, float],
    combination: str,
) -> dict[str, Any]:
    probability = max(float(probabilities[combination]), np.finfo(float).tiny)
    market_probability = max(float(market[combination]), np.finfo(float).tiny)
    rank_group = _rank_groups(dict(probabilities))[combination]
    probability_band = _band(probability, PROBABILITY_BANDS)
    divergence = math.log(probability / market_probability)
    return {
        "rank_group": rank_group,
        "probability_band": probability_band,
        "log_model_market_divergence": divergence,
        "market_implied_probability": market_probability,
        "in_registered_divergence_band": (
            REGISTERED_DIVERGENCE_LOWER
            <= divergence
            < REGISTERED_DIVERGENCE_UPPER
        ),
        "cell_key": "|".join((
            rank_group,
            probability_band,
            REGISTERED_DIVERGENCE_LABEL,
        )),
    }


def _daily_node_totals(
    races: Iterable[dict[str, Any]],
) -> tuple[
    list[str],
    dict[str, dict[str, list[float]]],
    dict[str, int],
    dict[str, int],
]:
    totals: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(lambda: [0.0, 0.0])
    )
    race_counts: dict[str, int] = defaultdict(int)
    rejection_reasons: dict[str, int] = defaultdict(int)
    dates: set[str] = set()
    for race in races:
        consistency = t300_snapshot_consistency(race)
        if not consistency["consistent"]:
            rejection_reasons[str(consistency["reason"])] += 1
            continue
        probabilities = race.get("model_probabilities") or {}
        market = consistency["market_probabilities"]
        actual = str(race.get("actual_combination") or "")
        if (
            len(probabilities) != 120
            or set(probabilities) != set(market)
            or actual not in probabilities
        ):
            rejection_reasons["incomplete_probability_or_result_mapping"] += 1
            continue
        race_date = str(race["race_date"])
        dates.add(race_date)
        race_counts[race_date] += 1
        for combination, raw_probability in probabilities.items():
            try:
                probability = float(raw_probability)
            except (TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(probability) or probability < 0.0:
                continue
            condition = _condition(probabilities, market, str(combination))
            if not condition["in_registered_divergence_band"]:
                continue
            hit = float(str(combination) == actual)
            for key in (
                f"rank:{condition['rank_group']}",
                f"cell:{condition['cell_key']}",
            ):
                totals[key][race_date][0] += probability
                totals[key][race_date][1] += hit
    return sorted(dates), totals, dict(race_counts), dict(rejection_reasons)


def _bootstrap_lower_observed_to_predicted_ratio(
    daily: Mapping[str, list[float]],
    *,
    dates: list[str],
    indices: np.ndarray,
) -> tuple[float, float, float, int]:
    predicted = np.asarray(
        [float((daily.get(day) or [0.0, 0.0])[0]) for day in dates],
        dtype=np.float64,
    )
    observed = np.asarray(
        [float((daily.get(day) or [0.0, 0.0])[1]) for day in dates],
        dtype=np.float64,
    )
    sampled_predicted = predicted[indices].sum(axis=1)
    sampled_observed = observed[indices].sum(axis=1)
    ratios = np.zeros(len(indices), dtype=np.float64)
    valid = sampled_predicted > 0.0
    ratios[valid] = sampled_observed[valid] / sampled_predicted[valid]
    # No pseudo-counts: a zero-hit resample remains exactly zero.
    lower = float(np.quantile(ratios, TARGET_LOWER_QUANTILE))
    active_days = int(np.count_nonzero((predicted > 0.0) | (observed > 0.0)))
    return (
        float(np.clip(lower, 0.0, 1.0)),
        float(predicted.sum()),
        float(observed.sum()),
        active_days,
    )


def fit_edge_conditional_probability_lcb_v14(
    crossfit_races: list[dict[str, Any]],
    *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Fit conservative rank/cell ratios on whole strict-prior days."""
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    dates, totals, race_counts, rejection_reasons = _daily_node_totals(
        crossfit_races
    )
    base = {
        "model_name": MODEL_NAME,
        "method": METHOD,
        "ready": len(dates) >= MIN_TRAINING_DAYS,
        "reason": (
            None
            if len(dates) >= MIN_TRAINING_DAYS
            else "insufficient_strict_prior_crossfit_days"
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
        "registered_divergence_band": {
            "definition": "log(model_probability / normalized_T300_market_probability)",
            "lower_inclusive": REGISTERED_DIVERGENCE_LOWER,
            "upper_exclusive": REGISTERED_DIVERGENCE_UPPER,
            "label": REGISTERED_DIVERGENCE_LABEL,
        },
        "uses_result_for_fit_only": True,
        "uses_payout": False,
        "optimistic_pseudo_counts": False,
        "double_shrinkage": False,
        "global_all_ticket_factor_used": False,
        "inconsistent_t300_snapshot_rejections": sum(rejection_reasons.values()),
        "t300_snapshot_rejection_reasons": rejection_reasons,
    }
    if not base["ready"]:
        return {**base, "rank_nodes": {}, "conditional_cells": {}}

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(dates), size=(bootstrap_samples, len(dates)))
    rank_nodes: dict[str, dict[str, Any]] = {}
    for rank in RANK_GROUPS:
        lower, predicted, observed, active_days = (
            _bootstrap_lower_observed_to_predicted_ratio(
                totals[f"rank:{rank}"], dates=dates, indices=indices
            )
        )
        usable = active_days >= MIN_ACTIVE_NODE_DAYS and predicted > 0.0
        rank_nodes[rank] = {
            "lower_observed_to_predicted_ratio": lower if usable else 0.0,
            "factor": lower if usable else 0.0,
            "sum_predicted_probability": predicted,
            "observed_hits": observed,
            "active_days": active_days,
            "usable": usable,
            "resolution": "rank_parent" if usable else "rank_parent_no_bet",
        }

    conditional_cells: dict[str, dict[str, Any]] = {}
    for node_key in sorted(key for key in totals if key.startswith("cell:")):
        cell_key = node_key.removeprefix("cell:")
        rank = cell_key.split("|", 1)[0]
        parent = rank_nodes.get(rank) or {}
        parent_factor = float(parent.get("factor") or 0.0)
        lower, predicted, observed, active_days = (
            _bootstrap_lower_observed_to_predicted_ratio(
                totals[node_key], dates=dates, indices=indices
            )
        )
        usable = (
            bool(parent.get("usable"))
            and active_days >= MIN_ACTIVE_NODE_DAYS
            and predicted > 0.0
        )
        factor = min(parent_factor, lower) if usable else 0.0
        conditional_cells[cell_key] = {
            "lower_observed_to_predicted_ratio": lower,
            "factor": factor,
            "parent_rank": rank,
            "parent_factor": parent_factor,
            "sum_predicted_probability": predicted,
            "observed_hits": observed,
            "active_days": active_days,
            "usable": usable,
            "resolution": (
                "cell_min_parent_and_cell_lower"
                if usable
                else "sparse_or_missing_parent_no_bet"
            ),
        }
    return {
        **base,
        "rank_nodes": rank_nodes,
        "conditional_cells": conditional_cells,
    }


def probability_lower_bound_details_v14(
    race: Mapping[str, Any],
    combination: str,
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    probabilities = race.get("model_probabilities") or {}
    raw_probability = float(probabilities.get(combination, 0.0) or 0.0)
    empty = {
        "probability": 0.0,
        "raw_probability": max(0.0, raw_probability),
        "factor": 0.0,
    }
    if artifact.get("method") != METHOD or not artifact.get("ready"):
        return {**empty, "resolution": "v14_not_ready"}
    consistency = t300_snapshot_consistency(race)
    if not consistency["consistent"]:
        return {
            **empty,
            "resolution": "inconsistent_t300_snapshot",
            "snapshot_consistency_reason": consistency["reason"],
        }
    market = consistency["market_probabilities"]
    if combination not in market or set(probabilities) != set(market):
        return {**empty, "resolution": "incomplete_t300_market_mapping"}
    condition = _condition(probabilities, market, combination)
    if not condition["in_registered_divergence_band"]:
        return {
            **empty,
            **condition,
            "resolution": "outside_registered_divergence_band",
        }
    cell = (artifact.get("conditional_cells") or {}).get(condition["cell_key"])
    if not isinstance(cell, Mapping) or not cell.get("usable"):
        return {
            **empty,
            **condition,
            "resolution": "sparse_or_missing_cell_no_bet",
        }
    parent = (artifact.get("rank_nodes") or {}).get(condition["rank_group"])
    parent_factor = float((parent or {}).get("factor") or 0.0)
    factor = min(parent_factor, float(cell.get("factor") or 0.0))
    factor = float(np.clip(factor, 0.0, 1.0))
    return {
        **condition,
        "probability": raw_probability * factor,
        "raw_probability": raw_probability,
        "factor": factor,
        "parent_factor": parent_factor,
        "resolution": "cell_min_parent_and_cell_lower",
    }


def artifact_fingerprint(artifact: Mapping[str, Any]) -> str:
    payload = repr((
        sorted((artifact.get("rank_nodes") or {}).items()),
        sorted((artifact.get("conditional_cells") or {}).items()),
    ))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
