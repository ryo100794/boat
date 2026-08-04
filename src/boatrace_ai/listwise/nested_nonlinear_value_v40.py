from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..bankroll_bootstrap import bootstrap_daily_roi
from .empirical_ev_calibration import fit_empirical_ev_calibration
from .empirical_lcb_policy import (
    empirical_bankroll_promotion_eligible,
    policy_edge_records,
    simulate_empirical_lcb_policy,
)
from .nonlinear_market_residual_v38 import (
    fit_temporal_nonlinear_market_residual,
    nonlinear_residual_metrics,
    nonlinear_residual_probabilities,
)


MODEL_NAME = "nested_nonlinear_value_calibration_v40"
MODEL_TRAINING_MINIMUM_DAYS = 20
VALUE_CALIBRATION_DAYS = 30
PURCHASE_SHRINKAGE = 1.0
PURCHASE_MAX_RANK = 5


def _value_decile_rows(
    records: list[dict[str, Any]],
    inner_edges: list[float],
) -> list[dict[str, Any]]:
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(10)]
    for record in records:
        raw_ev = float(record["raw_estimated_ev"])
        index = min(9, int(np.searchsorted(inner_edges, raw_ev, side="right")))
        buckets[index].append(record)
    result = []
    for index, bucket in enumerate(buckets):
        daily: dict[str, dict[str, Any]] = {}
        for record in bucket:
            day = str(record["race_date"])
            row = daily.setdefault(
                day,
                {"race_date": day, "stake_yen": 0, "return_yen": 0.0},
            )
            row["stake_yen"] += 100
            row["return_yen"] += 100.0 * float(
                record["gross_return_per_yen"]
            )
        confidence = (
            bootstrap_daily_roi(list(daily.values()))
            if bucket
            else {
                "roi": None,
                "roi_ci95_lower": None,
                "probability_roi_above_one": None,
            }
        )
        result.append({
            "decile": index + 1,
            "lower": None if index == 0 else inner_edges[index - 1],
            "upper": None if index == 9 else inner_edges[index],
            "candidates": len(bucket),
            "candidate_days": len(daily),
            "mean_predicted_raw_ev": (
                float(np.mean([
                    float(record["raw_estimated_ev"]) for record in bucket
                ]))
                if bucket
                else None
            ),
            "realized_roi": confidence.get("roi"),
            "realized_roi_lcb95": confidence.get("roi_ci95_lower"),
            "probability_roi_above_one": confidence.get(
                "probability_roi_above_one"
            ),
        })
    return result


def value_decile_audit(
    calibration_records: list[dict[str, Any]],
    evaluation_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Audit value monotonicity using edges learned on calibration only."""
    if not calibration_records:
        return {
            "status": "no_calibration_candidates",
            "edge_source": "value_calibration_only",
            "evaluation_used_for_edges": False,
            "calibration": [],
            "evaluation": [],
        }
    values = np.asarray(
        [float(record["raw_estimated_ev"]) for record in calibration_records],
        dtype=np.float64,
    )
    inner_edges = [
        float(value)
        for value in np.quantile(values, np.arange(1, 10) / 10.0)
    ]
    return {
        "status": "completed",
        "edge_source": "value_calibration_only",
        "evaluation_used_for_edges": False,
        "inner_edges": inner_edges,
        "calibration": _value_decile_rows(calibration_records, inner_edges),
        "evaluation": _value_decile_rows(evaluation_records, inner_edges),
    }


def _identity_probability_blender(
    model: Mapping[str, float],
    _market: Mapping[str, float],
    *,
    model_weight: float,
    temperature: float,
) -> dict[str, float]:
    if model_weight != 1.0 or temperature != 1.0:
        raise ValueError("V40 requires its frozen V38 distribution")
    return {str(key): float(value) for key, value in model.items()}


def _score(
    races: list[dict[str, Any]], artifact: Mapping[str, Any]
) -> list[dict[str, Any]]:
    return [
        {
            **race,
            "model_probabilities": nonlinear_residual_probabilities(
                race,
                artifact,
                shrinkage=PURCHASE_SHRINKAGE,
            ),
        }
        for race in races
    ]


def evaluate_nested_nonlinear_value_v40(
    calibration: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
    *,
    daily_budget_yen: int,
    num_threads: int = 4,
) -> dict[str, Any]:
    dates = sorted({str(race["race_date"]) for race in calibration})
    if len(dates) < MODEL_TRAINING_MINIMUM_DAYS + VALUE_CALIBRATION_DAYS:
        return {
            "model": MODEL_NAME,
            "status": "insufficient_nested_days",
            "calibration_days": len(dates),
            "required_days": (
                MODEL_TRAINING_MINIMUM_DAYS + VALUE_CALIBRATION_DAYS
            ),
            "promotion_eligible": False,
        }
    value_dates = set(dates[-VALUE_CALIBRATION_DAYS:])
    model_dates = set(dates[:-VALUE_CALIBRATION_DAYS])
    model_training = [
        race for race in calibration if str(race["race_date"]) in model_dates
    ]
    value_calibration = [
        race for race in calibration if str(race["race_date"]) in value_dates
    ]
    probability = fit_temporal_nonlinear_market_residual(
        model_training,
        [],
        num_threads=num_threads,
    )
    probability_artifact = probability["artifact"]
    value_scored = _score(value_calibration, probability_artifact)
    evaluation_scored = _score(evaluation, probability_artifact)
    calibrator = {"model_weight": 1.0, "temperature": 1.0}
    ledger = policy_edge_records(
        value_scored,
        calibrator,
        _identity_probability_blender,
        max_rank=PURCHASE_MAX_RANK,
    )
    evaluation_ledger = policy_edge_records(
        evaluation_scored,
        calibrator,
        _identity_probability_blender,
        max_rank=PURCHASE_MAX_RANK,
    )
    empirical = fit_empirical_ev_calibration(
        ledger,
        min_days=30,
        min_tickets=300,
        min_candidate_days=20,
        min_local_candidates=50,
        min_local_candidate_days=20,
        min_local_ess=10.0,
        candidate_min_raw_ev=1.0,
    )
    bankroll = simulate_empirical_lcb_policy(
        evaluation_scored,
        calibrator,
        _identity_probability_blender,
        empirical,
        daily_budget_yen,
        max_rank=PURCHASE_MAX_RANK,
    )
    confidence = (
        bootstrap_daily_roi(bankroll["daily"])
        if bankroll.get("stake_yen")
        else {
            "roi": None,
            "roi_ci95_lower": None,
            "probability_roi_above_one": None,
        }
    )
    bankroll.update({
        "roi": confidence.get("roi"),
        "roi_display": (
            confidence.get("roi")
            if confidence.get("roi") is not None
            else "N/A"
        ),
        "roi_ci95_lower": confidence.get("roi_ci95_lower"),
        "probability_roi_above_one": confidence.get(
            "probability_roi_above_one"
        ),
        "evaluation_days": len({str(race["race_date"]) for race in evaluation}),
    })
    result = {
        "model": MODEL_NAME,
        "status": "completed",
        "validation_design": (
            "earliest model-fit days; following 30 untouched days for all top5 "
            "value calibration; final outer days for purchase evaluation"
        ),
        "model_training_from": min(model_dates),
        "model_training_through": max(model_dates),
        "model_training_days": len(model_dates),
        "model_training_races": len(model_training),
        "value_calibration_from": min(value_dates),
        "value_calibration_through": max(value_dates),
        "value_calibration_days": len(value_dates),
        "value_calibration_races": len(value_calibration),
        "evaluation_from": min(
            str(race["race_date"]) for race in evaluation
        ) if evaluation else None,
        "evaluation_through": max(
            str(race["race_date"]) for race in evaluation
        ) if evaluation else None,
        "evaluation_races": len(evaluation),
        "purchase_shrinkage": PURCHASE_SHRINKAGE,
        "purchase_max_rank": PURCHASE_MAX_RANK,
        "candidate_population": "all_probability_top5_before_purchase_gate",
        "probability_selection": {
            key: probability.get(key)
            for key in (
                "inner_fit_through",
                "inner_validation_from",
                "selected_tree_preset",
                "selected_shrinkage",
            )
        },
        "probability_artifact": probability_artifact,
        "value_calibration_probability_metrics": nonlinear_residual_metrics(
            value_calibration,
            probability_artifact,
            shrinkage=PURCHASE_SHRINKAGE,
        ),
        "evaluation_probability_metrics": nonlinear_residual_metrics(
            evaluation,
            probability_artifact,
            shrinkage=PURCHASE_SHRINKAGE,
        ),
        "empirical_ev_calibration": empirical.as_dict(),
        "calibration_ledger_candidates": len(ledger),
        "evaluation_ledger_candidates": len(evaluation_ledger),
        "value_decile_audit": value_decile_audit(ledger, evaluation_ledger),
        "bankroll": bankroll,
        "promotion_eligible": False,
        "real_betting_enabled": False,
    }
    result["promotion_eligible"] = empirical_bankroll_promotion_eligible(
        bankroll
    )
    return result
