from __future__ import annotations

from typing import Any, Mapping

from ..bankroll_bootstrap import bootstrap_daily_roi
from .contextual_empirical_ev_calibration import (
    fit_contextual_empirical_ev_calibration,
)
from .empirical_lcb_policy import (
    empirical_bankroll_promotion_eligible,
    policy_edge_records,
    simulate_empirical_lcb_policy,
)
from .nested_nonlinear_value_v40 import value_decile_audit
from .stacked_market_residual_v42 import (
    fit_temporal_stacked_market_residual,
    stacked_metrics,
    stacked_probabilities,
)


MODEL_NAME = "mature_stacked_contextual_value"
MODEL_TRAINING_MINIMUM_DAYS = 60
VALUE_CALIBRATION_DAYS = 120
PURCHASE_MAX_RANK = 5


def _identity_probability_blender(
    model: Mapping[str, float],
    _market: Mapping[str, float],
    *,
    model_weight: float,
    temperature: float,
) -> dict[str, float]:
    if model_weight != 1.0 or temperature != 1.0:
        raise ValueError("mature stacked value requires its frozen distribution")
    return {str(key): float(value) for key, value in model.items()}


def _score(
    races: list[dict[str, Any]], artifact: Mapping[str, Any]
) -> list[dict[str, Any]]:
    return [
        {**race, "model_probabilities": stacked_probabilities(race, artifact)}
        for race in races
    ]


def evaluate_mature_stacked_value(
    calibration: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
    *,
    daily_budget_yen: int,
    num_threads: int = 4,
) -> dict[str, Any]:
    dates = sorted({str(race["race_date"]) for race in calibration})
    required_days = MODEL_TRAINING_MINIMUM_DAYS + VALUE_CALIBRATION_DAYS
    if len(dates) < required_days:
        return {
            "model": MODEL_NAME,
            "status": "insufficient_nested_days",
            "calibration_days": len(dates),
            "required_days": required_days,
            "model_training_minimum_days": MODEL_TRAINING_MINIMUM_DAYS,
            "value_calibration_required_days": VALUE_CALIBRATION_DAYS,
            "promotion_eligible": False,
            "real_betting_enabled": False,
        }
    if not evaluation:
        raise ValueError("mature stacked value requires untouched outer races")

    value_dates = set(dates[-VALUE_CALIBRATION_DAYS:])
    model_dates = set(dates[:-VALUE_CALIBRATION_DAYS])
    model_training = [
        race for race in calibration if str(race["race_date"]) in model_dates
    ]
    value_calibration = [
        race for race in calibration if str(race["race_date"]) in value_dates
    ]
    probability = fit_temporal_stacked_market_residual(
        model_training,
        [],
        num_threads=num_threads,
    )
    artifact = probability["artifact"]
    value_scored = _score(value_calibration, artifact)
    evaluation_scored = _score(evaluation, artifact)
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
    first_evaluation_date = min(
        str(race["race_date"]) for race in evaluation
    )
    empirical = fit_contextual_empirical_ev_calibration(
        ledger,
        prediction_date=first_evaluation_date,
        bootstrap_samples=5_000,
        min_days=VALUE_CALIBRATION_DAYS,
        min_tickets=1_000,
        min_candidate_days=80,
        candidate_min_raw_ev=1.0,
        min_rank_days=90,
        min_rank_tickets=1_000,
        min_cell_days=60,
        min_cell_tickets=200,
        rank_prior_tickets=500.0,
        cell_prior_tickets=200.0,
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
        "evaluation_days": len(
            {str(race["race_date"]) for race in evaluation}
        ),
    })
    result = {
        "model": MODEL_NAME,
        "status": "completed",
        "validation_design": (
            "earliest 60 or more days for nested V42 component and stack "
            "selection; following 120 untouched days for top5 contextual "
            "rank-by-odds value calibration; final outer days used once"
        ),
        "outer_period_used_for_selection": False,
        "model_training_from": min(model_dates),
        "model_training_through": max(model_dates),
        "model_training_days": len(model_dates),
        "model_training_races": len(model_training),
        "value_calibration_from": min(value_dates),
        "value_calibration_through": max(value_dates),
        "value_calibration_days": len(value_dates),
        "value_calibration_races": len(value_calibration),
        "evaluation_from": first_evaluation_date,
        "evaluation_through": max(
            str(race["race_date"]) for race in evaluation
        ),
        "evaluation_races": len(evaluation),
        "purchase_max_rank": PURCHASE_MAX_RANK,
        "candidate_population": (
            "all_stacked_probability_top5_before_purchase_gate"
        ),
        "probability_selection": {
            key: probability.get(key)
            for key in (
                "base_training_through",
                "stack_validation_from",
                "selected_stack",
                "selected_weights",
                "component_selection",
            )
        },
        "probability_artifact": artifact,
        "value_calibration_probability_metrics": stacked_metrics(
            value_calibration, artifact
        ),
        "evaluation_probability_metrics": stacked_metrics(
            evaluation, artifact
        ),
        "empirical_ev_calibration": empirical.as_dict(),
        "calibration_ledger_candidates": len(ledger),
        "evaluation_ledger_candidates": len(evaluation_ledger),
        "value_decile_audit": value_decile_audit(
            ledger, evaluation_ledger
        ),
        "bankroll": bankroll,
        "promotion_eligible": False,
        "real_betting_enabled": False,
    }
    result["promotion_eligible"] = empirical_bankroll_promotion_eligible(
        bankroll
    )
    return result
