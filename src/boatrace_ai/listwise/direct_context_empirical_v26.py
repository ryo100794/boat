from __future__ import annotations

from typing import Any

from .contextual_empirical_ev_calibration import (
    fit_contextual_empirical_ev_calibration,
)
from .direct_context_market_residual_v25 import (
    direct_context_metrics,
    direct_context_probabilities,
    fit_direct_context_residual,
)
from .empirical_lcb_policy import (
    policy_edge_records,
    simulate_empirical_lcb_policy,
)
from .market_calibration import blend_probabilities


MODEL_NAME = "direct_context_empirical_lcb_v26"
ROBUST_REGULARIZATION = 0.1
POLICY_CALIBRATION_DAYS = 30


def _replace_probabilities(
    races: list[dict[str, Any]],
    artifact: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            **race,
            "model_probabilities": direct_context_probabilities(race, artifact),
        }
        for race in races
    ]


def evaluate_temporal_direct_context_empirical(
    calibration: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
    *,
    daily_budget_yen: int,
    policy_calibration_days: int = POLICY_CALIBRATION_DAYS,
    bootstrap_samples: int = 2_000,
    min_tickets: int = 300,
    min_candidate_days: int = 20,
) -> dict[str, Any]:
    dates = sorted({str(race["race_date"]) for race in calibration})
    minimum_days = int(policy_calibration_days) + 10
    if len(dates) < minimum_days:
        return {
            "model": MODEL_NAME,
            "status": "insufficient_calibration_days",
            "calibration_days": len(dates),
            "required_calibration_days": minimum_days,
        }
    policy_dates = set(dates[-int(policy_calibration_days) :])
    probability_dates = set(dates[: -int(policy_calibration_days)])
    probability_training = [
        race for race in calibration if str(race["race_date"]) in probability_dates
    ]
    policy_calibration = [
        race for race in calibration if str(race["race_date"]) in policy_dates
    ]

    prior_probability_artifact = fit_direct_context_residual(
        probability_training,
        regularization=ROBUST_REGULARIZATION,
        max_iterations=100,
    )
    policy_probability_races = _replace_probabilities(
        policy_calibration,
        prior_probability_artifact,
    )
    policy_records = policy_edge_records(
        policy_probability_races,
        {"model_weight": 1.0, "temperature": 1.0},
        blend_probabilities,
    )
    first_evaluation_date = min(str(race["race_date"]) for race in evaluation)
    empirical_artifact = fit_contextual_empirical_ev_calibration(
        policy_records,
        prediction_date=first_evaluation_date,
        bootstrap_samples=bootstrap_samples,
        min_days=int(policy_calibration_days),
        min_tickets=int(min_tickets),
        min_candidate_days=int(min_candidate_days),
        min_rank_days=15,
        min_rank_tickets=150,
        min_cell_days=10,
        min_cell_tickets=50,
    )

    final_probability_artifact = fit_direct_context_residual(
        calibration,
        regularization=ROBUST_REGULARIZATION,
        max_iterations=100,
    )
    evaluation_probability_races = _replace_probabilities(
        evaluation,
        final_probability_artifact,
    )
    bankroll = simulate_empirical_lcb_policy(
        evaluation_probability_races,
        {"model_weight": 1.0, "temperature": 1.0},
        blend_probabilities,
        empirical_artifact,
        daily_budget_yen,
    )
    return {
        "model": MODEL_NAME,
        "status": "completed",
        "validation_design": (
            "Probability residual is trained on the earliest calibration days; "
            "empirical EV is fit only on the next 30 out-of-sample days; final "
            "probabilities are refit through the calibration boundary and both "
            "roles are scored once on later untouched days"
        ),
        "probability_training_from": dates[0],
        "probability_training_through": dates[-int(policy_calibration_days) - 1],
        "policy_calibration_from": dates[-int(policy_calibration_days)],
        "policy_calibration_through": dates[-1],
        "evaluation_from": first_evaluation_date,
        "evaluation_through": max(str(race["race_date"]) for race in evaluation),
        "regularization_selection": (
            "0.1 is the stronger convergent V25 candidate within 0.0001 inner "
            "validation LogLoss of the minimum"
        ),
        "prior_probability_artifact": prior_probability_artifact,
        "policy_calibration_probability_metrics": direct_context_metrics(
            policy_calibration,
            prior_probability_artifact,
        ),
        "empirical_ev_calibration": empirical_artifact.as_dict(),
        "final_probability_artifact": final_probability_artifact,
        "probability_metrics": direct_context_metrics(
            evaluation,
            final_probability_artifact,
        ),
        "bankroll": bankroll,
        "promotion_eligible": bool(
            bankroll.get("tickets", 0) >= 50
            and bankroll.get("profit_yen", 0) > 0
            and (bankroll.get("roi") or 0.0) >= 1.05
        ),
    }
