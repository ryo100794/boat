from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.optimize import minimize

from .contextual_empirical_ev_calibration import (
    fit_contextual_empirical_ev_calibration,
)
from .course_interaction_residual import (
    fit_structure_residual,
    structure_metrics,
)
from .empirical_lcb_policy import (
    policy_edge_records,
    simulate_empirical_lcb_policy,
)
from .market_calibration import blend_probabilities
from .pruned_direct_context_v27 import (
    FEATURE_VARIANTS,
    _feature_dimension,
    _prepare_race,
    _probability_vector,
    _validate_active_features,
    pruned_probabilities,
)


MODEL_NAME = "payout_weighted_role_model_v29"
FEATURE_VARIANT = "independent_core"
REGULARIZATION = 0.03
PAYOUT_WEIGHT_EXPONENTS = (0.0, 0.15, 0.30, 0.45)
PAYOUT_WEIGHT_CAP = 4.0
MAX_LOG_LOSS_REGRET = 0.02
POLICY_CALIBRATION_DAYS = 30
STAKE_YEN = 100
TOP_K = 5
EPSILON = 1e-12


def payout_race_weight(
    race: Mapping[str, Any],
    *,
    exponent: float,
    cap: float = PAYOUT_WEIGHT_CAP,
) -> float:
    if not math.isfinite(exponent) or exponent < 0.0:
        raise ValueError("payout weight exponent must be finite and non-negative")
    if not math.isfinite(cap) or cap < 1.0:
        raise ValueError("payout weight cap must be finite and at least one")
    try:
        payout_odds = float(race["actual_payout_yen"]) / STAKE_YEN
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("race requires a positive actual payout") from exc
    if not math.isfinite(payout_odds) or payout_odds <= 0.0:
        raise ValueError("race requires a positive actual payout")
    if exponent == 0.0:
        return 1.0
    return min(cap, max(0.5, (payout_odds / 20.0) ** exponent))


def _weighted_objective_gradient(
    coefficients: np.ndarray,
    prepared: list[Any],
    weights: np.ndarray,
    *,
    regularization: float,
) -> tuple[float, np.ndarray]:
    loss = 0.0
    gradient = np.zeros_like(coefficients)
    for race, weight in zip(prepared, weights):
        probabilities = _probability_vector(race, coefficients)
        loss -= float(weight) * math.log(
            max(EPSILON, float(probabilities[race.base.actual_index]))
        )
        errors = probabilities
        errors[race.base.actual_index] -= 1.0
        np.add.at(
            gradient[: race.base.indices.max() + 1],
            race.base.indices.reshape(-1),
            float(weight) * (errors[:, None] * race.base.values).reshape(-1),
        )
        context_gradient = gradient[
            int(coefficients.shape[0] - 3 * race.lane_context.shape[1]) :
        ].reshape(3, race.lane_context.shape[1])
        for stage in range(3):
            lane_errors = np.bincount(
                race.stage_lanes[:, stage], weights=errors, minlength=6
            )
            context_gradient[stage] += (
                float(weight) * race.lane_context.T @ lane_errors
            )
    scale = 1.0 / len(prepared)
    loss *= scale
    gradient *= scale
    loss += 0.5 * regularization * float(coefficients @ coefficients)
    gradient += regularization * coefficients
    return loss, gradient


def fit_payout_weighted_ranking(
    races: list[dict[str, Any]],
    *,
    exponent: float,
    regularization: float = REGULARIZATION,
    max_iterations: int = 120,
) -> dict[str, Any]:
    if not races:
        raise ValueError("at least one race is required")
    active = _validate_active_features(FEATURE_VARIANTS[FEATURE_VARIANT])
    prepared = [_prepare_race(race, active) for race in races]
    raw_weights = np.asarray(
        [payout_race_weight(race, exponent=exponent) for race in races],
        dtype=np.float64,
    )
    weights = raw_weights / float(np.mean(raw_weights))
    dimension = _feature_dimension(active)

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        return _weighted_objective_gradient(
            coefficients,
            prepared,
            weights,
            regularization=regularization,
        )

    result = minimize(
        objective,
        np.zeros(dimension, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": int(max_iterations),
            "ftol": 1e-10,
            "gtol": 1e-6,
            "maxls": 30,
        },
    )
    objective_value, gradient = objective(np.asarray(result.x, dtype=np.float64))
    return {
        "model": MODEL_NAME,
        "role": "payout_weighted_ranking_only",
        "feature_variant": FEATURE_VARIANT,
        "active_context_features": list(active),
        "active_context_feature_count": len(active),
        "feature_dimension": dimension,
        "base_feature_dimension": dimension - 3 * len(active) * 2,
        "regularization": float(regularization),
        "payout_weight_exponent": float(exponent),
        "payout_weight_cap": PAYOUT_WEIGHT_CAP,
        "normalized_weight_min": float(np.min(weights)),
        "normalized_weight_max": float(np.max(weights)),
        "coefficients": [float(value) for value in result.x],
        "objective": float(objective_value),
        "gradient_norm": float(np.linalg.norm(gradient)),
        "iterations": int(result.nit),
        "converged": bool(result.success),
        "message": str(result.message),
        "training_races": len(races),
    }


def ranking_probabilities(
    race: dict[str, Any], artifact: Mapping[str, Any]
) -> dict[str, float]:
    return pruned_probabilities(race, artifact)


def payout_ranking_metrics(
    races: list[dict[str, Any]], artifact: Mapping[str, Any]
) -> dict[str, Any]:
    loss = 0.0
    hits = 0
    return_yen = 0
    for race in races:
        probabilities = ranking_probabilities(race, artifact)
        actual = str(race["actual_combination"])
        loss -= math.log(max(EPSILON, float(probabilities.get(actual, 0.0))))
        top = sorted(probabilities, key=probabilities.get, reverse=True)[:TOP_K]
        hit = actual in top
        hits += int(hit)
        if hit:
            return_yen += int(race["actual_payout_yen"])
    count = len(races)
    stake_yen = count * TOP_K * STAKE_YEN
    return {
        "evaluated_races": count,
        "trifecta_log_loss": loss / count if count else None,
        "trifecta_top5_hit_rate": hits / count if count else None,
        "top5_flat_stake_yen": stake_yen,
        "top5_flat_return_yen": return_yen,
        "top5_flat_profit_yen": return_yen - stake_yen,
        "top5_flat_roi": return_yen / stake_yen if stake_yen else None,
    }


def _replace_with_ranking(
    races: list[dict[str, Any]], artifact: Mapping[str, Any]
) -> list[dict[str, Any]]:
    return [
        {**race, "model_probabilities": ranking_probabilities(race, artifact)}
        for race in races
    ]


def evaluate_temporal_payout_weighted_roles(
    calibration: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
    *,
    daily_budget_yen: int,
    policy_calibration_days: int = POLICY_CALIBRATION_DAYS,
    exponents: Iterable[float] = PAYOUT_WEIGHT_EXPONENTS,
    bootstrap_samples: int = 2_000,
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
    ranking_dates = dates[: -int(policy_calibration_days)]
    ranking_races = [
        race for race in calibration if str(race["race_date"]) in set(ranking_dates)
    ]
    policy_races = [
        race for race in calibration if str(race["race_date"]) in policy_dates
    ]
    split_index = max(1, min(len(ranking_dates) - 1, int(len(ranking_dates) * 0.8)))
    inner_fit_dates = set(ranking_dates[:split_index])
    inner_validation_dates = set(ranking_dates[split_index:])
    inner_fit = [
        race for race in ranking_races if str(race["race_date"]) in inner_fit_dates
    ]
    inner_validation = [
        race
        for race in ranking_races
        if str(race["race_date"]) in inner_validation_dates
    ]

    candidates = []
    for exponent in tuple(float(value) for value in exponents):
        artifact = fit_payout_weighted_ranking(
            inner_fit, exponent=exponent, max_iterations=120
        )
        candidates.append({
            "payout_weight_exponent": exponent,
            "converged": artifact["converged"],
            "gradient_norm": artifact["gradient_norm"],
            "metrics": payout_ranking_metrics(inner_validation, artifact),
        })
    baseline = next(
        (row for row in candidates if row["payout_weight_exponent"] == 0.0),
        None,
    )
    if baseline is None:
        raise ValueError("payout-weighted selection requires exponent zero baseline")
    loss_limit = float(baseline["metrics"]["trifecta_log_loss"]) + MAX_LOG_LOSS_REGRET
    eligible = [
        row
        for row in candidates
        if row["converged"]
        and float(row["metrics"]["trifecta_log_loss"]) <= loss_limit
    ] or candidates
    selected = max(
        eligible,
        key=lambda row: (
            float(row["metrics"]["top5_flat_roi"]),
            float(row["metrics"]["trifecta_top5_hit_rate"]),
            -float(row["metrics"]["trifecta_log_loss"]),
            -float(row["payout_weight_exponent"]),
        ),
    )
    selected_exponent = float(selected["payout_weight_exponent"])
    prior_ranking_artifact = fit_payout_weighted_ranking(
        ranking_races, exponent=selected_exponent, max_iterations=200
    )
    policy_ranked = _replace_with_ranking(policy_races, prior_ranking_artifact)
    policy_records = policy_edge_records(
        policy_ranked,
        {"model_weight": 1.0, "temperature": 1.0},
        blend_probabilities,
    )
    first_evaluation_date = min(str(race["race_date"]) for race in evaluation)
    empirical_artifact = fit_contextual_empirical_ev_calibration(
        policy_records,
        prediction_date=first_evaluation_date,
        bootstrap_samples=bootstrap_samples,
        min_days=int(policy_calibration_days),
        min_tickets=300,
        min_candidate_days=20,
        min_rank_days=15,
        min_rank_tickets=150,
        min_cell_days=10,
        min_cell_tickets=50,
    )

    probability_artifact = fit_structure_residual(
        calibration,
        structure_variant="shared_independent_core",
        regularization=REGULARIZATION,
        max_iterations=200,
    )
    final_ranking_artifact = fit_payout_weighted_ranking(
        calibration, exponent=selected_exponent, max_iterations=200
    )
    evaluation_ranked = _replace_with_ranking(evaluation, final_ranking_artifact)
    bankroll = simulate_empirical_lcb_policy(
        evaluation_ranked,
        {"model_weight": 1.0, "temperature": 1.0},
        blend_probabilities,
        empirical_artifact,
        daily_budget_yen,
    )
    return {
        "model": MODEL_NAME,
        "status": "completed",
        "validation_design": (
            "A proper probability head remains unweighted. Payout weighting is "
            "selected only for the ranking head on an inner prior-day block by "
            "Top5 flat ROI under a fixed LogLoss regret limit. Empirical EV is fit "
            "on the next 30 days and both heads are evaluated once on later days."
        ),
        "ranking_training_from": ranking_dates[0],
        "ranking_training_through": ranking_dates[-1],
        "inner_fit_through": ranking_dates[split_index - 1],
        "inner_validation_from": ranking_dates[split_index],
        "policy_calibration_from": dates[-int(policy_calibration_days)],
        "policy_calibration_through": dates[-1],
        "evaluation_from": first_evaluation_date,
        "evaluation_through": max(str(race["race_date"]) for race in evaluation),
        "selected_candidate": selected,
        "candidates": candidates,
        "probability_artifact": probability_artifact,
        "probability_metrics": structure_metrics(evaluation, probability_artifact),
        "prior_ranking_artifact": prior_ranking_artifact,
        "ranking_artifact": final_ranking_artifact,
        "ranking_metrics": payout_ranking_metrics(
            evaluation, final_ranking_artifact
        ),
        "empirical_ev_calibration": empirical_artifact.as_dict(),
        "bankroll": bankroll,
        "promotion_eligible": bool(
            bankroll.get("tickets", 0) >= 50
            and bankroll.get("profit_yen", 0) > 0
            and (bankroll.get("roi") or 0.0) >= 1.05
        ),
    }
