from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from .contextual_empirical_ev_calibration import (
    fit_contextual_empirical_ev_calibration,
)
from .empirical_lcb_policy import (
    policy_edge_records,
    simulate_empirical_lcb_policy,
)
from .market_calibration import blend_probabilities
from .pruned_direct_context_v27 import (
    BASE_FEATURE_DIMENSION,
    FEATURE_VARIANTS as LANE_FEATURE_VARIANTS,
    STAGES,
    _feature_dimension as lane_feature_dimension,
    _prepare_race,
    _validate_active_features,
)


MODEL_NAME = "conditional_ticket_residual_v30"
LANE_FEATURE_VARIANT = "independent_core"
REGULARIZATIONS = (0.03, 0.1)
POLICY_CALIBRATION_DAYS = 30
SELECTION_LOG_LOSS_TOLERANCE = 0.001
EPSILON = 1e-12

CORE_FEATURES = (
    "rank_delta",
    "absolute_rank_delta",
    "signed_residual_square",
    "residual_rank_delta",
    "model_top5_only",
    "market_top5_only",
    "joint_top5",
)
SHAPE_FEATURES = (
    "residual_model_entropy",
    "residual_market_entropy",
    "residual_entropy_gap",
    "residual_top5_mass_gap",
    "residual_top1_gap",
    "rank_delta_entropy_gap",
)
RACE_NUMBER_FEATURES = (
    "residual_rno_scaled",
    "residual_rno_early",
    "residual_rno_middle",
    "residual_rno_late",
)
VENUE_FEATURES = tuple(f"residual_venue_{jcd:02d}" for jcd in range(1, 25))
FEATURE_VARIANTS: dict[str, tuple[str, ...]] = {
    "rank_disagreement": CORE_FEATURES,
    "race_shape": CORE_FEATURES + SHAPE_FEATURES,
    "race_shape_number": CORE_FEATURES + SHAPE_FEATURES + RACE_NUMBER_FEATURES,
    "full_conditional": (
        CORE_FEATURES + SHAPE_FEATURES + RACE_NUMBER_FEATURES + VENUE_FEATURES
    ),
}


@dataclass(frozen=True)
class PreparedConditionalRace:
    lane: Any
    ticket_features: np.ndarray


def _normalized_entropy(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0.0]
    if positive.size == 0 or probabilities.size <= 1:
        return 0.0
    return float(-np.sum(positive * np.log(positive)) / math.log(probabilities.size))


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    return ranks


def _race_number(race: Mapping[str, Any]) -> int:
    try:
        return min(12, max(1, int(race.get("rno") or 1)))
    except (TypeError, ValueError, OverflowError):
        return 1


def _venue(race: Mapping[str, Any]) -> int:
    try:
        return min(24, max(1, int(str(race.get("jcd") or "1"))))
    except (TypeError, ValueError, OverflowError):
        return 1


def ticket_feature_matrix(
    race: Mapping[str, Any],
    combinations: Sequence[str],
    active_features: Sequence[str],
) -> np.ndarray:
    model = np.asarray(
        [max(EPSILON, float(race["model_probabilities"][key])) for key in combinations],
        dtype=np.float64,
    )
    market = np.asarray(
        [max(EPSILON, float(race["market_probabilities"][key])) for key in combinations],
        dtype=np.float64,
    )
    model /= float(np.sum(model))
    market /= float(np.sum(market))
    residual = np.clip(np.log(model) - np.log(market), -5.0, 5.0)
    residual -= float(np.mean(residual))
    model_ranks = _ranks(model)
    market_ranks = _ranks(market)
    rank_scale = max(1.0, float(len(combinations) - 1))
    rank_delta = (market_ranks - model_ranks) / rank_scale
    entropy_model = _normalized_entropy(model)
    entropy_market = _normalized_entropy(market)
    entropy_gap = entropy_model - entropy_market
    model_top = np.sort(model)[-5:]
    market_top = np.sort(market)[-5:]
    top5_mass_gap = float(np.sum(model_top) - np.sum(market_top))
    top1_gap = float(np.max(model) - np.max(market))
    rno = _race_number(race)
    venue = _venue(race)

    columns: dict[str, np.ndarray] = {
        "rank_delta": rank_delta,
        "absolute_rank_delta": np.abs(rank_delta),
        "signed_residual_square": np.sign(residual) * residual * residual / 5.0,
        "residual_rank_delta": residual * rank_delta,
        "model_top5_only": ((model_ranks <= 5) & (market_ranks > 5)).astype(float),
        "market_top5_only": ((market_ranks <= 5) & (model_ranks > 5)).astype(float),
        "joint_top5": ((model_ranks <= 5) & (market_ranks <= 5)).astype(float),
        "residual_model_entropy": residual * entropy_model,
        "residual_market_entropy": residual * entropy_market,
        "residual_entropy_gap": residual * entropy_gap,
        "residual_top5_mass_gap": residual * top5_mass_gap,
        "residual_top1_gap": residual * top1_gap,
        "rank_delta_entropy_gap": rank_delta * entropy_gap,
        "residual_rno_scaled": residual * ((rno - 1.0) / 11.0),
        "residual_rno_early": residual * float(rno <= 4),
        "residual_rno_middle": residual * float(5 <= rno <= 8),
        "residual_rno_late": residual * float(rno >= 9),
    }
    for jcd in range(1, 25):
        columns[f"residual_venue_{jcd:02d}"] = residual * float(venue == jcd)
    unknown = set(active_features).difference(columns)
    if unknown:
        raise ValueError(f"unknown V30 ticket features: {sorted(unknown)}")
    if not active_features:
        return np.zeros((len(combinations), 0), dtype=np.float64)
    return np.column_stack([columns[name] for name in active_features])


def _prepare(
    race: dict[str, Any], active_ticket_features: Sequence[str]
) -> PreparedConditionalRace:
    lane_features = _validate_active_features(
        LANE_FEATURE_VARIANTS[LANE_FEATURE_VARIANT]
    )
    lane = _prepare_race(race, lane_features)
    return PreparedConditionalRace(
        lane=lane,
        ticket_features=ticket_feature_matrix(
            race, lane.base.combinations, active_ticket_features
        ),
    )


def _dimension(active_ticket_features: Sequence[str]) -> int:
    lane_features = LANE_FEATURE_VARIANTS[LANE_FEATURE_VARIANT]
    return lane_feature_dimension(lane_features) + len(active_ticket_features)


def _probability_vector(
    race: PreparedConditionalRace, coefficients: np.ndarray
) -> np.ndarray:
    lane = race.lane
    lane_dimension = coefficients.size - race.ticket_features.shape[1]
    base_coefficients = coefficients[:BASE_FEATURE_DIMENSION]
    context_coefficients = coefficients[BASE_FEATURE_DIMENSION:lane_dimension].reshape(
        STAGES, lane.lane_context.shape[1]
    )
    logits = lane.base.market_log + np.sum(
        base_coefficients[lane.base.indices] * lane.base.values,
        axis=1,
    )
    for stage in range(STAGES):
        logits += lane.lane_context[lane.stage_lanes[:, stage]] @ context_coefficients[stage]
    if race.ticket_features.shape[1]:
        logits += race.ticket_features @ coefficients[lane_dimension:]
    logits -= float(np.max(logits))
    probabilities = np.exp(logits)
    probabilities /= float(np.sum(probabilities))
    return probabilities


def _objective_gradient(
    coefficients: np.ndarray,
    races: list[PreparedConditionalRace],
    *,
    regularization: float,
) -> tuple[float, np.ndarray]:
    loss = 0.0
    gradient = np.zeros_like(coefficients)
    ticket_columns = races[0].ticket_features.shape[1]
    lane_dimension = coefficients.size - ticket_columns
    base_gradient = gradient[:BASE_FEATURE_DIMENSION]
    context_gradient = gradient[BASE_FEATURE_DIMENSION:lane_dimension].reshape(
        STAGES, races[0].lane.lane_context.shape[1]
    )
    ticket_gradient = gradient[lane_dimension:]
    for race in races:
        probabilities = _probability_vector(race, coefficients)
        actual_index = race.lane.base.actual_index
        loss -= math.log(max(EPSILON, float(probabilities[actual_index])))
        errors = probabilities
        errors[actual_index] -= 1.0
        np.add.at(
            base_gradient,
            race.lane.base.indices.reshape(-1),
            (errors[:, None] * race.lane.base.values).reshape(-1),
        )
        for stage in range(STAGES):
            lane_errors = np.bincount(
                race.lane.stage_lanes[:, stage], weights=errors, minlength=6
            )
            context_gradient[stage] += race.lane.lane_context.T @ lane_errors
        if ticket_columns:
            ticket_gradient += race.ticket_features.T @ errors
    scale = 1.0 / len(races)
    loss *= scale
    gradient *= scale
    loss += 0.5 * regularization * float(coefficients @ coefficients)
    gradient += regularization * coefficients
    return loss, gradient


def fit_conditional_ticket_residual(
    races: list[dict[str, Any]],
    *,
    variant: str,
    regularization: float,
    max_iterations: int = 160,
) -> dict[str, Any]:
    if not races:
        raise ValueError("at least one race is required")
    if variant not in FEATURE_VARIANTS:
        raise ValueError(f"unknown V30 feature variant: {variant}")
    if regularization <= 0.0 or not math.isfinite(regularization):
        raise ValueError("regularization must be finite and positive")
    active = FEATURE_VARIANTS[variant]
    prepared = [_prepare(race, active) for race in races]
    dimension = _dimension(active)

    def objective(values: np.ndarray) -> tuple[float, np.ndarray]:
        return _objective_gradient(
            values, prepared, regularization=regularization
        )

    result = minimize(
        objective,
        np.zeros(dimension, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": int(max_iterations), "ftol": 1e-10, "gtol": 1e-6, "maxls": 30},
    )
    objective_value, gradient = objective(np.asarray(result.x, dtype=np.float64))
    return {
        "model": MODEL_NAME,
        "feature_variant": variant,
        "lane_feature_variant": LANE_FEATURE_VARIANT,
        "active_ticket_features": list(active),
        "active_ticket_feature_count": len(active),
        "feature_dimension": dimension,
        "regularization": float(regularization),
        "coefficients": [float(value) for value in result.x],
        "objective": float(objective_value),
        "gradient_norm": float(np.linalg.norm(gradient)),
        "iterations": int(result.nit),
        "converged": bool(result.success),
        "message": str(result.message),
        "training_races": len(races),
    }


def conditional_probabilities(
    race: dict[str, Any], artifact: Mapping[str, Any]
) -> dict[str, float]:
    variant = str(artifact["feature_variant"])
    active = FEATURE_VARIANTS[variant]
    prepared = _prepare(race, active)
    coefficients = np.asarray(artifact["coefficients"], dtype=np.float64)
    if coefficients.shape != (_dimension(active),):
        raise ValueError("V30 coefficient shape mismatch")
    probabilities = _probability_vector(prepared, coefficients)
    return {
        combination: float(value)
        for combination, value in zip(prepared.lane.base.combinations, probabilities)
    }


def conditional_metrics(
    races: list[dict[str, Any]], artifact: Mapping[str, Any]
) -> dict[str, Any]:
    loss = market_loss = raw_model_loss = 0.0
    top5_hits = market_top5_hits = 0
    for race in races:
        probabilities = conditional_probabilities(race, artifact)
        actual = str(race["actual_combination"])
        market = race["market_probabilities"]
        model = race["model_probabilities"]
        loss -= math.log(max(EPSILON, probabilities.get(actual, 0.0)))
        market_loss -= math.log(max(EPSILON, float(market.get(actual, 0.0))))
        raw_model_loss -= math.log(max(EPSILON, float(model.get(actual, 0.0))))
        top5_hits += int(actual in sorted(probabilities, key=probabilities.get, reverse=True)[:5])
        market_top5_hits += int(actual in sorted(market, key=market.get, reverse=True)[:5])
    count = len(races)
    return {
        "evaluated_races": count,
        "trifecta_log_loss": loss / count if count else None,
        "market_trifecta_log_loss": market_loss / count if count else None,
        "raw_model_trifecta_log_loss": raw_model_loss / count if count else None,
        "trifecta_top5_hit_rate": top5_hits / count if count else None,
        "market_trifecta_top5_hit_rate": market_top5_hits / count if count else None,
    }


def _replace_probabilities(
    races: list[dict[str, Any]], artifact: Mapping[str, Any]
) -> list[dict[str, Any]]:
    return [
        {**race, "model_probabilities": conditional_probabilities(race, artifact)}
        for race in races
    ]


def evaluate_temporal_conditional_ticket_residual(
    calibration: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
    *,
    daily_budget_yen: int,
    policy_calibration_days: int = POLICY_CALIBRATION_DAYS,
    variants: Iterable[str] = FEATURE_VARIANTS,
    regularizations: Iterable[float] = REGULARIZATIONS,
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
    ranking_dates = dates[:-int(policy_calibration_days)]
    policy_dates = set(dates[-int(policy_calibration_days):])
    split_index = max(1, min(len(ranking_dates) - 1, int(len(ranking_dates) * 0.8)))
    fit_dates = set(ranking_dates[:split_index])
    validation_dates = set(ranking_dates[split_index:])
    inner_fit = [race for race in calibration if str(race["race_date"]) in fit_dates]
    inner_validation = [
        race for race in calibration if str(race["race_date"]) in validation_dates
    ]

    variant_values = tuple(str(value) for value in variants)
    regularization_values = tuple(float(value) for value in regularizations)
    if not variant_values or not regularization_values:
        raise ValueError("V30 requires feature and regularization candidates")

    candidates = []
    for variant in variant_values:
        for regularization in regularization_values:
            artifact = fit_conditional_ticket_residual(
                inner_fit,
                variant=variant,
                regularization=regularization,
                max_iterations=120,
            )
            candidates.append({
                "feature_variant": variant,
                "regularization": regularization,
                "converged": artifact["converged"],
                "gradient_norm": artifact["gradient_norm"],
                "metrics": conditional_metrics(inner_validation, artifact),
            })
    converged = [row for row in candidates if row["converged"]]
    selection_pool = converged or candidates
    best_loss = min(
        float(row["metrics"]["trifecta_log_loss"])
        for row in selection_pool
    )
    eligible = [
        row for row in selection_pool
        if float(row["metrics"]["trifecta_log_loss"])
        <= best_loss + SELECTION_LOG_LOSS_TOLERANCE
    ]
    selected = max(
        eligible,
        key=lambda row: (
            float(row["metrics"]["trifecta_top5_hit_rate"]),
            -float(row["metrics"]["trifecta_log_loss"]),
            -len(FEATURE_VARIANTS[str(row["feature_variant"])]),
            -float(row["regularization"]),
        ),
    )
    variant = str(selected["feature_variant"])
    regularization = float(selected["regularization"])
    ranking_races = [
        race for race in calibration if str(race["race_date"]) in set(ranking_dates)
    ]
    policy_races = [
        race for race in calibration if str(race["race_date"]) in policy_dates
    ]
    prior_artifact = fit_conditional_ticket_residual(
        ranking_races,
        variant=variant,
        regularization=regularization,
        max_iterations=200,
    )
    policy_scored = _replace_probabilities(policy_races, prior_artifact)
    records = policy_edge_records(
        policy_scored,
        {"model_weight": 1.0, "temperature": 1.0},
        blend_probabilities,
    )
    first_evaluation_date = min(str(race["race_date"]) for race in evaluation)
    empirical = fit_contextual_empirical_ev_calibration(
        records,
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
    final_artifact = fit_conditional_ticket_residual(
        calibration,
        variant=variant,
        regularization=regularization,
        max_iterations=240,
    )
    evaluation_scored = _replace_probabilities(evaluation, final_artifact)
    bankroll = simulate_empirical_lcb_policy(
        evaluation_scored,
        {"model_weight": 1.0, "temperature": 1.0},
        blend_probabilities,
        empirical,
        daily_budget_yen,
    )
    return {
        "model": MODEL_NAME,
        "status": "completed",
        "validation_design": (
            "Candidate-level residual structure and regularization are selected "
            "on an inner later-day block by LogLoss then Top5. Empirical EV is "
            "fit on the next 30 days and scored once on untouched outer days."
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
        "artifact": final_artifact,
        "metrics": conditional_metrics(evaluation, final_artifact),
        "empirical_ev_calibration": empirical.as_dict(),
        "bankroll": bankroll,
        "promotion_eligible": bool(
            bankroll.get("tickets", 0) >= 50
            and bankroll.get("profit_yen", 0) > 0
            and (bankroll.get("roi") or 0.0) >= 1.05
        ),
    }


__all__ = [
    "FEATURE_VARIANTS",
    "conditional_metrics",
    "conditional_probabilities",
    "evaluate_temporal_conditional_ticket_residual",
    "fit_conditional_ticket_residual",
    "ticket_feature_matrix",
]
