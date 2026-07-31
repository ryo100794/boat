from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.optimize import minimize

from ..bankroll_bootstrap import bootstrap_daily_roi
from .contextual_market_residual_v24 import (
    EPSILON,
    FEATURE_DIMENSION as BASE_FEATURE_DIMENSION,
    PreparedRace,
    _lanes,
    prepare_race,
)
from .flat_policy import simulate_chronological_flat_policy
from .market_calibration import blend_probabilities
from .pruned_direct_context_v27 import (
    FEATURE_VARIANTS,
    _lane_context_matrix,
    _validate_active_features,
    fit_pruned_residual,
    pruned_probabilities,
)


MODEL_NAME = "course_interaction_market_residual_v28"
STAGES = 3
LANES = 6
SELECTION_TOLERANCE = 0.0005
TOP5_TOLERANCE = 1e-12
REGULARIZATIONS = (0.01, 0.03, 0.1)
STRUCTURE_VARIANTS: dict[str, tuple[str, str]] = {
    "shared_independent_core": ("stage_shared", "independent_core"),
    "course_ability_raw": ("course_gated", "ability_raw"),
    "course_equipment_history": ("course_gated", "equipment_history"),
    "course_independent_core": ("course_gated", "independent_core"),
}


@dataclass(frozen=True)
class PreparedCourseRace:
    base: PreparedRace
    stage_lanes: np.ndarray
    lane_context: np.ndarray


def _prepare_course_race(
    race: dict[str, Any], active_features: tuple[str, ...]
) -> PreparedCourseRace:
    base = prepare_race(race)
    return PreparedCourseRace(
        base=base,
        stage_lanes=np.asarray(
            [_lanes(combination) for combination in base.combinations],
            dtype=np.int8,
        ),
        lane_context=_lane_context_matrix(race, active_features),
    )


def _course_feature_dimension(active_features: tuple[str, ...]) -> int:
    return BASE_FEATURE_DIMENSION + STAGES * LANES * len(active_features) * 2


def _course_probability_vector(
    race: PreparedCourseRace, coefficients: np.ndarray
) -> np.ndarray:
    base_coefficients = coefficients[:BASE_FEATURE_DIMENSION]
    context_coefficients = coefficients[BASE_FEATURE_DIMENSION:].reshape(
        STAGES, LANES, race.lane_context.shape[1]
    )
    logits = race.base.market_log + np.sum(
        base_coefficients[race.base.indices] * race.base.values,
        axis=1,
    )
    for stage in range(STAGES):
        lanes = race.stage_lanes[:, stage]
        logits += np.einsum(
            "ij,ij->i",
            race.lane_context[lanes],
            context_coefficients[stage, lanes],
        )
    logits -= float(np.max(logits))
    probabilities = np.exp(logits)
    probabilities /= float(np.sum(probabilities))
    return probabilities


def _course_objective_gradient(
    coefficients: np.ndarray,
    races: list[PreparedCourseRace],
    *,
    regularization: float,
) -> tuple[float, np.ndarray]:
    loss = 0.0
    gradient = np.zeros_like(coefficients)
    base_gradient = gradient[:BASE_FEATURE_DIMENSION]
    context_gradient = gradient[BASE_FEATURE_DIMENSION:].reshape(
        STAGES, LANES, races[0].lane_context.shape[1]
    )
    for race in races:
        probabilities = _course_probability_vector(race, coefficients)
        loss -= math.log(
            max(EPSILON, float(probabilities[race.base.actual_index]))
        )
        errors = probabilities
        errors[race.base.actual_index] -= 1.0
        np.add.at(
            base_gradient,
            race.base.indices.reshape(-1),
            (errors[:, None] * race.base.values).reshape(-1),
        )
        for stage in range(STAGES):
            lanes = race.stage_lanes[:, stage]
            lane_errors = np.bincount(lanes, weights=errors, minlength=LANES)
            context_gradient[stage] += (
                lane_errors[:, None] * race.lane_context
            )
    scale = 1.0 / len(races)
    loss *= scale
    gradient *= scale
    loss += 0.5 * regularization * float(coefficients @ coefficients)
    gradient += regularization * coefficients
    return loss, gradient


def fit_course_residual(
    races: list[dict[str, Any]],
    *,
    feature_variant: str,
    regularization: float,
    max_iterations: int,
) -> dict[str, Any]:
    if not races:
        raise ValueError("at least one race is required")
    if feature_variant not in FEATURE_VARIANTS:
        raise ValueError(f"unknown V28 feature variant: {feature_variant}")
    if regularization <= 0.0 or not math.isfinite(regularization):
        raise ValueError("regularization must be finite and positive")
    active = _validate_active_features(FEATURE_VARIANTS[feature_variant])
    prepared = [_prepare_course_race(race, active) for race in races]
    dimension = _course_feature_dimension(active)

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        return _course_objective_gradient(
            coefficients, prepared, regularization=regularization
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
        "architecture": "finish_stage_by_starting_lane_context",
        "feature_variant": feature_variant,
        "active_context_features": list(active),
        "active_context_feature_count": len(active),
        "feature_dimension": dimension,
        "base_feature_dimension": BASE_FEATURE_DIMENSION,
        "regularization": float(regularization),
        "coefficients": [float(value) for value in result.x],
        "objective": float(objective_value),
        "gradient_norm": float(np.linalg.norm(gradient)),
        "iterations": int(result.nit),
        "converged": bool(result.success),
        "message": str(result.message),
        "training_races": len(races),
    }


def fit_structure_residual(
    races: list[dict[str, Any]],
    *,
    structure_variant: str,
    regularization: float,
    max_iterations: int,
) -> dict[str, Any]:
    try:
        architecture, feature_variant = STRUCTURE_VARIANTS[structure_variant]
    except KeyError as exc:
        raise ValueError(f"unknown V28 structure variant: {structure_variant}") from exc
    if architecture == "stage_shared":
        artifact = fit_pruned_residual(
            races,
            variant=feature_variant,
            regularization=regularization,
            max_iterations=max_iterations,
        )
        architecture_name = "finish_stage_shared_context"
    else:
        artifact = fit_course_residual(
            races,
            feature_variant=feature_variant,
            regularization=regularization,
            max_iterations=max_iterations,
        )
        architecture_name = "finish_stage_by_starting_lane_context"
    return {
        **artifact,
        "model": MODEL_NAME,
        "architecture": architecture_name,
        "structure_variant": structure_variant,
    }


def structure_probabilities(
    race: dict[str, Any], artifact: Mapping[str, Any]
) -> dict[str, float]:
    if artifact.get("architecture") == "finish_stage_shared_context":
        return pruned_probabilities(race, artifact)
    active = _validate_active_features(artifact.get("active_context_features") or ())
    prepared = _prepare_course_race(race, active)
    coefficients = np.asarray(artifact["coefficients"], dtype=np.float64)
    if coefficients.shape != (_course_feature_dimension(active),):
        raise ValueError("course-interaction residual coefficient shape mismatch")
    probabilities = _course_probability_vector(prepared, coefficients)
    return {
        combination: float(value)
        for combination, value in zip(prepared.base.combinations, probabilities)
    }


def structure_metrics(
    races: list[dict[str, Any]], artifact: Mapping[str, Any]
) -> dict[str, Any]:
    loss = market_loss = raw_model_loss = 0.0
    top5_hits = market_top5_hits = 0
    for race in races:
        probabilities = structure_probabilities(race, artifact)
        market = race["market_probabilities"]
        model = race["model_probabilities"]
        actual = str(race["actual_combination"])
        loss -= math.log(max(EPSILON, float(probabilities.get(actual, 0.0))))
        market_loss -= math.log(max(EPSILON, float(market.get(actual, 0.0))))
        raw_model_loss -= math.log(max(EPSILON, float(model.get(actual, 0.0))))
        top5_hits += int(
            actual in sorted(probabilities, key=probabilities.get, reverse=True)[:5]
        )
        market_top5_hits += int(
            actual in sorted(market, key=market.get, reverse=True)[:5]
        )
    count = len(races)
    return {
        "evaluated_races": count,
        "trifecta_log_loss": loss / count if count else None,
        "market_trifecta_log_loss": market_loss / count if count else None,
        "raw_model_trifecta_log_loss": raw_model_loss / count if count else None,
        "trifecta_top5_hit_rate": top5_hits / count if count else None,
        "market_trifecta_top5_hit_rate": market_top5_hits / count if count else None,
    }


def evaluate_temporal_course_interaction(
    calibration: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
    *,
    policies: Iterable[Mapping[str, Any]],
    daily_budget_yen: int,
    structure_variants: Mapping[str, tuple[str, str]] = STRUCTURE_VARIANTS,
    regularizations: Iterable[float] = REGULARIZATIONS,
    selection_tolerance: float = SELECTION_TOLERANCE,
) -> dict[str, Any]:
    dates = sorted({str(race["race_date"]) for race in calibration})
    if len(dates) < 2:
        raise ValueError("at least two calibration days are required")
    split_index = max(1, min(len(dates) - 1, int(len(dates) * 0.8)))
    fit_dates = set(dates[:split_index])
    validation_dates = set(dates[split_index:])
    inner_fit = [race for race in calibration if str(race["race_date"]) in fit_dates]
    inner_validation = [
        race for race in calibration if str(race["race_date"]) in validation_dates
    ]
    regularization_values = tuple(float(value) for value in regularizations)
    if not regularization_values:
        raise ValueError("at least one regularization is required")

    candidates = []
    for structure_variant, design in structure_variants.items():
        if STRUCTURE_VARIANTS.get(structure_variant) != tuple(design):
            raise ValueError("structure differs from the preregistered V28 design")
        for regularization in regularization_values:
            artifact = fit_structure_residual(
                inner_fit,
                structure_variant=structure_variant,
                regularization=regularization,
                max_iterations=120,
            )
            candidates.append({
                "structure_variant": structure_variant,
                "architecture": artifact["architecture"],
                "feature_variant": artifact["feature_variant"],
                "active_context_feature_count": artifact[
                    "active_context_feature_count"
                ],
                "feature_dimension": artifact["feature_dimension"],
                "regularization": regularization,
                "converged": artifact["converged"],
                "gradient_norm": artifact["gradient_norm"],
                "metrics": structure_metrics(inner_validation, artifact),
            })
    converged = [row for row in candidates if row["converged"]]
    eligible = converged or candidates
    best_loss = min(float(row["metrics"]["trifecta_log_loss"]) for row in eligible)
    near_best = [
        row
        for row in eligible
        if float(row["metrics"]["trifecta_log_loss"])
        <= best_loss + float(selection_tolerance)
    ]
    best_top5 = max(
        float(row["metrics"]["trifecta_top5_hit_rate"]) for row in near_best
    )
    top5_near_best = [
        row
        for row in near_best
        if float(row["metrics"]["trifecta_top5_hit_rate"])
        >= best_top5 - TOP5_TOLERANCE
    ]
    selected = min(
        top5_near_best,
        key=lambda row: (
            int(row["feature_dimension"]),
            -float(row["regularization"]),
            float(row["metrics"]["trifecta_log_loss"]),
        ),
    )
    artifact = fit_structure_residual(
        calibration,
        structure_variant=str(selected["structure_variant"]),
        regularization=float(selected["regularization"]),
        max_iterations=200,
    )
    scored = [
        {**race, "model_probabilities": structure_probabilities(race, artifact)}
        for race in evaluation
    ]
    purchase_diagnostics = []
    for policy in policies:
        simulation = simulate_chronological_flat_policy(
            scored,
            calibrator={"model_weight": 1.0, "temperature": 1.0},
            policy=dict(policy),
            probability_blender=blend_probabilities,
            initial_bankroll_yen=daily_budget_yen,
        )
        bootstrap = (
            bootstrap_daily_roi(simulation["daily"])
            if simulation["daily"]
            else {
                "days": 0,
                "roi": None,
                "roi_ci95_lower": None,
                "probability_roi_above_one": None,
            }
        )
        purchase_diagnostics.append({
            "policy": dict(policy),
            "simulation": simulation,
            "bootstrap": bootstrap,
        })
    return {
        "model": MODEL_NAME,
        "validation_design": (
            "Structure and regularization are selected only on the latest inner "
            "prior-day block. Converged candidates within 0.0005 LogLoss of the "
            "minimum maximize Top5 before preferring lower dimension; outer days "
            "remain untouched."
        ),
        "inner_fit_through": dates[split_index - 1],
        "inner_validation_from": dates[split_index],
        "selection_tolerance": float(selection_tolerance),
        "best_inner_log_loss": best_loss,
        "best_near_inner_top5": best_top5,
        "selected_candidate": selected,
        "candidates": candidates,
        "artifact": artifact,
        "metrics": structure_metrics(evaluation, artifact),
        "purchase_diagnostics": purchase_diagnostics,
    }
