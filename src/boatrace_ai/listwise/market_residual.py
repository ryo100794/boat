from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import random
from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from boatrace_ai.genetic_search import GeneticSearchSettings, evolve_population


EPSILON = 1e-12
DEFAULT_REGULARIZATION = (0.0, 0.001, 0.01, 0.1, 1.0, 10.0)
DECISION_OFFSET_SECONDS = 300
DECISION_INPUT_FIELDS = (
    "lane_context",
    "model_probabilities",
    "market_probabilities",
    "odds",
    "captured_at",
    "source_update_time",
    "input_snapshot_age_seconds",
    "odds_deadline_at",
    "earlier_market_probabilities",
    "earlier_snapshot_id",
    "earlier_captured_at",
    "earlier_snapshot_age_seconds",
    "momentum_interval_seconds",
    "momentum_scale",
    "odds_path",
    "odds_path_points",
)
POST_DECISION_TEACHER_FIELDS = (
    "actual_combination",
    "actual_payout_yen",
    "closing_odds",
    "official_closing_odds",
    "closing_snapshot_id",
    "closing_captured_at",
    "closing_source_update_time",
    "closing_snapshot_age_seconds",
    "closing_source_changed",
    "closing_odds_changed",
    "official_closing_source_key",
)
MARKET_RESIDUAL_GA_PROTOCOL = "genetic_t5_market_residual_v1"


@dataclass(frozen=True)
class MarketResidualGenome:
    family: str
    regularization: float
    lookback_days: int

    def validate(self) -> None:
        if self.family not in {"market_identity", "global_log_pool"}:
            raise ValueError(f"unsupported market residual family: {self.family}")
        if not math.isfinite(self.regularization) or self.regularization < 0.0:
            raise ValueError("regularization must be finite and non-negative")
        if self.lookback_days not in {0, 14, 30, 60, 120}:
            raise ValueError("lookback_days is outside the registered search space")


def _market_identity_calibrator(training_races: int) -> dict[str, Any]:
    return {
        "model_coefficient": 0.0,
        "market_coefficient": 1.0,
        "model_weight": 0.0,
        "temperature": 1.0,
        "regularization": 0.0,
        "objective": None,
        "gradient_norm": None,
        "iterations": 0,
        "converged": True,
        "training_races": int(training_races),
        "market_identity": True,
    }


def project_scored_race_for_residual(race: dict[str, Any]) -> dict[str, Any]:
    """Split an enriched scored race into immutable T-5 input and teachers."""
    missing = {
        "race_id",
        "race_date",
        "jcd",
        "rno",
        "model_probabilities",
        "market_probabilities",
        "odds",
    } - set(race)
    if missing:
        raise ValueError("scored race is missing: " + ", ".join(sorted(missing)))
    decision = {
        "race_id": str(race["race_id"]),
        "race_date": str(race["race_date"]),
        "jcd": str(race["jcd"]),
        "rno": int(race["rno"]),
    }
    for field in DECISION_INPUT_FIELDS:
        if field in race:
            decision[field] = deepcopy(race[field])
    checkpoints = race.get("odds_checkpoints")
    if isinstance(checkpoints, dict):
        decision["odds_checkpoints"] = {
            str(offset): deepcopy(point)
            for offset, point in checkpoints.items()
            if str(offset).isdigit() and int(offset) >= DECISION_OFFSET_SECONDS
        }
    path = decision.get("odds_path")
    if isinstance(path, list):
        decision["odds_path"] = [
            point
            for point in path
            if isinstance(point, dict)
            and float(point.get("minutes_before_decision", -1.0)) >= 0.0
        ]
        decision["odds_path_points"] = len(decision["odds_path"])
    teacher = {
        field: deepcopy(race[field])
        for field in POST_DECISION_TEACHER_FIELDS
        if field in race
    }
    return {
        "input_contract": "t300_residual_decision_v1",
        "decision": decision,
        "teacher": teacher,
    }


def residual_decision_fingerprint(projected: dict[str, Any]) -> str:
    decision = projected.get("decision")
    if not isinstance(decision, dict):
        raise ValueError("projected race must contain decision inputs")
    serialized = json.dumps(
        decision,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _raw_identity_calibrator(
    *,
    training_races: int,
    regularization: float,
    reason: str,
) -> dict[str, Any]:
    return {
        "model_coefficient": 1.0,
        "market_coefficient": 0.0,
        "model_weight": 1.0,
        "temperature": 1.0,
        "regularization": float(regularization),
        "objective": None,
        "gradient_norm": None,
        "iterations": 0,
        "converged": True,
        "training_races": int(training_races),
        "identity_fallback": True,
        "selection_reason": reason,
    }


def log_pool_probabilities(
    model: dict[str, float],
    market: dict[str, float],
    *,
    model_coefficient: float,
    market_coefficient: float,
) -> dict[str, float]:
    combinations = sorted(set(model) & set(market))
    if not combinations:
        return {}
    features = np.asarray(
        [
            [
                math.log(max(EPSILON, float(model[combination]))),
                math.log(max(EPSILON, float(market[combination]))),
            ]
            for combination in combinations
        ],
        dtype=np.float64,
    )
    logits = features @ np.asarray(
        [model_coefficient, market_coefficient], dtype=np.float64
    )
    logits -= float(np.max(logits))
    values = np.exp(logits)
    values /= float(np.sum(values))
    return {
        combination: float(value)
        for combination, value in zip(combinations, values)
    }


def _objective_gradient_hessian(
    races: list[dict[str, Any]],
    coefficients: np.ndarray,
    *,
    regularization: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    objective = 0.0
    gradient = np.zeros(2, dtype=np.float64)
    hessian = np.zeros((2, 2), dtype=np.float64)
    for race in races:
        combinations = sorted(
            set(race["model_probabilities"])
            & set(race["market_probabilities"])
        )
        actual = str(race["actual_combination"])
        if actual not in combinations:
            raise ValueError(f"actual combination {actual} is missing")
        features = np.asarray(
            [
                [
                    math.log(
                        max(
                            EPSILON,
                            float(race["model_probabilities"][combination]),
                        )
                    ),
                    math.log(
                        max(
                            EPSILON,
                            float(race["market_probabilities"][combination]),
                        )
                    ),
                ]
                for combination in combinations
            ],
            dtype=np.float64,
        )
        logits = features @ coefficients
        maximum = float(np.max(logits))
        exp_logits = np.exp(logits - maximum)
        probabilities = exp_logits / float(np.sum(exp_logits))
        actual_index = combinations.index(actual)
        log_partition = maximum + math.log(float(np.sum(exp_logits)))
        objective += log_partition - float(logits[actual_index])
        mean = probabilities @ features
        gradient += mean - features[actual_index]
        second_moment = (features.T * probabilities) @ features
        hessian += second_moment - np.outer(mean, mean)

    count = len(races)
    objective /= count
    gradient /= count
    hessian /= count
    prior = np.asarray([0.0, 1.0], dtype=np.float64)
    delta = coefficients - prior
    objective += 0.5 * regularization * float(delta @ delta)
    gradient += regularization * delta
    hessian += regularization * np.eye(2, dtype=np.float64)
    return objective, gradient, hessian


def fit_log_pool_newton(
    races: list[dict[str, Any]],
    *,
    regularization: float,
    max_iterations: int = 50,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    if not races:
        raise ValueError("at least one race is required")
    if regularization < 0.0 or not math.isfinite(regularization):
        raise ValueError("regularization must be finite and non-negative")
    coefficients = np.asarray([0.0, 1.0], dtype=np.float64)
    lower_bounds = np.asarray([0.0, 0.05], dtype=np.float64)
    converged = False
    objective = math.inf
    gradient = np.zeros(2, dtype=np.float64)
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        objective, gradient, hessian = _objective_gradient_hessian(
            races, coefficients, regularization=regularization
        )
        damped = hessian + 1e-9 * np.eye(2, dtype=np.float64)
        try:
            step = np.linalg.solve(damped, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(damped, gradient, rcond=None)[0]
        scale = 1.0
        accepted = False
        next_objective = objective
        next_coefficients = coefficients
        while scale >= 1e-8:
            candidate = np.maximum(coefficients - scale * step, lower_bounds)
            candidate_objective, _, _ = _objective_gradient_hessian(
                races, candidate, regularization=regularization
            )
            if candidate_objective <= objective + 1e-12:
                next_coefficients = candidate
                next_objective = candidate_objective
                accepted = True
                break
            scale *= 0.5
        parameter_change = float(np.max(np.abs(next_coefficients - coefficients)))
        objective_change = abs(objective - next_objective)
        coefficients = next_coefficients
        objective = next_objective
        if not accepted or (
            parameter_change <= tolerance and objective_change <= tolerance
        ):
            converged = True
            break
    objective, gradient, _ = _objective_gradient_hessian(
        races, coefficients, regularization=regularization
    )
    coefficient_sum = float(coefficients[0] + coefficients[1])
    return {
        "model_coefficient": float(coefficients[0]),
        "market_coefficient": float(coefficients[1]),
        "model_weight": float(coefficients[0]) / coefficient_sum,
        "temperature": 1.0 / coefficient_sum,
        "regularization": float(regularization),
        "objective": float(objective),
        "gradient_norm": float(np.linalg.norm(gradient)),
        "iterations": iterations,
        "converged": converged,
        "training_races": len(races),
    }


def residual_probability_metrics(
    races: list[dict[str, Any]],
    calibrator: dict[str, Any],
    *,
    include_raw_model: bool = False,
) -> dict[str, Any]:
    loss = 0.0
    market_loss = 0.0
    raw_model_loss = 0.0
    brier = 0.0
    market_brier = 0.0
    top5_hits = 0
    market_top5_hits = 0
    for race in races:
        probabilities = log_pool_probabilities(
            race["model_probabilities"],
            race["market_probabilities"],
            model_coefficient=float(calibrator["model_coefficient"]),
            market_coefficient=float(calibrator["market_coefficient"]),
        )
        actual = str(race["actual_combination"])
        loss -= math.log(max(EPSILON, probabilities.get(actual, 0.0)))
        market = race["market_probabilities"]
        market_loss -= math.log(max(EPSILON, float(market.get(actual, 0.0))))
        combinations = sorted(set(probabilities) & set(market))
        brier += sum(
            (
                float(probabilities[combination])
                - float(combination == actual)
            )
            ** 2
            for combination in combinations
        )
        market_brier += sum(
            (float(market[combination]) - float(combination == actual)) ** 2
            for combination in combinations
        )
        if include_raw_model:
            raw_model = race["model_probabilities"]
            raw_model_loss -= math.log(
                max(EPSILON, float(raw_model.get(actual, 0.0)))
            )
        top5_hits += int(
            actual in sorted(probabilities, key=probabilities.get, reverse=True)[:5]
        )
        market_top5_hits += int(
            actual in sorted(market, key=market.get, reverse=True)[:5]
        )
    count = len(races)
    result = {
        "evaluated_races": count,
        "trifecta_log_loss": loss / count if count else None,
        "market_trifecta_log_loss": market_loss / count if count else None,
        "trifecta_brier_score": brier / count if count else None,
        "market_trifecta_brier_score": market_brier / count if count else None,
        "trifecta_top5_hit_rate": top5_hits / count if count else None,
        "market_trifecta_top5_hit_rate": market_top5_hits / count if count else None,
    }
    if include_raw_model:
        result["raw_model_trifecta_log_loss"] = (
            raw_model_loss / count if count else None
        )
    return result


def _genome_key(genome: MarketResidualGenome) -> str:
    return json.dumps(
        {
            "family": genome.family,
            "regularization": round(genome.regularization, 12),
            "lookback_days": genome.lookback_days,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _serialize_genome(genome: MarketResidualGenome) -> dict[str, Any]:
    return {
        "family": genome.family,
        "regularization": float(genome.regularization),
        "lookback_days": int(genome.lookback_days),
    }


def _random_genome(rng: random.Random) -> MarketResidualGenome:
    return MarketResidualGenome(
        family="global_log_pool",
        regularization=10.0 ** rng.uniform(-5.0, 1.0),
        lookback_days=rng.choice((0, 14, 30, 60, 120)),
    )


def _crossover_genome(
    left: MarketResidualGenome,
    right: MarketResidualGenome,
    rng: random.Random,
) -> MarketResidualGenome:
    if left.family == "market_identity" and right.family == "market_identity":
        return left
    regularizations = [
        value
        for value in (left.regularization, right.regularization)
        if value > 0.0
    ]
    regularization = (
        math.sqrt(regularizations[0] * regularizations[-1])
        if regularizations
        else 1.0
    )
    return MarketResidualGenome(
        family="global_log_pool",
        regularization=regularization,
        lookback_days=rng.choice((left.lookback_days, right.lookback_days)),
    )


def _mutate_genome(
    genome: MarketResidualGenome,
    rng: random.Random,
    mutation_rate: float,
) -> MarketResidualGenome:
    if genome.family == "market_identity":
        return _random_genome(rng) if rng.random() < mutation_rate else genome
    regularization = genome.regularization
    lookback_days = genome.lookback_days
    if rng.random() < mutation_rate:
        regularization *= math.exp(rng.gauss(0.0, 1.0))
        regularization = min(10.0, max(1e-5, regularization))
    if rng.random() < mutation_rate:
        lookback_days = rng.choice((0, 14, 30, 60, 120))
    return MarketResidualGenome(
        family="global_log_pool",
        regularization=regularization,
        lookback_days=lookback_days,
    )


def _genetic_training_rows(races: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for race in races:
        projected = project_scored_race_for_residual(race)
        teacher = projected["teacher"]
        if "actual_combination" not in teacher:
            raise ValueError("actual_combination teacher is required")
        decision = projected["decision"]
        result.append(
            {
                "race_id": decision["race_id"],
                "race_date": decision["race_date"],
                "model_probabilities": decision["model_probabilities"],
                "market_probabilities": decision["market_probabilities"],
                "actual_combination": teacher["actual_combination"],
            }
        )
    return result


def _bootstrap_log_loss_delta_upper(
    folds: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> float:
    if samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    counts = np.asarray([row["evaluated_races"] for row in folds], dtype=np.float64)
    deltas = np.asarray([row["log_loss_delta"] for row in folds], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(folds), size=(samples, len(folds)))
    sampled_counts = counts[sampled]
    values = np.sum(deltas[sampled] * sampled_counts, axis=1) / np.sum(
        sampled_counts, axis=1
    )
    return float(np.quantile(values, 0.95))


def _evaluate_market_residual_genome(
    races: list[dict[str, Any]],
    genome: MarketResidualGenome,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    genome.validate()
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for race in races:
        by_day[str(race["race_date"])].append(race)
    dates = sorted(by_day)
    if len(dates) < 2:
        raise ValueError("at least two dates are required for genetic selection")
    folds = []
    for index in range(1, len(dates)):
        prior_dates = dates[:index]
        if genome.lookback_days:
            prior_dates = prior_dates[-genome.lookback_days :]
        training = [race for day in prior_dates for race in by_day[day]]
        calibrator = (
            _market_identity_calibrator(len(training))
            if genome.family == "market_identity"
            else fit_log_pool_newton(
                training, regularization=genome.regularization
            )
        )
        metrics = residual_probability_metrics(by_day[dates[index]], calibrator)
        folds.append(
            {
                "training_dates": prior_dates,
                "evaluation_date": dates[index],
                "evaluated_races": metrics["evaluated_races"],
                "log_loss_delta": (
                    metrics["trifecta_log_loss"]
                    - metrics["market_trifecta_log_loss"]
                ),
                "brier_delta": (
                    metrics["trifecta_brier_score"]
                    - metrics["market_trifecta_brier_score"]
                ),
                "top5_delta": (
                    metrics["trifecta_top5_hit_rate"]
                    - metrics["market_trifecta_top5_hit_rate"]
                ),
            }
        )
    total = sum(row["evaluated_races"] for row in folds)

    def weighted_mean(name: str) -> float:
        return sum(row[name] * row["evaluated_races"] for row in folds) / total

    return {
        "genome": _serialize_genome(genome),
        "prequential_races": total,
        "evaluation_days": len(folds),
        "mean_log_loss_delta": weighted_mean("log_loss_delta"),
        "log_loss_delta_ci95_upper": _bootstrap_log_loss_delta_upper(
            folds, samples=bootstrap_samples, seed=bootstrap_seed
        ),
        "worst_day_log_loss_delta": max(row["log_loss_delta"] for row in folds),
        "mean_brier_delta": weighted_mean("brier_delta"),
        "mean_top5_delta": weighted_mean("top5_delta"),
        "folds": folds,
    }


def _market_residual_fitness(
    metrics: dict[str, Any], genome: MarketResidualGenome
) -> float:
    del genome
    upper = float(metrics["log_loss_delta_ci95_upper"])
    mean = float(metrics["mean_log_loss_delta"])
    worst = float(metrics["worst_day_log_loss_delta"])
    brier = float(metrics["mean_brier_delta"])
    top5 = float(metrics["mean_top5_delta"])
    return (
        -upper
        - 0.25 * max(0.0, mean)
        - 0.25 * max(0.0, worst)
        - 0.10 * max(0.0, brier)
        + 0.05 * top5
    )


def select_market_residual_genetic(
    scored_races: list[dict[str, Any]],
    *,
    settings: GeneticSearchSettings | None = None,
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 33035,
) -> dict[str, Any]:
    """Select a T-5 residual structure on market-relative prior-day metrics."""
    training_rows = _genetic_training_rows(scored_races)
    selected_settings = settings or GeneticSearchSettings()

    def evaluator(genome: MarketResidualGenome) -> dict[str, Any]:
        return _evaluate_market_residual_genome(
            training_rows,
            genome,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )

    candidates, history = evolve_population(
        settings=selected_settings,
        evaluator=evaluator,
        fitness=_market_residual_fitness,
        random_candidate=_random_genome,
        crossover=_crossover_genome,
        mutate=_mutate_genome,
        candidate_key=_genome_key,
        serialize=_serialize_genome,
        immigrants=(
            MarketResidualGenome("market_identity", 0.0, 0),
            MarketResidualGenome("global_log_pool", 1.0, 0),
            MarketResidualGenome("global_log_pool", 0.01, 30),
        ),
    )
    champion = candidates[0]
    final_calibrator = (
        _market_identity_calibrator(len(training_rows))
        if champion.candidate.family == "market_identity"
        else fit_log_pool_newton(
            training_rows,
            regularization=champion.candidate.regularization,
        )
    )
    return {
        "protocol": MARKET_RESIDUAL_GA_PROTOCOL,
        "input_contract": "t300_residual_decision_v1",
        "selection_data": "strict_prior_daily_prequential_only",
        "outer_holdout_used": False,
        "fitness_definition": (
            "market-relative day-block LogLoss upper bound, mean/worst-day "
            "LogLoss, Brier and 3T5"
        ),
        "settings": {
            "population_size": selected_settings.population_size,
            "generations": selected_settings.generations,
            "elite_count": selected_settings.elite_count,
            "mutation_rate": selected_settings.mutation_rate,
            "random_injections": selected_settings.random_injections,
            "max_workers": selected_settings.max_workers,
            "seed": selected_settings.seed,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
        },
        "champion": _serialize_genome(champion.candidate),
        "champion_fitness": champion.fitness,
        "champion_metrics": dict(champion.metrics),
        "final_calibrator": final_calibrator,
        "generation_history": history,
        "candidate_metrics": [
            {
                "genome": _serialize_genome(row.candidate),
                "fitness": row.fitness,
                "first_generation": row.first_generation,
                **dict(row.metrics),
            }
            for row in candidates
        ],
    }


def select_regularization_prequential(
    races: list[dict[str, Any]],
    *,
    regularizations: Iterable[float] = DEFAULT_REGULARIZATION,
    enforce_raw_nonregression: bool = False,
) -> dict[str, Any]:
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for race in races:
        by_day[str(race["race_date"])].append(race)
    dates = sorted(by_day)
    if len(dates) < 2:
        raise ValueError("at least two dates are required for prequential selection")

    candidates = []
    for regularization in regularizations:
        folds = []
        weighted_loss = 0.0
        weighted_raw_model_loss = 0.0
        total = 0
        for index in range(1, len(dates)):
            training = [
                race for race_date in dates[:index] for race in by_day[race_date]
            ]
            holdout = by_day[dates[index]]
            calibrator = fit_log_pool_newton(
                training, regularization=float(regularization)
            )
            metrics = residual_probability_metrics(
                holdout,
                calibrator,
                include_raw_model=enforce_raw_nonregression,
            )
            count = int(metrics["evaluated_races"])
            weighted_loss += float(metrics["trifecta_log_loss"]) * count
            if enforce_raw_nonregression:
                weighted_raw_model_loss += (
                    float(metrics["raw_model_trifecta_log_loss"]) * count
                )
            total += count
            folds.append(
                {
                    "training_dates": dates[:index],
                    "evaluation_date": dates[index],
                    "calibrator": calibrator,
                    "metrics": metrics,
                }
            )
        candidate = {
            "regularization": float(regularization),
            "prequential_races": total,
            "prequential_log_loss": weighted_loss / total,
            "folds": folds,
        }
        if enforce_raw_nonregression:
            candidate["raw_model_prequential_log_loss"] = (
                weighted_raw_model_loss / total
            )
        candidates.append(candidate)
    selected = min(
        candidates,
        key=lambda row: (
            row["prequential_log_loss"],
            -row["regularization"],
        ),
    )
    fitted_calibrator = fit_log_pool_newton(
        races, regularization=float(selected["regularization"])
    )
    if not enforce_raw_nonregression:
        return {
            "validation_design": (
                "Regularization is selected on forward-only daily folds; final "
                "coefficients are refit on all calibration days"
            ),
            "dates": dates,
            "selected_regularization": selected["regularization"],
            "prequential_log_loss": selected["prequential_log_loss"],
            "final_calibrator": fitted_calibrator,
            "candidates": candidates,
        }
    fitted_metrics = residual_probability_metrics(
        races, fitted_calibrator, include_raw_model=True
    )
    prequential_regression = (
        float(selected["prequential_log_loss"])
        > float(selected["raw_model_prequential_log_loss"])
    )
    refit_regression = (
        float(fitted_metrics["trifecta_log_loss"])
        > float(fitted_metrics["raw_model_trifecta_log_loss"])
    )
    fallback_reason = (
        "calibrated_prequential_log_loss_worse_than_raw"
        if prequential_regression
        else "calibrated_prior_refit_log_loss_worse_than_raw"
        if refit_regression
        else None
    )
    final_calibrator = (
        _raw_identity_calibrator(
            training_races=len(races),
            regularization=float(selected["regularization"]),
            reason=str(fallback_reason),
        )
        if fallback_reason is not None
        else fitted_calibrator
    )
    return {
        "validation_design": (
            "Regularization is selected on forward-only daily folds; final coefficients "
            "are refit on all calibration days; raw identity fallback is selected "
            "without access to the outer holdout"
        ),
        "dates": dates,
        "selected_regularization": selected["regularization"],
        "prequential_log_loss": min(
            float(selected["prequential_log_loss"]),
            float(selected["raw_model_prequential_log_loss"]),
        ),
        "candidate_prequential_log_loss": selected["prequential_log_loss"],
        "raw_model_prequential_log_loss": selected[
            "raw_model_prequential_log_loss"
        ],
        "calibration_nonregression": {
            "selection_data": "strict_prior_prequential_and_prior_refit_only",
            "outer_holdout_used": False,
            "candidate_trifecta_log_loss": selected["prequential_log_loss"],
            "raw_trifecta_log_loss": selected[
                "raw_model_prequential_log_loss"
            ],
            "prior_refit_candidate_trifecta_log_loss": fitted_metrics[
                "trifecta_log_loss"
            ],
            "prior_refit_raw_trifecta_log_loss": fitted_metrics[
                "raw_model_trifecta_log_loss"
            ],
            "identity_fallback_applied": fallback_reason is not None,
            "reason": fallback_reason,
        },
        "final_calibrator": final_calibrator,
        "candidates": candidates,
    }


def fit_fixed_regularization(
    races: list[dict[str, Any]],
    *,
    regularization: float = 1.0,
    enforce_raw_nonregression: bool = False,
) -> dict[str, Any]:
    dates = sorted({str(race["race_date"]) for race in races})
    fitted_calibrator = fit_log_pool_newton(races, regularization=regularization)
    if not enforce_raw_nonregression:
        return {
            "validation_design": (
                "Regularization is preregistered because fewer than two calibration "
                "days are available; no holdout selection is performed"
            ),
            "dates": dates,
            "selected_regularization": float(regularization),
            "prequential_log_loss": None,
            "final_calibrator": fitted_calibrator,
            "candidates": [],
        }
    metrics = residual_probability_metrics(
        races, fitted_calibrator, include_raw_model=True
    )
    fallback = (
        float(metrics["trifecta_log_loss"])
        > float(metrics["raw_model_trifecta_log_loss"])
    )
    calibrator = (
        _raw_identity_calibrator(
            training_races=len(races),
            regularization=regularization,
            reason="calibrated_training_log_loss_worse_than_raw",
        )
        if fallback
        else fitted_calibrator
    )
    return {
        "validation_design": (
            "Regularization is preregistered because fewer than two calibration "
            "days are available; no holdout selection is performed"
        ),
        "dates": dates,
        "selected_regularization": float(regularization),
        "prequential_log_loss": None,
        "calibration_nonregression": {
            "selection_data": "single_prior_training_day_only",
            "outer_holdout_used": False,
            "candidate_trifecta_log_loss": metrics["trifecta_log_loss"],
            "raw_trifecta_log_loss": metrics["raw_model_trifecta_log_loss"],
            "identity_fallback_applied": fallback,
            "reason": (
                "calibrated_training_log_loss_worse_than_raw" if fallback else None
            ),
        },
        "final_calibrator": calibrator,
        "candidates": [],
    }
