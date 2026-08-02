from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
from math import fsum, isfinite, log
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .joint_market_value import (
    JointMarketScenario,
    validate_probability_simplex,
)


EPSILON = 1e-12
TEACHER_KIND = "strict_oof_terminal_probability_distribution"
MODEL_VERSION = "conditional_clr_joint_residual_v1"


@dataclass(frozen=True)
class JointScenarioObservation:
    race_date: str
    race_id: str
    teacher_trained_through_date: str
    terminal_probability_teacher_kind: str
    terminal_probability_teacher_source: str
    terminal_probability_artifact_sha256: str
    terminal_probability_fold_id: str
    terminal_probability_fold_manifest_sha256: str
    terminal_probability_prediction_sha256: str
    terminal_probability_outcome_schema_sha256: str
    terminal_probability_feature_cutoff_seconds: int
    venue: str
    decision_horizon_seconds: int
    popularity_band: str
    decision_probabilities: Mapping[str, float]
    terminal_probability_teacher: Mapping[str, float]
    decision_market_shares: Mapping[str, float]
    final_market_shares: Mapping[str, float]


@dataclass(frozen=True)
class ConditionalJointScenarioModel:
    version: str
    outcomes: tuple[str, ...]
    residual_mean: np.ndarray
    factor_components: np.ndarray
    factor_scales: np.ndarray
    diagonal_noise_scale: np.ndarray
    venue_means: Mapping[str, np.ndarray]
    horizon_means: Mapping[int, np.ndarray]
    popularity_means: Mapping[str, np.ndarray]
    group_means: Mapping[tuple[str, int, str], np.ndarray]
    group_counts: Mapping[tuple[str, int, str], int]
    pooling_strength: float
    training_races: int
    training_from: str
    training_through: str
    teacher_sources: tuple[str, ...]
    teacher_artifact_sha256s: tuple[str, ...]
    rank: int


def _iso_date(value: object, name: str) -> str:
    try:
        return date.fromisoformat(str(value)).isoformat()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an ISO date") from exc


def _context_key(
    venue: object,
    decision_horizon_seconds: object,
    popularity_band: object,
) -> tuple[str, int, str]:
    normalized_venue = str(venue).strip()
    normalized_band = str(popularity_band).strip()
    if not normalized_venue or not normalized_band:
        raise ValueError("venue and popularity_band must not be empty")
    if isinstance(decision_horizon_seconds, bool) or not isinstance(
        decision_horizon_seconds, (int, np.integer)
    ):
        raise ValueError("decision_horizon_seconds must be a positive integer")
    horizon = int(decision_horizon_seconds)
    if horizon <= 0:
        raise ValueError("decision_horizon_seconds must be a positive integer")
    return normalized_venue, horizon, normalized_band


def outcome_schema_fingerprint(outcomes: Sequence[str]) -> str:
    serialized = json.dumps(
        list(outcomes), ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def terminal_probability_prediction_fingerprint(
    *,
    race_id: str,
    probabilities: Mapping[str, float],
    artifact_sha256: str,
    fold_id: str,
    fold_manifest_sha256: str,
    feature_cutoff_seconds: int,
    outcomes: Sequence[str],
) -> str:
    payload = {
        "race_id": race_id,
        "artifact_sha256": artifact_sha256,
        "fold_id": fold_id,
        "fold_manifest_sha256": fold_manifest_sha256,
        "feature_cutoff_seconds": feature_cutoff_seconds,
        "outcomes": list(outcomes),
        "probabilities": [float(probabilities[outcome]) for outcome in outcomes],
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _sha256(value: object, name: str) -> str:
    normalized = str(value).strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    return normalized


def _clr(probabilities: Mapping[str, float], outcomes: Sequence[str]) -> np.ndarray:
    values = np.asarray(
        [max(EPSILON, float(probabilities[outcome])) for outcome in outcomes],
        dtype=np.float64,
    )
    logs = np.log(values)
    return logs - float(np.mean(logs))


def _softmax_mapping(values: np.ndarray, outcomes: Sequence[str]) -> dict[str, float]:
    centered = values - float(np.max(values))
    probabilities = np.exp(centered)
    probabilities /= float(np.sum(probabilities))
    return {
        outcome: float(probability)
        for outcome, probability in zip(outcomes, probabilities)
    }


def _validated_observation(
    observation: JointScenarioObservation,
    *,
    expected_outcomes: Sequence[str] | None,
) -> tuple[tuple[str, ...], tuple[str, int, str], np.ndarray]:
    race_date = _iso_date(observation.race_date, "race_date")
    trained_through = _iso_date(
        observation.teacher_trained_through_date,
        "teacher_trained_through_date",
    )
    if trained_through >= race_date:
        raise ValueError("terminal probability teacher must be trained strictly prior")
    if observation.terminal_probability_teacher_kind != TEACHER_KIND:
        raise ValueError("terminal probability teacher must be a strict OOF distribution")
    race_id = str(observation.race_id).strip()
    source = str(observation.terminal_probability_teacher_source).strip()
    fold_id = str(observation.terminal_probability_fold_id).strip()
    if not race_id or not source or not fold_id:
        raise ValueError("terminal teacher race, source and fold IDs must not be empty")
    if isinstance(observation.terminal_probability_feature_cutoff_seconds, bool):
        raise ValueError("terminal teacher feature cutoff must be zero seconds")
    if observation.terminal_probability_feature_cutoff_seconds != 0:
        raise ValueError("terminal teacher feature cutoff must be zero seconds")
    artifact_sha = _sha256(
        observation.terminal_probability_artifact_sha256,
        "terminal_probability_artifact_sha256",
    )
    prediction_sha = _sha256(
        observation.terminal_probability_prediction_sha256,
        "terminal_probability_prediction_sha256",
    )
    schema_sha = _sha256(
        observation.terminal_probability_outcome_schema_sha256,
        "terminal_probability_outcome_schema_sha256",
    )
    fold_manifest_sha = _sha256(
        observation.terminal_probability_fold_manifest_sha256,
        "terminal_probability_fold_manifest_sha256",
    )
    decision = validate_probability_simplex(
        observation.decision_probabilities,
        expected_outcomes=expected_outcomes,
    )
    outcomes = tuple(expected_outcomes or sorted(decision))
    terminal = validate_probability_simplex(
        observation.terminal_probability_teacher,
        expected_outcomes=outcomes,
    )
    terminal_values = np.asarray(list(terminal.values()), dtype=np.float64)
    if np.count_nonzero(terminal_values > EPSILON) < 2 or float(
        np.max(terminal_values)
    ) >= 1.0 - EPSILON:
        raise ValueError("literal one-hot terminal probability teachers are forbidden")
    if schema_sha != outcome_schema_fingerprint(outcomes):
        raise ValueError("terminal probability outcome schema hash mismatch")
    expected_prediction_sha = terminal_probability_prediction_fingerprint(
        race_id=race_id,
        probabilities=terminal,
        artifact_sha256=artifact_sha,
        fold_id=fold_id,
        fold_manifest_sha256=fold_manifest_sha,
        feature_cutoff_seconds=0,
        outcomes=outcomes,
    )
    if prediction_sha != expected_prediction_sha:
        raise ValueError("terminal probability prediction hash mismatch")
    market = validate_probability_simplex(
        observation.decision_market_shares,
        expected_outcomes=outcomes,
    )
    final_market = validate_probability_simplex(
        observation.final_market_shares,
        expected_outcomes=outcomes,
    )
    probability_residual = _clr(terminal, outcomes) - _clr(decision, outcomes)
    market_residual = _clr(final_market, outcomes) - _clr(market, outcomes)
    return (
        outcomes,
        _context_key(
            observation.venue,
            observation.decision_horizon_seconds,
            observation.popularity_band,
        ),
        np.concatenate((probability_residual, market_residual)),
    )


def fit_conditional_joint_scenario_model(
    observations: Sequence[JointScenarioObservation],
    *,
    expected_outcomes: Sequence[str] | None = None,
    rank: int = 8,
    pooling_strength: float = 20.0,
    diagonal_noise_fraction: float = 0.05,
) -> ConditionalJointScenarioModel:
    """Fit paired compositional residuals; never infer pi_T from one-hot results."""
    if len(observations) < 3:
        raise ValueError("at least three joint scenario observations are required")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("rank must be positive")
    if not isfinite(pooling_strength) or pooling_strength <= 0.0:
        raise ValueError("pooling_strength must be finite and positive")
    if not 0.0 <= diagonal_noise_fraction <= 1.0:
        raise ValueError("diagonal_noise_fraction must be in [0, 1]")
    parsed = [
        _validated_observation(row, expected_outcomes=expected_outcomes)
        for row in observations
    ]
    race_ids = [str(row.race_id) for row in observations]
    if len(set(race_ids)) != len(race_ids):
        raise ValueError("joint scenario observations must have unique race IDs")
    outcomes = parsed[0][0]
    if any(row[0] != outcomes for row in parsed):
        raise ValueError("all observations must use the same outcome order")
    residuals = np.vstack([row[2] for row in parsed])
    residual_mean = np.mean(residuals, axis=0)
    centered = residuals - residual_mean
    _u, singular_values, components = np.linalg.svd(
        centered, full_matrices=False
    )
    selected_rank = min(rank, len(observations) - 1, residuals.shape[1])
    factor_components = components[:selected_rank]
    factor_scales = singular_values[:selected_rank] / np.sqrt(
        max(1, len(observations) - 1)
    )
    reconstruction = (
        (centered @ factor_components.T) @ factor_components
        if selected_rank
        else np.zeros_like(centered)
    )
    diagonal_noise_scale = (
        np.std(centered - reconstruction, axis=0, ddof=1)
        * diagonal_noise_fraction
    )
    by_group: dict[tuple[str, int, str], list[np.ndarray]] = {}
    for _outcomes, key, residual in parsed:
        by_group.setdefault(key, []).append(residual)
    by_venue: dict[str, list[np.ndarray]] = {}
    by_horizon: dict[int, list[np.ndarray]] = {}
    by_popularity: dict[str, list[np.ndarray]] = {}
    for _outcomes, key, residual in parsed:
        by_venue.setdefault(key[0], []).append(residual)
        by_horizon.setdefault(key[1], []).append(residual)
        by_popularity.setdefault(key[2], []).append(residual)

    def pooled_main(rows: list[np.ndarray]) -> np.ndarray:
        count = len(rows)
        raw_mean = np.mean(np.vstack(rows), axis=0)
        weight = count / (count + pooling_strength)
        return residual_mean + weight * (raw_mean - residual_mean)

    venue_means = {key: pooled_main(rows) for key, rows in by_venue.items()}
    horizon_means = {key: pooled_main(rows) for key, rows in by_horizon.items()}
    popularity_means = {
        key: pooled_main(rows) for key, rows in by_popularity.items()
    }
    group_means = {}
    group_counts = {}
    for key, rows in by_group.items():
        count = len(rows)
        group_mean = np.mean(np.vstack(rows), axis=0)
        additive_main = (
            residual_mean
            + (venue_means[key[0]] - residual_mean)
            + (horizon_means[key[1]] - residual_mean)
            + (popularity_means[key[2]] - residual_mean)
        )
        weight = count / (count + pooling_strength)
        group_means[key] = additive_main + weight * (group_mean - additive_main)
        group_counts[key] = count
    dates = sorted(_iso_date(row.race_date, "race_date") for row in observations)
    return ConditionalJointScenarioModel(
        version=MODEL_VERSION,
        outcomes=outcomes,
        residual_mean=residual_mean,
        factor_components=factor_components,
        factor_scales=factor_scales,
        diagonal_noise_scale=diagonal_noise_scale,
        venue_means=venue_means,
        horizon_means=horizon_means,
        popularity_means=popularity_means,
        group_means=group_means,
        group_counts=group_counts,
        pooling_strength=float(pooling_strength),
        training_races=len(observations),
        training_from=dates[0],
        training_through=dates[-1],
        teacher_sources=tuple(sorted({
            str(row.terminal_probability_teacher_source) for row in observations
        })),
        teacher_artifact_sha256s=tuple(sorted({
            str(row.terminal_probability_artifact_sha256).lower()
            for row in observations
        })),
        rank=selected_rank,
    )


def generate_joint_market_scenarios(
    model: ConditionalJointScenarioModel,
    *,
    decision_probabilities: Mapping[str, float],
    decision_market_shares: Mapping[str, float],
    venue: str,
    decision_horizon_seconds: int,
    popularity_band: str,
    scenarios: int = 1_000,
    seed: int = 33036,
) -> list[JointMarketScenario]:
    if isinstance(scenarios, bool) or not isinstance(scenarios, int) or scenarios < 1:
        raise ValueError("scenarios must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    decision = validate_probability_simplex(
        decision_probabilities, expected_outcomes=model.outcomes
    )
    market = validate_probability_simplex(
        decision_market_shares, expected_outcomes=model.outcomes
    )
    key = _context_key(venue, decision_horizon_seconds, popularity_band)
    additive_main = (
        model.residual_mean
        + (model.venue_means.get(key[0], model.residual_mean) - model.residual_mean)
        + (
            model.horizon_means.get(key[1], model.residual_mean)
            - model.residual_mean
        )
        + (
            model.popularity_means.get(key[2], model.residual_mean)
            - model.residual_mean
        )
    )
    conditional_mean = model.group_means.get(key, additive_main)
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(scenarios, model.rank))
    residuals = conditional_mean + (
        factors * model.factor_scales
    ) @ model.factor_components
    if np.any(model.diagonal_noise_scale > 0.0):
        residuals += rng.normal(size=residuals.shape) * model.diagonal_noise_scale
    dimension = len(model.outcomes)
    base_probability = _clr(decision, model.outcomes)
    base_market = _clr(market, model.outcomes)
    weight = 1.0 / scenarios
    result = []
    for index, residual in enumerate(residuals):
        terminal_probability = _softmax_mapping(
            base_probability + residual[:dimension], model.outcomes
        )
        final_market = _softmax_mapping(
            base_market + residual[dimension:], model.outcomes
        )
        result.append(JointMarketScenario(
            probabilities=terminal_probability,
            market_state={
                "final_market_shares": final_market,
                "generator": model.version,
                "context_key": key,
                "context_fallback": key not in model.group_means,
                "scenario_index": index,
            },
            weight=weight,
        ))
    return result


def joint_scenario_model_diagnostics(
    model: ConditionalJointScenarioModel,
) -> dict[str, Any]:
    return {
        "version": model.version,
        "role": "diagnostic_joint_scenario_generator_not_yet_policy_connected",
        "training_races": model.training_races,
        "training_from": model.training_from,
        "training_through": model.training_through,
        "outcomes": len(model.outcomes),
        "rank": model.rank,
        "pooling_strength": model.pooling_strength,
        "groups": len(model.group_counts),
        "venue_main_effects": len(model.venue_means),
        "horizon_main_effects": len(model.horizon_means),
        "popularity_main_effects": len(model.popularity_means),
        "group_counts": {
            "|".join(map(str, key)): count
            for key, count in sorted(model.group_counts.items())
        },
        "teacher_kind": TEACHER_KIND,
        "teacher_sources": list(model.teacher_sources),
        "teacher_artifact_sha256s": list(model.teacher_artifact_sha256s),
        "actual_one_hot_used_as_terminal_probability_teacher": False,
        "parameter_uncertainty": (
            "not_in_single_fit; outer day-block refits are required"
        ),
    }


def _mean_scenario_simplex(
    scenarios: Sequence[JointMarketScenario],
    outcomes: Sequence[str],
    *,
    market: bool,
) -> dict[str, float]:
    values = {}
    for outcome in outcomes:
        if market:
            values[outcome] = fsum(
                scenario.weight
                * float(scenario.market_state["final_market_shares"][outcome])
                for scenario in scenarios
            )
        else:
            values[outcome] = fsum(
                scenario.weight * float(scenario.probabilities[outcome])
                for scenario in scenarios
            )
    total = fsum(values.values())
    return {outcome: value / total for outcome, value in values.items()}


def _walk_forward_metric_summary(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    count = len(rows)
    if not count:
        raise ValueError("walk-forward metric rows must not be empty")
    keys = tuple(rows[0])
    return {
        key: fsum(float(row[key]) for row in rows) / count
        for key in keys
    }


def evaluate_joint_scenario_walk_forward(
    observations: Sequence[JointScenarioObservation],
    actual_outcomes: Mapping[str, str],
    *,
    minimum_training_days: int = 3,
    scenarios_per_race: int = 128,
    rank: int = 8,
    pooling_strength: float = 20.0,
    seed: int = 33036,
) -> dict[str, Any]:
    """Evaluate the generator with strict prior-day refits; never tune policy."""
    if (
        isinstance(minimum_training_days, bool)
        or not isinstance(minimum_training_days, int)
        or minimum_training_days < 1
    ):
        raise ValueError("minimum_training_days must be a positive integer")
    if (
        isinstance(scenarios_per_race, bool)
        or not isinstance(scenarios_per_race, int)
        or scenarios_per_race < 1
    ):
        raise ValueError("scenarios_per_race must be a positive integer")
    by_day: dict[str, list[JointScenarioObservation]] = {}
    for observation in observations:
        day = _iso_date(observation.race_date, "race_date")
        by_day.setdefault(day, []).append(observation)
    dates = sorted(by_day)
    if len(dates) <= minimum_training_days:
        raise ValueError("insufficient days for joint scenario walk-forward")
    all_metric_rows = []
    daily = []
    outcomes: tuple[str, ...] | None = None
    for day_index in range(minimum_training_days, len(dates)):
        evaluation_date = dates[day_index]
        training_dates = dates[:day_index]
        training = [row for day in training_dates for row in by_day[day]]
        model = fit_conditional_joint_scenario_model(
            training,
            expected_outcomes=outcomes,
            rank=rank,
            pooling_strength=pooling_strength,
        )
        outcomes = model.outcomes
        day_metrics = []
        for observation in sorted(
            by_day[evaluation_date], key=lambda row: str(row.race_id)
        ):
            actual = str(actual_outcomes.get(str(observation.race_id)) or "")
            if actual not in outcomes:
                raise ValueError("walk-forward actual outcome is missing or invalid")
            race_seed = int.from_bytes(
                hashlib.sha256(
                    f"{seed}:{observation.race_id}".encode("utf-8")
                ).digest()[:8],
                byteorder="big",
                signed=False,
            )
            scenarios = generate_joint_market_scenarios(
                model,
                decision_probabilities=observation.decision_probabilities,
                decision_market_shares=observation.decision_market_shares,
                venue=observation.venue,
                decision_horizon_seconds=observation.decision_horizon_seconds,
                popularity_band=observation.popularity_band,
                scenarios=scenarios_per_race,
                seed=race_seed,
            )
            predicted_probability = _mean_scenario_simplex(
                scenarios, outcomes, market=False
            )
            predicted_market = _mean_scenario_simplex(
                scenarios, outcomes, market=True
            )
            decision_probability = validate_probability_simplex(
                observation.decision_probabilities,
                expected_outcomes=outcomes,
            )
            decision_market = validate_probability_simplex(
                observation.decision_market_shares,
                expected_outcomes=outcomes,
            )
            terminal_teacher = validate_probability_simplex(
                observation.terminal_probability_teacher,
                expected_outcomes=outcomes,
            )
            final_market = validate_probability_simplex(
                observation.final_market_shares,
                expected_outcomes=outcomes,
            )
            observed_pair = np.concatenate((
                _clr(terminal_teacher, outcomes) - _clr(decision_probability, outcomes),
                _clr(final_market, outcomes) - _clr(decision_market, outcomes),
            ))
            dimension = len(outcomes)
            generated_inner_products = []
            for scenario in scenarios:
                probability_residual = (
                    _clr(scenario.probabilities, outcomes)
                    - _clr(decision_probability, outcomes)
                )
                market_residual = (
                    _clr(
                        scenario.market_state["final_market_shares"], outcomes
                    )
                    - _clr(decision_market, outcomes)
                )
                generated_inner_products.append(
                    float(probability_residual @ market_residual) / dimension
                )
            one_hot = {
                outcome: float(outcome == actual) for outcome in outcomes
            }
            metrics = {
                "generated_log_loss": -log(max(EPSILON, predicted_probability[actual])),
                "decision_model_log_loss": -log(
                    max(EPSILON, decision_probability[actual])
                ),
                "decision_market_log_loss": -log(
                    max(EPSILON, decision_market[actual])
                ),
                "terminal_teacher_log_loss": -log(
                    max(EPSILON, terminal_teacher[actual])
                ),
                "generated_brier": fsum(
                    (predicted_probability[key] - one_hot[key]) ** 2
                    for key in outcomes
                ),
                "decision_model_brier": fsum(
                    (decision_probability[key] - one_hot[key]) ** 2
                    for key in outcomes
                ),
                "generated_top5": float(
                    actual
                    in sorted(
                        outcomes,
                        key=predicted_probability.get,
                        reverse=True,
                    )[:5]
                ),
                "decision_model_top5": float(
                    actual
                    in sorted(
                        outcomes,
                        key=decision_probability.get,
                        reverse=True,
                    )[:5]
                ),
                "closing_cross_entropy": -fsum(
                    final_market[key]
                    * log(max(EPSILON, predicted_market[key]))
                    for key in outcomes
                ),
                "decision_market_cross_entropy": -fsum(
                    final_market[key]
                    * log(max(EPSILON, decision_market[key]))
                    for key in outcomes
                ),
                "closing_total_variation": 0.5
                * fsum(
                    abs(predicted_market[key] - final_market[key])
                    for key in outcomes
                ),
                "decision_market_total_variation": 0.5
                * fsum(
                    abs(decision_market[key] - final_market[key])
                    for key in outcomes
                ),
                "observed_residual_inner_product": float(
                    observed_pair[:dimension] @ observed_pair[dimension:]
                )
                / dimension,
                "generated_residual_inner_product": fsum(
                    generated_inner_products
                )
                / len(generated_inner_products),
            }
            day_metrics.append(metrics)
            all_metric_rows.append(metrics)
        daily.append({
            "date": evaluation_date,
            "trained_through_date": training_dates[-1],
            "training_races": len(training),
            "evaluated_races": len(day_metrics),
            "metrics": _walk_forward_metric_summary(day_metrics),
        })
    return {
        "version": f"{MODEL_VERSION}_walk_forward_v1",
        "role": "diagnostic_joint_generator_not_policy_or_ga_fitness",
        "minimum_training_days": minimum_training_days,
        "scenarios_per_race": scenarios_per_race,
        "evaluated_days": len(daily),
        "evaluated_races": len(all_metric_rows),
        "evaluation_from": daily[0]["date"],
        "evaluation_through": daily[-1]["date"],
        "metrics": _walk_forward_metric_summary(all_metric_rows),
        "days": daily,
    }


__all__ = [
    "ConditionalJointScenarioModel",
    "JointScenarioObservation",
    "MODEL_VERSION",
    "TEACHER_KIND",
    "fit_conditional_joint_scenario_model",
    "evaluate_joint_scenario_walk_forward",
    "generate_joint_market_scenarios",
    "joint_scenario_model_diagnostics",
    "outcome_schema_fingerprint",
    "terminal_probability_prediction_fingerprint",
]
