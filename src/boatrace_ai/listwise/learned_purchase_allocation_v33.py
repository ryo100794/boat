from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from .four_head_nested_v22 import (
    DecisionRace,
    LabeledRace,
    RacePrediction,
    T5_ODDS_PATH_SCHEMA,
    validate_decision_odds_path,
)


MODEL_KEY = "learned_purchase_allocation_head_v33"
TEACHER = "daily_weighted_realized_log_bankroll_growth"
INFORMATION_BOUNDARY = "strict_prior_heads_and_decision_time_t5_only"
STAKE_UNIT_YEN = 100
MAX_RACE_EXPOSURE_FRACTION = 0.05
EPSILON = 1e-12


@dataclass(frozen=True)
class AllocationConfig:
    name: str
    regularization: float
    downside_penalty: float


DEFAULT_CONFIGS = (
    AllocationConfig("balanced", 0.03, 0.5),
    AllocationConfig("stable", 0.10, 1.0),
    AllocationConfig("conservative", 0.30, 2.0),
)
EXHAUSTIVE_CONFIGS_V1 = tuple(
    AllocationConfig(
        f"grid-r{regularization:g}-d{downside:g}",
        regularization,
        downside,
    )
    for regularization in (1e-5, 1e-4, 1e-3, 1e-2, 0.03, 0.1, 0.3)
    for downside in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
)


@dataclass(frozen=True)
class PreparedAllocationRace:
    race_id: str
    race_date: str
    ticket_features: np.ndarray
    race_features: np.ndarray
    base_log_probability: np.ndarray
    winner_index: int
    payout_odds: float


@dataclass(frozen=True)
class PreparedAllocationBatch:
    ticket: np.ndarray
    race: np.ndarray
    base_log_probability: np.ndarray
    winner_index: np.ndarray
    payout_odds: np.ndarray
    day_weight: np.ndarray


@dataclass(frozen=True)
class LearnedAllocationArtifact:
    model_key: str
    teacher: str
    trained_through_date: str
    base_predictions_trained_through_date: str
    training_race_ids: tuple[str, ...]
    ticket_feature_mean: tuple[float, ...]
    ticket_feature_scale: tuple[float, ...]
    race_feature_mean: tuple[float, ...]
    race_feature_scale: tuple[float, ...]
    allocation_coefficients: tuple[float, ...]
    gate_coefficients: tuple[float, ...]
    selected_config: AllocationConfig
    objective: float
    gradient_norm: float
    iterations: int
    converged: bool
    candidate_metrics: tuple[Mapping[str, Any], ...]
    training_input_sha256: str
    max_race_exposure_fraction: float = MAX_RACE_EXPOSURE_FRACTION
    information_boundary: str = INFORMATION_BOUNDARY
    outer_outcomes_used: bool = False
    odds_path_schema: str | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "teacher": self.teacher,
            "trained_through_date": self.trained_through_date,
            "base_predictions_trained_through_date": (
                self.base_predictions_trained_through_date
            ),
            "training_races": len(self.training_race_ids),
            "ticket_feature_count": len(self.ticket_feature_mean),
            "race_feature_count": len(self.race_feature_mean),
            "selected_config": self.selected_config.__dict__,
            "objective": self.objective,
            "gradient_norm": self.gradient_norm,
            "iterations": self.iterations,
            "converged": self.converged,
            "candidate_metrics": [dict(row) for row in self.candidate_metrics],
            "training_input_sha256": self.training_input_sha256,
            "max_race_exposure_fraction": self.max_race_exposure_fraction,
            "information_boundary": self.information_boundary,
            "outer_outcomes_used": self.outer_outcomes_used,
            "allocation_objective_version": 2,
            "gate_intercept_regularized": False,
            "odds_path_schema": self.odds_path_schema,
        }


@dataclass(frozen=True)
class AllocationDecision:
    race_id: str
    exposure_fraction: float
    proposed_stake_yen: int
    stakes_yen: tuple[int, ...]
    allocation_weights: tuple[float, ...]
    gate_probability: float

    @property
    def selected_indices(self) -> tuple[int, ...]:
        return tuple(index for index, stake in enumerate(self.stakes_yen) if stake > 0)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - float(np.max(values))
    result = np.exp(np.clip(shifted, -60.0, 0.0))
    return result / float(result.sum())


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    direct = math.exp(value)
    return direct / (1.0 + direct)


def _entropy(values: np.ndarray) -> float:
    positive = values[values > 0.0]
    return -float(np.sum(positive * np.log(positive)))


def _odds_path_feature_matrices(
    decision: DecisionRace, *, choices: int
) -> tuple[np.ndarray, np.ndarray]:
    path = decision.odds_path
    if path is None:
        return np.empty((choices, 0), dtype=np.float64), np.empty(0, dtype=np.float64)
    points = validate_decision_odds_path(
        path, choices=choices, expected_t300_odds=decision.current_odds
    )
    log_market: dict[int, np.ndarray] = {}
    for offset, point in points.items():
        implied = 1.0 / np.asarray(point.odds, dtype=np.float64)
        implied /= float(implied.sum())
        log_market[offset] = np.log(implied)
    availability = {
        offset: float(offset in log_market) for offset in (420, 600, 1200, 1800)
    }

    def slope(older: int, newer: int) -> np.ndarray:
        if older not in log_market or newer not in log_market:
            return np.zeros(choices, dtype=np.float64)
        minutes = (older - newer) / 60.0
        return (log_market[newer] - log_market[older]) / minutes

    recent = slope(420, 300)
    long = slope(1800, 300)
    acceleration = (
        recent - slope(600, 420)
        if 420 in log_market and 600 in log_market
        else np.zeros(choices, dtype=np.float64)
    )
    ordered = np.vstack([log_market[offset] for offset in sorted(log_market, reverse=True)])
    volatility = (
        np.std(ordered, axis=0)
        if len(ordered) >= 2
        else np.zeros(choices, dtype=np.float64)
    )
    mask_columns = np.tile(
        np.asarray(tuple(availability.values()), dtype=np.float64), (choices, 1)
    )
    ticket = np.column_stack((recent, long, acceleration, volatility, mask_columns))
    observed_offsets = sorted(log_market)
    span_minutes = (
        (max(observed_offsets) - min(observed_offsets)) / 60.0
        if len(observed_offsets) >= 2
        else 0.0
    )
    race = np.asarray(
        (
            len(observed_offsets) / 5.0,
            span_minutes / 25.0,
            float(np.mean(volatility)),
            float(np.std(volatility)),
            *availability.values(),
        ),
        dtype=np.float64,
    )
    return ticket, race


def decision_feature_matrices(
    decision: DecisionRace, prediction: RacePrediction
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if prediction.race_id != decision.race_id or prediction.race_date != decision.race_date:
        raise ValueError("allocation prediction does not match decision race")
    raw = np.asarray(decision.features, dtype=np.float64)
    odds = np.asarray(decision.current_odds, dtype=np.float64)
    probability = np.asarray(prediction.probabilities, dtype=np.float64)
    ranking = np.asarray(prediction.ranking_scores, dtype=np.float64)
    closing = np.asarray(prediction.predicted_closing_odds, dtype=np.float64)
    choices = len(odds)
    if (
        raw.ndim != 2
        or raw.shape[0] != choices
        or any(values.shape != (choices,) for values in (probability, ranking, closing))
    ):
        raise ValueError("allocation decision vectors have inconsistent shapes")
    if (
        not np.isfinite(raw).all()
        or not all(np.isfinite(values).all() for values in (odds, probability, ranking, closing))
        or np.any(odds <= 1.0)
        or np.any(probability <= 0.0)
        or np.any(closing <= 1.0)
    ):
        raise ValueError("allocation decision inputs are invalid")
    probability /= float(probability.sum())
    implied = 1.0 / odds
    implied /= float(implied.sum())
    ranking_order = np.argsort(-ranking, kind="stable")
    ranking_rank = np.empty(choices, dtype=np.float64)
    ranking_rank[ranking_order] = np.arange(choices, dtype=np.float64)
    ranking_rank /= max(1, choices - 1)
    path_ticket, path_race = _odds_path_feature_matrices(
        decision, choices=choices
    )
    ticket = np.column_stack(
        (
            raw,
            np.log(probability),
            np.log(implied),
            np.log(odds),
            np.log(closing),
            np.log(probability / implied),
            np.log(odds / closing),
            probability * odds,
            probability * closing,
            ranking_rank,
            path_ticket,
        )
    )
    sorted_probability = np.sort(probability)[::-1]
    sorted_implied = np.sort(implied)[::-1]
    race = np.concatenate(
        (
            np.mean(raw, axis=0),
            np.std(raw, axis=0),
            np.asarray(
                (
                    _entropy(probability) / math.log(choices),
                    _entropy(implied) / math.log(choices),
                    sorted_probability[0],
                    sorted_probability[:5].sum(),
                    sorted_probability[0] - sorted_probability[1],
                    sorted_implied[0],
                    sorted_implied[:5].sum(),
                    float(np.mean(np.log(odds))),
                    float(np.std(np.log(odds))),
                    float(np.mean(np.log(closing))),
                    float(np.std(np.log(closing))),
                ),
                dtype=np.float64,
            ),
        )
    )
    race = np.concatenate((race, path_race))
    if not np.isfinite(ticket).all() or not np.isfinite(race).all():
        raise ValueError("allocation model features contain non-finite values")
    return ticket, race, np.log(probability)


def _prepare_pairs(
    races: Sequence[LabeledRace],
    predictions: Sequence[RacePrediction],
    realized_payout_yen_by_race: Mapping[str, int],
) -> tuple[list[PreparedAllocationRace], str]:
    if not races or len(races) != len(predictions):
        raise ValueError("allocation training requires paired non-empty races")
    prepared: list[PreparedAllocationRace] = []
    payload: list[dict[str, Any]] = []
    previous: tuple[str, str] | None = None
    seen: set[str] = set()
    for labeled, prediction in zip(races, predictions, strict=True):
        decision = labeled.decision
        key = (decision.race_date, decision.race_id)
        if previous is not None and key <= previous:
            raise ValueError("allocation training races must be uniquely chronological")
        previous = key
        if decision.race_id in seen:
            raise ValueError("allocation training race ids must be unique")
        seen.add(decision.race_id)
        ticket, race, base_log_probability = decision_feature_matrices(
            decision, prediction
        )
        winner = int(labeled.outcome.winner_index)
        if not 0 <= winner < len(decision.current_odds):
            raise ValueError("allocation teacher winner index is invalid")
        payout_yen = realized_payout_yen_by_race.get(decision.race_id)
        if (
            isinstance(payout_yen, bool)
            or not isinstance(payout_yen, int)
            or payout_yen < STAKE_UNIT_YEN
        ):
            raise ValueError("allocation teacher requires official realized payout")
        payout = float(payout_yen) / STAKE_UNIT_YEN
        prepared.append(
            PreparedAllocationRace(
                decision.race_id,
                decision.race_date,
                ticket,
                race,
                base_log_probability,
                winner,
                payout,
            )
        )
        payload.append(
            {
                "race_id": decision.race_id,
                "winner": winner,
                "payout": payout,
                "prediction_sha256": hashlib.sha256(
                    np.concatenate((ticket.reshape(-1), race)).tobytes()
                ).hexdigest(),
            }
        )
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return prepared, digest


def _normalization(
    races: Sequence[PreparedAllocationRace],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ticket = np.vstack([race.ticket_features for race in races])
    race = np.vstack([item.race_features for item in races])
    ticket_mean = ticket.mean(axis=0)
    ticket_scale = ticket.std(axis=0)
    ticket_scale[ticket_scale < 1e-8] = 1.0
    race_mean = race.mean(axis=0)
    race_scale = race.std(axis=0)
    race_scale[race_scale < 1e-8] = 1.0
    return ticket_mean, ticket_scale, race_mean, race_scale


def _day_weights(races: Sequence[PreparedAllocationRace]) -> np.ndarray:
    counts = Counter(race.race_date for race in races)
    days = len(counts)
    return np.asarray(
        [1.0 / (days * counts[race.race_date]) for race in races],
        dtype=np.float64,
    )


def _prepared_batch(
    races: Sequence[PreparedAllocationRace],
    normalization: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> PreparedAllocationBatch:
    if not races:
        raise ValueError("allocation batch requires races")
    ticket_mean, ticket_scale, race_mean, race_scale = normalization
    choices = len(races[0].base_log_probability)
    if any(len(race.base_log_probability) != choices for race in races):
        raise ValueError("allocation batch requires a fixed ticket universe")
    ticket = np.ascontiguousarray(
        np.stack(
            [
                (race.ticket_features - ticket_mean) / ticket_scale
                for race in races
            ]
        ),
        dtype=np.float64,
    )
    race_matrix = np.ascontiguousarray(
        np.column_stack(
            (
                np.stack(
                    [
                        (race.race_features - race_mean) / race_scale
                        for race in races
                    ]
                ),
                np.ones(len(races), dtype=np.float64),
            )
        ),
        dtype=np.float64,
    )
    return PreparedAllocationBatch(
        ticket=ticket,
        race=race_matrix,
        base_log_probability=np.ascontiguousarray(
            np.stack([race.base_log_probability for race in races]),
            dtype=np.float64,
        ),
        winner_index=np.asarray(
            [race.winner_index for race in races], dtype=np.int64
        ),
        payout_odds=np.asarray(
            [race.payout_odds for race in races], dtype=np.float64
        ),
        day_weight=_day_weights(races),
    )


def _objective_gradient_batch(
    parameters: np.ndarray,
    batch: PreparedAllocationBatch,
    *,
    config: AllocationConfig,
) -> tuple[float, np.ndarray]:
    ticket_dimension = batch.ticket.shape[2]
    allocation = parameters[:ticket_dimension]
    gate = parameters[ticket_dimension:]

    scores = batch.base_log_probability + batch.ticket @ allocation
    scores -= np.max(scores, axis=1, keepdims=True)
    allocation_weight = np.exp(np.clip(scores, -60.0, 0.0))
    allocation_weight /= np.sum(allocation_weight, axis=1, keepdims=True)
    gate_score = np.clip(batch.race @ gate, -60.0, 60.0)
    gate_probability = 1.0 / (1.0 + np.exp(-gate_score))
    exposure = MAX_RACE_EXPOSURE_FRACTION * gate_probability
    row_index = np.arange(len(batch.winner_index))
    winner_allocation = allocation_weight[row_index, batch.winner_index]
    allocated_payout = batch.payout_odds * winner_allocation
    growth = np.maximum(
        EPSILON, 1.0 + exposure * (allocated_payout - 1.0)
    )
    log_growth = np.log(growth)
    downside = np.minimum(0.0, log_growth)
    loss = float(
        batch.day_weight
        @ (-log_growth + config.downside_penalty * downside * downside)
    )
    derivative_growth = (
        batch.day_weight
        * (-1.0 + 2.0 * config.downside_penalty * downside)
        / growth
    )

    gate_factor = (
        derivative_growth
        * (allocated_payout - 1.0)
        * MAX_RACE_EXPOSURE_FRACTION
        * gate_probability
        * (1.0 - gate_probability)
    )
    gate_gradient = batch.race.T @ gate_factor
    weighted_ticket = np.einsum(
        "nc,ncd->nd", allocation_weight, batch.ticket, optimize=True
    )
    winner_ticket = batch.ticket[row_index, batch.winner_index]
    allocation_factor = (
        derivative_growth * exposure * allocated_payout
    )
    allocation_gradient = (
        (winner_ticket - weighted_ticket).T @ allocation_factor
    )
    gradient = np.concatenate((allocation_gradient, gate_gradient))

    regularized = parameters.copy()
    regularized[-1] = 0.0
    loss += 0.5 * config.regularization * float(regularized @ regularized)
    gradient += config.regularization * regularized
    return loss, gradient


def _objective_gradient(
    parameters: np.ndarray,
    races: Sequence[PreparedAllocationRace],
    *,
    ticket_mean: np.ndarray,
    ticket_scale: np.ndarray,
    race_mean: np.ndarray,
    race_scale: np.ndarray,
    config: AllocationConfig,
) -> tuple[float, np.ndarray]:
    batch = _prepared_batch(
        races, (ticket_mean, ticket_scale, race_mean, race_scale)
    )
    return _objective_gradient_batch(parameters, batch, config=config)


def _fit(
    races: Sequence[PreparedAllocationRace],
    config: AllocationConfig,
    normalization: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    max_iterations: int,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    ticket_mean, ticket_scale, race_mean, race_scale = normalization
    dimension = ticket_mean.size + race_mean.size + 1
    batch = _prepared_batch(races, normalization)

    def objective(values: np.ndarray) -> tuple[float, np.ndarray]:
        return _objective_gradient_batch(values, batch, config=config)

    result = minimize(
        objective,
        np.zeros(dimension, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=[(None, None)] * (dimension - 1) + [(-12.0, 12.0)],
        options={"maxiter": max_iterations, "ftol": 1e-11, "gtol": 1e-6, "maxls": 40},
    )
    value, gradient = objective(np.asarray(result.x, dtype=np.float64))
    return np.asarray(result.x, dtype=np.float64), {
        "objective": float(value),
        "gradient_norm": float(np.linalg.norm(gradient)),
        "iterations": int(result.nit),
        "converged": bool(result.success),
        "message": str(result.message),
    }


def _continuous_metrics(
    parameters: np.ndarray,
    races: Sequence[PreparedAllocationRace],
    normalization: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, Any]:
    ticket_mean, ticket_scale, race_mean, race_scale = normalization
    ticket_dimension = ticket_mean.size
    allocation = parameters[:ticket_dimension]
    gate = parameters[ticket_dimension:]
    daily_growth: dict[str, float] = {}
    exposures: list[float] = []
    for race in races:
        ticket = (race.ticket_features - ticket_mean) / ticket_scale
        race_vector = np.concatenate(
            (((race.race_features - race_mean) / race_scale), np.ones(1))
        )
        weights = _softmax(race.base_log_probability + ticket @ allocation)
        gate_probability = _sigmoid(float(race_vector @ gate))
        exposure = MAX_RACE_EXPOSURE_FRACTION * gate_probability
        growth = max(
            EPSILON,
            1.0 + exposure * (race.payout_odds * weights[race.winner_index] - 1.0),
        )
        daily_growth[race.race_date] = daily_growth.get(race.race_date, 1.0) * growth
        exposures.append(exposure)
    values = np.asarray(list(daily_growth.values()), dtype=np.float64)
    return {
        "validation_days": len(values),
        "validation_races": len(races),
        "geometric_daily_growth": float(np.exp(np.mean(np.log(values)))) if len(values) else None,
        "profitable_day_fraction": float(np.mean(values > 1.0)) if len(values) else None,
        "mean_exposure_fraction": float(np.mean(exposures)) if exposures else None,
        "max_exposure_fraction": float(np.max(exposures)) if exposures else None,
    }


def fit_learned_allocation_head(
    races: Sequence[LabeledRace],
    predictions: Sequence[RacePrediction],
    realized_payout_yen_by_race: Mapping[str, int],
    *,
    base_predictions_trained_through_date: str,
    configs: Iterable[AllocationConfig] = DEFAULT_CONFIGS,
    validation_fraction: float = 0.25,
    max_iterations: int = 200,
    model_key: str = MODEL_KEY,
) -> LearnedAllocationArtifact:
    if not model_key:
        raise ValueError("allocation model_key is required")
    schemas = {
        race.decision.odds_path.schema
        for race in races
        if race.decision.odds_path is not None
    }
    missing_paths = sum(race.decision.odds_path is None for race in races)
    if len(schemas) > 1 or (schemas and missing_paths):
        raise ValueError("allocation training odds path schemas must be uniform")
    odds_path_schema = next(iter(schemas), None)
    if odds_path_schema not in (None, T5_ODDS_PATH_SCHEMA):
        raise ValueError("allocation training odds path schema is unsupported")
    prepared, digest = _prepare_pairs(
        races, predictions, realized_payout_yen_by_race
    )
    first_date = min(race.race_date for race in prepared)
    if str(base_predictions_trained_through_date) >= first_date:
        raise ValueError("allocation base heads must be trained strictly before teacher races")
    if not 0.1 <= validation_fraction <= 0.5:
        raise ValueError("allocation validation_fraction must be between 0.1 and 0.5")
    dates = sorted({race.race_date for race in prepared})
    if len(dates) < 4:
        raise ValueError("allocation model selection requires at least four dates")
    validation_days = max(1, int(math.ceil(len(dates) * validation_fraction)))
    validation_dates = set(dates[-validation_days:])
    fit_races = [race for race in prepared if race.race_date not in validation_dates]
    validation_races = [race for race in prepared if race.race_date in validation_dates]
    normalization = _normalization(fit_races)
    normalized_configs = tuple(configs)
    if not normalized_configs:
        raise ValueError("allocation model selection requires candidates")

    def fit_candidate(config: AllocationConfig) -> dict[str, Any]:
        parameters, diagnostics = _fit(
            fit_races, config, normalization, max_iterations=max_iterations
        )
        return {
            "config": config,
            **diagnostics,
            **_continuous_metrics(parameters, validation_races, normalization),
        }

    if len(normalized_configs) >= 8:
        with ThreadPoolExecutor(
            max_workers=min(4, len(normalized_configs)),
            thread_name_prefix="allocation-grid",
        ) as executor:
            candidates = list(executor.map(fit_candidate, normalized_configs))
    else:
        candidates = [fit_candidate(config) for config in normalized_configs]
    selected = max(
        candidates,
        key=lambda row: (
            float(row["geometric_daily_growth"] or 0.0),
            float(row["profitable_day_fraction"] or 0.0),
            -float(row["mean_exposure_fraction"] or 0.0),
        ),
    )
    final_normalization = _normalization(prepared)
    parameters, final_diagnostics = _fit(
        prepared,
        selected["config"],
        final_normalization,
        max_iterations=max_iterations,
    )
    ticket_mean, ticket_scale, race_mean, race_scale = final_normalization
    ticket_dimension = ticket_mean.size
    return LearnedAllocationArtifact(
        model_key=model_key,
        teacher=TEACHER,
        trained_through_date=max(race.race_date for race in prepared),
        base_predictions_trained_through_date=str(base_predictions_trained_through_date),
        training_race_ids=tuple(race.race_id for race in prepared),
        ticket_feature_mean=tuple(float(value) for value in ticket_mean),
        ticket_feature_scale=tuple(float(value) for value in ticket_scale),
        race_feature_mean=tuple(float(value) for value in race_mean),
        race_feature_scale=tuple(float(value) for value in race_scale),
        allocation_coefficients=tuple(float(value) for value in parameters[:ticket_dimension]),
        gate_coefficients=tuple(float(value) for value in parameters[ticket_dimension:]),
        selected_config=selected["config"],
        objective=float(final_diagnostics["objective"]),
        gradient_norm=float(final_diagnostics["gradient_norm"]),
        iterations=int(final_diagnostics["iterations"]),
        converged=bool(final_diagnostics["converged"]),
        candidate_metrics=tuple(
            {
                **{key: value for key, value in row.items() if key != "config"},
                "config": row["config"].__dict__,
            }
            for row in candidates
        ),
        training_input_sha256=digest,
        odds_path_schema=odds_path_schema,
    )


def allocation_decision(
    artifact: LearnedAllocationArtifact,
    decision: DecisionRace,
    prediction: RacePrediction,
    *,
    available_bankroll_yen: int,
    stake_unit_yen: int = STAKE_UNIT_YEN,
) -> AllocationDecision:
    if decision.race_date <= artifact.trained_through_date:
        raise ValueError("allocation inference must be strictly after training")
    if available_bankroll_yen < 0 or stake_unit_yen < 1:
        raise ValueError("allocation bankroll and stake unit must be valid")
    decision_schema = (
        decision.odds_path.schema if decision.odds_path is not None else None
    )
    if decision_schema != artifact.odds_path_schema:
        raise ValueError("allocation inference odds path schema mismatch")
    ticket, race, base_log_probability = decision_feature_matrices(decision, prediction)
    ticket_mean = np.asarray(artifact.ticket_feature_mean)
    ticket_scale = np.asarray(artifact.ticket_feature_scale)
    race_mean = np.asarray(artifact.race_feature_mean)
    race_scale = np.asarray(artifact.race_feature_scale)
    allocation = np.asarray(artifact.allocation_coefficients)
    gate = np.asarray(artifact.gate_coefficients)
    weights = _softmax(
        base_log_probability + ((ticket - ticket_mean) / ticket_scale) @ allocation
    )
    race_vector = np.concatenate((((race - race_mean) / race_scale), np.ones(1)))
    gate_probability = _sigmoid(float(race_vector @ gate))
    exposure = artifact.max_race_exposure_fraction * gate_probability
    stake = int((available_bankroll_yen * exposure) // stake_unit_yen) * stake_unit_yen
    units = stake // stake_unit_yen
    stakes = np.zeros(len(weights), dtype=np.int64)
    if units:
        raw_units = units * weights
        base_units = np.floor(raw_units).astype(np.int64)
        remaining = units - int(base_units.sum())
        if remaining:
            order = np.argsort(-(raw_units - base_units), kind="stable")
            base_units[order[:remaining]] += 1
        stakes = base_units * stake_unit_yen
    return AllocationDecision(
        race_id=decision.race_id,
        exposure_fraction=float(exposure),
        proposed_stake_yen=int(stakes.sum()),
        stakes_yen=tuple(int(value) for value in stakes),
        allocation_weights=tuple(float(value) for value in weights),
        gate_probability=float(gate_probability),
    )


__all__ = [
    "AllocationConfig",
    "AllocationDecision",
    "LearnedAllocationArtifact",
    "allocation_decision",
    "decision_feature_matrices",
    "fit_learned_allocation_head",
]
