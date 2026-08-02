from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from .four_head_nested_v22 import DecisionRace, LabeledRace, RacePrediction


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
    ticket_dimension = ticket_mean.size
    allocation = parameters[:ticket_dimension]
    gate = parameters[ticket_dimension:]
    gradient = np.zeros_like(parameters)
    allocation_gradient = gradient[:ticket_dimension]
    gate_gradient = gradient[ticket_dimension:]
    loss = 0.0
    for weight, race in zip(_day_weights(races), races, strict=True):
        ticket = (race.ticket_features - ticket_mean) / ticket_scale
        race_vector = np.concatenate(
            (((race.race_features - race_mean) / race_scale), np.ones(1))
        )
        allocation_weight = _softmax(race.base_log_probability + ticket @ allocation)
        gate_probability = _sigmoid(float(race_vector @ gate))
        exposure = MAX_RACE_EXPOSURE_FRACTION * gate_probability
        winner_allocation = float(allocation_weight[race.winner_index])
        allocated_payout = race.payout_odds * winner_allocation
        growth = max(EPSILON, 1.0 + exposure * (allocated_payout - 1.0))
        log_growth = math.log(growth)
        downside = min(0.0, log_growth)
        loss += weight * (-log_growth + config.downside_penalty * downside * downside)
        derivative_growth = weight * (
            -1.0 + 2.0 * config.downside_penalty * downside
        ) / growth
        gate_gradient += (
            derivative_growth
            * (allocated_payout - 1.0)
            * MAX_RACE_EXPOSURE_FRACTION
            * gate_probability
            * (1.0 - gate_probability)
            * race_vector
        )
        score_derivative = -allocation_weight
        score_derivative[race.winner_index] += 1.0
        allocation_gradient += (
            derivative_growth
            * exposure
            * allocated_payout
            * (ticket.T @ score_derivative)
        )
    loss += 0.5 * config.regularization * float(parameters @ parameters)
    gradient += config.regularization * parameters
    return float(loss), gradient


def _fit(
    races: Sequence[PreparedAllocationRace],
    config: AllocationConfig,
    normalization: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    max_iterations: int,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    ticket_mean, ticket_scale, race_mean, race_scale = normalization
    dimension = ticket_mean.size + race_mean.size + 1

    def objective(values: np.ndarray) -> tuple[float, np.ndarray]:
        return _objective_gradient(
            values,
            races,
            ticket_mean=ticket_mean,
            ticket_scale=ticket_scale,
            race_mean=race_mean,
            race_scale=race_scale,
            config=config,
        )

    result = minimize(
        objective,
        np.zeros(dimension, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
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
) -> LearnedAllocationArtifact:
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
    candidates: list[dict[str, Any]] = []
    normalized_configs = tuple(configs)
    if not normalized_configs:
        raise ValueError("allocation model selection requires candidates")
    for config in normalized_configs:
        parameters, diagnostics = _fit(
            fit_races, config, normalization, max_iterations=max_iterations
        )
        candidates.append(
            {
                "config": config,
                **diagnostics,
                **_continuous_metrics(parameters, validation_races, normalization),
            }
        )
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
        model_key=MODEL_KEY,
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
