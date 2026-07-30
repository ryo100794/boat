from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Iterable, Mapping

import numpy as np


CALIBRATION_VERSION = 1
EXPECTED_COMBINATIONS = 120
EPSILON = 1e-300
FEATURE_CLIP = 30.0


@dataclass(frozen=True)
class MarketOffsetPrediction:
    probabilities: dict[str, float]
    prediction_date: str
    artifact_prediction_date: str
    trained_through_date: str | None
    mode: str
    calibration_version: int = CALIBRATION_VERSION

    def as_dict(self) -> dict[str, object]:
        return {
            "probabilities": dict(self.probabilities),
            "prediction_date": self.prediction_date,
            "artifact_prediction_date": self.artifact_prediction_date,
            "trained_through_date": self.trained_through_date,
            "mode": self.mode,
            "calibration_version": self.calibration_version,
        }


@dataclass(frozen=True)
class MarketOffsetCalibrationArtifact:
    coefficients: tuple[float, float, float]
    feature_mean: tuple[float, float, float]
    feature_scale: tuple[float, float, float]
    prediction_date: str
    trained_through_date: str | None
    training_dates: tuple[str, ...]
    training_races: int
    excluded_non_past_races: int
    regularization: float
    objective: float | None
    gradient_norm: float | None
    iterations: int
    converged: bool
    fitted: bool
    fallback_reason: str | None
    calibration_version: int = CALIBRATION_VERSION

    def predict(
        self,
        model_probabilities: Mapping[str, object],
        market_probabilities: Mapping[str, object],
        forecast_odds: Mapping[str, object],
        *,
        prediction_date: object,
    ) -> MarketOffsetPrediction:
        target = _iso_date(prediction_date, "prediction_date")
        if target < self.prediction_date:
            raise ValueError("prediction_date precedes artifact boundary")
        if self.trained_through_date is not None and self.trained_through_date >= target:
            raise ValueError("artifact is not strictly prior to prediction_date")
        keys, model, market, odds = _inputs(
            model_probabilities, market_probabilities, forecast_odds
        )
        if self.fitted:
            features = (
                _features(keys, model, market, odds)
                - np.asarray(self.feature_mean)
            ) / np.asarray(self.feature_scale)
            probabilities = _softmax_offset(
                market, features @ np.asarray(self.coefficients)
            )
            mode = "market_offset"
        else:
            probabilities = market
            mode = "market_only"
        return MarketOffsetPrediction(
            probabilities={
                key: float(value) for key, value in zip(keys, probabilities)
            },
            prediction_date=target,
            artifact_prediction_date=self.prediction_date,
            trained_through_date=self.trained_through_date,
            mode=mode,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "calibration_version": self.calibration_version,
            "model": "market_offset_multinomial_log_loss",
            "formula": "q=softmax(log(p_market)+g(features))",
            "features": [
                "log_model_market_ratio",
                "model_probability_rank",
                "log_forecast_odds",
            ],
            "teacher": "one_hot_actual_combination",
            "uses_profit_teacher": False,
            "coefficients": list(self.coefficients),
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "prediction_date": self.prediction_date,
            "trained_through_date": self.trained_through_date,
            "training_dates": list(self.training_dates),
            "training_races": self.training_races,
            "excluded_non_past_races": self.excluded_non_past_races,
            "regularization": self.regularization,
            "objective": self.objective,
            "gradient_norm": self.gradient_norm,
            "iterations": self.iterations,
            "converged": self.converged,
            "fitted": self.fitted,
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class _Race:
    race_date: str
    identity: str
    features: np.ndarray
    market: np.ndarray
    actual: int


def _iso_date(value: object, name: str) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError as exc:
        raise ValueError(f"{name} must start with an ISO date") from exc


def _mass(
    values: Mapping[str, object], keys: tuple[str, ...], name: str
) -> np.ndarray:
    result = np.empty(len(keys), dtype=np.float64)
    for index, key in enumerate(keys):
        try:
            value = float(values[key])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name}[{key!r}] must be finite and non-negative") from exc
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name}[{key!r}] must be finite and non-negative")
        result[index] = value
    maximum = float(np.max(result))
    if maximum <= 0.0:
        raise ValueError(f"{name} must contain positive mass")
    result /= maximum
    total = math.fsum(float(value) for value in result)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(f"{name} cannot be normalized")
    return result / total


def _inputs(
    model_values: Mapping[str, object],
    market_values: Mapping[str, object],
    odds_values: Mapping[str, object],
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    model_keys, market_keys, odds_keys = (
        set(model_values), set(market_values), set(odds_values)
    )
    if model_keys != market_keys or model_keys != odds_keys:
        raise ValueError("model, market, and forecast odds keys must match")
    if len(model_keys) != EXPECTED_COMBINATIONS:
        raise ValueError(f"exactly {EXPECTED_COMBINATIONS} combinations are required")
    if any(not isinstance(key, str) for key in model_keys):
        raise ValueError("combination keys must be strings")
    keys = tuple(sorted(model_keys))
    odds = np.empty(len(keys), dtype=np.float64)
    for index, key in enumerate(keys):
        try:
            value = float(odds_values[key])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"forecast_odds[{key!r}] must be finite and positive") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"forecast_odds[{key!r}] must be finite and positive")
        odds[index] = value
    return (
        keys,
        _mass(model_values, keys, "model_probabilities"),
        _mass(market_values, keys, "market_probabilities"),
        odds,
    )


def _features(
    keys: tuple[str, ...],
    model: np.ndarray,
    market: np.ndarray,
    odds: np.ndarray,
) -> np.ndarray:
    ratio = np.clip(
        np.log(np.maximum(model, EPSILON))
        - np.log(np.maximum(market, EPSILON)),
        -FEATURE_CLIP,
        FEATURE_CLIP,
    )
    order = sorted(range(len(keys)), key=lambda i: (-model[i], keys[i]))
    ranks = np.empty(len(keys), dtype=np.float64)
    for rank, index in enumerate(order, start=1):
        ranks[index] = rank
    ranks = (ranks - (len(keys) + 1.0) / 2.0) / ((len(keys) - 1.0) / 2.0)
    log_odds = np.clip(np.log(odds), -FEATURE_CLIP, FEATURE_CLIP)
    return np.column_stack((ratio, ranks, log_odds))


def _softmax_offset(market: np.ndarray, correction: np.ndarray) -> np.ndarray:
    logits = np.log(np.maximum(market, EPSILON)) + np.clip(
        correction, -FEATURE_CLIP, FEATURE_CLIP
    )
    logits -= float(np.max(logits))
    probabilities = np.exp(logits)
    total = math.fsum(float(value) for value in probabilities)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("calibrated probabilities cannot be normalized")
    probabilities /= total
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("calibrated probabilities must be finite")
    return probabilities


def _prepare(race: Mapping[str, object], race_date: str) -> _Race:
    keys, model, market, odds = _inputs(
        race["model_probabilities"],
        race["market_probabilities"],
        race["forecast_odds"],
    )
    actual = str(race["actual_combination"])
    if actual not in keys:
        raise ValueError(f"actual combination {actual!r} is missing")
    identity = str(race.get(
        "race_id",
        "|".join((
            race_date,
            str(race.get("jcd", "")),
            str(race.get("race_no", race.get("rno", ""))),
            actual,
        )),
    ))
    return _Race(
        race_date, identity, _features(keys, model, market, odds),
        market, keys.index(actual)
    )


def _loss_derivatives(
    races: tuple[_Race, ...],
    coefficients: np.ndarray,
    regularization: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    losses: list[float] = []
    gradient = np.zeros(3)
    hessian = np.zeros((3, 3))
    for race in races:
        probability = _softmax_offset(
            race.market, race.features @ coefficients
        )
        losses.append(-math.log(max(EPSILON, float(probability[race.actual]))))
        mean = probability @ race.features
        gradient += mean - race.features[race.actual]
        second = (race.features.T * probability) @ race.features
        hessian += second - np.outer(mean, mean)
    count = len(races)
    objective = math.fsum(losses) / count
    gradient /= count
    hessian /= count
    objective += 0.5 * regularization * float(coefficients @ coefficients)
    gradient += regularization * coefficients
    hessian += regularization * np.eye(3)
    return objective, gradient, hessian


def _artifact(
    target: str,
    races: tuple[_Race, ...],
    excluded: int,
    regularization: float,
    *,
    fitted: bool,
    reason: str | None,
    coefficients: np.ndarray | None = None,
    mean: np.ndarray | None = None,
    scale: np.ndarray | None = None,
    objective: float | None = None,
    gradient_norm: float | None = None,
    iterations: int = 0,
    converged: bool = False,
) -> MarketOffsetCalibrationArtifact:
    dates = tuple(sorted({race.race_date for race in races}))
    return MarketOffsetCalibrationArtifact(
        coefficients=tuple(float(x) for x in (
            coefficients if coefficients is not None else np.zeros(3)
        )),
        feature_mean=tuple(float(x) for x in (
            mean if mean is not None else np.zeros(3)
        )),
        feature_scale=tuple(float(x) for x in (
            scale if scale is not None else np.ones(3)
        )),
        prediction_date=target,
        trained_through_date=dates[-1] if dates else None,
        training_dates=dates,
        training_races=len(races),
        excluded_non_past_races=excluded,
        regularization=regularization,
        objective=objective,
        gradient_norm=gradient_norm,
        iterations=iterations,
        converged=converged,
        fitted=fitted,
        fallback_reason=reason,
    )


def fit_market_offset_calibration(
    races: Iterable[Mapping[str, Any]],
    *,
    prediction_date: object,
    regularization: float = 1.0,
    min_training_races: int = 30,
    max_iterations: int = 50,
    tolerance: float = 1e-9,
) -> MarketOffsetCalibrationArtifact:
    """Fit one-hot multinomial loss on complete days strictly before prediction."""
    target = _iso_date(prediction_date, "prediction_date")
    if not math.isfinite(float(regularization)) or regularization <= 0.0:
        raise ValueError("regularization must be finite and positive")
    if min_training_races < 1 or max_iterations < 1:
        raise ValueError("training and iteration limits must be positive")
    if not math.isfinite(float(tolerance)) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    prepared: list[_Race] = []
    excluded = 0
    for race in races:
        race_date = _iso_date(race.get("race_date"), "race_date")
        if race_date >= target:
            excluded += 1
            continue
        prepared.append(_prepare(race, race_date))
    prepared.sort(key=lambda race: (
        race.race_date, race.identity, race.actual,
        race.features.tobytes(), race.market.tobytes(),
    ))
    training = tuple(prepared)
    if len(training) < min_training_races:
        return _artifact(
            target, training, excluded, float(regularization),
            fitted=False, reason="insufficient_strictly_prior_races",
        )

    stacked = np.concatenate([race.features for race in training])
    mean, scale = np.mean(stacked, axis=0), np.std(stacked, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    normalized = tuple(
        _Race(
            race.race_date, race.identity, (race.features - mean) / scale,
            race.market, race.actual
        )
        for race in training
    )
    coefficients = np.zeros(3)
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        objective, gradient, hessian = _loss_derivatives(
            normalized, coefficients, float(regularization)
        )
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        if not np.all(np.isfinite(step)):
            return _artifact(
                target, training, excluded, float(regularization),
                fitted=False, reason="non_finite_optimization_step",
            )
        multiplier, accepted = 1.0, False
        candidate, candidate_objective = coefficients, objective
        while multiplier >= 1e-10:
            proposal = np.clip(
                coefficients - multiplier * step, -FEATURE_CLIP, FEATURE_CLIP
            )
            proposal_objective, _, _ = _loss_derivatives(
                normalized, proposal, float(regularization)
            )
            if proposal_objective <= objective + 1e-12:
                candidate, candidate_objective = proposal, proposal_objective
                accepted = True
                break
            multiplier *= 0.5
        parameter_change = float(np.max(np.abs(candidate - coefficients)))
        objective_change = abs(candidate_objective - objective)
        coefficients = candidate
        if not accepted or (
            parameter_change <= tolerance and objective_change <= tolerance
        ):
            converged = accepted
            break

    objective, gradient, _ = _loss_derivatives(
        normalized, coefficients, float(regularization)
    )
    if not (
        math.isfinite(objective)
        and np.all(np.isfinite(coefficients))
        and np.all(np.isfinite(gradient))
    ):
        return _artifact(
            target, training, excluded, float(regularization),
            fitted=False, reason="non_finite_fit",
        )
    return _artifact(
        target, training, excluded, float(regularization),
        fitted=True, reason=None, coefficients=coefficients,
        mean=mean, scale=scale, objective=float(objective),
        gradient_norm=float(np.linalg.norm(gradient)),
        iterations=iterations, converged=converged,
    )
