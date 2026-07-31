from __future__ import annotations

import math
from typing import Any

import numpy as np

from .closing_odds import MAX_ODDS, MIN_ODDS, expected_odds_correction


DEFAULT_REGULARIZATION = 0.0001


def contextual_race_features(
    race: dict[str, Any],
) -> dict[str, tuple[float, ...]]:
    combinations = sorted(
        set(race["odds"])
        & set(race["market_probabilities"])
        & set(race["earlier_market_probabilities"])
    )
    log_odds = np.asarray(
        [math.log(float(race["odds"][key])) for key in combinations],
        dtype=np.float64,
    )
    ranks = np.argsort(np.argsort(log_odds)) / max(1, len(combinations) - 1)
    overround = math.log(
        sum(1.0 / float(race["odds"][key]) for key in combinations)
    )
    scale = float(race.get("momentum_scale") or 1.0)
    result = {}
    for index, combination in enumerate(combinations):
        current_log_odds = float(log_odds[index])
        momentum = scale * (
            math.log(float(race["market_probabilities"][combination]))
            - math.log(float(race["earlier_market_probabilities"][combination]))
        )
        centered_odds = (current_log_odds - 4.0) / 2.0
        result[combination] = (
            1.0,
            current_log_odds,
            momentum,
            centered_odds * centered_odds,
            momentum * momentum,
            centered_odds * momentum,
            float(ranks[index]),
            overround,
        )
    return result


def fit_contextual_closing_odds_model(
    races: list[dict[str, Any]],
    *,
    regularization: float = DEFAULT_REGULARIZATION,
) -> dict[str, Any]:
    if regularization < 0.0 or not math.isfinite(regularization):
        raise ValueError("regularization must be finite and non-negative")
    features = []
    targets = []
    race_offsets: list[tuple[int, int]] = []
    race_count = 0
    for race in races:
        closing = race.get("closing_odds") or {}
        feature_map = contextual_race_features(race)
        combinations = sorted(set(closing) & set(feature_map))
        if len(combinations) != 120:
            continue
        race_start = len(targets)
        for combination in combinations:
            closing_odds = float(closing[combination])
            if closing_odds <= 0.0:
                continue
            features.append(feature_map[combination])
            targets.append(math.log(closing_odds))
        if len(targets) > race_start:
            race_offsets.append((race_start, len(targets)))
            race_count += 1
    if not targets:
        raise ValueError("contextual closing-odds calibration requires complete snapshots")
    matrix = np.asarray(features, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    prior = np.zeros(matrix.shape[1], dtype=np.float64)
    prior[1] = 1.0
    gram = matrix.T @ matrix / len(targets)
    rhs = matrix.T @ target / len(targets)
    coefficients = np.linalg.solve(
        gram + regularization * np.eye(matrix.shape[1], dtype=np.float64),
        rhs + regularization * prior,
    )
    predicted = matrix @ coefficients
    correction = expected_odds_correction(target, predicted, race_offsets)
    return {
        "coefficients": coefficients.tolist(),
        "regularization": float(regularization),
        "training_races": race_count,
        "training_tickets": len(targets),
        "training_mean_absolute_log_error": float(
            np.mean(np.abs(target - predicted))
        ),
        **correction,
    }


def forecast_contextual_closing_odds(
    race: dict[str, Any],
    model: dict[str, Any],
    *,
    expected_value: bool = False,
) -> dict[str, float]:
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    multiplier = (
        float(model.get("expected_odds_multiplier") or 1.0)
        if expected_value
        else 1.0
    )
    return {
        combination: min(
            MAX_ODDS,
            max(
                MIN_ODDS,
                multiplier
                * math.exp(float(np.asarray(features) @ coefficients)),
            ),
        )
        for combination, features in contextual_race_features(race).items()
    }
