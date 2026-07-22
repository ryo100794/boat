from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


MIN_ODDS = 1.1
MAX_ODDS = 2_000.0
FEATURE_COUNT = 54


@dataclass(frozen=True)
class ConditionalPayoutRegressor:
    weights: np.ndarray
    residual_variance: float
    ridge: float
    training_samples: int


@dataclass
class ConditionalPayoutStatistics:
    gram: np.ndarray
    target_cross: np.ndarray
    target_square_sum: float
    samples: int

    @classmethod
    def empty(cls) -> "ConditionalPayoutStatistics":
        return cls(
            gram=np.zeros((FEATURE_COUNT, FEATURE_COUNT), dtype=np.float64),
            target_cross=np.zeros(FEATURE_COUNT, dtype=np.float64),
            target_square_sum=0.0,
            samples=0,
        )

    def update(
        self,
        probabilities: Sequence[float],
        combinations: Sequence[str],
        race_keys: Sequence[tuple[str, str, str, int]],
        payouts_yen: Sequence[float],
    ) -> None:
        matrix = payout_features(probabilities, combinations, race_keys)
        targets = payout_targets(payouts_yen, expected_rows=len(matrix))
        self.gram += matrix.T @ matrix
        self.target_cross += matrix.T @ targets
        self.target_square_sum += float(targets @ targets)
        self.samples += len(matrix)


def payout_features(
    probabilities: Sequence[float],
    combinations: Sequence[str],
    race_keys: Sequence[tuple[str, str, str, int]],
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if (
        values.ndim != 1
        or len(values) != len(combinations)
        or len(values) != len(race_keys)
    ):
        raise ValueError(
            "payout feature inputs must have matching one-dimensional lengths"
        )
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("payout probabilities must be finite and positive")
    matrix = np.zeros((len(values), FEATURE_COUNT), dtype=np.float64)
    surprise = np.clip(-np.log(values), 1.0, 12.0)
    matrix[:, 0] = 1.0
    matrix[:, 1] = surprise
    matrix[:, 2] = surprise * surprise
    for row_index, (combination, race_key) in enumerate(zip(combinations, race_keys)):
        try:
            lanes = tuple(int(value) for value in str(combination).split("-"))
        except ValueError as exc:
            raise ValueError(f"invalid trifecta combination: {combination}") from exc
        if (
            len(lanes) != 3
            or len(set(lanes)) != 3
            or any(lane < 1 or lane > 6 for lane in lanes)
        ):
            raise ValueError(f"invalid trifecta combination: {combination}")
        for stage, lane in enumerate(lanes):
            matrix[row_index, 3 + stage * 6 + lane - 1] = 1.0
        jcd = int(race_key[2])
        if 1 <= jcd <= 24:
            matrix[row_index, 21 + jcd - 1] = 1.0
        rno = int(race_key[3])
        race_bucket = 0 if rno <= 4 else 1 if rno <= 8 else 2
        matrix[row_index, 45 + race_bucket] = 1.0
        matrix[row_index, 48 + lanes[0] - 1] = surprise[row_index]
    return matrix


def fit_conditional_payout(
    probabilities: Sequence[float],
    combinations: Sequence[str],
    race_keys: Sequence[tuple[str, str, str, int]],
    payouts_yen: Sequence[float],
    *,
    ridge: float = 10.0,
) -> ConditionalPayoutRegressor:
    statistics = ConditionalPayoutStatistics.empty()
    statistics.update(probabilities, combinations, race_keys, payouts_yen)
    return fit_conditional_payout_statistics(statistics, ridge=ridge)


def payout_targets(
    payouts_yen: Sequence[float],
    *,
    expected_rows: int,
) -> np.ndarray:
    payouts = np.asarray(payouts_yen, dtype=np.float64)
    if payouts.shape != (expected_rows,) or not np.all(np.isfinite(payouts)):
        raise ValueError("payout targets must match features and be finite")
    if np.any(payouts <= 0.0):
        raise ValueError("payout targets must be positive")
    return np.log(np.clip(payouts / 100.0, MIN_ODDS, MAX_ODDS))


def fit_conditional_payout_statistics(
    statistics: ConditionalPayoutStatistics,
    *,
    ridge: float = 10.0,
) -> ConditionalPayoutRegressor:
    if ridge <= 0.0 or not np.isfinite(ridge):
        raise ValueError("ridge must be finite and positive")
    if statistics.samples <= 0:
        raise ValueError("conditional payout statistics must contain samples")
    if statistics.gram.shape != (FEATURE_COUNT, FEATURE_COUNT):
        raise ValueError("conditional payout gram matrix has an invalid shape")
    if statistics.target_cross.shape != (FEATURE_COUNT,):
        raise ValueError("conditional payout target cross-product has an invalid shape")
    penalty = np.eye(FEATURE_COUNT, dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(
        np.asarray(statistics.gram, dtype=np.float64) + penalty,
        np.asarray(statistics.target_cross, dtype=np.float64),
    )
    residual_sum_squares = float(
        statistics.target_square_sum
        - 2.0 * weights @ statistics.target_cross
        + weights @ statistics.gram @ weights
    )
    residual_variance = max(
        0.0,
        residual_sum_squares / max(1, statistics.samples - FEATURE_COUNT),
    )
    return ConditionalPayoutRegressor(
        weights=weights,
        residual_variance=residual_variance,
        ridge=float(ridge),
        training_samples=int(statistics.samples),
    )


def predict_conditional_odds(
    model: ConditionalPayoutRegressor,
    probabilities: Sequence[float],
    combinations: Sequence[str],
    race_keys: Sequence[tuple[str, str, str, int]],
    *,
    lognormal_mean_correction: bool = True,
) -> np.ndarray:
    matrix = payout_features(probabilities, combinations, race_keys)
    correction = (
        0.5 * float(model.residual_variance)
        if lognormal_mean_correction
        else 0.0
    )
    log_odds = matrix @ np.asarray(model.weights, dtype=np.float64) + correction
    return np.clip(np.exp(log_odds), MIN_ODDS, MAX_ODDS)
