from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


MIN_ODDS = 1.1
MAX_ODDS = 2_000.0
FEATURE_COUNT = 54
FEATURE_SCHEMA = "conditional_payout_additive_v1"
TAIL_PROBABILITY_THRESHOLDS = (0.02, 0.005, 0.001)
TAIL_BIN_LABELS = ("ge_0.02", "ge_0.005", "ge_0.001", "lt_0.001")
TAIL_BIN_COUNT = len(TAIL_BIN_LABELS)
TAIL_RATIO_MIN = 0.1
TAIL_RATIO_MAX = 4.0
ONE_SIDED_95_Z = 1.6448536269514722


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


def payout_tail_bin_indices(
    market_reference_probabilities: Sequence[float],
) -> np.ndarray:
    probabilities = np.asarray(market_reference_probabilities, dtype=np.float64)
    if probabilities.ndim != 1:
        raise ValueError("market reference probabilities must be one-dimensional")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("market reference probabilities must be finite and non-negative")
    return np.select(
        (
            probabilities >= TAIL_PROBABILITY_THRESHOLDS[0],
            probabilities >= TAIL_PROBABILITY_THRESHOLDS[1],
            probabilities >= TAIL_PROBABILITY_THRESHOLDS[2],
        ),
        (0, 1, 2),
        default=3,
    ).astype(np.int64)


@dataclass
class ConditionalPayoutTailStatistics:
    counts: np.ndarray
    ratio_sums: np.ndarray
    ratio_square_sums: np.ndarray

    @classmethod
    def empty(cls) -> "ConditionalPayoutTailStatistics":
        return cls(
            counts=np.zeros(TAIL_BIN_COUNT, dtype=np.int64),
            ratio_sums=np.zeros(TAIL_BIN_COUNT, dtype=np.float64),
            ratio_square_sums=np.zeros(TAIL_BIN_COUNT, dtype=np.float64),
        )

    @property
    def samples(self) -> int:
        return int(np.sum(self.counts, dtype=np.int64))

    def update(
        self,
        market_reference_probabilities: Sequence[float],
        raw_predicted_odds: Sequence[float],
        actual_odds: Sequence[float],
    ) -> None:
        probabilities = np.asarray(market_reference_probabilities, dtype=np.float64)
        predicted = np.asarray(raw_predicted_odds, dtype=np.float64)
        actual = np.asarray(actual_odds, dtype=np.float64)
        if actual.ndim != 1 or predicted.ndim != 1:
            raise ValueError("tail calibration odds must be one-dimensional")
        if actual.shape != probabilities.shape or predicted.shape != probabilities.shape:
            raise ValueError("tail calibration inputs must have matching lengths")
        if (
            not np.all(np.isfinite(actual))
            or not np.all(np.isfinite(predicted))
            or np.any(actual <= 0.0)
            or np.any(predicted <= 0.0)
        ):
            raise ValueError("tail calibration odds must be finite and positive")
        bin_indices = payout_tail_bin_indices(probabilities)
        ratios = np.clip(actual / predicted, TAIL_RATIO_MIN, TAIL_RATIO_MAX)
        np.add.at(self.counts, bin_indices, 1)
        np.add.at(self.ratio_sums, bin_indices, ratios)
        np.add.at(self.ratio_square_sums, bin_indices, ratios * ratios)


@dataclass
class ConditionalPayoutTailCalibrator:
    statistics: ConditionalPayoutTailStatistics = field(
        default_factory=ConditionalPayoutTailStatistics.empty
    )
    prior_samples: float = 20.0
    minimum_bin_samples: int = 20
    fallback_factor: float = 0.5
    confidence_z: float = ONE_SIDED_95_Z
    minimum_factor: float = TAIL_RATIO_MIN
    minimum_global_samples: int = 20

    def __post_init__(self) -> None:
        if not np.isfinite(self.prior_samples) or self.prior_samples <= 0.0:
            raise ValueError("tail calibration prior samples must be finite and positive")
        if self.minimum_bin_samples < 1:
            raise ValueError("tail calibration minimum bin samples must be positive")
        if self.minimum_global_samples < 1:
            raise ValueError("tail calibration minimum global samples must be positive")
        if (
            not np.isfinite(self.fallback_factor)
            or self.fallback_factor <= 0.0
            or self.fallback_factor > 1.0
        ):
            raise ValueError("tail calibration fallback factor must be in (0, 1]")
        if not np.isfinite(self.confidence_z) or self.confidence_z < 0.0:
            raise ValueError("tail calibration confidence z must be finite and non-negative")
        if (
            not np.isfinite(self.minimum_factor)
            or self.minimum_factor <= 0.0
            or self.minimum_factor > self.fallback_factor
        ):
            raise ValueError(
                "tail calibration minimum factor must be in (0, fallback factor]"
            )
        self._validate_statistics()

    @classmethod
    def empty(cls, **kwargs: float | int) -> "ConditionalPayoutTailCalibrator":
        return cls(statistics=ConditionalPayoutTailStatistics.empty(), **kwargs)

    @property
    def samples(self) -> int:
        return self.statistics.samples

    def update(
        self,
        market_reference_probabilities: Sequence[float],
        raw_predicted_odds: Sequence[float],
        actual_odds: Sequence[float],
    ) -> None:
        self.statistics.update(
            market_reference_probabilities,
            raw_predicted_odds,
            actual_odds,
        )

    def factors(self) -> np.ndarray:
        self._validate_statistics()
        counts = self.statistics.counts.astype(np.float64)
        total_count = float(np.sum(counts))
        if total_count <= 0.0:
            return np.full(TAIL_BIN_COUNT, self.fallback_factor, dtype=np.float64)

        total_sum = float(np.sum(self.statistics.ratio_sums))
        total_square_sum = float(np.sum(self.statistics.ratio_square_sums))
        global_mean = total_sum / total_count
        global_second_moment = total_square_sum / total_count
        global_variance = max(0.0, global_second_moment - global_mean * global_mean)
        global_lower = global_mean - self.confidence_z * np.sqrt(
            global_variance / total_count
        )

        factors = np.empty(TAIL_BIN_COUNT, dtype=np.float64)
        for bin_index in range(TAIL_BIN_COUNT):
            effective_count = counts[bin_index] + self.prior_samples
            posterior_mean = (
                self.statistics.ratio_sums[bin_index]
                + self.prior_samples * global_mean
            ) / effective_count
            posterior_second_moment = (
                self.statistics.ratio_square_sums[bin_index]
                + self.prior_samples * global_second_moment
            ) / effective_count
            posterior_variance = max(
                0.0,
                posterior_second_moment - posterior_mean * posterior_mean,
            )
            lower = posterior_mean - self.confidence_z * np.sqrt(
                posterior_variance / effective_count
            )
            if counts[bin_index] < self.minimum_bin_samples:
                lower = min(lower, global_lower, self.fallback_factor)
            factors[bin_index] = np.clip(
                lower,
                self.minimum_factor,
                1.0,
            )
        return factors

    def eligible_mask(
        self,
        market_reference_probabilities: Sequence[float],
    ) -> np.ndarray:
        self._validate_statistics()
        bin_indices = payout_tail_bin_indices(market_reference_probabilities)
        global_eligible = self.samples >= self.minimum_global_samples
        return np.logical_and(
            self.statistics.counts[bin_indices] >= self.minimum_bin_samples,
            global_eligible,
        )

    def calibrate_with_eligibility(
        self,
        market_reference_probabilities: Sequence[float],
        raw_predicted_odds: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        calibrated = self.calibrate(
            market_reference_probabilities,
            raw_predicted_odds,
        )
        eligible = self.eligible_mask(market_reference_probabilities)
        return calibrated, eligible

    def calibrate(
        self,
        market_reference_probabilities: Sequence[float],
        raw_predicted_odds: Sequence[float],
    ) -> np.ndarray:
        probabilities = np.asarray(market_reference_probabilities, dtype=np.float64)
        predicted = np.asarray(raw_predicted_odds, dtype=np.float64)
        if predicted.ndim != 1 or predicted.shape != probabilities.shape:
            raise ValueError("tail calibration inputs must have matching lengths")
        if not np.all(np.isfinite(predicted)) or np.any(predicted <= 0.0):
            raise ValueError("raw predicted odds must be finite and positive")
        bin_indices = payout_tail_bin_indices(probabilities)
        calibrated = predicted * self.factors()[bin_indices]
        return np.minimum(predicted, calibrated)

    def apply(
        self,
        market_reference_probabilities: Sequence[float],
        raw_predicted_odds: Sequence[float],
    ) -> np.ndarray:
        return self.calibrate(
            market_reference_probabilities,
            raw_predicted_odds,
        )

    def diagnostics(self) -> dict[str, object]:
        self._validate_statistics()
        samples = self.samples
        ratio_mean = (
            float(np.sum(self.statistics.ratio_sums) / samples)
            if samples
            else None
        )
        return {
            "samples": samples,
            "probability_bins": list(TAIL_BIN_LABELS),
            "bin_counts": self.statistics.counts.astype(int).tolist(),
            "bin_factors": self.factors().tolist(),
            "global_ratio_mean": ratio_mean,
            "ratio_winsor_limits": [TAIL_RATIO_MIN, TAIL_RATIO_MAX],
            "prior_samples": float(self.prior_samples),
            "minimum_bin_samples": int(self.minimum_bin_samples),
            "minimum_global_samples": int(self.minimum_global_samples),
            "fallback_factor": float(self.fallback_factor),
            "one_sided_confidence_z": float(self.confidence_z),
        }

    def _validate_statistics(self) -> None:
        statistics = self.statistics
        if (
            statistics.counts.shape != (TAIL_BIN_COUNT,)
            or statistics.ratio_sums.shape != (TAIL_BIN_COUNT,)
            or statistics.ratio_square_sums.shape != (TAIL_BIN_COUNT,)
        ):
            raise ValueError("tail calibration statistics have invalid shapes")
        if (
            np.any(statistics.counts < 0)
            or not np.all(np.isfinite(statistics.ratio_sums))
            or not np.all(np.isfinite(statistics.ratio_square_sums))
            or np.any(statistics.ratio_sums < 0.0)
            or np.any(statistics.ratio_square_sums < 0.0)
        ):
            raise ValueError("tail calibration statistics must be finite and non-negative")


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
    mean_correction_factor: float | None = None,
    tail_calibrator: ConditionalPayoutTailCalibrator | None = None,
) -> np.ndarray:
    matrix = payout_features(probabilities, combinations, race_keys)
    factor = (
        (1.0 if lognormal_mean_correction else 0.0)
        if mean_correction_factor is None
        else float(mean_correction_factor)
    )
    if not np.isfinite(factor) or factor < 0.0 or factor > 1.0:
        raise ValueError("mean correction factor must be between zero and one")
    correction = 0.5 * float(model.residual_variance) * factor
    log_odds = matrix @ np.asarray(model.weights, dtype=np.float64) + correction
    raw_odds = np.clip(np.exp(log_odds), MIN_ODDS, MAX_ODDS)
    if tail_calibrator is None:
        return raw_odds
    return tail_calibrator.calibrate(probabilities, raw_odds)
