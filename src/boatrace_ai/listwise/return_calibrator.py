from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


FEATURE_COUNT = 58
MIN_EXPECTED_RETURN = 1e-4
MAX_EXPECTED_RETURN = 20.0


@dataclass(frozen=True)
class ExpectedReturnCalibrator:
    weights: np.ndarray
    regularization: float
    training_samples: int
    iterations: int
    converged: bool
    objective: float
    gradient_norm: float


def expected_return_features(
    candidate_probabilities: np.ndarray,
    market_probabilities: np.ndarray,
    race_keys: Sequence[tuple[str, str, str, int]],
    combination_lanes: np.ndarray,
) -> np.ndarray:
    candidate = np.asarray(candidate_probabilities, dtype=np.float64)
    market = np.asarray(market_probabilities, dtype=np.float64)
    lanes = np.asarray(combination_lanes, dtype=np.int64)
    if candidate.shape != market.shape or candidate.ndim != 2:
        raise ValueError("candidate and market probabilities must be aligned matrices")
    if candidate.shape[0] != len(race_keys) or candidate.shape[1] != len(lanes):
        raise ValueError("probability matrices, race keys, and combinations must align")
    if lanes.ndim != 2 or lanes.shape[1] != 3:
        raise ValueError("combination lanes must have shape (combinations, 3)")
    if np.any(lanes < 0) or np.any(lanes > 5):
        raise ValueError("combination lanes must be zero-based")
    if not np.all(np.isfinite(candidate)) or not np.all(np.isfinite(market)):
        raise ValueError("probabilities must be finite")
    if np.any(candidate <= 0.0) or np.any(market <= 0.0):
        raise ValueError("probabilities must be positive")

    race_count, combination_count = candidate.shape
    rows = race_count * combination_count
    matrix = np.zeros((rows, FEATURE_COUNT), dtype=np.float64)
    flat_candidate = candidate.reshape(-1)
    flat_market = market.reshape(-1)
    market_surprise = np.clip(-np.log(flat_market) / 6.0, 0.0, 2.0)
    candidate_surprise = np.clip(-np.log(flat_candidate) / 6.0, 0.0, 2.0)
    log_edge = np.clip(np.log(flat_candidate / flat_market) / 3.0, -1.0, 1.0)
    tiled_lanes = np.tile(lanes, (race_count, 1))
    row_indices = np.arange(rows)

    matrix[:, 0] = 1.0
    matrix[:, 1] = market_surprise
    matrix[:, 2] = market_surprise * market_surprise
    for stage in range(3):
        matrix[row_indices, 3 + stage * 6 + tiled_lanes[:, stage]] = 1.0
    venues = np.repeat(
        np.asarray([int(key[2]) for key in race_keys], dtype=np.int64),
        combination_count,
    )
    valid_venues = (venues >= 1) & (venues <= 24)
    matrix[row_indices[valid_venues], 21 + venues[valid_venues] - 1] = 1.0
    race_numbers = np.repeat(
        np.asarray([int(key[3]) for key in race_keys], dtype=np.int64),
        combination_count,
    )
    race_buckets = np.where(race_numbers <= 4, 0, np.where(race_numbers <= 8, 1, 2))
    matrix[row_indices, 45 + race_buckets] = 1.0
    matrix[row_indices, 48 + tiled_lanes[:, 0]] = market_surprise
    matrix[:, 54] = candidate_surprise
    matrix[:, 55] = candidate_surprise * candidate_surprise
    matrix[:, 56] = log_edge
    matrix[:, 57] = log_edge * log_edge
    return matrix


def expected_return_targets(
    race_keys: Sequence[tuple[str, str, str, int]],
    payouts: dict[str, dict[str, Any]],
    combination_index: dict[str, int],
    combination_count: int,
) -> np.ndarray:
    targets = np.zeros((len(race_keys), combination_count), dtype=np.float64)
    for race_index, race_key in enumerate(race_keys):
        actual = payouts.get(str(race_key[0]))
        if actual is None:
            continue
        index = combination_index.get(str(actual["combination"]))
        if index is None:
            continue
        targets[race_index, index] = float(actual["payout_yen"]) / 100.0
    return targets


def _objective_gradient_hessian(
    weights: np.ndarray,
    candidate_probabilities: np.ndarray,
    market_probabilities: np.ndarray,
    race_keys: Sequence[tuple[str, str, str, int]],
    targets: np.ndarray,
    combination_lanes: np.ndarray,
    *,
    regularization: float,
    batch_races: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    gradient = np.zeros(FEATURE_COUNT, dtype=np.float64)
    hessian = np.zeros((FEATURE_COUNT, FEATURE_COUNT), dtype=np.float64)
    objective = 0.0
    sample_count = int(targets.size)
    for start in range(0, len(race_keys), batch_races):
        stop = min(len(race_keys), start + batch_races)
        matrix = expected_return_features(
            candidate_probabilities[start:stop],
            market_probabilities[start:stop],
            race_keys[start:stop],
            combination_lanes,
        )
        target = targets[start:stop].reshape(-1)
        linear = np.clip(matrix @ weights, -9.0, 3.0)
        mean = np.exp(linear)
        objective += float(np.sum(mean - target * linear))
        residual = mean - target
        gradient += matrix.T @ residual
        hessian += matrix.T @ (matrix * mean[:, None])

    scale = 1.0 / max(1, sample_count)
    gradient *= scale
    hessian *= scale
    objective *= scale
    penalty = np.asarray(weights, dtype=np.float64).copy()
    penalty[0] = 0.0
    objective += 0.5 * regularization * float(penalty @ penalty)
    gradient += regularization * penalty
    hessian += np.eye(FEATURE_COUNT, dtype=np.float64) * regularization
    hessian[0, 0] -= regularization
    return objective, gradient, hessian


def fit_expected_return_calibrator(
    candidate_probabilities: np.ndarray,
    market_probabilities: np.ndarray,
    race_keys: Sequence[tuple[str, str, str, int]],
    payouts: dict[str, dict[str, Any]],
    combination_lanes: np.ndarray,
    combination_index: dict[str, int],
    *,
    regularization: float = 0.01,
    max_iterations: int = 20,
    tolerance: float = 1e-6,
    batch_races: int = 500,
) -> ExpectedReturnCalibrator:
    if regularization <= 0.0 or not np.isfinite(regularization):
        raise ValueError("regularization must be finite and positive")
    if max_iterations < 1 or batch_races < 1:
        raise ValueError("iteration and batch sizes must be positive")
    targets = expected_return_targets(
        race_keys,
        payouts,
        combination_index,
        np.asarray(candidate_probabilities).shape[1],
    )
    sample_count = int(targets.size)
    mean_target = max(MIN_EXPECTED_RETURN, float(targets.sum()) / max(1, sample_count))
    weights = np.zeros(FEATURE_COUNT, dtype=np.float64)
    weights[0] = float(np.log(mean_target))
    converged = False
    objective = float("inf")
    gradient_norm = float("inf")
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        objective, gradient, hessian = _objective_gradient_hessian(
            weights,
            np.asarray(candidate_probabilities, dtype=np.float64),
            np.asarray(market_probabilities, dtype=np.float64),
            race_keys,
            targets,
            combination_lanes,
            regularization=float(regularization),
            batch_races=int(batch_races),
        )
        gradient_norm = float(np.linalg.norm(gradient))
        iterations = iteration
        if gradient_norm <= tolerance:
            converged = True
            break
        step = np.linalg.solve(hessian, gradient)
        max_step = float(np.max(np.abs(step)))
        if max_step > 1.0:
            step /= max_step
        weights -= step
        if float(np.linalg.norm(step)) <= tolerance:
            converged = True
            break

    return ExpectedReturnCalibrator(
        weights=weights,
        regularization=float(regularization),
        training_samples=sample_count,
        iterations=iterations,
        converged=converged,
        objective=float(objective),
        gradient_norm=float(gradient_norm),
    )


def predict_expected_returns(
    model: ExpectedReturnCalibrator,
    candidate_probabilities: np.ndarray,
    market_probabilities: np.ndarray,
    race_keys: Sequence[tuple[str, str, str, int]],
    combination_lanes: np.ndarray,
    *,
    batch_races: int = 500,
) -> np.ndarray:
    candidate = np.asarray(candidate_probabilities, dtype=np.float64)
    output = np.empty_like(candidate)
    for start in range(0, len(race_keys), batch_races):
        stop = min(len(race_keys), start + batch_races)
        matrix = expected_return_features(
            candidate[start:stop],
            np.asarray(market_probabilities, dtype=np.float64)[start:stop],
            race_keys[start:stop],
            combination_lanes,
        )
        predicted = np.exp(np.clip(matrix @ model.weights, -9.0, 3.0))
        output[start:stop] = np.clip(
            predicted.reshape(stop - start, candidate.shape[1]),
            MIN_EXPECTED_RETURN,
            MAX_EXPECTED_RETURN,
        )
    return output
