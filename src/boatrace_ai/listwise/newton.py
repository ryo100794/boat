from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg

from ..hashed_feature_dataset import HashedRaceDataset
from .model import ListwiseLinearModel, pl_loss_and_score_gradient, stable_softmax


def pl_hessian_score_product(
    scores: np.ndarray,
    ranks: np.ndarray,
    score_vector: np.ndarray,
    *,
    target: str,
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    rank_values = np.asarray(ranks)
    vector = np.asarray(score_vector, dtype=np.float64)
    if values.shape != rank_values.shape or values.shape != vector.shape:
        raise ValueError("scores, ranks, and score_vector must have identical shapes")
    stages = 1 if target == "winner" else 3 if target == "top3_pl" else 0
    if not stages:
        raise ValueError(f"unknown target: {target}")
    product = np.zeros_like(values)
    for race_index in range(values.shape[0]):
        order = np.argsort(rank_values[race_index])
        remaining = np.ones(6, dtype=bool)
        for stage in range(stages):
            lane_indices = np.flatnonzero(remaining)
            probabilities = stable_softmax(values[race_index, lane_indices])
            direction = vector[race_index, lane_indices]
            product[race_index, lane_indices] += probabilities * (
                direction - float(probabilities.dot(direction))
            )
            remaining[int(order[stage])] = False
    return product / max(1, values.shape[0] * stages)


def objective_gradient(
    dataset: HashedRaceDataset,
    model: ListwiseLinearModel,
    *,
    train_race_end: int,
    weights: np.ndarray,
    batch_races: int,
) -> tuple[float, np.ndarray]:
    train_end = min(dataset.race_count, int(train_race_end))
    gradient = np.zeros_like(weights, dtype=np.float64)
    loss_sum = 0.0
    seen = 0
    for start in range(0, train_end, max(1, int(batch_races))):
        stop = min(train_end, start + max(1, int(batch_races)))
        matrix = model.scaler.transform(dataset.matrix[dataset.row_slice(start, stop)])
        scores = np.asarray(matrix.dot(weights)).reshape(-1, 6)
        loss, score_gradient = pl_loss_and_score_gradient(
            scores, dataset.ranks[start:stop], target=model.target
        )
        count = stop - start
        loss_sum += loss * count
        gradient += np.asarray(matrix.T.dot(score_gradient.reshape(-1))).reshape(-1) * count
        seen += count
    objective = loss_sum / max(1, seen) + 0.5 * model.alpha * float(weights.dot(weights))
    gradient = gradient / max(1, seen) + model.alpha * weights
    return objective, gradient


def hessian_diagonal(
    dataset: HashedRaceDataset,
    model: ListwiseLinearModel,
    *,
    train_race_end: int,
    weights: np.ndarray,
    batch_races: int,
) -> np.ndarray:
    train_end = min(dataset.race_count, int(train_race_end))
    diagonal = np.zeros_like(weights, dtype=np.float64)
    seen = 0
    for start in range(0, train_end, max(1, int(batch_races))):
        stop = min(train_end, start + max(1, int(batch_races)))
        matrix = model.scaler.transform(
            dataset.matrix[dataset.row_slice(start, stop)]
        ).tocsr(copy=False)
        scores = np.asarray(matrix.dot(weights)).reshape(-1, 6)
        ranks = dataset.ranks[start:stop]
        count = stop - start
        stages = 1 if model.target == "winner" else 3
        order = np.argsort(ranks, axis=1)
        remaining = np.ones((count, 6), dtype=bool)
        squared = matrix.multiply(matrix)
        aggregate_indices = np.arange(count * 6, dtype=np.int32)
        aggregate_indptr = np.arange(0, count * 6 + 1, 6, dtype=np.int32)
        for stage in range(stages):
            probabilities = stable_softmax(
                np.where(remaining, scores, -np.inf)
            ).reshape(-1)
            expected = sparse.csr_matrix(
                (probabilities, aggregate_indices, aggregate_indptr),
                shape=(count, count * 6),
            ).dot(matrix)
            diagonal += (
                np.asarray(squared.T.dot(probabilities)).reshape(-1)
                - np.asarray(expected.multiply(expected).sum(axis=0)).reshape(-1)
            ) / stages
            remaining[np.arange(count), order[:, stage]] = False
        seen += count
    diagonal = diagonal / max(1, seen) + model.alpha
    numerical_floor = max(float(model.alpha), 1e-12 * float(diagonal.max()))
    diagonal = np.maximum(diagonal, numerical_floor)
    if not np.isfinite(diagonal).all() or np.any(diagonal <= 0.0):
        raise ValueError("invalid Hessian diagonal")
    return diagonal


def hessian_vector_product(
    dataset: HashedRaceDataset,
    model: ListwiseLinearModel,
    *,
    train_race_end: int,
    weights: np.ndarray,
    vector: np.ndarray,
    batch_races: int,
) -> np.ndarray:
    train_end = min(dataset.race_count, int(train_race_end))
    output = np.zeros_like(vector, dtype=np.float64)
    seen = 0
    for start in range(0, train_end, max(1, int(batch_races))):
        stop = min(train_end, start + max(1, int(batch_races)))
        matrix = model.scaler.transform(dataset.matrix[dataset.row_slice(start, stop)])
        scores = np.asarray(matrix.dot(weights)).reshape(-1, 6)
        score_vector = np.asarray(matrix.dot(vector)).reshape(-1, 6)
        score_product = pl_hessian_score_product(
            scores,
            dataset.ranks[start:stop],
            score_vector,
            target=model.target,
        )
        count = stop - start
        output += np.asarray(matrix.T.dot(score_product.reshape(-1))).reshape(-1) * count
        seen += count
    return output / max(1, seen) + model.alpha * vector


def refine_newton_cg(
    dataset: HashedRaceDataset,
    initial_model: ListwiseLinearModel,
    *,
    train_race_end: int,
    batch_races: int = 1_000,
    max_newton_iterations: int = 5,
    max_cg_iterations: int = 20,
    gradient_tolerance: float = 1e-4,
    cg_tolerance: float = 1e-3,
    use_jacobi_preconditioner: bool = True,
) -> tuple[ListwiseLinearModel, dict[str, Any]]:
    weights = np.asarray(initial_model.weights, dtype=np.float64).copy()
    history: list[dict[str, Any]] = []
    converged = False
    for iteration in range(max(1, int(max_newton_iterations))):
        objective, gradient = objective_gradient(
            dataset,
            initial_model,
            train_race_end=train_race_end,
            weights=weights,
            batch_races=batch_races,
        )
        gradient_norm = float(np.linalg.norm(gradient))
        row: dict[str, Any] = {
            "iteration": iteration,
            "objective": objective,
            "gradient_l2": gradient_norm,
        }
        if gradient_norm <= gradient_tolerance:
            row.update({"step": 0.0, "cg_info": 0, "converged": True})
            history.append(row)
            converged = True
            break
        operator = LinearOperator(
            shape=(len(weights), len(weights)),
            matvec=lambda vector: hessian_vector_product(
                dataset,
                initial_model,
                train_race_end=train_race_end,
                weights=weights,
                vector=np.asarray(vector),
                batch_races=batch_races,
            ),
            dtype=np.float64,
        )
        preconditioner = None
        diagonal_min = None
        diagonal_max = None
        if use_jacobi_preconditioner:
            diagonal = hessian_diagonal(
                dataset,
                initial_model,
                train_race_end=train_race_end,
                weights=weights,
                batch_races=batch_races,
            )
            inverse_diagonal = 1.0 / diagonal
            diagonal_min = float(diagonal.min())
            diagonal_max = float(diagonal.max())
            preconditioner = LinearOperator(
                shape=(len(weights), len(weights)),
                matvec=lambda vector: inverse_diagonal * np.asarray(vector),
                dtype=np.float64,
            )
        cg_absolute_tolerance = 0.1 * float(gradient_tolerance)
        cg_iterations = [0]

        def count_cg_iteration(_value: np.ndarray) -> None:
            cg_iterations[0] += 1

        direction, cg_info = cg(
            operator,
            -gradient,
            maxiter=max(1, int(max_cg_iterations)),
            rtol=float(cg_tolerance),
            atol=cg_absolute_tolerance,
            M=preconditioner,
            callback=count_cg_iteration,
        )
        residual = operator.matvec(direction) + gradient
        residual_norm = float(np.linalg.norm(residual))
        relative_residual = residual_norm / max(gradient_norm, np.finfo(float).tiny)
        directional_derivative = float(gradient.dot(direction))
        if not np.isfinite(direction).all() or directional_derivative >= 0.0:
            direction = -gradient
            directional_derivative = -gradient_norm * gradient_norm
            cg_info = -1
        step = 1.0
        accepted_objective = objective
        for _ in range(16):
            candidate = weights + step * direction
            candidate_objective, _ = objective_gradient(
                dataset,
                initial_model,
                train_race_end=train_race_end,
                weights=candidate,
                batch_races=batch_races,
            )
            if candidate_objective <= objective + 1e-4 * step * directional_derivative:
                weights = candidate
                accepted_objective = candidate_objective
                break
            step *= 0.5
        else:
            step = 0.0
        row.update({
            "step": step,
            "cg_info": int(cg_info),
            "cg_iterations": cg_iterations[0],
            "cg_residual_l2": residual_norm,
            "cg_relative_residual": relative_residual,
            "hessian_diagonal_min": diagonal_min,
            "hessian_diagonal_max": diagonal_max,
            "accepted_objective": accepted_objective,
            "converged": False,
        })
        history.append(row)
        if step == 0.0:
            break
    final_objective, final_gradient = objective_gradient(
        dataset,
        initial_model,
        train_race_end=train_race_end,
        weights=weights,
        batch_races=batch_races,
    )
    final_gradient_norm = float(np.linalg.norm(final_gradient))
    converged = converged or final_gradient_norm <= gradient_tolerance
    refined = replace(initial_model, weights=weights)
    return refined, {
        "method": "matrix_free_truncated_newton_cg",
        "materialized_hessian": False,
        "max_newton_iterations": int(max_newton_iterations),
        "max_cg_iterations": int(max_cg_iterations),
        "gradient_tolerance": float(gradient_tolerance),
        "cg_tolerance": float(cg_tolerance),
        "cg_absolute_tolerance": 0.1 * float(gradient_tolerance),
        "jacobi_preconditioner": bool(use_jacobi_preconditioner),
        "initial_objective": history[0]["objective"] if history else final_objective,
        "final_objective": final_objective,
        "final_gradient_l2": final_gradient_norm,
        "converged": converged,
        "history": history,
    }
