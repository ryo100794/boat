from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

from ..adaptive_allocation import allocate_adaptive_day, append_day_result, folds_by_full_day, zero_totals
from ..bankroll_backtest import _build_payout_model, _candidate_tickets, _load_trifecta_payouts
from ..calibrated_shadow_model import matrix_batch_ranges, stabilize_sparse_scaler
from ..db import connection, init_db
from ..feature_tuning import (
    _ensure_sparse_index32,
    iter_race_feature_rows,
    load_complete_race_ids,
    normalize_drop_feature_groups,
    to_hashable,
)
from ..hashed_feature_dataset import HashedRaceDataset, load_or_build_hashed_dataset
from ..modeling import _race_level_metrics


MODEL_NAME = "pastlog_listwise_pl_v1"
FEATURE_SET = "pastlog_v8_hashed_listwise"
TARGETS = ("winner", "top3_pl")


@dataclass
class ListwiseLinearModel:
    weights: np.ndarray
    scaler: StandardScaler
    target: str
    alpha: float
    learning_rate: float
    epochs: int
    loss_blend: float | None = None


def stable_softmax(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    shifted = values - np.max(values, axis=-1, keepdims=True)
    numerator = np.exp(shifted)
    return numerator / np.maximum(numerator.sum(axis=-1, keepdims=True), 1e-15)


def pl_loss_and_score_gradient(
    scores: np.ndarray,
    ranks: np.ndarray,
    *,
    target: str,
    loss_blend: float | None = None,
) -> tuple[float, np.ndarray]:
    """Mean loss and gradient for winner, top-three PL, or their convex blend."""
    if target not in TARGETS and not (
        target == "blended" and loss_blend is not None
    ):
        raise ValueError(f"unknown target: {target}")
    if loss_blend is not None:
        blend = float(loss_blend)
        if not 0.0 <= blend <= 1.0:
            raise ValueError("loss_blend must be between 0 and 1")
        if blend == 0.0:
            return pl_loss_and_score_gradient(scores, ranks, target="winner")
        if blend == 1.0:
            return pl_loss_and_score_gradient(scores, ranks, target="top3_pl")
        winner_loss, winner_gradient = pl_loss_and_score_gradient(
            scores, ranks, target="winner"
        )
        top3_loss, top3_gradient = pl_loss_and_score_gradient(
            scores, ranks, target="top3_pl"
        )
        return (
            (1.0 - blend) * winner_loss + blend * top3_loss,
            (1.0 - blend) * winner_gradient + blend * top3_gradient,
        )
    values = np.asarray(scores, dtype=np.float64)
    rank_values = np.asarray(ranks)
    if values.ndim != 2 or values.shape[1] != 6 or rank_values.shape != values.shape:
        raise ValueError("scores and ranks must both have shape (races, 6)")
    gradient = np.zeros_like(values)
    stages = 1 if target == "winner" else 3
    order = np.argsort(rank_values, axis=1)
    remaining = np.ones_like(values, dtype=bool)
    race_indices = np.arange(values.shape[0])
    stage_losses = np.empty((values.shape[0], stages), dtype=np.float64)
    for stage in range(stages):
        actual = order[:, stage]
        probabilities = stable_softmax(np.where(remaining, values, -np.inf))
        stage_losses[:, stage] = -np.log(
            np.maximum(1e-15, probabilities[race_indices, actual])
        )
        gradient += probabilities
        gradient[race_indices, actual] -= 1.0
        remaining[race_indices, actual] = False
    denominator = max(1, values.shape[0] * stages)
    return float(stage_losses.sum() / denominator), gradient / denominator


def fit_scaler(dataset: HashedRaceDataset, *, race_end: int, batch_rows: int) -> StandardScaler:
    row_end = min(dataset.race_count, max(0, int(race_end))) * 6
    if row_end <= 0:
        raise ValueError("no races available for scaler")
    scaler = StandardScaler(with_mean=False)
    for start, stop in matrix_batch_ranges(row_end, batch_rows):
        scaler.partial_fit(dataset.matrix[start:stop])
    return stabilize_sparse_scaler(scaler)


def train_listwise_model(
    dataset: HashedRaceDataset,
    *,
    train_race_end: int,
    target: str = "top3_pl",
    alpha: float = 1e-4,
    learning_rate: float = 0.02,
    epochs: int = 3,
    batch_races: int = 1_000,
    scaler: StandardScaler | None = None,
    loss_blend: float | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
) -> tuple[ListwiseLinearModel, list[dict[str, Any]]]:
    if target not in TARGETS and not (
        target == "blended" and loss_blend is not None
    ):
        raise ValueError(f"unknown target: {target}")
    if loss_blend is not None and not 0.0 <= float(loss_blend) <= 1.0:
        raise ValueError("loss_blend must be between 0 and 1")
    if early_stopping_patience is not None and int(early_stopping_patience) < 1:
        raise ValueError("early_stopping_patience must be positive")
    if float(early_stopping_min_delta) < 0.0:
        raise ValueError("early_stopping_min_delta must be non-negative")
    train_end = min(dataset.race_count, max(0, int(train_race_end)))
    if train_end <= 0:
        raise ValueError("no races available for training")
    batch_size = max(1, int(batch_races))
    scaler = scaler or fit_scaler(dataset, race_end=train_end, batch_rows=batch_size * 6)
    weights = np.zeros(dataset.n_features, dtype=np.float64)
    first_moment = np.zeros_like(weights)
    second_moment = np.zeros_like(weights)
    beta1, beta2, step = 0.9, 0.999, 0
    history: list[dict[str, Any]] = []
    best_loss = math.inf
    best_weights: np.ndarray | None = None
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(max(1, int(epochs))):
        loss_sum = 0.0
        seen = 0
        for race_start in range(0, train_end, batch_size):
            race_stop = min(train_end, race_start + batch_size)
            matrix = scaler.transform(dataset.matrix[dataset.row_slice(race_start, race_stop)])
            scores = np.asarray(matrix.dot(weights)).reshape(-1, 6)
            loss, score_gradient = pl_loss_and_score_gradient(
                scores,
                dataset.ranks[race_start:race_stop],
                target=target,
                loss_blend=loss_blend,
            )
            gradient = np.asarray(matrix.T.dot(score_gradient.reshape(-1))).reshape(-1)
            gradient += float(alpha) * weights
            norm = float(np.linalg.norm(gradient))
            if norm > 25.0:
                gradient *= 25.0 / norm
            step += 1
            first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
            second_moment = beta2 * second_moment + (1.0 - beta2) * gradient * gradient
            weights -= float(learning_rate) * (first_moment / (1.0 - beta1**step)) / (
                np.sqrt(second_moment / (1.0 - beta2**step)) + 1e-8
            )
            count = race_stop - race_start
            loss_sum += loss * count
            seen += count
        epoch_loss = loss_sum / max(1, seen)
        improved = epoch_loss < best_loss - float(early_stopping_min_delta)
        if improved:
            best_loss = epoch_loss
            best_weights = weights.copy()
            best_epoch = epoch + 1
            stale_epochs = 0
        else:
            stale_epochs += 1
        history.append({
            "epoch": float(epoch + 1),
            "training_ranking_log_loss": epoch_loss,
            "weight_l2": float(np.linalg.norm(weights)),
            "improved": improved,
        })
        if (
            early_stopping_patience is not None
            and stale_epochs >= int(early_stopping_patience)
        ):
            history[-1]["early_stopped"] = True
            break
    if best_weights is not None:
        weights = best_weights
    return ListwiseLinearModel(
        weights,
        scaler,
        target,
        float(alpha),
        float(learning_rate),
        best_epoch,
        None if loss_blend is None else float(loss_blend),
    ), history


def predict_race_probabilities(
    dataset: HashedRaceDataset,
    model: ListwiseLinearModel,
    *,
    race_start: int,
    race_end: int,
    batch_races: int,
) -> Iterable[np.ndarray]:
    batch_size = max(1, int(batch_races))
    for start in range(max(0, race_start), min(dataset.race_count, race_end), batch_size):
        stop = min(dataset.race_count, race_end, start + batch_size)
        matrix = model.scaler.transform(dataset.matrix[dataset.row_slice(start, stop)])
        scores = np.asarray(matrix.dot(model.weights)).reshape(-1, 6)
        yield from stable_softmax(scores)


def evaluate_range(
    dataset: HashedRaceDataset,
    model: ListwiseLinearModel,
    *,
    race_start: int,
    race_end: int,
    batch_races: int,
    keep_rows: bool = False,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    labels: list[int] = []
    probabilities: list[float] = []
    predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ranking_loss = 0.0
    count = 0
    rows = predict_race_probabilities(
        dataset, model, race_start=race_start, race_end=race_end, batch_races=batch_races
    )
    for race_index, lane_probabilities in zip(range(race_start, race_end), rows):
        race_id, race_date, jcd, rno = dataset.race_keys[race_index]
        ranks = dataset.ranks[race_index]
        loss, _ = pl_loss_and_score_gradient(
            np.log(np.maximum(lane_probabilities, 1e-15))[None, :],
            ranks[None, :],
            target="top3_pl",
        )
        ranking_loss += loss
        count += 1
        for lane in range(1, 7):
            label = int(ranks[lane - 1] == 1)
            probability = float(lane_probabilities[lane - 1])
            labels.append(label)
            probabilities.append(probability)
            predictions[race_id].append({
                "race_id": race_id,
                "race_date": race_date,
                "jcd": jcd,
                "rno": int(rno),
                "lane": lane,
                "rank": int(ranks[lane - 1]),
                "probability": probability,
            })
    metrics = {
        "evaluated_races": count,
        "entry_log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "entry_brier": float(brier_score_loss(labels, probabilities)),
        "ranking_log_loss": ranking_loss / max(1, count),
        **_race_level_metrics(predictions),
    }
    return metrics, predictions if keep_rows else {}
