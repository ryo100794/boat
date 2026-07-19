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

from .adaptive_allocation import allocate_adaptive_day, append_day_result, folds_by_full_day, zero_totals
from .bankroll_backtest import _build_payout_model, _candidate_tickets, _load_trifecta_payouts
from .calibrated_shadow_model import matrix_batch_ranges
from .db import connection, init_db
from .feature_tuning import (
    _ensure_sparse_index32,
    iter_race_feature_rows,
    load_complete_race_ids,
    normalize_drop_feature_groups,
    to_hashable,
)
from .hashed_feature_dataset import HashedRaceDataset, load_or_build_hashed_dataset
from .modeling import _race_level_metrics


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
) -> tuple[float, np.ndarray]:
    """Mean loss and score gradient for a winner or PL top-three target."""
    if target not in TARGETS:
        raise ValueError(f"unknown target: {target}")
    values = np.asarray(scores, dtype=np.float64)
    rank_values = np.asarray(ranks)
    if values.ndim != 2 or values.shape[1] != 6 or rank_values.shape != values.shape:
        raise ValueError("scores and ranks must both have shape (races, 6)")
    gradient = np.zeros_like(values)
    total_loss = 0.0
    stages = 1 if target == "winner" else 3
    for race_index in range(values.shape[0]):
        order = np.argsort(rank_values[race_index])
        remaining = np.ones(6, dtype=bool)
        for stage in range(stages):
            actual = int(order[stage])
            lane_indices = np.flatnonzero(remaining)
            probabilities = stable_softmax(values[race_index, lane_indices])
            actual_position = int(np.flatnonzero(lane_indices == actual)[0])
            total_loss -= math.log(max(1e-15, float(probabilities[actual_position])))
            gradient[race_index, lane_indices] += probabilities
            gradient[race_index, actual] -= 1.0
            remaining[actual] = False
    denominator = max(1, values.shape[0] * stages)
    return total_loss / denominator, gradient / denominator


def fit_scaler(dataset: HashedRaceDataset, *, race_end: int, batch_rows: int) -> StandardScaler:
    row_end = min(dataset.race_count, max(0, int(race_end))) * 6
    if row_end <= 0:
        raise ValueError("no races available for scaler")
    scaler = StandardScaler(with_mean=False)
    for start, stop in matrix_batch_ranges(row_end, batch_rows):
        scaler.partial_fit(dataset.matrix[start:stop])
    return scaler


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
) -> tuple[ListwiseLinearModel, list[dict[str, float]]]:
    if target not in TARGETS:
        raise ValueError(f"unknown target: {target}")
    train_end = min(dataset.race_count, max(0, int(train_race_end)))
    if train_end <= 0:
        raise ValueError("no races available for training")
    batch_size = max(1, int(batch_races))
    scaler = scaler or fit_scaler(dataset, race_end=train_end, batch_rows=batch_size * 6)
    weights = np.zeros(dataset.n_features, dtype=np.float64)
    first_moment = np.zeros_like(weights)
    second_moment = np.zeros_like(weights)
    beta1, beta2, step = 0.9, 0.999, 0
    history: list[dict[str, float]] = []

    for epoch in range(max(1, int(epochs))):
        loss_sum = 0.0
        seen = 0
        for race_start in range(0, train_end, batch_size):
            race_stop = min(train_end, race_start + batch_size)
            matrix = scaler.transform(dataset.matrix[dataset.row_slice(race_start, race_stop)])
            scores = np.asarray(matrix.dot(weights)).reshape(-1, 6)
            loss, score_gradient = pl_loss_and_score_gradient(
                scores, dataset.ranks[race_start:race_stop], target=target
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
        history.append({
            "epoch": float(epoch + 1),
            "training_ranking_log_loss": loss_sum / max(1, seen),
            "weight_l2": float(np.linalg.norm(weights)),
        })
    return ListwiseLinearModel(
        weights, scaler, target, float(alpha), float(learning_rate), max(1, int(epochs))
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
