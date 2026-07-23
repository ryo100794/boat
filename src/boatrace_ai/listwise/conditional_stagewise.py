from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import StandardScaler

from ..calibrated_shadow_model import matrix_batch_ranges
from ..db import connection, init_db
from ..feature_tuning import load_complete_race_ids
from ..hashed_feature_dataset import HashedRaceDataset, load_hashed_dataset
from .model import ListwiseLinearModel, fit_scaler, stable_softmax
from .newton_refine import dump_joblib_atomic
from .stagewise_mlp import (
    COMBINATION_LANES,
    EPSILON,
    actual_combination_indices,
    cutoff_boundaries,
    evaluate_listwise_baseline,
    stagewise_trifecta_probabilities,
)


MODEL_NAME = "pastlog_conditional_stagewise_pl_v1"
STAGES = 3


@dataclass
class ConditionalStagewiseModel:
    weights: np.ndarray
    scaler: StandardScaler
    alpha: float
    learning_rate: float
    epochs: int


def conditional_loss_and_score_gradient(
    scores: np.ndarray,
    ranks: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Mean top-three conditional choice loss and score gradient."""
    values = np.asarray(scores, dtype=np.float64)
    rank_values = np.asarray(ranks)
    if values.ndim != 3 or values.shape[1:] != (6, STAGES):
        raise ValueError("scores must have shape (races, 6, 3)")
    if rank_values.shape != values.shape[:2]:
        raise ValueError("ranks must have shape (races, 6)")

    gradient = np.zeros_like(values)
    total_loss = 0.0
    for race_index in range(values.shape[0]):
        order = np.argsort(rank_values[race_index])
        remaining = np.ones(6, dtype=bool)
        for stage in range(STAGES):
            actual = int(order[stage])
            candidates = np.flatnonzero(remaining)
            probabilities = stable_softmax(values[race_index, candidates, stage])
            actual_position = int(np.flatnonzero(candidates == actual)[0])
            total_loss -= math.log(max(EPSILON, float(probabilities[actual_position])))
            gradient[race_index, candidates, stage] += probabilities
            gradient[race_index, actual, stage] -= 1.0
            remaining[actual] = False
    denominator = max(1, values.shape[0] * STAGES)
    return total_loss / denominator, gradient / denominator


def fit_conditional_stagewise_model(
    dataset: HashedRaceDataset,
    *,
    train_race_end: int,
    epochs: int,
    alpha: float,
    learning_rate: float,
    batch_races: int,
    scaler: StandardScaler | None = None,
) -> tuple[ConditionalStagewiseModel, list[dict[str, float]]]:
    train_end = min(dataset.race_count, max(0, int(train_race_end)))
    if train_end <= 0:
        raise ValueError("no races available for training")
    batch_size = max(1, int(batch_races))
    scaler = scaler or fit_scaler(
        dataset,
        race_end=train_end,
        batch_rows=batch_size * 6,
    )
    weights = np.zeros((STAGES, dataset.n_features), dtype=np.float64)
    first_moment = np.zeros_like(weights)
    second_moment = np.zeros_like(weights)
    beta1, beta2, step = 0.9, 0.999, 0
    history: list[dict[str, float]] = []

    for epoch in range(max(1, int(epochs))):
        loss_sum = 0.0
        seen = 0
        started = time.perf_counter()
        for race_start in range(0, train_end, batch_size):
            race_stop = min(train_end, race_start + batch_size)
            matrix = scaler.transform(
                dataset.matrix[dataset.row_slice(race_start, race_stop)]
            )
            flat_scores = np.column_stack(
                [np.asarray(matrix.dot(stage_weights)).reshape(-1) for stage_weights in weights]
            )
            scores = flat_scores.reshape(-1, 6, STAGES)
            loss, score_gradient = conditional_loss_and_score_gradient(
                scores,
                dataset.ranks[race_start:race_stop],
            )
            gradient = np.vstack(
                [
                    np.asarray(
                        matrix.T.dot(score_gradient[:, :, stage].reshape(-1))
                    ).reshape(-1)
                    for stage in range(STAGES)
                ]
            )
            gradient += float(alpha) * weights
            norm = float(np.linalg.norm(gradient))
            if norm > 25.0:
                gradient *= 25.0 / norm
            step += 1
            first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
            second_moment = beta2 * second_moment + (1.0 - beta2) * gradient * gradient
            first_unbiased = first_moment / (1.0 - beta1**step)
            second_unbiased = second_moment / (1.0 - beta2**step)
            weights -= float(learning_rate) * first_unbiased / (
                np.sqrt(second_unbiased) + 1e-8
            )
            count = race_stop - race_start
            loss_sum += loss * count
            seen += count
        history.append(
            {
                "epoch": float(epoch + 1),
                "training_conditional_log_loss": loss_sum / max(1, seen),
                "weight_l2": float(np.linalg.norm(weights)),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
    return (
        ConditionalStagewiseModel(
            weights=weights,
            scaler=scaler,
            alpha=float(alpha),
            learning_rate=float(learning_rate),
            epochs=max(1, int(epochs)),
        ),
        history,
    )


def conditional_position_utilities(
    model: ConditionalStagewiseModel,
    matrix,
) -> np.ndarray:
    transformed = model.scaler.transform(matrix)
    flat_scores = np.column_stack(
        [
            np.asarray(transformed.dot(stage_weights)).reshape(-1)
            for stage_weights in model.weights
        ]
    )
    scores = flat_scores.reshape(-1, 6, STAGES)
    scores -= np.max(scores, axis=1, keepdims=True)
    return np.exp(np.clip(scores, -50.0, 0.0))


def _new_metrics() -> dict[str, float | int]:
    return {
        "races": 0,
        "trifecta_loss": 0.0,
        "conditional_loss": 0.0,
        "winner_hits": 0,
        "trifecta_top1_hits": 0,
        "trifecta_top5_hits": 0,
    }


def _update_metrics(
    accumulator: dict[str, float | int],
    *,
    utilities: np.ndarray,
    ranks: np.ndarray,
) -> None:
    race_count = int(ranks.shape[0])
    trifecta = stagewise_trifecta_probabilities(utilities)
    actual = actual_combination_indices(ranks)
    rows = np.arange(race_count)
    accumulator["races"] += race_count
    accumulator["trifecta_loss"] += float(
        -np.log(np.maximum(trifecta[rows, actual], EPSILON)).sum()
    )
    conditional_loss, _gradient = conditional_loss_and_score_gradient(
        np.log(np.maximum(utilities, EPSILON)),
        ranks,
    )
    accumulator["conditional_loss"] += conditional_loss * race_count
    first_marginals = np.column_stack(
        [trifecta[:, COMBINATION_LANES[:, 0] == lane].sum(axis=1) for lane in range(6)]
    )
    winners = np.argmin(ranks, axis=1)
    accumulator["winner_hits"] += int(
        np.sum(np.argmax(first_marginals, axis=1) == winners)
    )
    order = np.argsort(-trifecta, axis=1)
    accumulator["trifecta_top1_hits"] += int(np.sum(order[:, 0] == actual))
    accumulator["trifecta_top5_hits"] += int(
        np.sum(np.any(order[:, :5] == actual[:, None], axis=1))
    )


def _finalize_metrics(accumulator: dict[str, float | int]) -> dict[str, Any]:
    races = int(accumulator["races"])
    return {
        "evaluated_races": races,
        "conditional_log_loss": float(accumulator["conditional_loss"]) / max(1, races),
        "trifecta_log_loss": float(accumulator["trifecta_loss"]) / max(1, races),
        "winner_top1_accuracy": int(accumulator["winner_hits"]) / max(1, races),
        "trifecta_top1_hit_rate": int(accumulator["trifecta_top1_hits"]) / max(1, races),
        "trifecta_top5_hit_rate": int(accumulator["trifecta_top5_hits"])
        / max(1, races),
    }


def evaluate_conditional_stagewise_model(
    dataset: HashedRaceDataset,
    model: ConditionalStagewiseModel,
    *,
    race_start: int,
    race_end: int,
    batch_races: int,
) -> dict[str, Any]:
    accumulator = _new_metrics()
    for start in range(race_start, race_end, max(1, int(batch_races))):
        stop = min(race_end, start + max(1, int(batch_races)))
        utilities = conditional_position_utilities(
            model,
            dataset.matrix[dataset.row_slice(start, stop)],
        )
        _update_metrics(
            accumulator,
            utilities=utilities,
            ranks=dataset.ranks[start:stop],
        )
    return _finalize_metrics(accumulator)


def _load_dataset(conn, cache_prefix: Path) -> tuple[HashedRaceDataset, dict[str, Any], list[Any]]:
    manifest = json.loads(
        Path(f"{cache_prefix}.manifest.json").read_text(encoding="utf-8")
    )
    last_date = str(manifest["last_race_id"])[:10]
    race_keys = [
        row for row in load_complete_race_ids(conn) if str(row[1]) <= last_date
    ]
    n_features = int(manifest["n_features"])
    dropped = tuple(str(value) for value in manifest.get("drop_feature_groups") or ())
    hasher = FeatureHasher(
        n_features=n_features,
        input_type="dict",
        alternate_sign=False,
    )
    dataset = load_hashed_dataset(
        cache_prefix,
        race_keys=race_keys,
        n_features=n_features,
        drop_feature_groups=dropped,
        hasher=hasher,
    )
    if dataset is None:
        raise ValueError("hashed feature cache failed contract or integrity validation")
    return dataset, manifest, race_keys


def run(conn, *, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    cache_prefix = Path(args.cache_prefix)
    dataset, manifest, race_keys = _load_dataset(conn, cache_prefix)
    train_end, evaluation_start, evaluation_end = cutoff_boundaries(
        race_keys,
        training_through=args.training_through,
        evaluation_from=args.evaluation_from,
        evaluation_through=args.evaluation_through,
    )
    model, history = fit_conditional_stagewise_model(
        dataset,
        train_race_end=train_end,
        epochs=args.epochs,
        alpha=args.alpha,
        learning_rate=args.learning_rate,
        batch_races=args.batch_races,
    )
    metrics = evaluate_conditional_stagewise_model(
        dataset,
        model,
        race_start=evaluation_start,
        race_end=evaluation_end,
        batch_races=args.batch_races,
    )
    baseline = None
    if args.baseline_model:
        baseline_artifact = joblib.load(args.baseline_model)
        baseline_model = baseline_artifact.get("model")
        if not isinstance(baseline_model, ListwiseLinearModel):
            raise ValueError("baseline artifact does not contain a listwise model")
        baseline_dropped = tuple(
            str(value) for value in baseline_artifact.get("drop_feature_groups") or ()
        )
        if (
            len(baseline_model.weights) != dataset.n_features
            or baseline_dropped != dataset.drop_feature_groups
        ):
            raise ValueError("baseline and conditional model feature contracts differ")
        trained_through = baseline_artifact.get("trained_through")
        if (
            isinstance(trained_through, (list, tuple))
            and len(trained_through) >= 2
            and str(trained_through[1]) >= args.evaluation_from
        ):
            raise ValueError("baseline training overlaps the evaluation period")
        baseline = evaluate_listwise_baseline(
            dataset,
            baseline_model,
            race_start=evaluation_start,
            race_end=evaluation_end,
            batch_races=args.batch_races,
        )

    hasher = FeatureHasher(
        n_features=dataset.n_features,
        input_type="dict",
        alternate_sign=False,
    )
    artifact = {
        "model": model,
        "hasher": hasher,
        "model_name": MODEL_NAME,
        "drop_feature_groups": dataset.drop_feature_groups,
        "n_features": dataset.n_features,
        "feature_schema_version": manifest["feature_schema_version"],
        "trained_races": train_end,
        "trained_through": race_keys[train_end - 1],
        "training_cutoff": args.training_through,
        "architecture": {
            "target": "conditional_top3_pl",
            "stage_weights": STAGES,
        },
    }
    dump_joblib_atomic(Path(args.model_output), artifact)
    result = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "model": MODEL_NAME,
        "comparison_role": "fixed-cutoff conditional top-three Plackett-Luce research",
        "model_artifact": args.model_output,
        "training_through": args.training_through,
        "evaluation_from": args.evaluation_from,
        "evaluation_through": args.evaluation_through,
        "trained_races": train_end,
        "evaluation_races": evaluation_end - evaluation_start,
        "n_features": dataset.n_features,
        "drop_feature_groups": list(dataset.drop_feature_groups),
        "epochs": args.epochs,
        "alpha": args.alpha,
        "learning_rate": args.learning_rate,
        "training_history": history,
        "conditional_stagewise": metrics,
        "listwise_baseline": baseline,
        "comparison": (
            {
                "trifecta_log_loss_improved": metrics["trifecta_log_loss"]
                < baseline["trifecta_log_loss"],
                "winner_top1_not_worse": metrics["winner_top1_accuracy"]
                >= baseline["winner_top1_accuracy"],
                "trifecta_top5_not_worse": metrics["trifecta_top5_hit_rate"]
                >= baseline["trifecta_top5_hit_rate"],
            }
            if baseline is not None
            else None
        ),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate conditional stagewise Plackett-Luce probabilities."
    )
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--cache-prefix", required=True)
    parser.add_argument("--training-through", required=True)
    parser.add_argument("--evaluation-from", required=True)
    parser.add_argument("--evaluation-through", required=True)
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.0001)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--batch-races", type=int, default=2_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    init_db(args.db)
    with connection(args.db) as conn:
        result = run(conn, args=args)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
