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
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from ..calibrated_shadow_model import matrix_batch_ranges
from ..db import connection, init_db
from ..fast_math import TRIFECTA_COMBINATIONS
from ..feature_tuning import load_complete_race_ids
from ..hashed_feature_dataset import HashedRaceDataset, load_hashed_dataset
from .model import ListwiseLinearModel, fit_scaler, stable_softmax
from .newton_refine import dump_joblib_atomic


MODEL_NAME = "pastlog_stagewise_mlp_v1"
EPSILON = 1e-15
COMBINATION_INDEX = {
    tuple(int(lane) for lane in combination): index
    for index, combination in enumerate(TRIFECTA_COMBINATIONS)
}
COMBINATION_LANES = np.asarray(TRIFECTA_COMBINATIONS, dtype=np.int8) - 1


@dataclass
class StagewiseMLPModel:
    scaler: StandardScaler
    classifier: MLPClassifier
    epochs: int
    alpha: float


def cutoff_boundaries(
    race_keys: list[tuple[str, str, str, int]],
    *,
    training_through: str,
    evaluation_from: str,
    evaluation_through: str,
) -> tuple[int, int, int]:
    if not training_through < evaluation_from <= evaluation_through:
        raise ValueError(
            "dates must satisfy training_through < evaluation_from <= evaluation_through"
        )
    train_end = sum(str(row[1]) <= training_through for row in race_keys)
    evaluation_start = sum(str(row[1]) < evaluation_from for row in race_keys)
    evaluation_end = sum(str(row[1]) <= evaluation_through for row in race_keys)
    if train_end <= 0 or train_end != evaluation_start:
        raise ValueError("training and evaluation ranges must be adjacent full days")
    if evaluation_end <= evaluation_start:
        raise ValueError("evaluation period contains no complete races")
    return train_end, evaluation_start, evaluation_end


def rank_class_labels(ranks: np.ndarray) -> np.ndarray:
    values = np.asarray(ranks)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("ranks must have shape (races, 6)")
    return np.where(values <= 3, values, 0).astype(np.int64).reshape(-1)


def stagewise_trifecta_probabilities(position_scores: np.ndarray) -> np.ndarray:
    """Convert positive first/second/third lane scores into 120 ordered probabilities."""
    scores = np.asarray(position_scores, dtype=np.float64)
    squeeze = scores.ndim == 2
    if squeeze:
        scores = scores[None, ...]
    if scores.ndim != 3 or scores.shape[1:] != (6, 3):
        raise ValueError("position_scores must have shape (races, 6, 3) or (6, 3)")
    scores = np.maximum(scores, EPSILON)
    first = COMBINATION_LANES[:, 0]
    second = COMBINATION_LANES[:, 1]
    third = COMBINATION_LANES[:, 2]
    first_scores = scores[:, :, 0]
    second_scores = scores[:, :, 1]
    third_scores = scores[:, :, 2]
    first_probability = first_scores[:, first] / np.maximum(
        first_scores.sum(axis=1, keepdims=True), EPSILON
    )
    second_denominator = second_scores.sum(axis=1, keepdims=True) - second_scores[:, first]
    second_probability = second_scores[:, second] / np.maximum(
        second_denominator, EPSILON
    )
    third_denominator = (
        third_scores.sum(axis=1, keepdims=True)
        - third_scores[:, first]
        - third_scores[:, second]
    )
    third_probability = third_scores[:, third] / np.maximum(
        third_denominator, EPSILON
    )
    probabilities = first_probability * second_probability * third_probability
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), EPSILON)
    return probabilities[0] if squeeze else probabilities


def actual_combination_indices(ranks: np.ndarray) -> np.ndarray:
    values = np.asarray(ranks)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("ranks must have shape (races, 6)")
    orders = np.argsort(values, axis=1)[:, :3] + 1
    return np.asarray(
        [COMBINATION_INDEX[tuple(int(value) for value in order)] for order in orders],
        dtype=np.int64,
    )


def fit_stagewise_model(
    dataset: HashedRaceDataset,
    *,
    train_race_end: int,
    epochs: int,
    alpha: float,
    batch_rows: int,
    hidden_layer_sizes: tuple[int, ...] = (64, 16),
) -> tuple[StagewiseMLPModel, list[dict[str, Any]]]:
    train_end = min(dataset.race_count, max(0, int(train_race_end)))
    if train_end <= 0:
        raise ValueError("no races available for training")
    row_end = train_end * 6
    scaler = fit_scaler(
        dataset,
        race_end=train_end,
        batch_rows=max(6, int(batch_rows)),
    )
    labels = rank_class_labels(dataset.ranks[:train_end])
    classifier = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=float(alpha),
        batch_size="auto",
        learning_rate_init=0.001,
        max_iter=1,
        random_state=42,
    )
    first = True
    history = []
    for epoch in range(max(1, int(epochs))):
        started = time.perf_counter()
        for start, stop in matrix_batch_ranges(row_end, max(6, int(batch_rows))):
            matrix = scaler.transform(dataset.matrix[start:stop])
            kwargs = {"classes": np.asarray([0, 1, 2, 3])} if first else {}
            classifier.partial_fit(matrix, labels[start:stop], **kwargs)
            first = False
        history.append(
            {
                "epoch": epoch + 1,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "loss": float(getattr(classifier, "loss_", math.nan)),
            }
        )
    return StagewiseMLPModel(
        scaler=scaler,
        classifier=classifier,
        epochs=max(1, int(epochs)),
        alpha=float(alpha),
    ), history


def classifier_position_scores(
    model: StagewiseMLPModel,
    matrix,
) -> tuple[np.ndarray, np.ndarray]:
    class_probabilities = np.asarray(
        model.classifier.predict_proba(model.scaler.transform(matrix)),
        dtype=np.float64,
    )
    class_indices = {int(value): index for index, value in enumerate(model.classifier.classes_)}
    if set(class_indices) != {0, 1, 2, 3}:
        raise ValueError("stagewise classifier must expose rank classes 0, 1, 2, and 3")
    position_scores = np.column_stack(
        [class_probabilities[:, class_indices[rank]] for rank in (1, 2, 3)]
    )
    return class_probabilities, position_scores


def _metric_accumulator() -> dict[str, Any]:
    return {
        "races": 0,
        "trifecta_loss": 0.0,
        "winner_hits": 0,
        "trifecta_top1_hits": 0,
        "trifecta_top5_hits": 0,
        "labels": [],
        "class_probabilities": [],
    }


def _update_metrics(
    accumulator: dict[str, Any],
    *,
    ranks: np.ndarray,
    position_scores: np.ndarray,
    class_probabilities: np.ndarray | None = None,
) -> None:
    race_count = int(ranks.shape[0])
    scores = np.asarray(position_scores, dtype=np.float64).reshape(race_count, 6, 3)
    trifecta = stagewise_trifecta_probabilities(scores)
    actual = actual_combination_indices(ranks)
    row_indices = np.arange(race_count)
    accumulator["races"] += race_count
    accumulator["trifecta_loss"] += float(
        -np.log(np.maximum(trifecta[row_indices, actual], EPSILON)).sum()
    )
    winners = np.argmin(ranks, axis=1)
    accumulator["winner_hits"] += int(
        np.sum(np.argmax(scores[:, :, 0], axis=1) == winners)
    )
    order = np.argsort(-trifecta, axis=1)
    accumulator["trifecta_top1_hits"] += int(np.sum(order[:, 0] == actual))
    accumulator["trifecta_top5_hits"] += int(
        np.sum(np.any(order[:, :5] == actual[:, None], axis=1))
    )
    if class_probabilities is not None:
        accumulator["labels"].append(rank_class_labels(ranks))
        accumulator["class_probabilities"].append(class_probabilities)


def _finalize_metrics(accumulator: dict[str, Any]) -> dict[str, Any]:
    races = int(accumulator["races"])
    metrics = {
        "evaluated_races": races,
        "trifecta_log_loss": accumulator["trifecta_loss"] / max(1, races),
        "winner_top1_accuracy": accumulator["winner_hits"] / max(1, races),
        "trifecta_top1_hit_rate": accumulator["trifecta_top1_hits"] / max(1, races),
        "trifecta_top5_hit_rate": accumulator["trifecta_top5_hits"] / max(1, races),
    }
    if accumulator["labels"]:
        labels = np.concatenate(accumulator["labels"])
        probabilities = np.vstack(accumulator["class_probabilities"])
        metrics["entry_rank_class_log_loss"] = float(
            log_loss(labels, probabilities, labels=[0, 1, 2, 3])
        )
    return metrics


def evaluate_stagewise_model(
    dataset: HashedRaceDataset,
    model: StagewiseMLPModel,
    *,
    race_start: int,
    race_end: int,
    batch_races: int,
) -> dict[str, Any]:
    accumulator = _metric_accumulator()
    for start in range(race_start, race_end, max(1, int(batch_races))):
        stop = min(race_end, start + max(1, int(batch_races)))
        matrix = dataset.matrix[dataset.row_slice(start, stop)]
        class_probabilities, position_scores = classifier_position_scores(model, matrix)
        _update_metrics(
            accumulator,
            ranks=dataset.ranks[start:stop],
            position_scores=position_scores,
            class_probabilities=class_probabilities,
        )
    return _finalize_metrics(accumulator)


def evaluate_listwise_baseline(
    dataset: HashedRaceDataset,
    model: ListwiseLinearModel,
    *,
    race_start: int,
    race_end: int,
    batch_races: int,
) -> dict[str, Any]:
    accumulator = _metric_accumulator()
    for start in range(race_start, race_end, max(1, int(batch_races))):
        stop = min(race_end, start + max(1, int(batch_races)))
        matrix = model.scaler.transform(dataset.matrix[dataset.row_slice(start, stop)])
        lane_probabilities = stable_softmax(
            np.asarray(matrix.dot(model.weights)).reshape(-1, 6)
        )
        position_scores = np.repeat(lane_probabilities[:, :, None], 3, axis=2)
        _update_metrics(
            accumulator,
            ranks=dataset.ranks[start:stop],
            position_scores=position_scores,
        )
    return _finalize_metrics(accumulator)


def run(conn, *, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    manifest_path = Path(f"{args.cache_prefix}.manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
        Path(args.cache_prefix),
        race_keys=race_keys,
        n_features=n_features,
        drop_feature_groups=dropped,
        hasher=hasher,
    )
    if dataset is None:
        raise ValueError("hashed feature cache failed contract or integrity validation")
    train_end, evaluation_start, evaluation_end = cutoff_boundaries(
        race_keys,
        training_through=args.training_through,
        evaluation_from=args.evaluation_from,
        evaluation_through=args.evaluation_through,
    )
    model, history = fit_stagewise_model(
        dataset,
        train_race_end=train_end,
        epochs=args.epochs,
        alpha=args.alpha,
        batch_rows=args.batch_rows,
        hidden_layer_sizes=tuple(args.hidden_layers),
    )
    stagewise = evaluate_stagewise_model(
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
            int(baseline_artifact.get("n_features") or len(baseline_model.weights))
            != n_features
            or len(baseline_model.weights) != n_features
            or baseline_dropped != dropped
        ):
            raise ValueError("baseline and stagewise feature contracts differ")
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
    artifact = {
        "model": model,
        "hasher": hasher,
        "model_name": MODEL_NAME,
        "feature_variant": "drop_base_pastlog",
        "drop_feature_groups": dropped,
        "n_features": n_features,
        "trained_races": train_end,
        "trained_through": race_keys[train_end - 1],
        "training_cutoff": args.training_through,
        "architecture": {
            "target": "entry_rank_class_0_1_2_3",
            "conditional_order": "stage-specific Plackett-Luce",
            "hidden_layers": list(args.hidden_layers),
        },
    }
    dump_joblib_atomic(Path(args.model_output), artifact)
    result = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "model": MODEL_NAME,
        "comparison_role": "fixed-cutoff stage-specific order probability shadow",
        "model_artifact": args.model_output,
        "training_through": args.training_through,
        "evaluation_from": args.evaluation_from,
        "evaluation_through": args.evaluation_through,
        "trained_races": train_end,
        "evaluation_races": evaluation_end - evaluation_start,
        "n_features": n_features,
        "drop_feature_groups": list(dropped),
        "hidden_layers": list(args.hidden_layers),
        "epochs": args.epochs,
        "alpha": args.alpha,
        "training_history": history,
        "stagewise": stagewise,
        "listwise_baseline": baseline,
        "comparison": (
            {
                "trifecta_log_loss_improved": stagewise["trifecta_log_loss"]
                < baseline["trifecta_log_loss"],
                "winner_top1_not_worse": stagewise["winner_top1_accuracy"]
                >= baseline["winner_top1_accuracy"],
                "trifecta_top5_not_worse": stagewise["trifecta_top5_hit_rate"]
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
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate stage-specific MLP trifecta probabilities."
    )
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--cache-prefix", required=True)
    parser.add_argument("--training-through", required=True)
    parser.add_argument("--evaluation-from", required=True)
    parser.add_argument("--evaluation-through", required=True)
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-model")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.0001)
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[64, 16])
    parser.add_argument("--batch-rows", type=int, default=12_000)
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
