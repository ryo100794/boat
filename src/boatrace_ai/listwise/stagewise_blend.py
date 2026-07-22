from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.feature_extraction import FeatureHasher

from ..db import connection, init_db
from ..feature_tuning import load_complete_race_ids
from ..hashed_feature_dataset import HashedRaceDataset, load_hashed_dataset
from .model import ListwiseLinearModel, stable_softmax
from .newton_refine import dump_joblib_atomic
from .stagewise_mlp import (
    COMBINATION_LANES,
    EPSILON,
    StagewiseMLPModel,
    actual_combination_indices,
    classifier_position_scores,
    stagewise_trifecta_probabilities,
)


MODEL_NAME = "pastlog_listwise_stagewise_blend_v1"


@dataclass
class StagewiseBlendModel:
    listwise_model: ListwiseLinearModel
    stagewise_model: StagewiseMLPModel
    stagewise_weight: float


def period_boundaries(
    race_keys: list[tuple[str, str, str, int]],
    *,
    date_from: str,
    date_through: str,
) -> tuple[int, int]:
    if date_from > date_through:
        raise ValueError("date_from must not exceed date_through")
    start = sum(str(row[1]) < date_from for row in race_keys)
    end = sum(str(row[1]) <= date_through for row in race_keys)
    if end <= start:
        raise ValueError("evaluation period contains no complete races")
    return start, end


def blend_probabilities(
    listwise: np.ndarray,
    stagewise: np.ndarray,
    *,
    stagewise_weight: float,
) -> np.ndarray:
    weight = float(stagewise_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("stagewise_weight must be between zero and one")
    blended = (1.0 - weight) * np.asarray(listwise) + weight * np.asarray(stagewise)
    return blended / np.maximum(blended.sum(axis=1, keepdims=True), EPSILON)


def _new_metrics() -> dict[str, float | int]:
    return {
        "races": 0,
        "trifecta_loss": 0.0,
        "winner_hits": 0,
        "trifecta_top1_hits": 0,
        "trifecta_top5_hits": 0,
    }


def update_metrics(
    accumulator: dict[str, float | int],
    *,
    probabilities: np.ndarray,
    ranks: np.ndarray,
) -> None:
    values = np.asarray(probabilities, dtype=np.float64)
    race_count = int(values.shape[0])
    if values.shape != (race_count, 120):
        raise ValueError("trifecta probabilities must have shape (races, 120)")
    actual = actual_combination_indices(ranks)
    rows = np.arange(race_count)
    order = np.argsort(-values, axis=1)
    first_marginals = np.column_stack(
        [values[:, COMBINATION_LANES[:, 0] == lane].sum(axis=1) for lane in range(6)]
    )
    winners = np.argmin(ranks, axis=1)
    accumulator["races"] += race_count
    accumulator["trifecta_loss"] += float(
        -np.log(np.maximum(values[rows, actual], EPSILON)).sum()
    )
    accumulator["winner_hits"] += int(
        np.sum(np.argmax(first_marginals, axis=1) == winners)
    )
    accumulator["trifecta_top1_hits"] += int(np.sum(order[:, 0] == actual))
    accumulator["trifecta_top5_hits"] += int(
        np.sum(np.any(order[:, :5] == actual[:, None], axis=1))
    )


def finalize_metrics(accumulator: dict[str, float | int]) -> dict[str, Any]:
    races = int(accumulator["races"])
    return {
        "evaluated_races": races,
        "trifecta_log_loss": float(accumulator["trifecta_loss"]) / max(1, races),
        "winner_top1_accuracy": int(accumulator["winner_hits"]) / max(1, races),
        "trifecta_top1_hit_rate": int(accumulator["trifecta_top1_hits"])
        / max(1, races),
        "trifecta_top5_hit_rate": int(accumulator["trifecta_top5_hits"])
        / max(1, races),
    }


def validate_artifact_pair(
    stagewise_artifact: dict[str, Any],
    listwise_artifact: dict[str, Any],
    *,
    n_features: int,
    dropped: tuple[str, ...],
    period_from: str,
) -> tuple[StagewiseMLPModel, ListwiseLinearModel]:
    stagewise_model = stagewise_artifact.get("model")
    listwise_model = listwise_artifact.get("model")
    if not isinstance(stagewise_model, StagewiseMLPModel):
        raise ValueError("stagewise artifact has the wrong model type")
    if not isinstance(listwise_model, ListwiseLinearModel):
        raise ValueError("listwise artifact has the wrong model type")
    for name, artifact in (
        ("stagewise", stagewise_artifact),
        ("listwise", listwise_artifact),
    ):
        artifact_features = int(artifact.get("n_features") or 0)
        artifact_dropped = tuple(
            str(value) for value in artifact.get("drop_feature_groups") or ()
        )
        trained_through = artifact.get("trained_through")
        if artifact_features != n_features or artifact_dropped != dropped:
            raise ValueError(f"{name} artifact feature contract differs from cache")
        if not isinstance(trained_through, (list, tuple)) or len(trained_through) < 2:
            raise ValueError(f"{name} artifact lacks training cutoff metadata")
        if str(trained_through[1]) >= period_from:
            raise ValueError(f"{name} training overlaps evaluation period")
    if len(listwise_model.weights) != n_features:
        raise ValueError("listwise coefficient count differs from cache")
    return stagewise_model, listwise_model


def evaluate_weights(
    dataset: HashedRaceDataset,
    *,
    stagewise_model: StagewiseMLPModel,
    listwise_model: ListwiseLinearModel,
    race_start: int,
    race_end: int,
    weights: Iterable[float],
    batch_races: int,
) -> dict[float, dict[str, Any]]:
    selected_weights = tuple(dict.fromkeys(float(value) for value in weights))
    if not selected_weights or any(not 0.0 <= value <= 1.0 for value in selected_weights):
        raise ValueError("weights must contain values between zero and one")
    accumulators = {weight: _new_metrics() for weight in selected_weights}
    batch_size = max(1, int(batch_races))
    for start in range(race_start, race_end, batch_size):
        stop = min(race_end, start + batch_size)
        raw_matrix = dataset.matrix[dataset.row_slice(start, stop)]
        _classes, stage_scores = classifier_position_scores(stagewise_model, raw_matrix)
        stagewise_probabilities = stagewise_trifecta_probabilities(
            stage_scores.reshape(-1, 6, 3)
        )
        listwise_matrix = listwise_model.scaler.transform(raw_matrix)
        lane_probabilities = stable_softmax(
            np.asarray(listwise_matrix.dot(listwise_model.weights)).reshape(-1, 6)
        )
        listwise_probabilities = stagewise_trifecta_probabilities(
            np.repeat(lane_probabilities[:, :, None], 3, axis=2)
        )
        ranks = dataset.ranks[start:stop]
        for weight in selected_weights:
            update_metrics(
                accumulators[weight],
                probabilities=blend_probabilities(
                    listwise_probabilities,
                    stagewise_probabilities,
                    stagewise_weight=weight,
                ),
                ranks=ranks,
            )
    return {
        weight: finalize_metrics(accumulator)
        for weight, accumulator in accumulators.items()
    }


def select_weight(results: dict[float, dict[str, Any]]) -> float:
    if not results:
        raise ValueError("blend selection requires evaluated weights")
    return min(
        results,
        key=lambda weight: (
            float(results[weight]["trifecta_log_loss"]),
            -float(results[weight]["trifecta_top5_hit_rate"]),
            abs(float(weight) - 0.5),
        ),
    )


def run(conn, *, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    manifest = json.loads(
        Path(f"{args.cache_prefix}.manifest.json").read_text(encoding="utf-8")
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
        Path(args.cache_prefix),
        race_keys=race_keys,
        n_features=n_features,
        drop_feature_groups=dropped,
        hasher=hasher,
    )
    if dataset is None:
        raise ValueError("hashed feature cache failed contract or integrity validation")

    selection_start, selection_end = period_boundaries(
        race_keys,
        date_from=args.selection_from,
        date_through=args.selection_through,
    )
    selection_stage_artifact = joblib.load(args.selection_stagewise_model)
    selection_list_artifact = joblib.load(args.selection_listwise_model)
    selection_stage, selection_list = validate_artifact_pair(
        selection_stage_artifact,
        selection_list_artifact,
        n_features=n_features,
        dropped=dropped,
        period_from=args.selection_from,
    )
    grid = [index / 20 for index in range(21)]
    selection_results = evaluate_weights(
        dataset,
        stagewise_model=selection_stage,
        listwise_model=selection_list,
        race_start=selection_start,
        race_end=selection_end,
        weights=grid,
        batch_races=args.batch_races,
    )
    selected_weight = select_weight(selection_results)

    evaluation_start, evaluation_end = period_boundaries(
        race_keys,
        date_from=args.evaluation_from,
        date_through=args.evaluation_through,
    )
    evaluation_stage_artifact = joblib.load(args.evaluation_stagewise_model)
    evaluation_list_artifact = joblib.load(args.evaluation_listwise_model)
    evaluation_stage, evaluation_list = validate_artifact_pair(
        evaluation_stage_artifact,
        evaluation_list_artifact,
        n_features=n_features,
        dropped=dropped,
        period_from=args.evaluation_from,
    )
    evaluation_results = evaluate_weights(
        dataset,
        stagewise_model=evaluation_stage,
        listwise_model=evaluation_list,
        race_start=evaluation_start,
        race_end=evaluation_end,
        weights=(0.0, selected_weight, 1.0),
        batch_races=args.batch_races,
    )
    model_artifact = {
        "model": StagewiseBlendModel(
            listwise_model=evaluation_list,
            stagewise_model=evaluation_stage,
            stagewise_weight=selected_weight,
        ),
        "hasher": evaluation_stage_artifact["hasher"],
        "model_name": MODEL_NAME,
        "feature_variant": evaluation_stage_artifact.get("feature_variant"),
        "drop_feature_groups": dropped,
        "n_features": n_features,
        "feature_schema_version": evaluation_stage_artifact.get(
            "feature_schema_version"
        ),
        "trained_races": evaluation_stage_artifact.get("trained_races"),
        "trained_through": evaluation_stage_artifact.get("trained_through"),
        "training_cutoff": evaluation_stage_artifact.get("training_cutoff"),
        "stagewise_weight": selected_weight,
        "weight_selection_from": args.selection_from,
        "weight_selection_through": args.selection_through,
        "final_holdout_from": args.evaluation_from,
        "final_holdout_through": args.evaluation_through,
    }
    dump_joblib_atomic(Path(args.model_output), model_artifact)
    result = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "model": MODEL_NAME,
        "comparison_role": "preselected blend weight on untouched final temporal holdout",
        "model_artifact": args.model_output,
        "selection_period": {
            "from": args.selection_from,
            "through": args.selection_through,
            "races": selection_end - selection_start,
        },
        "selection_grid": [
            {"stagewise_weight": weight, **selection_results[weight]}
            for weight in sorted(selection_results)
        ],
        "selected_stagewise_weight": selected_weight,
        "evaluation_period": {
            "from": args.evaluation_from,
            "through": args.evaluation_through,
            "races": evaluation_end - evaluation_start,
        },
        "final_evaluation": {
            "listwise": evaluation_results[0.0],
            "selected_blend": evaluation_results[selected_weight],
            "stagewise": evaluation_results[1.0],
        },
        "final_period_not_used_for_weight_selection": True,
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
        description="Select a listwise/stagewise blend before a final temporal holdout."
    )
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--cache-prefix", required=True)
    parser.add_argument("--selection-stagewise-model", required=True)
    parser.add_argument("--selection-listwise-model", required=True)
    parser.add_argument("--selection-from", required=True)
    parser.add_argument("--selection-through", required=True)
    parser.add_argument("--evaluation-stagewise-model", required=True)
    parser.add_argument("--evaluation-listwise-model", required=True)
    parser.add_argument("--evaluation-from", required=True)
    parser.add_argument("--evaluation-through", required=True)
    parser.add_argument("--batch-races", type=int, default=2_000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-output", required=True)
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
