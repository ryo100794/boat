from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ..db import connection, init_db
from ..feature_tuning import load_complete_race_ids
from ..hashed_feature_dataset import race_ids_sha256
from .feature_search import load_variant_dataset_with_cache
from .model import ListwiseLinearModel, evaluate_range, fit_scaler
from .newton import refine_newton_cg
from .newton_refine import dump_joblib_atomic


MODEL_NAME = "pastlog_listwise_newton_cutoff_refit_v1"


def cutoff_boundaries(
    race_keys: list[tuple[str, str, str, int]],
    *,
    training_cutoff: str,
    evaluation_from: str,
    evaluation_through: str,
) -> tuple[int, int, int]:
    if not training_cutoff < evaluation_from <= evaluation_through:
        raise ValueError("dates must satisfy training_cutoff < evaluation_from <= evaluation_through")
    train_end = sum(str(row[1]) <= training_cutoff for row in race_keys)
    evaluation_start = sum(str(row[1]) < evaluation_from for row in race_keys)
    evaluation_end = sum(str(row[1]) <= evaluation_through for row in race_keys)
    if train_end <= 0 or evaluation_start != train_end:
        raise ValueError("cutoff and evaluation period must be adjacent complete-day ranges")
    if evaluation_end <= evaluation_start:
        raise ValueError("evaluation period contains no complete races")
    return train_end, evaluation_start, evaluation_end


def rescale_weights_preserving_scores(
    weights: np.ndarray,
    *,
    old_scale: np.ndarray,
    new_scale: np.ndarray,
) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    old = np.asarray(old_scale, dtype=np.float64)
    new = np.asarray(new_scale, dtype=np.float64)
    if values.ndim != 1 or old.shape != values.shape or new.shape != values.shape:
        raise ValueError("weights and scaler vectors must have identical one-dimensional shape")
    if (
        not np.isfinite(values).all()
        or not np.isfinite(old).all()
        or not np.isfinite(new).all()
        or np.any(old <= 0.0)
        or np.any(new <= 0.0)
    ):
        raise ValueError("weights and scaler vectors must be finite with positive scales")
    return values * new / old


def validate_source_artifact(
    artifact: dict[str, Any],
    *,
    training_cutoff: str,
    evaluation_from: str,
) -> ListwiseLinearModel:
    model = artifact.get("model")
    hasher = artifact.get("hasher")
    trained_through = artifact.get("trained_through")
    if not isinstance(model, ListwiseLinearModel) or hasher is None:
        raise ValueError("source artifact must contain listwise model and hasher")
    weights = np.asarray(model.weights)
    scale = np.asarray(getattr(model.scaler, "scale_", ()))
    artifact_features = int(artifact.get("n_features") or weights.size)
    hasher_features = int(getattr(hasher, "n_features", -1))
    if (
        weights.ndim != 1
        or scale.shape != weights.shape
        or artifact_features != weights.size
        or hasher_features != weights.size
    ):
        raise ValueError("source model, scaler, artifact, and hasher dimensions differ")
    if (
        getattr(hasher, "input_type", None) != "dict"
        or bool(getattr(hasher, "alternate_sign", True))
    ):
        raise ValueError("source hasher settings do not match the fixed feature pipeline")
    if not isinstance(trained_through, (list, tuple)) or len(trained_through) < 2:
        raise ValueError("source artifact lacks trained_through metadata")
    if str(trained_through[1]) > training_cutoff:
        raise ValueError("source artifact was trained beyond the requested cutoff")
    if training_cutoff >= evaluation_from:
        raise ValueError("training cutoff overlaps evaluation period")
    return model


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(conn, *, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    source_path = Path(args.source_model)
    model_output = Path(args.model_output)
    if source_path.resolve() == model_output.resolve():
        raise ValueError("model output must not overwrite the source model")
    artifact = joblib.load(source_path)
    source_model = validate_source_artifact(
        artifact,
        training_cutoff=args.training_cutoff,
        evaluation_from=args.evaluation_from,
    )
    all_keys = load_complete_race_ids(conn)
    race_keys = [row for row in all_keys if str(row[1]) <= args.evaluation_through]
    train_end, evaluation_start, evaluation_end = cutoff_boundaries(
        race_keys,
        training_cutoff=args.training_cutoff,
        evaluation_from=args.evaluation_from,
        evaluation_through=args.evaluation_through,
    )
    dropped = tuple(str(value) for value in artifact.get("drop_feature_groups") or ())
    feature_variant = str(artifact.get("feature_variant") or "full")
    n_features = int(artifact.get("n_features") or len(source_model.weights))
    dataset, cache_source, cache_prefix = load_variant_dataset_with_cache(
        conn,
        race_keys=race_keys,
        cache_dir=Path(args.cache_dir),
        name=feature_variant,
        dropped=dropped,
        n_features=n_features,
        batch_races=args.batch_races,
        write_cache=True,
    )
    scaler = fit_scaler(
        dataset,
        race_end=train_end,
        batch_rows=args.batch_races * 6,
    )
    warm_weights = rescale_weights_preserving_scores(
        source_model.weights,
        old_scale=np.asarray(source_model.scaler.scale_),
        new_scale=np.asarray(scaler.scale_),
    )
    warm_model = ListwiseLinearModel(
        weights=warm_weights,
        scaler=scaler,
        target=source_model.target,
        alpha=float(source_model.alpha),
        learning_rate=float(source_model.learning_rate),
        epochs=int(source_model.epochs),
    )
    before, _ = evaluate_range(
        dataset,
        warm_model,
        race_start=evaluation_start,
        race_end=evaluation_end,
        batch_races=args.batch_races,
    )
    refined, convergence = refine_newton_cg(
        dataset,
        warm_model,
        train_race_end=train_end,
        batch_races=args.batch_races,
        max_newton_iterations=args.max_newton_iterations,
        max_cg_iterations=args.max_cg_iterations,
        gradient_tolerance=args.gradient_tolerance,
        cg_tolerance=args.cg_tolerance,
    )
    after, _ = evaluate_range(
        dataset,
        refined,
        race_start=evaluation_start,
        race_end=evaluation_end,
        batch_races=args.batch_races,
    )
    source_hash = file_sha256(source_path)
    dump_joblib_atomic(
        model_output,
        {
            "model": refined,
            "hasher": artifact["hasher"],
            "feature_variant": feature_variant,
            "drop_feature_groups": dropped,
            "n_features": n_features,
            "trained_races": train_end,
            "trained_through": race_keys[train_end - 1],
            "target": refined.target,
            "alpha": refined.alpha,
            "race_universe_sha256": race_ids_sha256(race_keys),
            "training_cutoff": args.training_cutoff,
            "source_model": str(source_path),
            "source_model_sha256": source_hash,
            "coefficient_optimizer": "cutoff scaler transfer + matrix-free Newton-CG",
        },
    )
    result = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "model": MODEL_NAME,
        "comparison_role": "fixed-structure pre-market-cutoff refit shadow",
        "source_model": str(source_path),
        "source_model_sha256": source_hash,
        "model_artifact": str(model_output),
        "training_cutoff": args.training_cutoff,
        "evaluation_from": args.evaluation_from,
        "evaluation_through": args.evaluation_through,
        "trained_races": train_end,
        "trained_through": race_keys[train_end - 1],
        "evaluation_races": evaluation_end - evaluation_start,
        "evaluation_race_universe_sha256": race_ids_sha256(
            race_keys[evaluation_start:evaluation_end]
        ),
        "feature_variant": feature_variant,
        "drop_feature_groups": list(dropped),
        "target": refined.target,
        "alpha": refined.alpha,
        "n_features": n_features,
        "cache_source": cache_source,
        "cache_prefix": str(cache_prefix) if cache_prefix is not None else None,
        "before_refit": before,
        "after_refit": after,
        "newton_convergence": convergence,
        "ranking_loss_improved": (
            float(after["ranking_log_loss"]) < float(before["ranking_log_loss"])
        ),
        "top1_not_worse": (
            float(after["winner_top1_accuracy"])
            >= float(before["winner_top1_accuracy"])
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
        description="Refit a fixed listwise structure through a pre-market cutoff."
    )
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--source-model", default="data/models/listwise_newton_cg_v1.joblib")
    parser.add_argument(
        "--model-output",
        default="data/models/listwise_newton_cutoff_20260717.joblib",
    )
    parser.add_argument(
        "--output",
        default="data/models/listwise_newton_cutoff_20260717.json",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/models/listwise_cutoff_cache",
    )
    parser.add_argument("--training-cutoff", default="2026-07-17")
    parser.add_argument("--evaluation-from", default="2026-07-18")
    parser.add_argument("--evaluation-through", default="2026-07-21")
    parser.add_argument("--batch-races", type=int, default=2_000)
    parser.add_argument("--max-newton-iterations", type=int, default=5)
    parser.add_argument("--max-cg-iterations", type=int, default=30)
    parser.add_argument("--gradient-tolerance", type=float, default=1e-4)
    parser.add_argument("--cg-tolerance", type=float, default=1e-3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    init_db(args.db)
    with connection(args.db) as conn:
        result = run(conn, args=args)
    compact = {
        key: value
        for key, value in result.items()
        if key not in {"newton_convergence"}
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
