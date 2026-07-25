from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .calibrated_shadow_model import recency_sample_weights
from .db import connection
from .hashed_feature_dataset import HashedRaceDataset
from .recency_mlp_evaluation import (
    build_parser as build_recency_parser,
    evaluate_recency_mlp,
)


MODEL_NAME = "calibrated_lightgbm_recency_selected"
MODEL_KIND = "lightgbm"
FEATURE_SET = "pastlog_lightgbm_hash_v1"
DEFAULT_DROP_FEATURE_GROUPS = ("legacy_composites",)
DEFAULT_FEATURE_CACHE = Path(
    "data/models/lightgbm_features_16384_drop_legacy_composites"
)
DEFAULT_HALF_LIVES: tuple[float | None, ...] = (None, 365.0)


def train_lightgbm_bundle_from_dataset(
    dataset: HashedRaceDataset,
    *,
    train_race_count: int,
    model_kind: str,
    batch_size: int,
    epochs: int,
    alpha: float,
    recency_half_life_days: float | None,
    n_estimators: int = 300,
    num_leaves: int = 31,
    max_depth: int = -1,
    min_child_samples: int = 100,
    feature_fraction: float = 0.6,
    max_bin: int = 63,
    n_jobs: int = 16,
) -> dict[str, Any]:
    del batch_size, epochs, alpha
    if model_kind != MODEL_KIND:
        raise ValueError(f"LightGBM trainer requires model_kind={MODEL_KIND}")
    _validate_parameters(
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_child_samples=min_child_samples,
        feature_fraction=feature_fraction,
        max_bin=max_bin,
        n_jobs=n_jobs,
    )
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise RuntimeError(
            "LightGBM evaluation requires the gbdt optional dependency"
        ) from exc

    train_count = min(dataset.race_count, int(train_race_count))
    train_end = train_count * 6
    if train_end <= 0:
        raise ValueError("no cached training examples")
    labels = (dataset.ranks[:train_count].reshape(-1) == 1).astype(np.int8)
    if len(labels) != train_end:
        raise ValueError("cached label shape mismatch")
    sample_weights = recency_sample_weights(
        dataset,
        train_race_count=train_count,
        recency_half_life_days=recency_half_life_days,
    )
    classifier = LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=int(n_estimators),
        learning_rate=0.05,
        num_leaves=int(num_leaves),
        max_depth=int(max_depth),
        min_child_samples=int(min_child_samples),
        max_bin=int(max_bin),
        colsample_bytree=float(feature_fraction),
        subsample=1.0,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=int(n_jobs),
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    classifier.fit(
        dataset.matrix[:train_end],
        labels,
        sample_weight=sample_weights,
    )
    return {
        "scaler": None,
        "classifier": classifier,
        "model_kind": MODEL_KIND,
        "drop_feature_groups": list(dataset.drop_feature_groups),
        "examples": train_end,
        "n_features": dataset.n_features,
        "matrix_cached": True,
        "recency_half_life_days": (
            None
            if recency_half_life_days is None
            else float(recency_half_life_days)
        ),
        "lightgbm_parameters": {
            "n_estimators": int(n_estimators),
            "num_leaves": int(num_leaves),
            "max_depth": int(max_depth),
            "min_child_samples": int(min_child_samples),
            "feature_fraction": float(feature_fraction),
            "max_bin": int(max_bin),
            "n_jobs": int(n_jobs),
        },
    }


def _validate_parameters(
    *,
    n_estimators: int,
    num_leaves: int,
    max_depth: int,
    min_child_samples: int,
    feature_fraction: float,
    max_bin: int,
    n_jobs: int,
) -> None:
    if not 10 <= int(n_estimators) <= 2_000:
        raise ValueError("n_estimators must be between 10 and 2000")
    if not 2 <= int(num_leaves) <= 512:
        raise ValueError("num_leaves must be between 2 and 512")
    if int(max_depth) != -1 and not 1 <= int(max_depth) <= 20:
        raise ValueError("max_depth must be -1 or between 1 and 20")
    if not 1 <= int(min_child_samples) <= 100_000:
        raise ValueError("min_child_samples must be between 1 and 100000")
    if not np.isfinite(feature_fraction) or not 0.05 <= feature_fraction <= 1.0:
        raise ValueError("feature_fraction must be between 0.05 and 1.0")
    if not 15 <= int(max_bin) <= 255:
        raise ValueError("max_bin must be between 15 and 255")
    if not 1 <= int(n_jobs) <= 128:
        raise ValueError("n_jobs must be between 1 and 128")


def evaluate_lightgbm_recency(
    conn: Any,
    *,
    output_path: Path,
    evaluation_date: date,
    feature_cache: Path | None = DEFAULT_FEATURE_CACHE,
    half_lives: Sequence[float | None] = DEFAULT_HALF_LIVES,
    calibration_days: int = 180,
    drop_feature_groups: Sequence[str] = DEFAULT_DROP_FEATURE_GROUPS,
    model_output_path: Path | None = None,
    deployment_model_output_path: Path | None = None,
    incumbent_prediction_path: Path | None = None,
    incumbent_bankroll_path: Path | None = None,
    **trainer_kwargs: Any,
) -> dict[str, Any]:
    return evaluate_recency_mlp(
        conn,
        output_path=output_path,
        evaluation_date=evaluation_date,
        feature_cache=feature_cache,
        half_lives=half_lives,
        calibration_days=calibration_days,
        epochs=1,
        alpha=0.0,
        drop_feature_groups=drop_feature_groups,
        model_output_path=model_output_path,
        deployment_model_output_path=deployment_model_output_path,
        incumbent_prediction_path=incumbent_prediction_path,
        incumbent_bankroll_path=incumbent_bankroll_path,
        model_name=MODEL_NAME,
        model_kind=MODEL_KIND,
        feature_set=FEATURE_SET,
        bundle_trainer=train_lightgbm_bundle_from_dataset,
        trainer_kwargs=trainer_kwargs,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = build_recency_parser()
    parser.description = (
        "Select LightGBM recency decay on training-only calibration data"
    )
    parser.set_defaults(
        feature_cache=DEFAULT_FEATURE_CACHE,
        drop_feature_groups=DEFAULT_DROP_FEATURE_GROUPS,
        half_lives=DEFAULT_HALF_LIVES,
    )
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--min-child-samples", type=int, default=100)
    parser.add_argument("--feature-fraction", type=float, default=0.6)
    parser.add_argument("--max-bin", type=int, default=63)
    parser.add_argument("--n-jobs", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    trainer_kwargs = {
        "n_estimators": args.n_estimators,
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "min_child_samples": args.min_child_samples,
        "feature_fraction": args.feature_fraction,
        "max_bin": args.max_bin,
        "n_jobs": args.n_jobs,
    }
    _validate_parameters(**trainer_kwargs)
    with connection(args.db) as conn:
        result = evaluate_lightgbm_recency(
            conn,
            output_path=args.output,
            evaluation_date=args.evaluation_date,
            feature_cache=args.feature_cache,
            half_lives=args.half_lives,
            calibration_days=args.calibration_days,
            drop_feature_groups=args.drop_feature_groups,
            model_output_path=args.model_output,
            deployment_model_output_path=args.deployment_model_output,
            incumbent_prediction_path=args.incumbent_prediction,
            incumbent_bankroll_path=args.incumbent_bankroll,
            **trainer_kwargs,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
