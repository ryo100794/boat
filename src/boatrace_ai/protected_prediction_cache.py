from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import joblib

from . import historical_model
from .protected_historical_blend import prediction_metrics
from .protected_historical_evaluation import (
    file_sha256,
    score_historical_baseline_range,
)
from .standard_evaluation import race_set_sha256


CACHE_VERSION = 1


def cached_historical_baseline_range(
    conn: Any,
    race_keys: Sequence[tuple[str, str, str, int]],
    *,
    train_end: int,
    score_start: int,
    score_end: int,
    cache_dir: Path,
    model_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Reuse leakage-safe baseline predictions for an identical race boundary."""
    if not 0 < train_end <= score_start < score_end <= len(race_keys):
        raise ValueError("invalid historical baseline train/score boundary")
    training_ids = {str(row[0]) for row in race_keys[:train_end]}
    score_ids = {str(row[0]) for row in race_keys[score_start:score_end]}
    metadata = {
        "cache_version": CACHE_VERSION,
        "feature_set": historical_model.FEATURE_SET,
        "train_end": int(train_end),
        "score_start": int(score_start),
        "score_end": int(score_end),
        "training_race_set_sha256": race_set_sha256(training_ids),
        "score_race_set_sha256": race_set_sha256(score_ids),
        "model_sha256": file_sha256(model_path) if model_path is not None else None,
        "include_research": False,
    }
    digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    cache_path = Path(cache_dir) / f"baseline-{digest}.joblib"
    if cache_path.is_file():
        try:
            payload = joblib.load(cache_path)
            predictions = payload["predictions"]
            if payload.get("metadata") != metadata or set(predictions) != score_ids:
                raise ValueError("protected prediction cache metadata mismatch")
            return prediction_metrics(predictions), predictions
        except (EOFError, OSError, TypeError, ValueError, KeyError):
            pass

    metrics, predictions = score_historical_baseline_range(
        conn,
        race_keys,
        train_end=train_end,
        score_start=score_start,
        score_end=score_end,
        model_path=model_path,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
    try:
        joblib.dump(
            {"metadata": metadata, "predictions": predictions},
            temporary,
            compress=3,
        )
        temporary.replace(cache_path)
    finally:
        temporary.unlink(missing_ok=True)
    return metrics, predictions
