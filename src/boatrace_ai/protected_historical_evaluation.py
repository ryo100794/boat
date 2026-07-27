from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import joblib

from . import historical_model
from .protected_historical_blend import prediction_metrics
from .standard_evaluation import race_set_sha256


def score_historical_baseline_range(
    conn: Any,
    race_keys: Sequence[tuple[str, str, str, int]],
    *,
    train_end: int,
    score_start: int,
    score_end: int,
    model_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    if not 0 < train_end <= score_start < score_end <= len(race_keys):
        raise ValueError("invalid historical baseline train/score boundary")
    training_races = {str(row[0]) for row in race_keys[:train_end]}
    expected_training_hash = race_set_sha256(training_races)
    if model_path is None:
        bundle = historical_model.fit_streaming_pipeline(
            conn,
            train_races=training_races,
            include_research=False,
        )
    else:
        bundle = joblib.load(model_path)
        if not isinstance(bundle, dict) or "pipeline" not in bundle:
            raise ValueError("historical baseline artifact lacks a pipeline")
        metadata = bundle.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("historical baseline artifact lacks metadata")
        if int(metadata.get("train_races") or 0) != train_end:
            raise ValueError("historical baseline training race count mismatch")
        if metadata.get("train_race_set_sha256") != expected_training_hash:
            raise ValueError("historical baseline training race hash mismatch")
        if metadata.get("include_odds") is not False:
            raise ValueError("historical baseline artifact must exclude odds")

    score_rows = race_keys[score_start:score_end]
    score_races = {str(row[0]) for row in score_rows}
    predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for probability, meta in historical_model.iter_scored_entries(
        conn,
        pipeline=bundle["pipeline"],
        include_races=score_races,
        from_date=str(score_rows[0][1]),
        through_date=str(score_rows[-1][1]),
        include_research=False,
    ):
        race_id = str(meta["race_id"])
        predictions[race_id].append(
            {
                "race_id": race_id,
                "race_date": str(meta["race_date"]),
                "jcd": str(meta["jcd"]),
                "rno": int(meta["rno"]),
                "lane": int(meta["lane"]),
                "rank": int(meta["rank"]),
                "label": 1 if int(meta["rank"]) == 1 else 0,
                "probability": float(probability),
            }
        )
    if set(predictions) != score_races:
        raise ValueError("historical baseline scored race set mismatch")
    for race_id, rows in predictions.items():
        total = sum(float(row["probability"]) for row in rows)
        if len(rows) != 6 or total <= 0.0:
            raise ValueError(f"historical baseline race is incomplete: {race_id}")
        for row in rows:
            row["probability"] = float(row["probability"]) / total
    normalized = dict(predictions)
    return prediction_metrics(normalized), normalized
