from __future__ import annotations

from pathlib import Path

import pytest

from boatrace_ai import protected_historical_evaluation as evaluation
from boatrace_ai.standard_evaluation import race_set_sha256


RACE_KEYS = [
    ("train", "2025-01-01", "01", 1),
    ("holdout-1", "2026-01-01", "01", 1),
    ("holdout-2", "2026-01-02", "02", 2),
]


def scored_entries(include_races: set[str]):
    for race_id, race_date, jcd, rno in RACE_KEYS:
        if race_id not in include_races:
            continue
        for lane in range(1, 7):
            yield 0.5 if lane == 1 else 0.1, {
                "race_id": race_id,
                "race_date": race_date,
                "jcd": jcd,
                "rno": rno,
                "lane": lane,
                "rank": lane,
            }


def test_scores_exact_range_and_normalizes_each_race(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}
    monkeypatch.setattr(
        evaluation.historical_model,
        "fit_streaming_pipeline",
        lambda _conn, **kwargs: captured.update(kwargs) or {"pipeline": object()},
    )
    monkeypatch.setattr(
        evaluation.historical_model,
        "iter_scored_entries",
        lambda _conn, *, include_races, **_kwargs: scored_entries(include_races),
    )

    metrics, predictions = evaluation.score_historical_baseline_range(
        None,
        RACE_KEYS,
        train_end=1,
        score_start=1,
        score_end=3,
    )

    assert captured["train_races"] == {"train"}
    assert captured["include_research"] is False
    assert set(predictions) == {"holdout-1", "holdout-2"}
    assert metrics["evaluated_races"] == 2
    assert all(
        sum(row["probability"] for row in rows) == pytest.approx(1.0)
        for rows in predictions.values()
    )


def test_rejects_artifact_from_different_training_universe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        evaluation.joblib,
        "load",
        lambda _path: {
            "pipeline": object(),
            "metadata": {
                "train_races": 1,
                "train_race_set_sha256": race_set_sha256({"different"}),
                "include_odds": False,
            },
        },
    )

    with pytest.raises(ValueError, match="training race hash mismatch"):
        evaluation.score_historical_baseline_range(
            None,
            RACE_KEYS,
            train_end=1,
            score_start=1,
            score_end=3,
            model_path=tmp_path / "baseline.joblib",
        )


def test_file_sha256_tracks_exact_artifact_bytes(tmp_path: Path) -> None:
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"protected-model")

    assert evaluation.file_sha256(artifact) == (
        "8260c04747c023d3fcd9f8ab8677e8abb097638af32a57ecf1cfcde51eaba612"
    )
