from __future__ import annotations

import argparse
from datetime import date, timedelta

import numpy as np
import pytest
from scipy import sparse

pytest.importorskip("lightgbm")

from boatrace_ai.hashed_feature_dataset import HashedRaceDataset
from boatrace_ai import lightgbm_recency_evaluation as lightgbm_eval
from boatrace_ai.recency_mlp_evaluation import (
    score_range,
    select_recency_half_life,
)


def _dataset(race_count: int = 40) -> HashedRaceDataset:
    rows = race_count * 6
    matrix = sparse.lil_matrix((rows, 16), dtype=np.float32)
    ranks = np.empty((race_count, 6), dtype=np.int8)
    start = date(2026, 1, 1)
    race_keys = []
    for race in range(race_count):
        winner = race % 6
        race_keys.append(
            (
                f"race-{race:03d}",
                (start + timedelta(days=race)).isoformat(),
                "01",
                race % 12 + 1,
            )
        )
        for lane in range(6):
            row = race * 6 + lane
            matrix[row, lane] = 1.0
            matrix[row, 6] = float(lane + 1)
            matrix[row, 7] = float(race % 12 + 1)
            matrix[row, 8 + race % 8] = 1.0
            ranks[race, lane] = (lane - winner) % 6 + 1
    return HashedRaceDataset(
        matrix=matrix.tocsr(),
        race_keys=race_keys,
        ranks=ranks,
        n_features=16,
        drop_feature_groups=("legacy_composites",),
    )


def _trainer_kwargs() -> dict[str, object]:
    return {
        "n_estimators": 10,
        "num_leaves": 7,
        "max_depth": 4,
        "min_child_samples": 2,
        "feature_fraction": 1.0,
        "max_bin": 31,
        "n_jobs": 1,
    }


def test_lightgbm_bundle_scores_complete_normalized_races() -> None:
    dataset = _dataset()
    bundle = lightgbm_eval.train_lightgbm_bundle_from_dataset(
        dataset,
        train_race_count=30,
        model_kind="lightgbm",
        batch_size=64,
        epochs=1,
        alpha=0.0,
        recency_half_life_days=20.0,
        **_trainer_kwargs(),
    )

    metrics, predictions = score_range(
        dataset,
        bundle=bundle,
        race_start=30,
        race_end=40,
        batch_size=12,
    )

    assert bundle["scaler"] is None
    assert bundle["model_kind"] == "lightgbm"
    assert metrics["evaluated_races"] == 10
    assert len(predictions) == 10
    for rows in predictions.values():
        assert len(rows) == 6
        assert sum(float(row["probability"]) for row in rows) == pytest.approx(1.0)


def test_lightgbm_recency_selection_uses_injected_trainer() -> None:
    dataset = _dataset()
    selected, candidates, split = select_recency_half_life(
        dataset,
        outer_train_end=30,
        half_lives=(None, 20.0),
        calibration_days=10,
        batch_size=12,
        epochs=1,
        alpha=0.0,
        bundle_trainer=lightgbm_eval.train_lightgbm_bundle_from_dataset,
        model_kind="lightgbm",
        trainer_kwargs=_trainer_kwargs(),
    )

    assert selected in (None, 20.0)
    assert len(candidates) == 2
    assert split["calibration_races"] == 10
    assert all(np.isfinite(row["entry_log_loss"]) for row in candidates)


def test_lightgbm_structural_selection_is_nested_inside_training_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset()
    trained: list[tuple[int, int, float | None]] = []

    def fake_trainer(
        _dataset: HashedRaceDataset,
        **kwargs: object,
    ) -> dict[str, object]:
        leaves = int(kwargs["num_leaves"])
        trained.append(
            (
                int(kwargs["train_race_count"]),
                leaves,
                kwargs["recency_half_life_days"],
            )
        )
        return {"leaves": leaves}

    def fake_score(
        _dataset: HashedRaceDataset,
        *,
        bundle: dict[str, object],
        race_start: int,
        race_end: int,
        batch_size: int,
    ) -> tuple[dict[str, float | int], dict[str, list[dict[str, object]]]]:
        del batch_size
        loss = 0.3 if bundle["leaves"] == 31 else 0.4
        return (
            {
                "entry_log_loss": loss,
                "entry_brier": 0.1,
                "trifecta_log_loss": 3.8,
                "winner_top1_accuracy": 0.5,
                "trifecta_top1_hit_rate": 0.1,
                "trifecta_top5_hit_rate": 0.3,
                "evaluated_races": race_end - race_start,
            },
            {"calibration": [{"leaves": bundle["leaves"]}]},
        )

    monkeypatch.setattr(
        "boatrace_ai.recency_mlp_evaluation.score_range",
        fake_score,
    )
    selected, candidates, split = select_recency_half_life(
        dataset,
        outer_train_end=30,
        half_lives=(None, 20.0),
        calibration_days=10,
        bundle_trainer=fake_trainer,
        model_kind="lightgbm",
        trainer_parameter_candidates=[
            {"num_leaves": 15},
            {"num_leaves": 31},
        ],
    )

    assert selected is None
    assert len(candidates) == 4
    assert split["selected_trainer_parameters"] == {"num_leaves": 31}
    assert split["trainer_parameter_candidate_count"] == 2
    assert {row[0] for row in trained} == {20}


def test_architecture_presets_are_bounded_and_validated() -> None:
    assert lightgbm_eval.parse_architecture_presets(
        "compact,balanced,compact"
    ) == ("compact", "balanced")
    with pytest.raises(
        argparse.ArgumentTypeError,
        match="unknown LightGBM architecture preset",
    ):
        lightgbm_eval.parse_architecture_presets("unbounded")


def test_multimetric_selection_protects_top5_within_loss_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset()

    def fake_trainer(
        _dataset: HashedRaceDataset,
        **kwargs: object,
    ) -> dict[str, object]:
        return {"leaves": int(kwargs["num_leaves"])}

    def fake_score(
        _dataset: HashedRaceDataset,
        *,
        bundle: dict[str, object],
        race_start: int,
        race_end: int,
        batch_size: int,
    ) -> tuple[dict[str, float | int], dict[str, list[dict[str, object]]]]:
        del batch_size
        interaction = bundle["leaves"] == 63
        return (
            {
                "entry_log_loss": 0.3205 if interaction else 0.3208,
                "entry_brier": 0.1,
                "trifecta_log_loss": 3.8,
                "winner_top1_accuracy": 0.56 if interaction else 0.57,
                "trifecta_top1_hit_rate": 0.1,
                "trifecta_top5_hit_rate": 0.318 if interaction else 0.322,
                "evaluated_races": race_end - race_start,
            },
            {"calibration": [{"leaves": bundle["leaves"]}]},
        )

    monkeypatch.setattr(
        "boatrace_ai.recency_mlp_evaluation.score_range",
        fake_score,
    )
    _, _, split = select_recency_half_life(
        dataset,
        outer_train_end=30,
        half_lives=(None,),
        calibration_days=10,
        bundle_trainer=fake_trainer,
        model_kind="lightgbm",
        trainer_parameter_candidates=[
            {"num_leaves": 31},
            {"num_leaves": 63},
        ],
        selection_entry_log_loss_tolerance=0.0005,
    )

    assert split["selected_trainer_parameters"] == {"num_leaves": 31}
    assert "trifecta_top5_hit_rate" in split["selection_criterion"]
    assert split["selection_entry_log_loss_tolerance"] == 0.0005


def test_lightgbm_trainer_rejects_wrong_model_kind() -> None:
    with pytest.raises(ValueError, match="requires model_kind"):
        lightgbm_eval.train_lightgbm_bundle_from_dataset(
            _dataset(2),
            train_race_count=1,
            model_kind="mlp",
            batch_size=12,
            epochs=1,
            alpha=0.0,
            recency_half_life_days=None,
            **_trainer_kwargs(),
        )


def test_lightgbm_cli_uses_separate_cache_and_safe_feature_drop() -> None:
    args = lightgbm_eval.build_parser().parse_args(
        [
            "--db",
            "fixture.sqlite",
            "--output",
            "result.json",
            "--evaluation-date",
            "2026-07-24",
        ]
    )

    assert args.feature_cache == lightgbm_eval.DEFAULT_FEATURE_CACHE
    assert args.write_feature_cache is True
    assert args.drop_feature_groups == ("legacy_composites",)
    assert args.half_lives == (None, 365.0)
    assert args.n_jobs == 4
    assert args.selection_entry_log_loss_tolerance == 0.0005


def test_lightgbm_wrapper_injects_model_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_evaluate(_conn: object, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"model": kwargs["model_name"]}

    monkeypatch.setattr(lightgbm_eval, "evaluate_recency_mlp", fake_evaluate)
    result = lightgbm_eval.evaluate_lightgbm_recency(
        None,
        output_path=lightgbm_eval.Path("result.json"),
        evaluation_date=date(2026, 7, 24),
        write_feature_cache=False,
        n_estimators=20,
    )

    assert result["model"] == lightgbm_eval.MODEL_NAME
    assert captured["model_kind"] == "lightgbm"
    assert captured["write_feature_cache"] is False
    assert captured["feature_set"] == lightgbm_eval.FEATURE_SET
    assert captured["feature_schema_version"] == (
        lightgbm_eval.LIGHTGBM_FEATURE_SCHEMA_VERSION
    )
    assert captured["bundle_trainer"] is (
        lightgbm_eval.train_lightgbm_bundle_from_dataset
    )
    assert captured["trainer_kwargs"] == {"n_estimators": 20}


def test_lightgbm_wrapper_forwards_progress_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_evaluate(_conn: object, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"status": "completed"}

    callback = lambda _row: None
    monkeypatch.setattr(lightgbm_eval, "evaluate_recency_mlp", fake_evaluate)

    lightgbm_eval.evaluate_lightgbm_recency(
        None,
        output_path=lightgbm_eval.Path("result.json"),
        evaluation_date=date(2026, 7, 24),
        progress_callback=callback,
    )

    assert captured["progress_callback"] is callback


def test_multimetric_wrapper_uses_versioned_model_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_evaluate(_conn: object, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"model": kwargs["model_name"]}

    monkeypatch.setattr(lightgbm_eval, "evaluate_recency_mlp", fake_evaluate)
    result = lightgbm_eval.evaluate_lightgbm_recency(
        None,
        output_path=lightgbm_eval.Path("result.json"),
        evaluation_date=date(2026, 7, 24),
        architecture_presets=("balanced",),
        selection_entry_log_loss_tolerance=0.0005,
    )

    assert result["model"] == lightgbm_eval.MULTIMETRIC_MODEL_NAME
