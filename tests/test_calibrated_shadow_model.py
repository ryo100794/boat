from unittest.mock import patch

import joblib
import numpy as np
import pytest
from scipy import sparse

from boatrace_ai.calibrated_shadow_model import (
    backtest_model,
    normalize_model_kind,
    predict_probabilities,
    score_dataset_fold,
    train_bundle,
    train_bundle_from_dataset,
)
from boatrace_ai.hashed_feature_dataset import HashedRaceDataset


def synthetic_entries(*_args, **_kwargs):
    for race in range(8):
        for lane in range(1, 7):
            yield (
                {
                    "lane": str(lane),
                    "lane_num": lane,
                    "ability": 7.0 - lane + race * 0.01,
                    "origin": "A" if lane % 2 else "B",
                },
                1 if lane == 1 else 0,
                {"race_id": f"r{race}", "lane": lane},
            )


@pytest.mark.parametrize("model_kind", ["linear", "mlp"])
def test_calibrated_shadow_trains_in_batches(model_kind) -> None:
    with patch(
        "boatrace_ai.calibrated_shadow_model.iter_training_entries",
        side_effect=synthetic_entries,
    ):
        bundle = train_bundle(
            None,
            include_races={f"r{race}" for race in range(8)},
            model_kind=model_kind,
            n_features=256,
            batch_size=12,
            epochs=1,
        )

    probabilities = predict_probabilities(
        bundle,
        [
            {"lane": "1", "lane_num": 1, "ability": 6.0, "origin": "A"},
            {"lane": "6", "lane_num": 6, "ability": 1.0, "origin": "B"},
        ],
    )

    assert bundle["model_kind"] == model_kind
    assert len(probabilities) == 2
    assert all(0.0 <= value <= 1.0 for value in probabilities)


def test_unknown_model_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown model kind"):
        normalize_model_kind("tree")


def test_backtest_artifact_requires_single_outer_fold(tmp_path) -> None:
    with pytest.raises(ValueError, match="single outer fold"):
        backtest_model(
            None,
            output_path=tmp_path / "result.json",
            model_output_path=tmp_path / "model.joblib",
            model_kind="mlp",
            folds=2,
        )


def test_backtest_persists_exact_training_artifact(tmp_path, monkeypatch) -> None:
    race_keys = [
        ("r0", "2026-01-01", "01", 1),
        ("r1", "2026-01-02", "01", 1),
        ("r2", "2026-01-03", "01", 1),
    ]
    matrix = np.zeros((18, 256), dtype=np.float64)
    for race in range(3):
        for lane in range(6):
            matrix[race * 6 + lane, lane] = 1.0
    dataset = HashedRaceDataset(
        matrix=sparse.csr_matrix(matrix),
        race_keys=race_keys,
        ranks=np.asarray([[1, 2, 3, 4, 5, 6]] * 3, dtype=np.int8),
        n_features=256,
        drop_feature_groups=(),
    )
    monkeypatch.setattr(
        "boatrace_ai.calibrated_shadow_model.load_complete_race_ids",
        lambda _conn: race_keys,
    )
    monkeypatch.setattr(
        "boatrace_ai.calibrated_shadow_model.load_or_build_hashed_dataset",
        lambda **_kwargs: (dataset, "test-cache"),
    )
    monkeypatch.setattr(
        "boatrace_ai.calibrated_shadow_model._load_trifecta_payouts",
        lambda _conn: {},
    )
    monkeypatch.setattr(
        "boatrace_ai.listwise.validation.evaluate_bankroll_fold",
        lambda **_kwargs: (
            {"stake_yen": 0, "return_yen": 0, "profit_yen": 0, "roi": 0.0},
            (0, 0, 0),
        ),
    )
    output = tmp_path / "result.json"
    model_output = tmp_path / "model.joblib"

    result = backtest_model(
        None,
        output_path=output,
        model_output_path=model_output,
        model_kind="linear",
        folds=1,
        min_train_races=2,
        n_features=256,
        epochs=1,
        feature_cache=tmp_path / "features",
    )

    artifact = joblib.load(model_output)
    assert result["model_artifact_saved"] is True
    assert artifact["hasher"].n_features == 256
    assert artifact["training_races"] == 2
    assert artifact["trained_through"] == race_keys[1]
    assert artifact["feature_schema_version"] == dataset.feature_schema_version
    assert artifact["metadata"]["evaluation_races"] == 1
    assert artifact["metadata"]["evaluation_race_set_sha256"] == result[
        "evaluation_race_set_sha256"
    ]


@pytest.mark.parametrize("model_kind", ["linear", "mlp"])
def test_calibrated_shadow_trains_and_scores_cached_matrix(model_kind) -> None:
    rows = []
    ranks = []
    race_keys = []
    for race in range(8):
        race_keys.append((f"r{race}", f"2026-01-{race + 1:02d}", "01", 1))
        ranks.append([1, 2, 3, 4, 5, 6])
        for lane in range(1, 7):
            rows.append([float(lane == 1), float(lane), float(race) / 10.0])
    dataset = HashedRaceDataset(
        matrix=sparse.csr_matrix(np.asarray(rows)),
        race_keys=race_keys,
        ranks=np.asarray(ranks, dtype=np.int8),
        n_features=3,
        drop_feature_groups=(),
    )
    bundle = train_bundle_from_dataset(
        dataset,
        train_race_count=6,
        model_kind=model_kind,
        batch_size=12,
        epochs=1,
    )
    scored = list(
        score_dataset_fold(
            dataset,
            bundle=bundle,
            race_start=6,
            race_end=8,
            batch_size=6,
        )
    )

    assert bundle["matrix_cached"] is True
    assert len(scored) == 2
    assert all(len(race) == 6 for race in scored)
    assert all(
        sum(row["probability"] for row in race) == pytest.approx(1.0)
        for race in scored
    )
