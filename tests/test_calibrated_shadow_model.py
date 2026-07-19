from unittest.mock import patch

import numpy as np
import pytest
from scipy import sparse

from boatrace_ai.calibrated_shadow_model import (
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
