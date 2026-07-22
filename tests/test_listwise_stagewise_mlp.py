from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from boatrace_ai.hashed_feature_dataset import HashedRaceDataset

from boatrace_ai.listwise.stagewise_mlp import (
    COMBINATION_INDEX,
    actual_combination_indices,
    cutoff_boundaries,
    evaluate_stagewise_model,
    fit_stagewise_model,
    rank_class_labels,
    stagewise_trifecta_probabilities,
)


def test_stagewise_probabilities_are_normalized_and_use_position_scores() -> None:
    scores = np.ones((6, 3), dtype=np.float64)
    scores[0, 0] = 20.0
    scores[1, 1] = 20.0
    scores[2, 2] = 20.0

    probabilities = stagewise_trifecta_probabilities(scores)

    assert probabilities.shape == (120,)
    assert probabilities.sum() == pytest.approx(1.0)
    assert np.all(probabilities > 0.0)
    assert int(np.argmax(probabilities)) == COMBINATION_INDEX[(1, 2, 3)]


def test_stagewise_batch_matches_single_race_conversion() -> None:
    first = np.arange(1, 19, dtype=np.float64).reshape(6, 3)
    second = np.flip(first, axis=0)

    batch = stagewise_trifecta_probabilities(np.stack((first, second)))

    assert batch.shape == (2, 120)
    assert np.allclose(batch[0], stagewise_trifecta_probabilities(first))
    assert np.allclose(batch.sum(axis=1), 1.0)


def test_rank_targets_and_actual_combination_follow_finish_order() -> None:
    ranks = np.asarray([[3, 1, 5, 2, 6, 4]], dtype=np.int8)

    assert rank_class_labels(ranks).tolist() == [3, 1, 0, 2, 0, 0]
    assert actual_combination_indices(ranks).tolist() == [
        COMBINATION_INDEX[(2, 4, 1)]
    ]


def test_stagewise_model_can_fit_and_score_cached_race_rows() -> None:
    race_count = 8
    ranks = np.asarray(
        [np.roll(np.arange(1, 7, dtype=np.int8), shift) for shift in range(race_count)]
    )
    matrix = sparse.csr_matrix(
        np.arange(race_count * 6 * 12, dtype=np.float64).reshape(race_count * 6, 12)
        % 17
    )
    dataset = HashedRaceDataset(
        matrix=matrix,
        race_keys=[
            (f"2026-01-{index + 1:02d}-01-01", f"2026-01-{index + 1:02d}", "01", 1)
            for index in range(race_count)
        ],
        ranks=ranks,
        n_features=12,
        drop_feature_groups=(),
    )

    model, history = fit_stagewise_model(
        dataset,
        train_race_end=6,
        epochs=1,
        alpha=0.0001,
        batch_rows=12,
        hidden_layer_sizes=(4,),
    )
    metrics = evaluate_stagewise_model(
        dataset,
        model,
        race_start=6,
        race_end=8,
        batch_races=1,
    )

    assert len(history) == 1
    assert metrics["evaluated_races"] == 2
    assert np.isfinite(metrics["trifecta_log_loss"])
    assert 0.0 <= metrics["trifecta_top5_hit_rate"] <= 1.0


def test_cutoff_boundaries_require_adjacent_full_days() -> None:
    race_keys = [
        ("2026-07-17-01-01", "2026-07-17", "01", 1),
        ("2026-07-18-01-01", "2026-07-18", "01", 1),
        ("2026-07-19-01-01", "2026-07-19", "01", 1),
    ]

    assert cutoff_boundaries(
        race_keys,
        training_through="2026-07-17",
        evaluation_from="2026-07-18",
        evaluation_through="2026-07-19",
    ) == (1, 1, 3)
    with pytest.raises(ValueError, match="adjacent"):
        cutoff_boundaries(
            race_keys,
            training_through="2026-07-17",
            evaluation_from="2026-07-19",
            evaluation_through="2026-07-19",
        )
