import math

import numpy as np
import pytest
from scipy import sparse

from boatrace_ai.hashed_feature_dataset import HashedRaceDataset
from boatrace_ai.listwise.conditional_stagewise import (
    conditional_loss_and_score_gradient,
    evaluate_conditional_stagewise_model,
    fit_conditional_stagewise_model,
)


def _tiny_dataset() -> HashedRaceDataset:
    race_keys = [
        (f"2026-01-0{index + 1}-01-01", f"2026-01-0{index + 1}", "01", 1)
        for index in range(4)
    ]
    rows = []
    ranks = []
    for winner in range(4):
        race_ranks = np.arange(1, 7)
        race_ranks[0], race_ranks[winner] = race_ranks[winner], race_ranks[0]
        ranks.append(race_ranks)
        for lane in range(6):
            rows.append([1.0, float(lane == winner), float(lane + 1) / 6.0])
    return HashedRaceDataset(
        matrix=sparse.csr_matrix(np.asarray(rows)),
        race_keys=race_keys,
        ranks=np.asarray(ranks, dtype=np.int8),
        n_features=3,
        drop_feature_groups=(),
    )


def test_uniform_conditional_loss_matches_candidate_counts() -> None:
    ranks = np.asarray([[1, 2, 3, 4, 5, 6]], dtype=np.int8)
    loss, gradient = conditional_loss_and_score_gradient(
        np.zeros((1, 6, 3)),
        ranks,
    )

    assert loss == pytest.approx((math.log(6) + math.log(5) + math.log(4)) / 3)
    assert gradient.shape == (1, 6, 3)
    assert np.allclose(gradient.sum(axis=1), 0.0)
    assert gradient[0, 0, 1] == 0.0
    assert gradient[0, 0, 2] == 0.0
    assert gradient[0, 1, 2] == 0.0


def test_correct_stage_scores_reduce_conditional_loss() -> None:
    ranks = np.asarray([[1, 2, 3, 4, 5, 6]], dtype=np.int8)
    scores = np.zeros((1, 6, 3))
    scores[0, 0, 0] = 5.0
    scores[0, 1, 1] = 5.0
    scores[0, 2, 2] = 5.0

    uniform, _ = conditional_loss_and_score_gradient(np.zeros_like(scores), ranks)
    informed, _ = conditional_loss_and_score_gradient(scores, ranks)

    assert informed < uniform


def test_fit_and_evaluate_conditional_model_on_tiny_dataset() -> None:
    dataset = _tiny_dataset()
    model, history = fit_conditional_stagewise_model(
        dataset,
        train_race_end=3,
        epochs=2,
        alpha=0.0001,
        learning_rate=0.02,
        batch_races=2,
    )
    metrics = evaluate_conditional_stagewise_model(
        dataset,
        model,
        race_start=3,
        race_end=4,
        batch_races=1,
    )

    assert model.weights.shape == (3, 3)
    assert len(history) == 2
    assert metrics["evaluated_races"] == 1
    assert math.isfinite(metrics["conditional_log_loss"])
    assert math.isfinite(metrics["trifecta_log_loss"])
    assert 0.0 <= metrics["trifecta_top5_hit_rate"] <= 1.0
