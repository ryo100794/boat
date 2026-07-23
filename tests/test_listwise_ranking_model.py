from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import sparse

from boatrace_ai.hashed_feature_dataset import HashedRaceDataset
from boatrace_ai.listwise.model import (
    evaluate_range,
    pl_loss_and_score_gradient,
    stable_softmax,
    train_listwise_model,
)
from boatrace_ai.listwise.validation import full_day_fold_boundaries, nested_select_candidate


def reference_pl_loss_and_score_gradient(
    scores: np.ndarray,
    ranks: np.ndarray,
    *,
    target: str,
) -> tuple[float, np.ndarray]:
    values = np.asarray(scores, dtype=np.float64)
    rank_values = np.asarray(ranks)
    gradient = np.zeros_like(values)
    total_loss = 0.0
    stages = 1 if target == "winner" else 3
    for race_index in range(values.shape[0]):
        order = np.argsort(rank_values[race_index])
        remaining = np.ones(6, dtype=bool)
        for stage in range(stages):
            actual = int(order[stage])
            lane_indices = np.flatnonzero(remaining)
            probabilities = stable_softmax(values[race_index, lane_indices])
            actual_position = int(np.flatnonzero(lane_indices == actual)[0])
            total_loss -= math.log(max(1e-15, float(probabilities[actual_position])))
            gradient[race_index, lane_indices] += probabilities
            gradient[race_index, actual] -= 1.0
            remaining[actual] = False
    denominator = max(1, values.shape[0] * stages)
    return total_loss / denominator, gradient / denominator


def assert_matches_reference(
    scores: np.ndarray,
    ranks: np.ndarray,
    target: str,
) -> None:
    expected_loss, expected_gradient = reference_pl_loss_and_score_gradient(
        scores, ranks, target=target
    )
    loss, gradient = pl_loss_and_score_gradient(scores, ranks, target=target)
    np.testing.assert_allclose(loss, expected_loss, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-14, atol=1e-14)


def synthetic_dataset(races: int = 90) -> HashedRaceDataset:
    matrix_rows = []
    ranks = []
    keys = []
    for race in range(races):
        winner = race % 6
        order = [(winner + offset) % 6 for offset in range(6)]
        rank_by_lane = [order.index(lane) + 1 for lane in range(6)]
        ranks.append(rank_by_lane)
        for lane in range(6):
            matrix_rows.append([7.0 - rank_by_lane[lane], float(lane == 0), float(lane + 1)])
        day = race // 6 + 1
        keys.append((f"r{race:04d}", f"2026-01-{day:02d}", "01", race % 12 + 1))
    return HashedRaceDataset(
        matrix=sparse.csr_matrix(np.asarray(matrix_rows, dtype=np.float64)),
        race_keys=keys,
        ranks=np.asarray(ranks, dtype=np.int8),
        n_features=3,
        drop_feature_groups=(),
    )


def test_softmax_is_stable_and_sums_to_one() -> None:
    probabilities = stable_softmax(np.asarray([[1_000.0, 999.0, -1_000.0]]))
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert probabilities[0, 0] > probabilities[0, 1]


@pytest.mark.parametrize("target", ("winner", "top3_pl"))
def test_pl_loss_and_gradient_match_reference_for_random_batch(target: str) -> None:
    random = np.random.default_rng(20260723)
    scores = random.normal(loc=0.0, scale=4.0, size=(257, 6))
    ranks = np.asarray(
        [random.permutation(np.arange(1, 7)) for _ in range(scores.shape[0])]
    )
    assert_matches_reference(scores, ranks, target)


@pytest.mark.parametrize("target", ("winner", "top3_pl"))
def test_pl_loss_and_gradient_match_reference_for_extreme_scores(target: str) -> None:
    scores = np.asarray(
        [
            [1_000.0, -1_000.0, 999.0, -999.0, 0.0, 500.0],
            [-1_000.0, 1_000.0, -999.0, 999.0, -500.0, 0.0],
            [700.0, 700.0, -700.0, -700.0, 1e-12, -1e-12],
        ]
    )
    ranks = np.asarray(
        [[6, 1, 5, 2, 4, 3], [1, 6, 2, 5, 3, 4], [3, 2, 6, 5, 1, 4]]
    )
    assert_matches_reference(scores, ranks, target)


@pytest.mark.parametrize("target", ("winner", "top3_pl"))
def test_pl_loss_and_gradient_match_reference_for_single_race_batch(target: str) -> None:
    scores = np.asarray([[0.4, -0.1, 0.2, 0.0, -0.3, 0.1]])
    ranks = np.asarray([[2, 4, 1, 3, 6, 5]], dtype=np.int8)
    assert_matches_reference(scores, ranks, target)


@pytest.mark.parametrize(
    ("scores", "ranks"),
    (
        (np.zeros(6), np.zeros(6)),
        (np.zeros((2, 5)), np.zeros((2, 5))),
        (np.zeros((2, 6)), np.zeros((1, 6))),
    ),
)
def test_pl_loss_and_gradient_reject_invalid_shapes(
    scores: np.ndarray, ranks: np.ndarray
) -> None:
    with pytest.raises(ValueError, match="shape \\(races, 6\\)"):
        pl_loss_and_score_gradient(scores, ranks, target="winner")


def test_pl_loss_and_gradient_reject_unknown_target() -> None:
    with pytest.raises(ValueError, match="unknown target: invalid"):
        pl_loss_and_score_gradient(np.zeros((1, 6)), np.ones((1, 6)), target="invalid")


def test_pl_gradient_matches_finite_difference() -> None:
    scores = np.asarray([[0.4, -0.1, 0.2, 0.0, -0.3, 0.1]], dtype=np.float64)
    ranks = np.asarray([[2, 4, 1, 3, 6, 5]], dtype=np.int8)
    _, gradient = pl_loss_and_score_gradient(scores, ranks, target="top3_pl")
    epsilon = 1e-6
    numerical = np.zeros(6)
    for lane in range(6):
        plus = scores.copy()
        minus = scores.copy()
        plus[0, lane] += epsilon
        minus[0, lane] -= epsilon
        plus_loss, _ = pl_loss_and_score_gradient(plus, ranks, target="top3_pl")
        minus_loss, _ = pl_loss_and_score_gradient(minus, ranks, target="top3_pl")
        numerical[lane] = (plus_loss - minus_loss) / (2 * epsilon)
    assert np.allclose(gradient[0], numerical, atol=1e-6)


def test_listwise_training_learns_race_relative_order() -> None:
    dataset = synthetic_dataset()
    model, history = train_listwise_model(
        dataset,
        train_race_end=60,
        target="top3_pl",
        alpha=1e-5,
        learning_rate=0.03,
        epochs=5,
        batch_races=12,
    )
    metrics, _ = evaluate_range(
        dataset,
        model,
        race_start=60,
        race_end=90,
        batch_races=12,
    )
    assert history[-1]["training_ranking_log_loss"] < history[0]["training_ranking_log_loss"]
    assert metrics["winner_top1_accuracy"] > 0.95
    assert metrics["trifecta_top5_hit_rate"] > 0.95


def test_nested_selection_and_outer_folds_are_time_scoped() -> None:
    dataset = synthetic_dataset()
    selected, candidates = nested_select_candidate(
        dataset,
        outer_train_end=60,
        targets=("winner", "top3_pl"),
        alphas=(1e-5,),
        learning_rate=0.03,
        epochs=2,
        batch_races=12,
        validation_fraction=0.2,
        min_validation_races=6,
    )
    assert len(candidates) == 2
    assert selected["inner_train_races"] == 48
    assert selected["validation_races"] == 12

    boundaries = full_day_fold_boundaries(dataset.race_keys, folds=3, min_train_races=18)
    assert boundaries
    for train_end, test_end, test_dates in boundaries:
        assert train_end < test_end
        assert dataset.race_keys[train_end - 1][1] not in test_dates
        assert dataset.race_keys[train_end][1] in test_dates
