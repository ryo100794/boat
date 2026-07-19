from __future__ import annotations

import numpy as np
from scipy import sparse

from boatrace_ai.hashed_feature_dataset import HashedRaceDataset
from boatrace_ai.listwise.newton import (
    hessian_vector_product,
    objective_gradient,
    pl_hessian_score_product,
    refine_newton_cg,
)
from boatrace_ai.listwise.model import train_listwise_model


def dataset() -> HashedRaceDataset:
    rows = []
    ranks = []
    keys = []
    for race in range(36):
        winner = race % 6
        order = [(winner + offset) % 6 for offset in range(6)]
        race_ranks = [order.index(lane) + 1 for lane in range(6)]
        ranks.append(race_ranks)
        rows.extend([[7.0 - race_ranks[lane], float(lane + 1)] for lane in range(6)])
        keys.append((f"r{race}", f"2026-01-{race // 6 + 1:02d}", "01", race % 12 + 1))
    return HashedRaceDataset(
        sparse.csr_matrix(np.asarray(rows)),
        keys,
        np.asarray(ranks, dtype=np.int8),
        2,
        (),
    )


def test_score_hessian_product_matches_gradient_difference() -> None:
    scores = np.asarray([[0.2, -0.1, 0.4, 0.0, -0.3, 0.1]])
    ranks = np.asarray([[2, 4, 1, 3, 6, 5]])
    vector = np.asarray([[0.3, -0.2, 0.1, 0.5, -0.4, 0.2]])
    analytic = pl_hessian_score_product(scores, ranks, vector, target="top3_pl")
    from boatrace_ai.listwise.model import pl_loss_and_score_gradient

    epsilon = 1e-6
    _, plus = pl_loss_and_score_gradient(scores + epsilon * vector, ranks, target="top3_pl")
    _, minus = pl_loss_and_score_gradient(scores - epsilon * vector, ranks, target="top3_pl")
    assert np.allclose(analytic, (plus - minus) / (2 * epsilon), atol=1e-6)


def test_feature_hessian_product_matches_gradient_difference() -> None:
    data = dataset()
    model, _ = train_listwise_model(
        data, train_race_end=24, target="top3_pl", epochs=1, batch_races=6
    )
    vector = np.asarray([0.7, -0.2])
    analytic = hessian_vector_product(
        data,
        model,
        train_race_end=24,
        weights=model.weights,
        vector=vector,
        batch_races=6,
    )
    epsilon = 1e-6
    _, plus = objective_gradient(
        data,
        model,
        train_race_end=24,
        weights=model.weights + epsilon * vector,
        batch_races=6,
    )
    _, minus = objective_gradient(
        data,
        model,
        train_race_end=24,
        weights=model.weights - epsilon * vector,
        batch_races=6,
    )
    assert np.allclose(analytic, (plus - minus) / (2 * epsilon), atol=1e-5)


def test_newton_cg_reduces_regularized_objective() -> None:
    data = dataset()
    model, _ = train_listwise_model(
        data,
        train_race_end=24,
        target="top3_pl",
        alpha=1e-3,
        epochs=1,
        batch_races=6,
    )
    refined, report = refine_newton_cg(
        data,
        model,
        train_race_end=24,
        batch_races=6,
        max_newton_iterations=4,
        max_cg_iterations=10,
        gradient_tolerance=1e-5,
    )
    assert report["final_objective"] < report["initial_objective"]
    assert report["final_gradient_l2"] < report["history"][0]["gradient_l2"]
    assert not np.allclose(refined.weights, model.weights)
