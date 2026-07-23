from __future__ import annotations

import numpy as np
import pytest
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import StandardScaler

from boatrace_ai.listwise.conditional_stagewise import ConditionalStagewiseModel
from boatrace_ai.listwise.market_calibration import artifact_model_probabilities
from boatrace_ai.listwise.model import ListwiseLinearModel

from boatrace_ai.listwise.stagewise_blend import (
    StagewiseBlendModel,
    blend_probabilities,
    period_boundaries,
    select_weight,
    update_metrics,
)
from boatrace_ai.listwise.stagewise_mlp import (
    COMBINATION_INDEX,
    StagewiseMLPModel,
)


class _StaticRankClassifier:
    classes_ = np.asarray([0, 1, 2, 3])

    def predict_proba(self, matrix):
        return np.tile(np.asarray([0.5, 0.2, 0.2, 0.1]), (matrix.shape[0], 1))


def test_probability_blend_preserves_normalization_and_endpoints() -> None:
    listwise = np.full((2, 120), 1.0 / 120)
    stagewise = listwise.copy()
    stagewise[:, COMBINATION_INDEX[(1, 2, 3)]] += 0.1
    stagewise /= stagewise.sum(axis=1, keepdims=True)

    assert np.allclose(
        blend_probabilities(listwise, stagewise, stagewise_weight=0.0),
        listwise,
    )
    assert np.allclose(
        blend_probabilities(listwise, stagewise, stagewise_weight=1.0),
        stagewise,
    )
    assert np.allclose(
        blend_probabilities(listwise, stagewise, stagewise_weight=0.4).sum(axis=1),
        1.0,
    )
    with pytest.raises(ValueError, match="between zero and one"):
        blend_probabilities(listwise, stagewise, stagewise_weight=1.1)


def test_weight_selection_uses_loss_before_top5() -> None:
    results = {
        0.0: {"trifecta_log_loss": 4.1, "trifecta_top5_hit_rate": 0.35},
        0.5: {"trifecta_log_loss": 4.0, "trifecta_top5_hit_rate": 0.30},
        1.0: {"trifecta_log_loss": 4.2, "trifecta_top5_hit_rate": 0.40},
    }

    assert select_weight(results) == 0.5


def test_metric_update_scores_actual_order_and_first_marginal() -> None:
    probabilities = np.full((1, 120), 1e-6)
    probabilities[0, COMBINATION_INDEX[(2, 4, 1)]] = 1.0
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    ranks = np.asarray([[3, 1, 5, 2, 6, 4]], dtype=np.int8)
    accumulator = {
        "races": 0,
        "trifecta_loss": 0.0,
        "winner_hits": 0,
        "trifecta_top1_hits": 0,
        "trifecta_top5_hits": 0,
    }

    update_metrics(accumulator, probabilities=probabilities, ranks=ranks)

    assert accumulator["races"] == 1
    assert accumulator["winner_hits"] == 1
    assert accumulator["trifecta_top1_hits"] == 1
    assert accumulator["trifecta_top5_hits"] == 1


def test_market_scorer_accepts_stagewise_blend_artifact() -> None:
    hasher = FeatureHasher(
        n_features=16,
        input_type="dict",
        alternate_sign=False,
    )
    feature_rows = [
        {"features": {f"lane_{lane}": 1.0}}
        for lane in range(1, 7)
    ]
    matrix = hasher.transform([row["features"] for row in feature_rows])
    scaler = StandardScaler(with_mean=False).fit(matrix)
    listwise = ListwiseLinearModel(
        weights=np.zeros(16),
        scaler=scaler,
        target="top3_pl",
        alpha=0.0001,
        learning_rate=0.01,
        epochs=1,
    )
    stagewise = StagewiseMLPModel(
        scaler=scaler,
        classifier=_StaticRankClassifier(),
        epochs=1,
        alpha=0.0001,
    )
    artifact = {
        "hasher": hasher,
        "model": StagewiseBlendModel(listwise, stagewise, 0.5),
    }

    probabilities = artifact_model_probabilities(artifact, feature_rows)

    assert len(probabilities) == 120
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_market_scorer_accepts_conditional_stagewise_artifact() -> None:
    hasher = FeatureHasher(
        n_features=16,
        input_type="dict",
        alternate_sign=False,
    )
    feature_rows = [
        {"features": {f"lane_{lane}": 1.0}}
        for lane in range(1, 7)
    ]
    matrix = hasher.transform([row["features"] for row in feature_rows])
    scaler = StandardScaler(with_mean=False).fit(matrix)
    weights = np.zeros((3, 16))
    weights[0, :] = 0.1
    artifact = {
        "hasher": hasher,
        "model": ConditionalStagewiseModel(
            weights=weights,
            scaler=scaler,
            alpha=0.0001,
            learning_rate=0.01,
            epochs=3,
        ),
    }

    probabilities = artifact_model_probabilities(artifact, feature_rows)

    assert len(probabilities) == 120
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert all(value > 0.0 for value in probabilities.values())


def test_period_boundaries_select_full_dates() -> None:
    race_keys = [
        ("2026-07-17-01-01", "2026-07-17", "01", 1),
        ("2026-07-18-01-01", "2026-07-18", "01", 1),
        ("2026-07-19-01-01", "2026-07-19", "01", 1),
    ]

    assert period_boundaries(
        race_keys,
        date_from="2026-07-18",
        date_through="2026-07-19",
    ) == (1, 3)
