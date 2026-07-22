from __future__ import annotations

import numpy as np
import pytest

from boatrace_ai.listwise.stagewise_blend import (
    blend_probabilities,
    period_boundaries,
    select_weight,
    update_metrics,
)
from boatrace_ai.listwise.stagewise_mlp import COMBINATION_INDEX


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
