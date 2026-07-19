from __future__ import annotations

import pytest

from boatrace_ai.listwise.live_shadow import evaluate_predictions


def test_evaluate_predictions_ignores_unfinished_races() -> None:
    rows = [
        {
            "actual_order": [1, 2, 3, 4, 5, 6],
            "winner_hit": True,
            "trifecta_top5_hit": True,
            "lane_probabilities": [
                {"lane": lane, "probability": probability}
                for lane, probability in enumerate(
                    [0.4, 0.2, 0.15, 0.1, 0.08, 0.07],
                    start=1,
                )
            ],
        },
        {
            "actual_order": None,
            "winner_hit": False,
            "trifecta_top5_hit": False,
            "lane_probabilities": [],
        },
    ]

    metrics = evaluate_predictions(rows)

    assert metrics["evaluated_races"] == 1
    assert metrics["winner_top1_accuracy"] == pytest.approx(1.0)
    assert metrics["trifecta_top5_hit_rate"] == pytest.approx(1.0)
    assert metrics["entry_log_loss"] is not None
    assert metrics["entry_brier"] is not None


def test_evaluate_predictions_returns_none_before_results() -> None:
    metrics = evaluate_predictions([])
    assert metrics == {
        "evaluated_races": 0,
        "entry_log_loss": None,
        "entry_brier": None,
        "winner_top1_accuracy": None,
        "trifecta_top5_hit_rate": None,
    }
