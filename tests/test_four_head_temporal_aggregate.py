from __future__ import annotations

import pytest

from boatrace_ai.listwise.four_head_temporal_aggregate import aggregate_four_head_folds


def _fold(day: str, *, stake: int, returned: int, largest: int, races: int) -> dict:
    return {
        "purchase_loss": "learned_test",
        "periods": {"outer_from": day, "outer_through": day},
        "purchase_value_diagnostics": {
            "positive_predicted_tickets": 10,
            "positive_observed_capped_roi": returned / stake,
        },
        "formal_bankroll": {
            "races": races,
            "tickets": 2,
            "hit_tickets": int(returned > 0),
            "stake_yen": stake,
            "return_yen": returned,
            "profit_yen": returned - stake,
            "roi": returned / stake,
            "winner_log_loss": 1.5,
            "winner_top1_accuracy": 0.6,
            "trifecta_log_loss": 4.4,
            "trifecta_top1_accuracy": 0.1,
            "trifecta_top5_hit_rate": 0.35,
            "closing_odds_log_mae": 0.3,
            "daily": [
                {
                    "race_date": day,
                    "evaluated_races": races,
                    "tickets": 2,
                    "hit_tickets": int(returned > 0),
                    "stake_yen": stake,
                    "return_yen": returned,
                    "profit_yen": returned - stake,
                    "max_drawdown_yen": max(stake - returned, 0),
                    "largest_hit_return_yen": largest,
                    "hit_return_square_sum_yen2": largest * largest,
                }
            ],
        },
    }


def test_recomputes_combined_bankroll_instead_of_averaging_fold_roi() -> None:
    result = aggregate_four_head_folds(
        [
            _fold("2026-07-01", stake=100, returned=300, largest=300, races=100),
            _fold("2026-07-02", stake=900, returned=0, largest=0, races=200),
        ],
        source_job_ids=[1, 2],
    )

    assert result["source_job_ids"] == [1, 2]
    assert result["evaluation_days"] == 2
    assert result["evaluated_races"] == 300
    assert result["roi"] == 0.3
    assert result["profit_yen"] == -700
    assert result["roi_without_largest_hit"] == 0.0
    assert result["effective_hit_count"] == 1.0
    assert result["purchase_value_positive_observed_capped_roi"] == 1.5
    assert result["promotion_eligible"] is False


def test_rejects_overlapping_daily_folds() -> None:
    fold = _fold("2026-07-01", stake=100, returned=0, largest=0, races=100)
    with pytest.raises(ValueError, match="overlap"):
        aggregate_four_head_folds([fold, fold])
