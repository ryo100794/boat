import pytest

from boatrace_ai.listwise.market_calibration import bankroll_reliability_metrics


def test_reliability_metrics_expose_purchase_funnel_and_fluke_dependence() -> None:
    metrics = bankroll_reliability_metrics(
        [
            {
                "tickets": 16,
                "hit_tickets": 1,
                "races_bet": 16,
                "hit_races": 1,
                "stake_yen": 1600,
                "return_yen": 1590,
                "largest_hit_return_yen": 1590,
                "hit_return_square_sum_yen2": 1590**2,
            },
            {
                "tickets": 18,
                "hit_tickets": 1,
                "races_bet": 16,
                "hit_races": 1,
                "stake_yen": 1800,
                "return_yen": 1490,
                "largest_hit_return_yen": 1490,
                "hit_return_square_sum_yen2": 1490**2,
            },
        ],
        evaluated_races=340,
    )

    assert metrics["selected_races"] == 32
    assert metrics["hit_races"] == 2
    assert metrics["race_selection_rate"] == pytest.approx(32 / 340)
    assert metrics["avg_tickets_per_selected_race"] == pytest.approx(34 / 32)
    assert metrics["ticket_hit_rate"] == pytest.approx(2 / 34)
    assert metrics["race_hit_rate"] == pytest.approx(2 / 32)
    assert metrics["ticket_hit_rate_ci95_lower"] < metrics["ticket_hit_rate"]
    assert metrics["ticket_hit_rate_ci95_upper"] > metrics["ticket_hit_rate"]
    assert metrics["largest_hit_return_share"] == pytest.approx(1590 / 3080)
    assert metrics["roi_without_largest_hit"] == pytest.approx(1490 / 3400)
    assert metrics["profit_without_largest_hit_yen"] == -1910
    assert 1.9 < metrics["effective_hit_count"] < 2.0
