from __future__ import annotations

import pytest

from boatrace_ai.listwise.tail_portfolio_diagnostics import (
    diagnose_tail_portfolio,
)


def test_empty_input_is_distinct_from_rows_without_purchases() -> None:
    empty = diagnose_tail_portfolio([], bootstrap_samples=100)
    no_purchases = diagnose_tail_portfolio(
        [
            {
                "date": "2026-07-01",
                "race_id": "01-01",
                "odds": 150.0,
                "stake": 0,
                "return": 0,
            }
        ],
        bootstrap_samples=100,
    )

    assert empty["status"] == "empty"
    assert empty["input_rows"] == 0
    assert no_purchases["status"] == "no_purchases"
    assert no_purchases["input_rows"] == 1
    assert no_purchases["purchased_tickets"] == 0
    assert no_purchases["tail"]["status"] == "no_purchases"
    assert no_purchases["tail"]["roi"] is None
    assert no_purchases["tail"]["daily_cluster_bootstrap_roi_lower_95"] is None


def test_reports_ordinary_and_tail_metrics_at_101_odds_boundary() -> None:
    result = diagnose_tail_portfolio(
        [
            {
                "date": "2026-07-01",
                "race_id": "01-01",
                "odds": 100.9,
                "stake": 100,
                "return": 0,
            },
            {
                "date": "2026-07-02",
                "race_id": "02-01",
                "odds": 5.0,
                "stake": 100,
                "return": 300,
            },
            {
                "date": "2026-07-01",
                "race_id": "01-02",
                "odds": 101.0,
                "stake": 100,
                "return": 10_100,
            },
            {
                "date": "2026-07-01",
                "race_id": "01-03",
                "odds": 180.0,
                "stake": 100,
                "return": 0,
            },
            {
                "date": "2026-07-02",
                "race_id": "02-02",
                "odds": 250.0,
                "stake": 100,
                "return": 0,
            },
        ],
        bootstrap_samples=1_000,
        seed=19,
    )

    normal = result["normal"]
    assert normal["tickets"] == 2
    assert normal["hits"] == 1
    assert normal["hit_days"] == 1
    assert normal["stake"] == 200
    assert normal["return"] == 300
    assert normal["profit"] == 100
    assert normal["roi"] == pytest.approx(1.5)
    assert normal["roi_excluding_largest_hit"] == 0.0

    tail = result["tail"]
    assert tail["tickets"] == 3
    assert tail["hits"] == 1
    assert tail["hit_days"] == 1
    assert tail["stake"] == 300
    assert tail["return"] == 10_100
    assert tail["profit"] == 9_800
    assert tail["roi"] == pytest.approx(10_100 / 300)
    assert tail["roi_excluding_largest_hit"] == 0.0
    assert tail["largest_hit_return"] == 10_100
    assert tail["largest_hit_race_id"] == "01-02"


def test_daily_cluster_bootstrap_is_reproducible_and_uses_active_days() -> None:
    rows = [
        {
            "date": "2026-07-01",
            "race_id": "01-01",
            "odds": 10.0,
            "stake": 100,
            "return": 200,
        },
        {
            "date": "2026-07-02",
            "race_id": "02-01",
            "odds": 12.0,
            "stake": 300,
            "return": 0,
        },
        {
            "date": "2026-07-03",
            "race_id": "03-01",
            "odds": 20.0,
            "stake": 0,
            "return": 0,
        },
        {
            "date": "2026-07-01",
            "race_id": "01-02",
            "odds": 101.0,
            "stake": 100,
            "return": 0,
        },
    ]

    first = diagnose_tail_portfolio(rows, bootstrap_samples=2_000, seed=73)
    second = diagnose_tail_portfolio(rows, bootstrap_samples=2_000, seed=73)

    assert first == second
    assert first["normal"]["daily_cluster_bootstrap_roi_lower_95"] == 0.0
    assert first["tail"]["daily_cluster_bootstrap_roi_lower_95"] == 0.0


def test_no_hit_keeps_roi_when_there_is_no_largest_hit_to_remove() -> None:
    result = diagnose_tail_portfolio(
        [
            {
                "date": "2026-07-01",
                "race_id": "01-01",
                "odds": 150.0,
                "stake": 100,
                "return": 0,
            }
        ],
        bootstrap_samples=100,
    )

    assert result["tail"]["roi"] == 0.0
    assert result["tail"]["roi_excluding_largest_hit"] == 0.0
    assert result["tail"]["largest_hit_return"] is None


@pytest.mark.parametrize(
    ("field", "value"),
    [("odds", 0), ("stake", -100), ("return", -1)],
)
def test_rejects_invalid_ticket_values(field: str, value: float) -> None:
    row = {
        "date": "2026-07-01",
        "race_id": "01-01",
        "odds": 10.0,
        "stake": 100,
        "return": 0,
    }
    row[field] = value

    with pytest.raises(ValueError):
        diagnose_tail_portfolio([row], bootstrap_samples=100)


def test_rejects_return_for_an_unpurchased_ticket() -> None:
    with pytest.raises(ValueError, match="stake is zero"):
        diagnose_tail_portfolio(
            [
                {
                    "date": "2026-07-01",
                    "race_id": "01-01",
                    "odds": 101.0,
                    "stake": 0,
                    "return": 10_100,
                }
            ],
            bootstrap_samples=100,
        )
