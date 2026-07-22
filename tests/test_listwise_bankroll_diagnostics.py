from __future__ import annotations

import pytest

from boatrace_ai.listwise.bankroll_diagnostics import (
    sequential_top5_ev_kelly_diagnostic,
)


def _winning_race(race_date: str, rno: int) -> dict:
    minute = (rno - 1) * 20
    return {
        "race_id": f"{race_date}-01-{rno:02d}",
        "race_date": race_date,
        "jcd": "01",
        "rno": rno,
        "odds_deadline_at": f"{race_date}T12:{minute:02d}:00+09:00",
        "actual_combination": "1-2-3",
        "actual_payout_yen": 500,
        "model_probabilities": {
            "1-2-3": 0.35,
            "1-2-4": 0.20,
            "1-3-2": 0.15,
            "2-1-3": 0.10,
            "2-3-1": 0.08,
            "6-5-4": 0.01,
        },
        "odds": {
            "1-2-3": 4.0,
            "1-2-4": 2.0,
            "1-3-2": 2.0,
            "2-1-3": 2.0,
            "2-3-1": 2.0,
            "6-5-4": 50.0,
        },
    }


def test_sequential_kelly_reinvests_profit_with_daily_reset() -> None:
    races = [
        _winning_race("2026-07-20", 1),
        _winning_race("2026-07-20", 2),
        _winning_race("2026-07-21", 1),
    ]
    result = sequential_top5_ev_kelly_diagnostic(
        races,
        daily_budget_yen=10_000,
    )
    assert result["profit_reinvestment"] is True
    assert result["settlement_delay_minutes"] == 10
    assert result["evaluation_days"] == 2
    assert result["stake_yen"] == 600
    assert result["return_yen"] == 3_000
    assert result["profit_yen"] == 2_400
    assert result["roi"] == pytest.approx(5.0)
    assert result["units"] == 6
    assert result["bets"] == 3
    assert result["daily"][0]["opening_balance_yen"] == 10_000
    assert result["daily"][0]["closing_balance_yen"] == 11_600
    assert result["daily"][1]["opening_balance_yen"] == 10_000
    assert result["daily"][1]["closing_balance_yen"] == 10_800


def test_sequential_kelly_does_not_reuse_unsettled_payout() -> None:
    first = _winning_race("2026-07-20", 1)
    second = _winning_race("2026-07-20", 2)
    second["odds_deadline_at"] = "2026-07-20T12:05:00+09:00"
    result = sequential_top5_ev_kelly_diagnostic(
        [first, second],
        daily_budget_yen=10_000,
    )
    assert result["stake_yen"] == 300
    assert result["return_yen"] == 1_500
    assert result["profit_yen"] == 1_200
    assert result["units"] == 3
    assert result["daily"][0]["closing_balance_yen"] == 11_200


def test_sequential_kelly_rejects_unfunded_budget() -> None:
    with pytest.raises(ValueError, match="at least one"):
        sequential_top5_ev_kelly_diagnostic([], daily_budget_yen=99)
