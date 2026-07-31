from __future__ import annotations

from datetime import datetime, timedelta, timezone

from boatrace_ai.runtime.intraday_t300_shadow import (
    ModelIdentity,
    RaceWindow,
    V18ScheduleQuotaModelAdapter,
)


JST = timezone(timedelta(hours=9))


class Result:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows


class Connection:
    def __init__(self):
        self.calls = 0

    def execute(self, query, parameters):
        self.calls += 1
        if self.calls == 1:
            return Result([
                {"race_id": "race-1", "deadline_at": "2026-07-30T09:00:00+09:00"},
                {"race_id": "race-2", "deadline_at": "2026-07-30T09:10:00+09:00"},
                {"race_id": "race-3", "deadline_at": "2026-07-30T09:20:00+09:00"},
                {"race_id": "race-4", "deadline_at": "2026-07-30T09:30:00+09:00"},
            ])
        return Result([
            {
                "selected_candidates": [{"combination": "1-2-3"}, {"combination": "1-3-2"}],
                "total_stake_yen": 300,
                "profit_yen": 500,
            },
            {
                "selected_candidates": '[{"combination":"2-1-3"}]',
                "total_stake_yen": 100,
                "profit_yen": -200,
            },
            {
                "selected_candidates": [],
                "total_stake_yen": 0,
                "profit_yen": None,
            },
        ])


def test_runtime_uses_artifact_selected_ceil_rounding() -> None:
    model = object.__new__(V18ScheduleQuotaModelAdapter)
    model._ticket_limit = 3
    model._quota_rounding = "ceil"
    model._identity = ModelIdentity("v18_daily", "a" * 64, model.strategy_name)
    race = RaceWindow(
        "race-2",
        "2026-07-30",
        "01",
        2,
        datetime(2026, 7, 30, 9, 10, tzinfo=JST),
    )

    limits = model._runtime_limits(Connection(), race, bankroll_yen=9_700)

    assert limits["cumulative_ticket_quota"] == 2
    assert limits["remaining_ticket_quota"] == 0


def test_v18_runtime_limits_apply_schedule_and_realized_net_profit_only() -> None:
    model = object.__new__(V18ScheduleQuotaModelAdapter)
    model._ticket_limit = 26
    model._identity = ModelIdentity("v18_daily", "a" * 64, model.strategy_name)
    race = RaceWindow(
        "race-2",
        "2026-07-30",
        "01",
        2,
        datetime(2026, 7, 30, 9, 10, tzinfo=JST),
    )
    limits = model._runtime_limits(Connection(), race, bankroll_yen=9_700)
    assert limits == {
        "schedule_races_elapsed": 2,
        "schedule_races_total": 4,
        "cumulative_ticket_quota": 13,
        "used_tickets": 3,
        "remaining_ticket_quota": 10,
        "gross_stake_yen": 400,
        "realized_cumulative_profit_yen": 300,
        "gross_stake_allowance_yen": 10_300,
        "remaining_gross_stake_allowance_yen": 9_900,
        "allocatable_bankroll_yen": 9_700,
    }
