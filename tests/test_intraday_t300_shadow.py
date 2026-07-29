from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from boatrace_ai.runtime.intraday_t300_shadow import (
    DEFAULT_MAX_CHECKPOINT_AGE_SECONDS,
    DEFAULT_MAX_SOURCE_UPDATE_STALENESS_SECONDS,
    ModelIdentity,
    RaceWindow,
    ShadowDecision,
    SnapshotCheck,
    T300Snapshot,
    build_parser,
    resolve_race_date,
    run_cycle,
)


JST = timezone(timedelta(hours=9))
COMBINATIONS = [
    f"{first}-{second}-{third}"
    for first in range(1, 7)
    for second in range(1, 7)
    if second != first
    for third in range(1, 7)
    if third not in (first, second)
]


def race(day: str, *, suffix: str = "01", hour: int = 12) -> RaceWindow:
    return RaceWindow(
        f"{day.replace('-', '')}-01-{suffix}", day, "01", int(suffix),
        datetime.fromisoformat(f"{day}T{hour:02d}:00:00+09:00"),
    )


def snapshot(target: datetime, *, age: float = 30.0, source_stale: float = 20.0,
             count: int = 120) -> T300Snapshot:
    captured = target - timedelta(seconds=age)
    return T300Snapshot(
        101, captured, (captured - timedelta(seconds=source_stale)).isoformat(), {},
        {combination: 5.0 + index / 10 for index, combination in enumerate(COMBINATIONS[:count])},
    )


class Adapter:
    strategy_name = "test_v12_role"

    def __init__(self, key: str, digest: str = "a" * 64, *, selected: bool = True) -> None:
        self._identity = ModelIdentity(key, digest, self.strategy_name)
        self.selected = selected
        self.calls = 0

    @property
    def identity(self) -> ModelIdentity:
        return self._identity

    def decide(self, conn: Any, row: RaceWindow, point: T300Snapshot,
               *, bankroll_yen: int) -> ShadowDecision:
        self.calls += 1
        probabilities = {key: 1.0 / 120.0 for key in COMBINATIONS}
        closing = dict(point.odds)
        candidates = (
            ({"race_id": row.race_id, "race_date": row.race_date,
              "combination": "1-2-3", "probability": 0.08,
              "estimated_odds": 15.0, "estimated_ev": 1.2,
              "stake_yen": 100},)
            if self.selected else ()
        )
        return ShadowDecision(
            probabilities, closing, candidates,
            None if candidates else "no_positive_discrete_log_growth",
        )


class MemoryStore:
    conn = object()

    def __init__(self, races: list[RaceWindow], snapshots: dict[str, T300Snapshot]) -> None:
        self.races = races
        self.snapshots = snapshots
        self.decisions: dict[tuple[str, str], dict[str, Any]] = {}
        self.results: dict[str, tuple[str, int]] = {}
        self.settlements: dict[tuple[str, str], dict[str, Any]] = {}
        self.schema_calls = 0

    def ensure_schema(self) -> None:
        self.schema_calls += 1

    def due_races(self, *, race_date: str, now: datetime) -> list[RaceWindow]:
        return [row for row in self.races
                if row.race_date == race_date and row.target_t300_at <= now.astimezone(JST)]

    def decision_identity(self, *, race_id: str, model_key: str) -> ModelIdentity | None:
        row = self.decisions.get((race_id, model_key))
        return row["identity"] if row else None

    def latest_complete_snapshot(self, row: RaceWindow) -> T300Snapshot | None:
        return self.snapshots.get(row.race_id)

    def bankroll_yen(self, *, race_date: str, model_key: str, starting_yen: int) -> int:
        stakes = sum(value["decision"].total_stake_yen for value in self.decisions.values()
                     if value["race"].race_date == race_date
                     and value["identity"].model_key == model_key)
        returns = sum(value["return_yen"] for key, value in self.settlements.items()
                      if self.decisions[key]["race"].race_date == race_date
                      and key[1] == model_key)
        return starting_yen - stakes + returns

    def insert_decision(self, **values: Any) -> bool:
        key = (values["race"].race_id, values["identity"].model_key)
        if key in self.decisions:
            if self.decisions[key]["identity"] != values["identity"]:
                raise ValueError("model identity conflict")
            return False
        self.decisions[key] = copy.deepcopy(values)
        return True

    def append_available_settlements(self, *, race_date: str, now: datetime) -> int:
        inserted = 0
        for key, row in self.decisions.items():
            if row["race"].race_date != race_date or key in self.settlements:
                continue
            result = self.results.get(row["race"].race_id)
            if result is None:
                continue
            actual, payout = result
            returned = sum(
                int(item["stake_yen"]) * payout // 100
                for item in row["decision"].selected_candidates
                if item["combination"] == actual
            )
            stake = row["decision"].total_stake_yen
            self.settlements[key] = {
                "settled_at": now, "actual": actual, "payout": payout,
                "stake_yen": stake, "return_yen": returned,
                "profit_yen": returned - stake,
            }
            inserted += 1
        return inserted


def cycle_time(row: RaceWindow) -> datetime:
    return row.target_t300_at + timedelta(seconds=1)


def test_restart_is_idempotent_and_decision_is_immutable() -> None:
    row = race("2026-07-30")
    store = MemoryStore([row], {row.race_id: snapshot(row.target_t300_at)})
    adapter = Adapter("v12-jul30")
    first = run_cycle(store, [adapter], now=cycle_time(row))
    original = copy.deepcopy(store.decisions[(row.race_id, "v12-jul30")])
    second = run_cycle(store, [adapter], now=cycle_time(row) + timedelta(seconds=30))

    assert first["decisions_inserted"] == 1
    assert second["decisions_inserted"] == 0
    assert second["existing_decisions"] == 1
    assert adapter.calls == 1
    assert store.decisions[(row.race_id, "v12-jul30")] == original


def test_automatic_day_rollover_uses_jst_and_does_not_pin_jul30() -> None:
    first = race("2026-07-30")
    second = race("2026-07-31")
    store = MemoryStore(
        [first, second],
        {first.race_id: snapshot(first.target_t300_at),
         second.race_id: snapshot(second.target_t300_at)},
    )
    adapter = Adapter("v12")
    run_cycle(store, [adapter], now=cycle_time(first))
    run_cycle(store, [adapter], now=cycle_time(second))

    assert resolve_race_date(cycle_time(second), None) == "2026-07-31"
    assert {key[0] for key in store.decisions} == {first.race_id, second.race_id}


def test_settlement_is_appended_later_without_changing_decision() -> None:
    row = race("2026-07-30")
    store = MemoryStore([row], {row.race_id: snapshot(row.target_t300_at)})
    adapter = Adapter("v12")
    now = cycle_time(row)
    run_cycle(store, [adapter], now=now)
    original = copy.deepcopy(store.decisions[(row.race_id, "v12")])
    assert store.settlements == {}

    store.results[row.race_id] = ("1-2-3", 1_500)
    result = run_cycle(store, [adapter], now=now + timedelta(minutes=20))

    assert result["settlements_inserted"] == 1
    assert store.settlements[(row.race_id, "v12")]["profit_yen"] == 1_400
    assert store.decisions[(row.race_id, "v12")] == original


@pytest.mark.parametrize(
    ("age", "source_stale", "reason"),
    [(91.0, 20.0, "stale_t300_checkpoint"),
     (30.0, 121.0, "stale_t300_source_update")],
)
def test_stale_snapshot_is_recorded_as_no_bet_with_actual_ages(
    age: float, source_stale: float, reason: str,
) -> None:
    row = race("2026-07-30")
    store = MemoryStore(
        [row], {row.race_id: snapshot(row.target_t300_at, age=age, source_stale=source_stale)}
    )
    adapter = Adapter("v12")
    run_cycle(store, [adapter], now=cycle_time(row))
    saved = store.decisions[(row.race_id, "v12")]

    assert saved["decision"].no_bet_reason == reason
    assert saved["decision"].total_stake_yen == 0
    assert saved["snapshot_check"].checkpoint_age_before_target_seconds == age
    assert saved["snapshot_check"].source_update_staleness_seconds == source_stale
    assert adapter.calls == 0


def test_exactly_120_combinations_are_required() -> None:
    row = race("2026-07-30")
    store = MemoryStore([row], {row.race_id: snapshot(row.target_t300_at, count=119)})
    adapter = Adapter("v12")
    run_cycle(store, [adapter], now=cycle_time(row))
    saved = store.decisions[(row.race_id, "v12")]
    assert saved["decision"].no_bet_reason == "incomplete_t300_snapshot"
    assert adapter.calls == 0


def test_model_keys_coexist_and_hash_change_under_same_key_fails() -> None:
    row = race("2026-07-30")
    store = MemoryStore([row], {row.race_id: snapshot(row.target_t300_at)})
    run_cycle(store, [Adapter("v12-a"), Adapter("v12-b", "b" * 64)], now=cycle_time(row))
    assert len(store.decisions) == 2

    with pytest.raises(ValueError, match="model identity conflict"):
        run_cycle(store, [Adapter("v12-a", "c" * 64)], now=cycle_time(row))


def test_default_staleness_limits_and_daemon_cli_are_explicit() -> None:
    assert DEFAULT_MAX_CHECKPOINT_AGE_SECONDS == 90.0
    assert DEFAULT_MAX_SOURCE_UPDATE_STALENESS_SECONDS == 120.0
    args = build_parser().parse_args([
        "--db", "host=localhost dbname=boatrace",
        "--model-spec", "v12:v12_role_t300:bundle.joblib:base.joblib",
        "--max-checkpoint-age-seconds", "75",
        "--max-source-update-staleness-seconds", "100",
        "--once",
    ])
    assert args.max_checkpoint_age_seconds == 75.0
    assert args.max_source_update_staleness_seconds == 100.0
    assert args.once is True
