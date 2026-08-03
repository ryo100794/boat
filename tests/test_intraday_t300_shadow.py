from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import pytest

from boatrace_ai.listwise.edge_conditional_probability_lcb_v14 import METHOD
from boatrace_ai.runtime import intraday_t300_shadow as shadow_runtime
from boatrace_ai.runtime.intraday_t300_shadow import (
    DEFAULT_MAX_CHECKPOINT_AGE_SECONDS,
    DEFAULT_MAX_DECISION_DELAY_SECONDS,
    DEFAULT_MAX_SOURCE_UPDATE_STALENESS_SECONDS,
    ModelIdentity,
    PostgresShadowStore,
    RaceWindow,
    ShadowDecision,
    SnapshotCheck,
    T300Snapshot,
    V14RegisteredBandModelAdapter,
    build_adapter,
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


def test_postgresql_snapshot_boundary_compares_instants_not_timestamp_text() -> None:
    executed = {}

    class Cursor:
        def fetchall(self):
            return []

    class Conn:
        dialect = "postgresql"

        def execute(self, statement, params):
            executed["statement"] = statement
            executed["params"] = params
            return Cursor()

    row = race("2026-07-30", suffix="01", hour=8)
    assert PostgresShadowStore(Conn()).latest_complete_snapshot(row) is None

    statement = executed["statement"]
    assert "CAST(candidate.captured_at AS timestamptz)" in statement
    assert "CAST(? AS timestamptz)" in statement
    assert "candidate.captured_at <= ?" not in statement
    assert executed["params"][-1] == row.target_t300_at.isoformat()


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


def test_cycle_backfills_unsettled_decision_from_previous_week() -> None:
    prior = race("2026-07-30")
    current = race("2026-08-03")
    store = MemoryStore(
        [prior, current],
        {
            prior.race_id: snapshot(prior.target_t300_at),
            current.race_id: snapshot(current.target_t300_at),
        },
    )
    adapter = Adapter("v12")
    run_cycle(store, [adapter], now=cycle_time(prior))
    store.results[prior.race_id] = ("1-2-3", 1_500)

    result = run_cycle(store, [adapter], now=cycle_time(current))

    assert result["settlements_inserted"] == 1
    assert store.settlements[(prior.race_id, "v12")]["profit_yen"] == 1_400


def test_postgres_store_appends_full_refund_for_terminal_nonevaluable_race() -> None:
    class Result:
        def __init__(self, rows=(), rowcount=0):
            self._rows = list(rows)
            self.rowcount = rowcount

        def fetchall(self):
            return self._rows

    class Connection:
        dialect = "postgresql"

        def __init__(self):
            self.insert_params = None

        def execute(self, sql, params):
            if str(sql).lstrip().startswith("SELECT"):
                assert "LEFT JOIN payouts" in sql
                assert "rs.trifecta_evaluable = 0" in sql
                return Result([{
                    "decision_id": 17,
                    "selected_candidates": json.dumps([
                        {"combination": "1-2-3", "stake_yen": 200}
                    ]),
                    "total_stake_yen": 200,
                    "actual_combination": None,
                    "payout_yen": None,
                    "result_status": "final",
                    "trifecta_evaluable": 0,
                }])
            self.insert_params = params
            return Result(rowcount=1)

    conn = Connection()
    store = PostgresShadowStore(conn)
    now = datetime(2026, 8, 3, 10, 0, tzinfo=timezone.utc)

    assert store.append_available_settlements(
        race_date="2026-08-03", now=now
    ) == 1
    assert conn.insert_params == (
        17,
        now.isoformat(),
        "refund",
        "__refund__",
        100,
        "{}",
        200,
        200,
        0,
    )


def test_postgres_store_sums_every_winning_payout_for_dead_heat() -> None:
    class Result:
        def __init__(self, rows=(), rowcount=0):
            self._rows = list(rows)
            self.rowcount = rowcount

        def fetchall(self):
            return self._rows

    class Connection:
        dialect = "postgresql"

        def __init__(self):
            self.insert_params = None

        def execute(self, sql, params):
            if str(sql).lstrip().startswith("SELECT"):
                base = {
                    "decision_id": 18,
                    "selected_candidates": [
                        {"combination": "1-2-3", "stake_yen": 100},
                        {"combination": "2-1-3", "stake_yen": 200},
                    ],
                    "total_stake_yen": 300,
                    "result_status": "final",
                    "trifecta_evaluable": 1,
                }
                return Result([
                    {**base, "actual_combination": "1-2-3", "payout_yen": 300},
                    {**base, "actual_combination": "2-1-3", "payout_yen": 500},
                ])
            self.insert_params = params
            return Result(rowcount=1)

    conn = Connection()
    store = PostgresShadowStore(conn)
    now = datetime(2026, 8, 3, 10, 0, tzinfo=timezone.utc)

    assert store.append_available_settlements(
        race_date="2026-08-03", now=now
    ) == 1
    assert conn.insert_params == (
        18,
        now.isoformat(),
        "final",
        "__multiple__",
        0,
        '{"1-2-3": 300, "2-1-3": 500}',
        300,
        1_300,
        1_000,
    )


@pytest.mark.parametrize(
    ("age", "source_stale", "reason"),
    [(91.0, 20.0, "stale_t300_checkpoint"),
     (30.0, 131.0, "stale_t300_source_update")],
)
def test_stale_snapshot_is_recorded_as_no_bet_with_actual_ages(
    age: float, source_stale: float, reason: str,
) -> None:
    row = race("2026-07-30")
    store = MemoryStore(
        [row], {row.race_id: snapshot(row.target_t300_at, age=age, source_stale=source_stale)}
    )
    adapter = Adapter("v12")
    first = run_cycle(store, [adapter], now=cycle_time(row))
    assert first["deferred_decisions"] == 1
    assert store.decisions == {}

    run_cycle(
        store,
        [adapter],
        now=row.target_t300_at + timedelta(seconds=90),
    )
    saved = store.decisions[(row.race_id, "v12")]

    assert saved["decision"].no_bet_reason == reason
    assert saved["decision"].total_stake_yen == 0
    assert saved["snapshot_check"].checkpoint_age_before_target_seconds == age
    assert saved["snapshot_check"].source_update_staleness_seconds == source_stale
    assert adapter.calls == 0


def test_two_minute_official_update_with_small_poll_jitter_is_accepted() -> None:
    row = race("2026-07-30")
    store = MemoryStore(
        [row], {row.race_id: snapshot(row.target_t300_at, source_stale=122.0)}
    )
    adapter = Adapter("v12")
    run_cycle(store, [adapter], now=row.target_t300_at + timedelta(seconds=90))
    saved = store.decisions[(row.race_id, "v12")]

    assert saved["decision"].status == "selected"
    assert adapter.calls == 1


def test_exactly_120_combinations_are_required() -> None:
    row = race("2026-07-30")
    store = MemoryStore([row], {row.race_id: snapshot(row.target_t300_at, count=119)})
    adapter = Adapter("v12")
    run_cycle(
        store,
        [adapter],
        now=row.target_t300_at + timedelta(seconds=90),
    )
    saved = store.decisions[(row.race_id, "v12")]
    assert saved["decision"].no_bet_reason == "incomplete_t300_snapshot"
    assert adapter.calls == 0


def test_missing_snapshot_is_retried_and_recovers_within_decision_delay() -> None:
    row = race("2026-07-30")
    store = MemoryStore([row], {})
    adapter = Adapter("v12")

    first = run_cycle(store, [adapter], now=cycle_time(row))
    assert first["decisions_inserted"] == 0
    assert first["deferred_decisions"] == 1
    assert store.decisions == {}

    store.snapshots[row.race_id] = snapshot(row.target_t300_at)
    recovered = run_cycle(
        store,
        [adapter],
        now=row.target_t300_at + timedelta(seconds=6),
    )
    assert recovered["decisions_inserted"] == 1
    assert recovered["deferred_decisions"] == 0
    assert adapter.calls == 1
    assert store.decisions[(row.race_id, "v12")]["decision"].status == "selected"


def test_missing_snapshot_is_finalized_as_no_bet_at_decision_deadline() -> None:
    row = race("2026-07-30")
    store = MemoryStore([row], {})
    adapter = Adapter("v12")

    result = run_cycle(
        store,
        [adapter],
        now=row.target_t300_at + timedelta(seconds=90),
    )
    saved = store.decisions[(row.race_id, "v12")]
    assert result["deferred_decisions"] == 0
    assert saved["decision"].no_bet_reason == "missing_complete_t300_snapshot"
    assert adapter.calls == 0


def test_model_keys_coexist_and_hash_change_under_same_key_fails() -> None:
    row = race("2026-07-30")
    store = MemoryStore([row], {row.race_id: snapshot(row.target_t300_at)})
    run_cycle(store, [Adapter("v12-a"), Adapter("v12-b", "b" * 64)], now=cycle_time(row))
    assert len(store.decisions) == 2

    with pytest.raises(ValueError, match="model identity conflict"):
        run_cycle(store, [Adapter("v12-a", "c" * 64)], now=cycle_time(row))


def test_cycle_reports_backlog_and_only_configured_model_timings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = race("2026-07-30")
    store = MemoryStore([row], {row.race_id: snapshot(row.target_t300_at)})
    adapters = [Adapter("v12"), Adapter("v14"), Adapter("v16")]
    ticks = iter((0.0, 1.0, 3.0, 3.0, 7.0, 7.0, 8.0, 9.0))
    monkeypatch.setattr(shadow_runtime.time, "perf_counter", lambda: next(ticks))

    result = run_cycle(
        store, adapters, now=row.target_t300_at + timedelta(seconds=120)
    )

    assert result["models"] == ["v12", "v14", "v16"]
    assert result["timing"] == {
        "cycle_elapsed_seconds": 9.0,
        "due_races_scanned": 1,
        "pending_decisions": 3,
        "initial_pending_backlog_max_seconds": 120.0,
        "model_decide": {
            "v12": {"calls": 1, "total_seconds": 2.0, "max_seconds": 2.0},
            "v14": {"calls": 1, "total_seconds": 4.0, "max_seconds": 4.0},
            "v16": {"calls": 1, "total_seconds": 1.0, "max_seconds": 1.0},
        },
    }
    assert "v20" not in result["timing"]["model_decide"]


def test_default_staleness_limits_and_daemon_cli_are_explicit() -> None:
    assert DEFAULT_MAX_CHECKPOINT_AGE_SECONDS == 90.0
    assert DEFAULT_MAX_DECISION_DELAY_SECONDS == 90.0
    assert DEFAULT_MAX_SOURCE_UPDATE_STALENESS_SECONDS == 130.0
    args = build_parser().parse_args([
        "--db", "host=localhost dbname=boatrace",
        "--model-spec", "v12:v12_role_t300:bundle.joblib:base.joblib",
        "--max-checkpoint-age-seconds", "75",
        "--max-source-update-staleness-seconds", "100",
        "--max-decision-delay-seconds", "80",
        "--once",
    ])
    assert args.max_checkpoint_age_seconds == 75.0
    assert args.max_source_update_staleness_seconds == 100.0
    assert args.max_decision_delay_seconds == 80.0
    assert args.once is True


def _v14_evaluation() -> dict[str, Any]:
    return {
        "deployment_configuration": {
            "calibrator_strategy": (
                "odds_path_role_integrated_registered_band_lcb_v14"
            ),
            "probability_lcb": {
                "model_name": "edge_conditional_probability_lcb_v14",
                "method": METHOD,
                "ready": True,
                "trained_through_date": "2026-07-29",
                "uses_result_for_fit_only": True,
                "uses_payout": False,
                "registered_divergence_band": {
                    "lower_inclusive": 0.5,
                    "upper_exclusive": 1.0,
                },
                "rank_nodes": {},
                "conditional_cells": {},
            },
            "candidate_policy": {
                "registered_divergence_lower_inclusive": 0.5,
                "registered_divergence_upper_exclusive": 1.0,
            },
        }
    }


def _write_v14_artifacts(
    tmp_path: Path, *, conformal_ready: bool = False,
) -> tuple[Path, Path]:
    merged_bundle = tmp_path / "v14-merged.joblib"
    base_model = tmp_path / "base.joblib"
    deployment = _v14_evaluation()["deployment_configuration"]
    deployment.update({
        "operational_model": {"trained_through_date": "2026-07-29"},
        "closing_t300_v12_model": {
            "model_name": "closing_odds_t300_nonlinear_v12",
            "ready": True,
            "trained_through_date": "2026-07-29",
            "point_model": {"estimator": object()},
        },
        "selection_conformal": {
            "ready": conformal_ready,
            "trained_through_date": "2026-07-29",
            "haircut": 0.9,
        },
    })
    joblib.dump({
        "deployment": deployment,
        "sources": {
            "v14_evaluation": "job-00007396.json",
            "closing_estimator": "v12_live_bundle",
        },
    }, merged_bundle)
    joblib.dump({"feature_schema_version": 1}, base_model)
    return merged_bundle, base_model


def test_v14_evaluation_deployment_is_composed_with_v12_closing(
    tmp_path: Path,
) -> None:
    merged_bundle, base_model = _write_v14_artifacts(tmp_path)
    adapter = build_adapter(
        f"v14-jul30:v14_registered_band_t300:{merged_bundle}:{base_model}"
    )

    assert isinstance(adapter, V14RegisteredBandModelAdapter)
    assert adapter.identity.strategy_name == "v14_registered_band_t300"
    assert len(adapter.identity.model_hash) == 64
    assert adapter._component("probability_lcb")["method"] == METHOD
    assert adapter._component("closing_t300_v12_model")["model_name"] == (
        "closing_odds_t300_nonlinear_v12"
    )


def test_v12_and_v14_series_coexist_with_distinct_model_hashes(tmp_path: Path) -> None:
    merged_bundle, base_model = _write_v14_artifacts(tmp_path)
    v14 = V14RegisteredBandModelAdapter(
        model_key="v14", bundle_path=merged_bundle, base_model_path=base_model,
    )
    row = race("2026-07-30")
    store = MemoryStore([row], {row.race_id: snapshot(row.target_t300_at)})
    v12 = Adapter("v12", "1" * 64, selected=False)

    v14.decide = lambda *args, **kwargs: ShadowDecision(  # type: ignore[method-assign]
        {}, {}, (), "v14_sparse_or_missing_cell_no_bet"
    )
    run_cycle(store, [v12, v14], now=cycle_time(row))

    assert set(store.decisions) == {(row.race_id, "v12"), (row.race_id, "v14")}
    assert store.decisions[(row.race_id, "v14")]["decision"].no_bet_reason == (
        "v14_sparse_or_missing_cell_no_bet"
    )
    assert v12.identity.model_hash != v14.identity.model_hash


def test_v14_decision_uses_only_pre_result_race_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    merged_bundle, base_model = _write_v14_artifacts(tmp_path)
    adapter = V14RegisteredBandModelAdapter(
        model_key="v14", bundle_path=merged_bundle, base_model_path=base_model,
    )
    row = race("2026-07-30")
    point = snapshot(row.target_t300_at)
    uniform = {key: 1.0 / 120.0 for key in COMBINATIONS}
    seen: list[dict[str, Any]] = []

    monkeypatch.setattr(adapter, "_base_probabilities", lambda conn, race: uniform)
    monkeypatch.setattr(
        "boatrace_ai.runtime.intraday_t300_shadow.attach_odds_path_probability_v8",
        lambda races, model: races,
    )
    def capture(
        race_payload: dict[str, Any], combination: str, artifact: dict[str, Any],
    ) -> dict[str, Any]:
        if not seen:
            seen.append(copy.deepcopy(race_payload))
        return {
            "probability": 0.0, "raw_probability": 1.0 / 120.0,
            "factor": 0.0, "in_registered_divergence_band": False,
            "resolution": "outside_registered_divergence_band",
        }

    monkeypatch.setattr(
        "boatrace_ai.runtime.intraday_t300_shadow.probability_lower_bound_details_v14",
        capture,
    )
    decision = adapter.decide(object(), row, point, bankroll_yen=10_000)

    assert decision.no_bet_reason == "selection_conformal_not_ready"
    assert decision.diagnostics["v14_registered_band"]["status"] == "recorded"
    assert decision.diagnostics["v14_registered_band"]["uses_result"] is False
    assert decision.diagnostics["v14_registered_band"]["uses_payout"] is False
    assert len(seen) == 1
    assert "actual_combination" not in seen[0]
    assert "result" not in seen[0]
    assert "payout" not in seen[0]
    assert seen[0]["snapshot_id"] == point.snapshot_id
    assert seen[0]["odds_checkpoints"]["300"]["snapshot_id"] == point.snapshot_id


def test_v14_ready_path_never_calls_v13_and_allocates_v14_top2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    merged_bundle, base_model = _write_v14_artifacts(
        tmp_path, conformal_ready=True
    )
    adapter = V14RegisteredBandModelAdapter(
        model_key="v14", bundle_path=merged_bundle, base_model_path=base_model,
    )
    row = race("2026-07-30")
    point = snapshot(row.target_t300_at)
    uniform = {key: 1.0 / 120.0 for key in COMBINATIONS}
    v14_calls: list[str] = []
    allocator_candidates: list[dict[str, Any]] = []

    monkeypatch.setattr(adapter, "_base_probabilities", lambda conn, race: uniform)
    monkeypatch.setattr(
        "boatrace_ai.runtime.intraday_t300_shadow.attach_odds_path_probability_v8",
        lambda races, model: races,
    )
    monkeypatch.setattr(
        "boatrace_ai.runtime.intraday_t300_shadow.forecast_closing_odds_t300_nonlinear_v12",
        lambda race_payload, model, prediction_date: {
            "ready": True,
            "future_checkpoint_offsets_used": [],
            "lower_final_odds": {key: 10.0 for key in COMBINATIONS},
        },
    )

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("V13 probability API must not be called by V14")

    monkeypatch.setattr(
        "boatrace_ai.runtime.intraday_t300_shadow.selected_safe_ev_candidates",
        forbidden,
    )
    monkeypatch.setattr(
        "boatrace_ai.listwise.selection_conformal.probability_lower_bound_details",
        forbidden,
    )

    def v14_detail(
        race_payload: dict[str, Any], combination: str, artifact: dict[str, Any],
    ) -> dict[str, Any]:
        v14_calls.append(combination)
        assert "actual_combination" not in race_payload
        assert "payout" not in race_payload
        probability = 0.2 if combination in COMBINATIONS[:3] else 0.0
        return {
            "probability": probability,
            "raw_probability": uniform[combination],
            "factor": 1.0 if probability else 0.0,
            "in_registered_divergence_band": bool(probability),
            "resolution": "cell_min_parent_and_cell_lower",
        }

    monkeypatch.setattr(
        "boatrace_ai.runtime.intraday_t300_shadow.probability_lower_bound_details_v14",
        v14_detail,
    )

    def allocate(
        day: str, candidates: list[dict[str, Any]], race_ids: set[str], **kwargs: Any,
    ) -> dict[str, Any]:
        allocator_candidates.extend(copy.deepcopy(candidates))
        return {
            "selected_sample": [
                {**candidate, "stake_yen": 100} for candidate in candidates
            ],
            "allocation_candidate_tickets": len(candidates),
        }

    monkeypatch.setattr(
        "boatrace_ai.runtime.intraday_t300_shadow.allocate_discrete_log_day",
        allocate,
    )
    decision = adapter.decide(object(), row, point, bankroll_yen=10_000)

    assert v14_calls == sorted(COMBINATIONS)
    assert [item["combination"] for item in allocator_candidates] == COMBINATIONS[:2]
    assert all(item["estimated_odds"] == 9.0 for item in allocator_candidates)
    assert all(
        item["odds_source"] == "v12_t300_lower_v14_lcb_times_selection_conformal"
        for item in allocator_candidates
    )
    assert decision.status == "selected"
    assert decision.no_bet_reason is None
    assert decision.diagnostics["v14_registered_band"]["top2_candidates"] == 2


@pytest.mark.parametrize("field", ["uses_payout", "registered_divergence_band"])
def test_v14_rejects_unsafe_or_unregistered_deployment(
    tmp_path: Path, field: str,
) -> None:
    merged_bundle, base_model = _write_v14_artifacts(tmp_path)
    payload = joblib.load(merged_bundle)
    lcb = payload["deployment"]["probability_lcb"]
    if field == "uses_payout":
        lcb[field] = True
    else:
        lcb[field]["upper_exclusive"] = 1.1
    joblib.dump(payload, merged_bundle)

    with pytest.raises(ValueError):
        V14RegisteredBandModelAdapter(
            model_key="v14", bundle_path=merged_bundle,
            base_model_path=base_model,
        )
