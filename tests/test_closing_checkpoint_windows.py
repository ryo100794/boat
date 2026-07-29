from __future__ import annotations

import hashlib
import inspect
import json

import pytest
from datetime import date, datetime, timedelta, timezone

from boatrace_ai.db import connection, init_db
from boatrace_ai.odds_quality import (
    TRIFECTA_COMBINATION_KEYS,
    TRIFECTA_PARSER_VERSION,
)
from boatrace_ai.runtime import collector
from boatrace_ai.runtime.t5_spool import (
    T5DurabilityWorker,
    T5Spool,
    build_capture,
    decorate_checkpoint_capture,
    replay_spool,
)
from boatrace_ai.runtime.time_semantics import JST


RACE_DATE = date(2026, 7, 30)


def _row(cutoff: datetime) -> dict:
    return {
        "race_id": f"{RACE_DATE.isoformat()}-01-01",
        "jcd": "01",
        "rno": 1,
        "deadline_at": (cutoff + timedelta(minutes=5)).astimezone(JST).isoformat(),
        "latest_odds_at": None,
    }


def _capture(captured_at: datetime) -> tuple[dict, bytes]:
    raw = b"stable official odds"
    parsed = {
        "parser_version": TRIFECTA_PARSER_VERSION,
        "parsed_count": 120,
        "source_update_time": (
            captured_at.astimezone(JST) - timedelta(seconds=7)
        ).strftime("%H:%M:%S"),
        "odds": {
            combination: float(index + 10)
            for index, combination in enumerate(TRIFECTA_COMBINATION_KEYS)
        },
    }
    return (
        build_capture(
            race_date=RACE_DATE,
            jcd="01",
            rno=1,
            captured_at=captured_at.isoformat(),
            source_url="https://example.invalid/odds",
            parsed=parsed,
            raw_sha256=hashlib.sha256(raw).hexdigest(),
            raw_bytes=len(raw),
        ),
        raw,
    )


def test_restart_expires_old_offsets_and_captures_only_current_checkpoint(tmp_path) -> None:
    cutoff = datetime(2026, 7, 30, 6, 0, tzinfo=timezone.utc)
    now = cutoff - timedelta(seconds=60)
    spool = T5Spool(tmp_path / "spool")
    spool.save_schedule(RACE_DATE, [_row(cutoff)])
    calls = []

    def fetch(**kwargs):
        calls.append(kwargs)
        return _capture(now)

    worker = T5DurabilityWorker(
        spool,
        date_provider=lambda: RACE_DATE,
        fetch=fetch,
        checkpoint_window_seconds=10,
        closing_window_seconds=0,
        request_interval_seconds=0,
    )

    assert worker.capture_due_once(now=now) == 1
    assert len(calls) == 1
    assert spool.pending_events()[0]["target_offset_seconds"] == 60
    for expired_offset in (300, 120):
        record = spool.checkpoint_record(
            RACE_DATE, _row(cutoff)["race_id"], expired_offset
        )
        assert record["status"] == "expired"
        assert record["attempts"] == 0
    assert worker.capture_due_once(now=now + timedelta(seconds=1)) == 0


def test_checkpoint_capture_uses_only_the_ten_seconds_before_target(tmp_path) -> None:
    cutoff = datetime(2026, 7, 30, 6, 0, tzinfo=timezone.utc)
    target = cutoff - timedelta(seconds=60)

    within = T5Spool(tmp_path / "within")
    within.save_schedule(RACE_DATE, [_row(cutoff)])
    captured_at = target - timedelta(seconds=10)
    worker = T5DurabilityWorker(
        within,
        date_provider=lambda: RACE_DATE,
        fetch=lambda **kwargs: _capture(captured_at),
        checkpoint_offsets=(60,),
        checkpoint_window_seconds=10,
        closing_window_seconds=0,
        request_interval_seconds=0,
    )
    assert worker.capture_due_once(now=captured_at) == 1
    event = within.pending_events()[0]
    assert event["target_offset_seconds"] == 60
    assert event["captured_age_seconds"] == 70.0
    assert within.checkpoint_record(
        RACE_DATE, _row(cutoff)["race_id"], 60
    )["success"] is True

    expired = T5Spool(tmp_path / "expired")
    expired.save_schedule(RACE_DATE, [_row(cutoff)])
    calls = []
    expired_worker = T5DurabilityWorker(
        expired,
        date_provider=lambda: RACE_DATE,
        fetch=lambda **kwargs: calls.append(kwargs),
        checkpoint_offsets=(60,),
        checkpoint_window_seconds=10,
        closing_window_seconds=0,
    )
    after_target = target + timedelta(microseconds=1)
    assert expired_worker.capture_due_once(now=after_target) == 0
    assert calls == []
    record = expired.checkpoint_record(RACE_DATE, _row(cutoff)["race_id"], 60)
    assert record["status"] == "expired"


def test_fetch_finishing_after_target_is_late_and_never_checkpoint_training_data(
    tmp_path,
) -> None:
    cutoff = datetime(2026, 7, 30, 6, 0, tzinfo=timezone.utc)
    target = cutoff - timedelta(seconds=60)
    attempted_at = target - timedelta(seconds=1)
    captured_at = target + timedelta(seconds=1)
    spool = T5Spool(tmp_path / "spool")
    spool.save_schedule(RACE_DATE, [_row(cutoff)])
    worker = T5DurabilityWorker(
        spool,
        date_provider=lambda: RACE_DATE,
        fetch=lambda **kwargs: _capture(captured_at),
        checkpoint_offsets=(60,),
        checkpoint_window_seconds=10,
        closing_window_seconds=75,
        request_interval_seconds=0,
    )

    assert worker.capture_due_once(now=attempted_at) == 1

    event = spool.pending_events()[0]
    assert event["captured_age_seconds"] == 59.0
    assert event["target_offset_seconds"] is None
    assert event["observation_label"] == "closing_cadence"
    assert event["requested_checkpoint_offset_seconds"] == 60
    collection = event["parsed"]["_collection"]
    assert collection["target_offset_seconds"] is None
    assert collection["requested_checkpoint_offset_seconds"] == 60

    record = spool.checkpoint_record(RACE_DATE, _row(cutoff)["race_id"], 60)
    assert record["success"] is False
    assert record["expired"] is True
    assert record["status"] == "late"
    assert record["captured_age_seconds"] == 59.0
    monitor = worker.status(now=captured_at)["checkpoints"]["60"]
    assert monitor["success"] == 0
    assert monitor["late"] == 1
    assert monitor["age_seconds_p50"] is None

    database = tmp_path / "collector.sqlite"
    init_db(database)
    with connection(database) as conn:
        assert replay_spool(spool, conn)["replayed"] == 1
        raw_json = conn.execute(
            "SELECT raw_json FROM odds_snapshots"
        ).fetchone()[0]
    persisted = json.loads(raw_json)["_collection"]
    assert persisted["target_offset_seconds"] is None
    assert persisted["requested_checkpoint_offset_seconds"] == 60


def test_closing_window_keeps_one_observation_per_five_second_cadence(tmp_path) -> None:
    cutoff = datetime(2026, 7, 30, 6, 0, tzinfo=timezone.utc)
    current = [cutoff - timedelta(seconds=75)]
    spool = T5Spool(tmp_path / "spool")
    spool.save_schedule(RACE_DATE, [_row(cutoff)])
    worker = T5DurabilityWorker(
        spool,
        date_provider=lambda: RACE_DATE,
        fetch=lambda **kwargs: _capture(current[0]),
        checkpoint_offsets=(60, 30, 10),
        checkpoint_window_seconds=10,
        closing_window_seconds=75,
        closing_cadence_seconds=5,
        request_interval_seconds=0,
    )

    expected = {
        75: "closing_cadence",
        70: "T1",
        65: "closing_cadence",
        60: "closing_cadence",
        55: "closing_cadence",
    }
    for age, label in expected.items():
        current[0] = cutoff - timedelta(seconds=age)
        assert worker.capture_due_once(now=current[0]) == 1
        assert spool.pending_events()[-1]["observation_label"] == label
        assert worker.capture_due_once(now=current[0] + timedelta(seconds=2)) == 0

    assert len(spool.pending_events()) == len(expected)
    status = worker.status(now=current[0])
    assert status["closing_cadence"]["attempt"] == len(expected)
    assert status["closing_cadence"]["success"] == len(expected)


def test_closing_guard_remains_active_after_a_fresh_worker_capture() -> None:
    cutoff = datetime(2026, 7, 30, 6, 0, tzinfo=timezone.utc)
    now = cutoff - timedelta(seconds=70)
    row = {**_row(cutoff), "latest_odds_at": now.isoformat()}

    guarded = collector.closing_cadence_guard_rows([row], now=now)

    assert guarded == [(70.0, row)]


def test_main_has_no_legacy_t5_or_closing_http_fetch_loop() -> None:
    source = inspect.getsource(collector.main)
    assert "for priority_row in t5_priority_rows" not in source
    assert "for closing_seconds, closing_row in closing_priority_rows" not in source
    assert "t5_worker.manages_row" in source
    assert "closing_cadence_guard_rows" in source


def test_same_timestamp_same_signature_keeps_checkpoint_observations_idempotent(
    tmp_path,
) -> None:
    cutoff = datetime(2026, 7, 30, 6, 0, tzinfo=timezone.utc)
    captured_at = cutoff - timedelta(seconds=60)
    base, raw = _capture(captured_at)
    future, _ = _capture(cutoff - timedelta(seconds=59))
    with pytest.raises(
        ValueError, match="after its decision target"
    ):
        decorate_checkpoint_capture(
            future,
            offset_seconds=60,
            attempt=1,
            deadline_at=cutoff,
        )

    spool = T5Spool(tmp_path / "spool")
    events = [
        decorate_checkpoint_capture(
            base,
            offset_seconds=offset,
            attempt=1,
            deadline_at=cutoff,
        )
        for offset in (60, 30)
    ]
    for event in events:
        spool.enqueue(event, raw_payload=raw)

    database = tmp_path / "collector.sqlite"
    init_db(database)
    with connection(database) as conn:
        assert replay_spool(spool, conn)["replayed"] == 2
        assert conn.execute("SELECT COUNT(*) FROM odds_snapshots").fetchone()[0] == 2
        for event in events:
            spool.enqueue(event, raw_payload=raw)
        assert replay_spool(spool, conn)["replayed"] == 2
        assert conn.execute("SELECT COUNT(*) FROM odds_snapshots").fetchone()[0] == 2
