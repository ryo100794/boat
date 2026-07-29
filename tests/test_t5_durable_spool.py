from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone

import pytest

from boatrace_ai.db import connection, init_db
from boatrace_ai.odds_quality import TRIFECTA_COMBINATION_KEYS, TRIFECTA_PARSER_VERSION
from boatrace_ai.runtime import postgresql_collector
from boatrace_ai.runtime.t5_spool import (
    SpoolCapacityError,
    T5DurabilityWorker,
    T5Spool,
    build_capture,
    replay_spool,
)
from boatrace_ai.runtime.time_semantics import JST


RACE_DATE = date(2026, 7, 29)


def _odds(seed: int = 0) -> dict[str, float]:
    return {
        combination: float(index + 10 + seed)
        for index, combination in enumerate(TRIFECTA_COMBINATION_KEYS)
    }


def _capture(rno: int, captured_at: datetime) -> tuple[dict, bytes]:
    payload = f"<html>race-{rno}</html>".encode()
    parsed = {
        "parser_version": TRIFECTA_PARSER_VERSION,
        "parsed_count": 120,
        "source_update_time": captured_at.astimezone(JST).strftime("%H:%M"),
        "odds": _odds(rno),
    }
    return (
        build_capture(
            race_date=RACE_DATE,
            jcd="01",
            rno=rno,
            captured_at=captured_at.isoformat(),
            source_url=f"https://example.invalid/odds?rno={rno}",
            parsed=parsed,
            raw_sha256=hashlib.sha256(payload).hexdigest(),
            raw_bytes=len(payload),
        ),
        payload,
    )


def _schedule(start: datetime, count: int = 4) -> list[dict]:
    return [
        {
            "race_id": f"{RACE_DATE.isoformat()}-01-{rno:02d}",
            "jcd": "01",
            "rno": rno,
            "deadline_at": (start + timedelta(minutes=5 * (rno - 1))).isoformat(),
            "latest_odds_at": None,
        }
        for rno in range(1, count + 1)
    ]


def test_fifteen_minute_db_outage_spools_all_due_t5_and_replays_after_restart(
    tmp_path,
) -> None:
    spool_dir = tmp_path / "workspace" / "runtime_spool"
    archive_dir = tmp_path / "workspace" / "raw"
    spool = T5Spool(spool_dir, archive_raw_dir=archive_dir)
    first_t5 = datetime(2026, 7, 29, 3, 0, tzinfo=timezone.utc)
    spool.save_schedule(
        RACE_DATE,
        _schedule(first_t5.astimezone(JST) + timedelta(minutes=10)),
    )
    active_capture_time = [first_t5]

    def fetch(*, race_date, jcd, rno):
        assert race_date == RACE_DATE
        return _capture(rno, active_capture_time[0])

    worker = T5DurabilityWorker(
        spool,
        date_provider=lambda: RACE_DATE,
        fetch=fetch,
        checkpoint_offsets=(300,),
        closing_window_seconds=0,
    )
    for minute in (0, 5, 10, 15):
        active_capture_time[0] = first_t5 + timedelta(minutes=minute)
        assert worker.capture_due_once(now=active_capture_time[0]) == 1

    assert spool.status()["pending"] == 4
    assert len(list((spool_dir / "raw").glob("*.html"))) == 4

    database = tmp_path / "collector.sqlite"
    init_db(database)
    restarted_spool = T5Spool(spool_dir, archive_raw_dir=archive_dir)
    with connection(database) as conn:
        result = replay_spool(restarted_spool, conn)
        assert result == {"replayed": 4, "failed": 0, "corrupt_tail_records": 0}
        assert conn.execute("SELECT COUNT(*) FROM odds_snapshots").fetchone()[0] == 4
        assert conn.execute("SELECT COUNT(*) FROM odds_trifecta").fetchone()[0] == 480
        assert conn.execute("SELECT COUNT(*) FROM raw_pages").fetchone()[0] == 4

    assert restarted_spool.status()["pending"] == 0
    assert len(list(archive_dir.rglob("odds3t-cp300-*.html"))) == 4


def test_replay_is_idempotent_when_commit_succeeds_before_spool_ack(tmp_path) -> None:
    from boatrace_ai.runtime.t5_spool import persist_capture

    database = tmp_path / "collector.sqlite"
    spool = T5Spool(tmp_path / "spool", archive_raw_dir=tmp_path / "raw")
    init_db(database)
    event, payload = _capture(1, datetime(2026, 7, 29, 3, 0, tzinfo=timezone.utc))
    spool.enqueue(event, raw_payload=payload)

    with connection(database) as conn:
        def committed_then_crashed(item):
            persist_capture(conn, item)
            conn.commit()
            raise RuntimeError("process stopped before queue ack")

        first = spool.replay(committed_then_crashed)
        assert first["failed"] == 1
        assert spool.status()["pending"] == 1

    restarted = T5Spool(tmp_path / "spool", archive_raw_dir=tmp_path / "raw")
    with connection(database) as conn:
        assert replay_spool(restarted, conn)["replayed"] == 1
        assert conn.execute("SELECT COUNT(*) FROM odds_snapshots").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM odds_trifecta").fetchone()[0] == 120
        assert conn.execute("SELECT COUNT(*) FROM raw_pages").fetchone()[0] == 1


def test_corrupt_trailing_jsonl_is_quarantined_without_blocking_valid_replay(
    tmp_path,
) -> None:
    spool = T5Spool(tmp_path / "spool")
    for rno in (1, 2):
        event, payload = _capture(
            rno,
            datetime(2026, 7, 29, 3, rno, tzinfo=timezone.utc),
        )
        spool.enqueue(event, raw_payload=payload)
    with spool.journal_path.open("ab") as handle:
        handle.write(b'{"event_id":"torn-tail"')

    replayed = []
    restarted = T5Spool(tmp_path / "spool")
    result = restarted.replay(lambda event: replayed.append(event["event_id"]))

    assert len(replayed) == 2
    assert result["corrupt_tail_records"] == 1
    assert restarted.status()["pending"] == 0
    assert restarted.status()["corrupt_files"] == 1
    assert list(restarted.corrupt_dir.glob("tail-*.jsonl"))[0].read_bytes().startswith(
        b'{"event_id":"torn-tail"'
    )


def test_capacity_limit_rejects_without_partial_record_and_reports_usage(tmp_path) -> None:
    event, payload = _capture(1, datetime(2026, 7, 29, 3, 0, tzinfo=timezone.utc))
    spool = T5Spool(tmp_path / "spool", max_bytes=128)

    with pytest.raises(SpoolCapacityError, match="capacity exceeded"):
        spool.enqueue(event, raw_payload=payload)

    status = spool.status()
    assert status["pending"] == 0
    assert status["max_bytes"] == 128
    assert status["used_bytes"] <= 128
    assert status["usage_ratio"] <= 1.0
    assert list(spool.raw_dir.glob("*.html")) == []


def test_schedule_cache_is_date_scoped_and_survives_restart(tmp_path) -> None:
    root = tmp_path / "spool"
    spool = T5Spool(root)
    rows = _schedule(datetime(2026, 7, 29, 13, 0, tzinfo=JST), count=2)
    spool.save_schedule(RACE_DATE, rows)

    restarted = T5Spool(root)
    assert restarted.load_schedule(RACE_DATE) == rows
    assert restarted.load_schedule(date(2026, 7, 30)) == []


def test_retrying_postgresql_connection_keeps_waiting_until_db_recovers(
    monkeypatch,
) -> None:
    attempts = []
    closed = []
    retry_hooks = []

    @contextmanager
    def recovered_connection(_dsn):
        attempts.append("attempt")
        if len(attempts) < 3:
            raise ConnectionError("database unavailable")
        try:
            yield "connection"
        finally:
            closed.append(True)

    monkeypatch.setattr(postgresql_collector, "connection", recovered_connection)
    monkeypatch.setattr(postgresql_collector.time, "sleep", lambda _seconds: None)

    with postgresql_collector.retrying_connection(
        "dsn", on_retry=lambda: retry_hooks.append("retry")
    ) as conn:
        assert conn == "connection"

    assert len(attempts) == 3
    assert retry_hooks == ["retry", "retry"]
    assert closed == [True]


def test_monitor_status_is_json_serializable(tmp_path) -> None:
    spool = T5Spool(tmp_path / "spool", max_bytes=1024 * 1024)
    event, payload = _capture(1, datetime(2026, 7, 29, 3, 0, tzinfo=timezone.utc))
    spool.enqueue(event, raw_payload=payload)
    worker = T5DurabilityWorker(spool, date_provider=lambda: RACE_DATE)

    status = worker.status()
    assert status["pending"] == 1
    assert status["used_bytes"] > 0
    assert status["oldest_captured_at"] == event["captured_at"]
    assert json.loads(json.dumps(status))["max_bytes"] == 1024 * 1024
