from __future__ import annotations

import hashlib
import json
import threading
import time
from datetime import date, datetime, timedelta, timezone

from boatrace_ai.db import connection, init_db
from boatrace_ai.odds_quality import (
    TRIFECTA_COMBINATION_KEYS,
    TRIFECTA_PARSER_VERSION,
)
from boatrace_ai.runtime.t5_spool import (
    DEFAULT_CHECKPOINT_OFFSETS,
    T5DurabilityWorker,
    T5Spool,
    build_capture,
    parse_checkpoint_offsets,
    replay_spool,
)
from boatrace_ai.runtime.time_semantics import JST


RACE_DATE = date(2026, 7, 30)


def _schedule(cutoff: datetime, *, races: int = 1) -> list[dict]:
    return [
        {
            "race_id": f"{RACE_DATE.isoformat()}-01-{rno:02d}",
            "jcd": "01",
            "rno": rno,
            # The stored value is race start; estimated cutoff is five minutes earlier.
            "deadline_at": (cutoff + timedelta(minutes=5)).astimezone(JST).isoformat(),
            "latest_odds_at": None,
        }
        for rno in range(1, races + 1)
    ]


def _capture(
    rno: int,
    captured_at: datetime,
    *,
    same_payload: bool = False,
    race_date: date = RACE_DATE,
):
    raw = b"same official response" if same_payload else f"race-{rno}-{captured_at}".encode()
    odds = {
        combination: float(index + 10)
        for index, combination in enumerate(TRIFECTA_COMBINATION_KEYS)
    }
    parsed = {
        "parser_version": TRIFECTA_PARSER_VERSION,
        "parsed_count": 120,
        "source_update_time": captured_at.astimezone(JST).strftime("%H:%M:%S"),
        "odds": odds,
    }
    return (
        build_capture(
            race_date=race_date,
            jcd="01",
            rno=rno,
            captured_at=captured_at.isoformat(),
            source_url=f"https://example.invalid/odds?rno={rno}",
            parsed=parsed,
            raw_sha256=hashlib.sha256(raw).hexdigest(),
            raw_bytes=len(raw),
        ),
        raw,
    )


def test_default_checkpoint_parser_preserves_five_closing_offsets() -> None:
    assert parse_checkpoint_offsets("300,120,60,30,10") == DEFAULT_CHECKPOINT_OFFSETS
    assert parse_checkpoint_offsets([10, 300, 60]) == (300, 60, 10)


def test_all_five_checkpoints_keep_same_signature_as_distinct_observations(tmp_path) -> None:
    cutoff = datetime(2026, 7, 30, 6, 0, tzinfo=timezone.utc)
    current = [cutoff - timedelta(seconds=300)]
    spool = T5Spool(tmp_path / "spool", archive_raw_dir=tmp_path / "raw")
    spool.save_schedule(RACE_DATE, _schedule(cutoff))

    def fetch(*, race_date, jcd, rno):
        return _capture(rno, current[0], same_payload=True)

    worker = T5DurabilityWorker(
        spool,
        date_provider=lambda: RACE_DATE,
        fetch=fetch,
        request_interval_seconds=0,
    )
    for offset in DEFAULT_CHECKPOINT_OFFSETS:
        current[0] = cutoff - timedelta(seconds=offset)
        assert worker.capture_due_once(now=current[0]) == 1

    pending = spool.pending_events()
    assert [event["target_offset_seconds"] for event in pending] == list(
        DEFAULT_CHECKPOINT_OFFSETS
    )
    assert [event["observation_label"] for event in pending] == [
        "T5",
        "T2",
        "T1",
        "T30",
        "T10",
    ]
    assert len({event["odds_signature"] for event in pending}) == 1
    assert all(event["checkpoint_attempt"] == 1 for event in pending)

    database = tmp_path / "collector.sqlite"
    init_db(database)
    with connection(database) as conn:
        assert replay_spool(spool, conn)["replayed"] == 5
        assert conn.execute("SELECT COUNT(*) FROM odds_snapshots").fetchone()[0] == 5
        assert conn.execute("SELECT COUNT(*) FROM raw_pages").fetchone()[0] == 5
        metadata = [
            json.loads(row[0])["_collection"]
            for row in conn.execute(
                "SELECT raw_json FROM odds_snapshots ORDER BY captured_at"
            ).fetchall()
        ]
    assert {item["target_offset_seconds"] for item in metadata} == set(
        DEFAULT_CHECKPOINT_OFFSETS
    )
    assert all("captured_age_seconds" in item for item in metadata)
    assert all(item["source_update_staleness_seconds"] == 0.0 for item in metadata)


def test_failed_checkpoint_retries_every_five_seconds_before_next_checkpoint(
    tmp_path,
) -> None:
    cutoff = datetime(2026, 7, 30, 6, 0, tzinfo=timezone.utc)
    first = cutoff - timedelta(seconds=300)
    current = [first]
    outcomes = [None, None, "success"]
    calls = []
    spool = T5Spool(tmp_path / "spool")
    spool.save_schedule(RACE_DATE, _schedule(cutoff))

    def fetch(*, race_date, jcd, rno):
        calls.append(current[0])
        outcome = outcomes.pop(0)
        return _capture(rno, current[0]) if outcome else None

    worker = T5DurabilityWorker(
        spool,
        date_provider=lambda: RACE_DATE,
        fetch=fetch,
        retry_seconds=5,
        request_interval_seconds=0,
    )
    assert worker.capture_due_once(now=first) == 0
    current[0] = first + timedelta(seconds=4)
    assert worker.capture_due_once(now=current[0]) == 0
    assert len(calls) == 1
    current[0] = first + timedelta(seconds=5)
    assert worker.capture_due_once(now=current[0]) == 0
    current[0] = first + timedelta(seconds=10)
    assert worker.capture_due_once(now=current[0]) == 1

    record = spool.checkpoint_record(RACE_DATE, _schedule(cutoff)[0]["race_id"], 300)
    assert record["attempts"] == 3
    assert record["success"] is True
    assert record["captured_age_seconds"] == 290.0
    assert calls == [first, first + timedelta(seconds=5), first + timedelta(seconds=10)]


def test_ten_second_checkpoint_has_priority_when_collector_starts_late(tmp_path) -> None:
    cutoff = datetime(2026, 7, 30, 6, 0, tzinfo=timezone.utc)
    now = cutoff - timedelta(seconds=10)
    spool = T5Spool(tmp_path / "spool")
    spool.save_schedule(RACE_DATE, _schedule(cutoff))
    worker = T5DurabilityWorker(
        spool,
        date_provider=lambda: RACE_DATE,
        fetch=lambda **kwargs: _capture(kwargs["rno"], now),
        request_interval_seconds=0,
    )

    assert worker.capture_due_once(now=now) == 1
    assert spool.pending_events()[0]["target_offset_seconds"] == 10


def test_completed_checkpoint_is_restored_after_restart_and_date_rollover(tmp_path) -> None:
    cutoff = datetime(2026, 7, 30, 6, 0, tzinfo=timezone.utc)
    now = cutoff - timedelta(seconds=300)
    spool = T5Spool(tmp_path / "spool")
    spool.save_schedule(RACE_DATE, _schedule(cutoff))
    calls = []

    def fetch(*, race_date, jcd, rno):
        calls.append((race_date, rno))
        return _capture(rno, now)

    first = T5DurabilityWorker(
        spool,
        date_provider=lambda: RACE_DATE,
        fetch=fetch,
        request_interval_seconds=0,
    )
    assert first.capture_due_once(now=now) == 1

    restarted = T5DurabilityWorker(
        T5Spool(tmp_path / "spool"),
        date_provider=lambda: RACE_DATE,
        fetch=fetch,
        request_interval_seconds=0,
    )
    assert restarted.capture_due_once(now=now) == 0
    assert len(calls) == 1

    next_date = RACE_DATE + timedelta(days=1)
    next_cutoff = cutoff + timedelta(days=1)
    next_spool = T5Spool(tmp_path / "spool")
    next_spool.save_schedule(
        next_date,
        [
            {
                **_schedule(next_cutoff)[0],
                "race_id": f"{next_date.isoformat()}-01-01",
            }
        ],
    )
    next_worker = T5DurabilityWorker(
        next_spool,
        date_provider=lambda: next_date,
        fetch=lambda **kwargs: _capture(
            kwargs["rno"],
            next_cutoff - timedelta(seconds=300),
            race_date=next_date,
        ),
        request_interval_seconds=0,
    )
    assert next_worker.capture_due_once(now=next_cutoff - timedelta(seconds=300)) == 1


def test_same_day_schedule_refresh_keeps_completed_races_for_monitoring(tmp_path) -> None:
    cutoff = datetime(2026, 7, 30, 6, 0, tzinfo=timezone.utc)
    spool = T5Spool(tmp_path / "spool")
    rows = _schedule(cutoff, races=2)
    spool.save_schedule(RACE_DATE, rows)

    # The DB query later omits race 1 after its result becomes final.
    spool.save_schedule(RACE_DATE, [rows[1]])

    assert [row["race_id"] for row in spool.load_schedule(RACE_DATE)] == [
        rows[0]["race_id"],
        rows[1]["race_id"],
    ]


def test_fifteen_minute_db_outage_spools_every_checkpoint_then_replays(tmp_path) -> None:
    cutoff = datetime(2026, 7, 30, 6, 0, tzinfo=timezone.utc)
    current = [cutoff - timedelta(seconds=300)]
    spool = T5Spool(tmp_path / "workspace" / "spool")
    spool.save_schedule(RACE_DATE, _schedule(cutoff))
    worker = T5DurabilityWorker(
        spool,
        date_provider=lambda: RACE_DATE,
        fetch=lambda **kwargs: _capture(kwargs["rno"], current[0]),
        request_interval_seconds=0,
    )

    for offset in DEFAULT_CHECKPOINT_OFFSETS:
        current[0] = cutoff - timedelta(seconds=offset)
        worker.capture_due_once(now=current[0])
    # The database remains unavailable for ten more minutes after the cutoff.
    current[0] = cutoff + timedelta(minutes=10)
    assert worker.capture_due_once(now=current[0]) == 0
    assert spool.status()["pending"] == 5

    database = tmp_path / "collector.sqlite"
    init_db(database)
    with connection(database) as conn:
        assert replay_spool(spool, conn)["replayed"] == 5
        assert conn.execute("SELECT COUNT(*) FROM odds_snapshots").fetchone()[0] == 5


def test_collection_stops_after_deadline_and_monitor_reports_missing(tmp_path) -> None:
    cutoff = datetime(2026, 7, 30, 6, 0, tzinfo=timezone.utc)
    spool = T5Spool(tmp_path / "spool")
    spool.save_schedule(RACE_DATE, _schedule(cutoff))

    def unexpected_fetch(**kwargs):
        raise AssertionError("official endpoint must not be called after cutoff")

    worker = T5DurabilityWorker(
        spool,
        date_provider=lambda: RACE_DATE,
        fetch=unexpected_fetch,
    )
    after = cutoff + timedelta(seconds=1)
    assert worker.capture_due_once(now=after) == 0
    checkpoints = worker.status(now=after)["checkpoints"]
    assert all(item["eligible"] == 1 for item in checkpoints.values())
    assert all(item["missing"] == 1 for item in checkpoints.values())
    assert all(item["expired"] == 1 for item in checkpoints.values())
    assert all(item["attempt"] == 0 for item in checkpoints.values())


def test_official_requests_are_serialized_and_respect_page_interval(tmp_path) -> None:
    cutoff = datetime(2026, 7, 30, 6, 0, tzinfo=timezone.utc)
    now = cutoff - timedelta(seconds=300)
    spool = T5Spool(tmp_path / "spool")
    spool.save_schedule(RACE_DATE, _schedule(cutoff, races=2))
    active = 0
    max_active = 0
    starts = []
    lock = threading.Lock()

    def fetch(*, race_date, jcd, rno):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
            starts.append(time.monotonic())
        time.sleep(0.01)
        with lock:
            active -= 1
        return _capture(rno, now)

    worker = T5DurabilityWorker(
        spool,
        date_provider=lambda: RACE_DATE,
        fetch=fetch,
        request_interval_seconds=0.04,
    )
    threads = [
        threading.Thread(target=worker.capture_due_once, kwargs={"now": now})
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert max_active == 1
    assert len(starts) == 2
    assert starts[1] - starts[0] >= 0.035


def test_legacy_t5_is_the_300_second_checkpoint(tmp_path) -> None:
    cutoff = datetime(2026, 7, 30, 6, 0, tzinfo=timezone.utc)
    t5_at = cutoff - timedelta(seconds=300)
    spool = T5Spool(tmp_path / "spool")
    spool.save_schedule(RACE_DATE, _schedule(cutoff))
    worker = T5DurabilityWorker(
        spool,
        date_provider=lambda: RACE_DATE,
        fetch=lambda **kwargs: _capture(kwargs["rno"], t5_at),
        checkpoint_offsets=(300,),
        request_interval_seconds=0,
    )

    assert worker.capture_due_once(now=t5_at - timedelta(seconds=1)) == 0
    assert worker.capture_due_once(now=t5_at) == 1
    event = spool.pending_events()[0]
    assert event["target_offset_seconds"] == 300
    assert event["captured_age_seconds"] == 300.0


def test_checkpoint_monitor_has_age_percentiles_and_pending_counts(tmp_path) -> None:
    cutoff = datetime(2026, 7, 30, 6, 0, tzinfo=timezone.utc)
    now = cutoff - timedelta(seconds=300)
    spool = T5Spool(tmp_path / "spool")
    spool.save_schedule(RACE_DATE, _schedule(cutoff, races=2))
    worker = T5DurabilityWorker(
        spool,
        date_provider=lambda: RACE_DATE,
        fetch=lambda **kwargs: _capture(kwargs["rno"], now),
        checkpoint_offsets=(300,),
        request_interval_seconds=0,
    )
    assert worker.capture_due_once(now=now) == 2

    monitor = worker.status(now=now)["checkpoints"]["300"]
    assert monitor == {
        "eligible": 2,
        "attempt": 2,
        "success": 2,
        "expired": 0,
        "missing": 0,
        "age_seconds_p50": 300.0,
        "age_seconds_p90": 300.0,
        "source_update_staleness_seconds_p50": 0.0,
        "source_update_staleness_seconds_p90": 0.0,
        "source_update_staleness_missing": 0,
        "pending": 2,
    }
