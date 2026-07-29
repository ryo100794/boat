from __future__ import annotations

import json
import os
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from ..constants import VENUE_BY_CODE
from ..db import race_id, trifecta_odds_signature, upsert_race
from ..features import MODEL_DECISION_LEAD_MINUTES
from ..http import fetch_text, sha256_bytes
from ..official import race_page_url
from ..odds_quality import TRIFECTA_PARSER_VERSION, plausible_trifecta_odds
from ..storage import record_raw_page
from ..ingestion.parsers import parse_odds3t_html, result_page_is_cancelled
from .time_semantics import estimated_deadline_from_start, stored_start_time


DEFAULT_MAX_BYTES = 512 * 1024 * 1024
SCHEMA_VERSION = 1


class SpoolCapacityError(RuntimeError):
    pass


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    _fsync_directory(path.parent)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _row_dict(row: Any) -> dict[str, Any]:
    keys = row.keys() if hasattr(row, "keys") else row
    return {str(key): row[key] for key in keys}


class T5Spool:
    """Durable write-ahead queue for the small number of T-5 captures."""

    def __init__(
        self,
        root: str | Path,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
        archive_raw_dir: str | Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.max_bytes = int(max_bytes)
        if self.max_bytes <= 0:
            raise ValueError("T-5 spool max_bytes must be positive")
        self.raw_dir = self.root / "raw"
        self.corrupt_dir = self.root / "corrupt"
        self.journal_path = self.root / "pending.jsonl"
        self.schedule_path = self.root / "schedule.json"
        self.archive_raw_dir = Path(archive_raw_dir) if archive_raw_dir else None
        self._lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.corrupt_dir.mkdir(parents=True, exist_ok=True)

    def save_schedule(self, race_date: date, rows: Iterable[Any]) -> None:
        keep = (
            "race_id",
            "jcd",
            "rno",
            "deadline_at",
            "latest_odds_at",
        )
        serialized = []
        for row in rows:
            values = _row_dict(row)
            serialized.append({key: values.get(key) for key in keep})
        with self._lock:
            _atomic_json(
                self.schedule_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "race_date": race_date.isoformat(),
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                    "rows": serialized,
                },
            )

    def load_schedule(self, race_date: date) -> list[dict[str, Any]]:
        with self._lock:
            try:
                payload = json.loads(self.schedule_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, OSError, json.JSONDecodeError):
                return []
        if payload.get("race_date") != race_date.isoformat():
            return []
        rows = payload.get("rows")
        return list(rows) if isinstance(rows, list) else []

    def enqueue(
        self,
        capture: dict[str, Any],
        *,
        raw_payload: bytes,
    ) -> str:
        event = dict(capture)
        event["schema_version"] = SCHEMA_VERSION
        event_id = str(event["event_id"])
        raw_path = self.raw_dir / f"{event_id}.html"
        event["raw_local_path"] = str(raw_path)
        event_line = (
            json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")

        with self._lock:
            pending, _ = self._read_pending_locked(repair_tail=True)
            if any(item.get("event_id") == event_id for item in pending):
                return event_id
            current_bytes = self._disk_bytes_locked()
            existing_raw_bytes = raw_path.stat().st_size if raw_path.exists() else 0
            growth = len(event_line) + max(0, len(raw_payload) - existing_raw_bytes)
            if current_bytes + growth > self.max_bytes:
                raise SpoolCapacityError(
                    f"T-5 spool capacity exceeded: {current_bytes + growth}>{self.max_bytes}"
                )
            if sha256_bytes(raw_payload) != event.get("raw_sha256"):
                raise ValueError("raw payload sha256 does not match capture metadata")
            _atomic_bytes(raw_path, raw_payload)
            with self.journal_path.open("ab") as handle:
                handle.write(event_line)
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_directory(self.journal_path.parent)
        return event_id

    def replay(self, persist: Callable[[dict[str, Any]], None]) -> dict[str, int]:
        with self._lock:
            pending, corrupt_tail = self._read_pending_locked(repair_tail=True)
        replayed: list[dict[str, Any]] = []
        failed = 0
        for event in pending:
            try:
                persisted_event = self._promote_raw(event)
                persist(persisted_event)
            except Exception:
                failed = 1
                break
            replayed.append(event)
        if replayed:
            replayed_ids = {str(item["event_id"]) for item in replayed}
            with self._lock:
                current, extra_corrupt = self._read_pending_locked(repair_tail=True)
                remaining = [
                    item for item in current if str(item.get("event_id")) not in replayed_ids
                ]
                self._write_pending_locked(remaining)
                corrupt_tail += extra_corrupt
                for event in replayed:
                    Path(str(event.get("raw_local_path", ""))).unlink(missing_ok=True)
        return {
            "replayed": len(replayed),
            "failed": failed,
            "corrupt_tail_records": corrupt_tail,
        }

    def _promote_raw(self, event: dict[str, Any]) -> dict[str, Any]:
        if self.archive_raw_dir is None:
            return event
        captured = str(event["captured_at"]).replace(":", "").replace("+", "_")
        target = (
            self.archive_raw_dir
            / "pages"
            / str(event["race_date"]).replace("-", "")
            / str(event["jcd"]).zfill(2)
            / f"{int(event['rno']):02d}"
            / f"odds3t-t5-{captured}.html"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        source = Path(str(event["raw_local_path"]))
        if not target.exists():
            _atomic_bytes(target, source.read_bytes())
        promoted = dict(event)
        promoted["raw_local_path"] = str(target)
        return promoted

    def pending_event_ids(self) -> set[str]:
        with self._lock:
            pending, _ = self._read_pending_locked(repair_tail=True)
        return {str(item.get("event_id")) for item in pending}

    def status(self) -> dict[str, Any]:
        with self._lock:
            pending, corrupt_tail = self._read_pending_locked(repair_tail=True)
            used_bytes = self._disk_bytes_locked()
            corrupt_files = list(self.corrupt_dir.glob("*.jsonl"))
        captured = [str(item.get("captured_at")) for item in pending if item.get("captured_at")]
        return {
            "spool_dir": str(self.root),
            "pending": len(pending),
            "used_bytes": used_bytes,
            "max_bytes": self.max_bytes,
            "usage_ratio": round(used_bytes / self.max_bytes, 6),
            "oldest_captured_at": min(captured) if captured else None,
            "corrupt_files": len(corrupt_files),
            "repaired_corrupt_tail": corrupt_tail,
        }

    def _disk_bytes_locked(self) -> int:
        total = 0
        for path in self.root.rglob("*"):
            if path.is_file():
                try:
                    total += path.stat().st_size
                except OSError:
                    continue
        return total

    def _read_pending_locked(self, *, repair_tail: bool) -> tuple[list[dict[str, Any]], int]:
        try:
            raw = self.journal_path.read_bytes()
        except FileNotFoundError:
            return [], 0
        if not raw:
            return [], 0
        lines = raw.splitlines(keepends=True)
        events: list[dict[str, Any]] = []
        valid_bytes = bytearray()
        for index, line in enumerate(lines):
            complete = line.endswith((b"\n", b"\r"))
            try:
                event = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                is_tail = index == len(lines) - 1
                if not (repair_tail and is_tail):
                    raise ValueError(f"corrupt T-5 spool record at line {index + 1}")
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                _atomic_bytes(self.corrupt_dir / f"tail-{stamp}.jsonl", line)
                _atomic_bytes(self.journal_path, bytes(valid_bytes))
                return events, 1
            if not isinstance(event, dict) or not event.get("event_id"):
                raise ValueError(f"invalid T-5 spool record at line {index + 1}")
            events.append(event)
            valid_bytes.extend(line if complete else line + b"\n")
        return events, 0

    def _write_pending_locked(self, events: list[dict[str, Any]]) -> None:
        temporary = self.journal_path.with_name(f".{self.journal_path.name}.tmp")
        with temporary.open("wb") as handle:
            for event in events:
                handle.write(
                    (
                        json.dumps(
                            event,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    ).encode("utf-8")
                )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.journal_path)
        _fsync_directory(self.journal_path.parent)


def build_capture(
    *,
    race_date: date,
    jcd: str,
    rno: int,
    captured_at: str,
    source_url: str,
    parsed: dict[str, Any],
    raw_sha256: str,
    raw_bytes: int,
) -> dict[str, Any]:
    rid = race_id(race_date.isoformat(), jcd, rno)
    signature = trifecta_odds_signature(parsed["odds"])
    event_id = f"{rid}-t5-{captured_at.replace(':', '').replace('+', '_')}"
    return {
        "event_id": event_id,
        "kind": "t5_trifecta_snapshot",
        "race_id": rid,
        "race_date": race_date.isoformat(),
        "jcd": jcd.zfill(2),
        "rno": int(rno),
        "captured_at": captured_at,
        "source_url": source_url,
        "source_update_time": parsed.get("source_update_time"),
        "parser_version": parsed.get("parser_version"),
        "parsed_count": parsed.get("parsed_count"),
        "odds_signature": signature,
        "odds": parsed["odds"],
        "parsed": parsed,
        "raw_sha256": raw_sha256,
        "raw_bytes": int(raw_bytes),
    }


def fetch_t5_capture(
    *,
    race_date: date,
    jcd: str,
    rno: int,
    timeout: float = 5.0,
    retries: int = 0,
) -> tuple[dict[str, Any], bytes] | None:
    url = race_page_url("odds3t", race_date, jcd, rno)
    status_code, html, payload = fetch_text(
        url,
        cache_bust=True,
        timeout=timeout,
        retries=retries,
    )
    if status_code != 200 or result_page_is_cancelled(html):
        return None
    parsed = parse_odds3t_html(html)
    if (
        parsed.get("parser_version") != TRIFECTA_PARSER_VERSION
        or parsed.get("parsed_count") != 120
        or not plausible_trifecta_odds(parsed.get("odds") or {})
    ):
        return None
    captured_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return (
        build_capture(
            race_date=race_date,
            jcd=jcd,
            rno=rno,
            captured_at=captured_at,
            source_url=url,
            parsed=parsed,
            raw_sha256=sha256_bytes(payload),
            raw_bytes=len(payload),
        ),
        payload,
    )


def persist_capture(conn: Any, event: dict[str, Any]) -> int:
    venue = VENUE_BY_CODE.get(str(event["jcd"]).zfill(2))
    upsert_race(
        conn,
        {
            "race_id": event["race_id"],
            "race_date": event["race_date"],
            "jcd": str(event["jcd"]).zfill(2),
            "venue_name": venue.name if venue else str(event["jcd"]).zfill(2),
            "rno": int(event["rno"]),
            "status": "scheduled",
        },
    )
    raw_path = str(event["raw_local_path"])
    exists = conn.execute(
        "SELECT 1 FROM raw_pages WHERE page_type = ? AND race_id = ? "
        "AND sha256 = ? LIMIT 1",
        ("odds3t", event["race_id"], event["raw_sha256"]),
    ).fetchone()
    if exists is None:
        record_raw_page(
            conn,
            page_type="odds3t",
            race_id=event["race_id"],
            source_url=event["source_url"],
            local_path=raw_path,
            sha256=event["raw_sha256"],
            bytes_count=int(event["raw_bytes"]),
        )

    row = conn.execute(
        "SELECT snapshot_id FROM odds_snapshots WHERE race_id = ? "
        "AND bet_type = 'trifecta' AND captured_at = ? AND odds_signature = ? "
        "ORDER BY snapshot_id LIMIT 1",
        (event["race_id"], event["captured_at"], event["odds_signature"]),
    ).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO odds_snapshots (race_id, bet_type, captured_at, "
            "source_update_time, odds_signature, parser_version, raw_json, source_url) "
            "VALUES (?, 'trifecta', ?, ?, ?, ?, ?, ?)",
            (
                event["race_id"],
                event["captured_at"],
                event.get("source_update_time"),
                event["odds_signature"],
                event.get("parser_version"),
                json.dumps(event["parsed"], ensure_ascii=False, sort_keys=True),
                event["source_url"],
            ),
        )
        snapshot_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
    else:
        snapshot_id = int(row[0])
    conn.executemany(
        "INSERT OR REPLACE INTO odds_trifecta "
        "(snapshot_id, race_id, combination, odds) VALUES (?, ?, ?, ?)",
        [
            (snapshot_id, event["race_id"], combination, odds)
            for combination, odds in event["odds"].items()
        ],
    )
    return snapshot_id


class T5DurabilityWorker:
    def __init__(
        self,
        spool: T5Spool,
        *,
        date_provider: Callable[[], date],
        fetch: Callable[..., tuple[dict[str, Any], bytes] | None] = fetch_t5_capture,
        poll_seconds: float = 2.0,
    ) -> None:
        self.spool = spool
        self.date_provider = date_provider
        self.fetch = fetch
        self.poll_seconds = poll_seconds
        self._captured: set[str] = set()
        self._capture_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.counters = {"captures": 0, "failed": 0, "capacity_rejected": 0}

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="t5-durability", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.poll_seconds * 2))

    def capture_due_once(self, *, now: datetime) -> int:
        with self._capture_lock:
            return self._capture_due_once(now=now)

    def _capture_due_once(self, *, now: datetime) -> int:
        race_date = self.date_provider()
        rows = self.spool.load_schedule(race_date)
        pending = self.spool.pending_event_ids()
        captured = 0
        for row in rows:
            start_at = stored_start_time(row.get("deadline_at"))
            cutoff_at = estimated_deadline_from_start(start_at)
            if cutoff_at is None:
                continue
            t5_at = cutoff_at - timedelta(minutes=MODEL_DECISION_LEAD_MINUTES)
            seconds = (t5_at - now).total_seconds()
            capture_key = f"{row['race_id']}:{t5_at.isoformat()}"
            if capture_key in self._captured or not -1.0 <= seconds <= 60.0:
                continue
            if any(event_id.startswith(f"{row['race_id']}-t5-") for event_id in pending):
                self._captured.add(capture_key)
                continue
            try:
                result = self.fetch(
                    race_date=race_date,
                    jcd=str(row["jcd"]),
                    rno=int(row["rno"]),
                )
                if result is None:
                    self.counters["failed"] += 1
                    continue
                event, raw_payload = result
                self.spool.enqueue(event, raw_payload=raw_payload)
            except SpoolCapacityError:
                self.counters["capacity_rejected"] += 1
                continue
            except Exception:
                self.counters["failed"] += 1
                continue
            self._captured.add(capture_key)
            pending.add(str(event["event_id"]))
            self.counters["captures"] += 1
            captured += 1
        return captured

    def status(self) -> dict[str, Any]:
        return {**self.spool.status(), **self.counters}

    def _run(self) -> None:
        while not self._stop.is_set():
            now = datetime.now(timezone.utc)
            self.capture_due_once(now=now)
            self._stop.wait(self.poll_seconds)


def replay_spool(spool: T5Spool, conn: Any) -> dict[str, int]:
    def persist(event: dict[str, Any]) -> None:
        try:
            persist_capture(conn, event)
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    return spool.replay(persist)
