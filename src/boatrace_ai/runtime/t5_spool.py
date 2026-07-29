from __future__ import annotations

import json
import os
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from ..constants import VENUE_BY_CODE
from ..db import race_id, trifecta_odds_signature, upsert_race
from ..http import fetch_text, sha256_bytes
from ..official import race_page_url
from ..odds_quality import TRIFECTA_PARSER_VERSION, plausible_trifecta_odds
from ..storage import record_raw_page
from ..ingestion.parsers import parse_odds3t_html, result_page_is_cancelled
from .time_semantics import JST, estimated_deadline_from_start, stored_start_time


DEFAULT_MAX_BYTES = 512 * 1024 * 1024
DEFAULT_CHECKPOINT_OFFSETS = (300, 120, 60, 30, 10)
DEFAULT_RETRY_SECONDS = 5.0
DEFAULT_CHECKPOINT_WINDOW_SECONDS = 10.0
DEFAULT_CLOSING_CADENCE_SECONDS = 5.0
DEFAULT_CLOSING_WINDOW_SECONDS = 75.0
SCHEMA_VERSION = 2


class SpoolCapacityError(RuntimeError):
    pass


def parse_checkpoint_offsets(value: str | Iterable[int]) -> tuple[int, ...]:
    parts = value.split(",") if isinstance(value, str) else value
    offsets = tuple(int(part) for part in parts)
    if not offsets or any(offset <= 0 for offset in offsets):
        raise ValueError("closing checkpoint offsets must be positive seconds")
    if len(set(offsets)) != len(offsets):
        raise ValueError("closing checkpoint offsets must be unique")
    return tuple(sorted(offsets, reverse=True))


def checkpoint_label(offset_seconds: int) -> str:
    if offset_seconds >= 60 and offset_seconds % 60 == 0:
        return f"T{offset_seconds // 60}"
    return f"T{offset_seconds}"


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
        self.state_dir = self.root / "checkpoint_state"
        self.archive_raw_dir = Path(archive_raw_dir) if archive_raw_dir else None
        self._lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.corrupt_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def save_schedule(self, race_date: date, rows: Iterable[Any]) -> None:
        keep = (
            "race_id",
            "jcd",
            "rno",
            "deadline_at",
            "latest_odds_at",
        )
        incoming = []
        for row in rows:
            values = _row_dict(row)
            incoming.append({key: values.get(key) for key in keep})
        with self._lock:
            merged = {
                str(row["race_id"]): row
                for row in self.load_schedule(race_date)
                if row.get("race_id")
            }
            for row in incoming:
                race_id_value = str(row.get("race_id") or "")
                if not race_id_value:
                    continue
                merged[race_id_value] = {
                    **merged.get(race_id_value, {}),
                    **row,
                }
            serialized = sorted(
                merged.values(),
                key=lambda row: (str(row.get("deadline_at") or ""), str(row["race_id"])),
            )
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

    def load_checkpoint_state(self, race_date: date) -> dict[str, Any]:
        path = self._checkpoint_state_path(race_date)
        with self._lock:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (FileNotFoundError, OSError, json.JSONDecodeError):
                return {
                    "schema_version": SCHEMA_VERSION,
                    "race_date": race_date.isoformat(),
                    "races": {},
                }
        if payload.get("race_date") != race_date.isoformat():
            return {
                "schema_version": SCHEMA_VERSION,
                "race_date": race_date.isoformat(),
                "races": {},
            }
        if not isinstance(payload.get("races"), dict):
            payload["races"] = {}
        return payload

    def record_checkpoint_attempt(
        self,
        *,
        race_date: date,
        race_id_value: str,
        offset_seconds: int,
        attempted_at: datetime,
        retry_seconds: float,
        success: bool,
        captured_at: str | None = None,
        captured_age_seconds: float | None = None,
        source_update_time: str | None = None,
        source_update_staleness_seconds: float | None = None,
        event_id: str | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            state = self.load_checkpoint_state(race_date)
            races = state.setdefault("races", {})
            checkpoints = races.setdefault(race_id_value, {})
            key = str(int(offset_seconds))
            record = checkpoints.setdefault(key, {})
            record["attempts"] = int(record.get("attempts") or 0) + 1
            record["last_attempt_at"] = attempted_at.isoformat()
            record["next_retry_at"] = (
                attempted_at + timedelta(seconds=retry_seconds)
            ).isoformat()
            record["success"] = bool(success)
            record["expired"] = False
            record["status"] = "success" if success else "retrying"
            record["last_error"] = error
            if success:
                record["completed_at"] = attempted_at.isoformat()
                record["captured_at"] = captured_at
                record["captured_age_seconds"] = captured_age_seconds
                record["source_update_time"] = source_update_time
                record["source_update_staleness_seconds"] = source_update_staleness_seconds
                record["event_id"] = event_id
                record["next_retry_at"] = None
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            _atomic_json(self._checkpoint_state_path(race_date), state)
            return dict(record)

    def mark_checkpoint_expired(
        self,
        *,
        race_date: date,
        race_id_value: str,
        offset_seconds: int,
        expired_at: datetime,
        status: str = "expired",
        error: str = "missed_checkpoint_window",
        event_id: str | None = None,
        captured_at: str | None = None,
        captured_age_seconds: float | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            state = self.load_checkpoint_state(race_date)
            record = (
                state.setdefault("races", {})
                .setdefault(race_id_value, {})
                .setdefault(str(int(offset_seconds)), {})
            )
            if not record.get("success"):
                record.update(
                    {
                        "success": False,
                        "attempts": int(record.get("attempts") or 0),
                        "expired": True,
                        "status": status,
                        "expired_at": expired_at.isoformat(),
                        "next_retry_at": None,
                        "last_error": error,
                        "event_id": event_id,
                        "captured_at": captured_at,
                        "captured_age_seconds": captured_age_seconds,
                    }
                )
                state["updated_at"] = datetime.now(timezone.utc).isoformat()
                _atomic_json(self._checkpoint_state_path(race_date), state)
            return dict(record)

    def record_closing_attempt(
        self,
        *,
        race_date: date,
        race_id_value: str,
        attempted_at: datetime,
        success: bool,
        retry_seconds: float,
        event_id: str | None,
        error: str | None,
    ) -> dict[str, Any]:
        with self._lock:
            state = self.load_checkpoint_state(race_date)
            record = (
                state.setdefault("races", {})
                .setdefault(race_id_value, {})
                .setdefault("_closing_cadence", {})
            )
            record["attempts"] = int(record.get("attempts") or 0) + 1
            record["last_attempt_at"] = attempted_at.isoformat()
            record["next_retry_at"] = (
                attempted_at + timedelta(seconds=retry_seconds)
            ).isoformat()
            record["last_success"] = bool(success)
            record["last_error"] = error
            if success:
                record["successes"] = int(record.get("successes") or 0) + 1
                record["last_event_id"] = event_id
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            _atomic_json(self._checkpoint_state_path(race_date), state)
            return dict(record)

    def checkpoint_record(
        self,
        race_date: date,
        race_id_value: str,
        offset_seconds: int,
    ) -> dict[str, Any]:
        state = self.load_checkpoint_state(race_date)
        return dict(
            state.get("races", {})
            .get(race_id_value, {})
            .get(str(int(offset_seconds)), {})
        )

    def pending_events(self) -> list[dict[str, Any]]:
        with self._lock:
            pending, _ = self._read_pending_locked(repair_tail=True)
        return pending

    def _checkpoint_state_path(self, race_date: date) -> Path:
        return self.state_dir / f"{race_date.isoformat()}.json"

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
        checkpoint = event.get("target_offset_seconds")
        if checkpoint is None:
            checkpoint = event.get("observation_label", "closing")
        attempt = event.get("checkpoint_attempt", 1)
        target = (
            self.archive_raw_dir
            / "pages"
            / str(event["race_date"]).replace("-", "")
            / str(event["jcd"]).zfill(2)
            / f"{int(event['rno']):02d}"
            / f"odds3t-cp{checkpoint}-a{attempt}-{captured}.html"
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


def source_update_staleness_seconds(
    value: Any,
    *,
    captured_at: datetime,
) -> float | None:
    if not value:
        return None
    try:
        parts = [int(part) for part in str(value).split(":")]
    except ValueError:
        return None
    if len(parts) not in (2, 3):
        return None
    hour, minute = parts[:2]
    second = parts[2] if len(parts) == 3 else 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        return None
    captured_jst = captured_at.astimezone(JST)
    source_at = captured_jst.replace(
        hour=hour, minute=minute, second=second, microsecond=0
    )
    if source_at - captured_jst > timedelta(minutes=1):
        source_at -= timedelta(days=1)
    return max(0.0, (captured_jst - source_at).total_seconds())


def decorate_checkpoint_capture(
    event: dict[str, Any],
    *,
    offset_seconds: int | None,
    attempt: int,
    deadline_at: datetime,
    observation_label: str | None = None,
) -> dict[str, Any]:
    decorated = dict(event)
    captured_at = datetime.fromisoformat(str(event["captured_at"]))
    if captured_at.tzinfo is None:
        captured_at = captured_at.replace(tzinfo=timezone.utc)
    captured_age = (deadline_at - captured_at).total_seconds()
    if offset_seconds is not None and captured_age < int(offset_seconds):
        raise ValueError(
            "checkpoint capture occurred after its decision target"
        )
    source_staleness = source_update_staleness_seconds(
        event.get("source_update_time"), captured_at=captured_at
    )
    label = observation_label or (
        checkpoint_label(int(offset_seconds))
        if offset_seconds is not None
        else "closing_cadence"
    )
    offset_token = int(offset_seconds) if offset_seconds is not None else label
    event_id = (
        f"{event['race_id']}-closing-{offset_token}-"
        f"a{int(attempt)}-{str(event['captured_at']).replace(':', '').replace('+', '_')}"
    )
    collection = {
        "event_id": event_id,
        "target_offset_seconds": (
            int(offset_seconds) if offset_seconds is not None else None
        ),
        "observation_label": label,
        "attempt": int(attempt),
        "deadline_at": deadline_at.isoformat(),
        "captured_age_seconds": captured_age,
        "source_update_time": event.get("source_update_time"),
        "source_update_staleness_seconds": source_staleness,
    }
    parsed = dict(event["parsed"])
    parsed["_collection"] = collection
    decorated.update(
        {
            "event_id": event_id,
            "kind": (
                "closing_checkpoint_trifecta_snapshot"
                if offset_seconds is not None
                else "closing_cadence_trifecta_snapshot"
            ),
            "target_offset_seconds": (
                int(offset_seconds) if offset_seconds is not None else None
            ),
            "observation_label": label,
            "checkpoint_attempt": int(attempt),
            "deadline_at": deadline_at.isoformat(),
            "captured_age_seconds": captured_age,
            "source_update_staleness_seconds": source_staleness,
            "parsed": parsed,
        }
    )
    return decorated


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
    captured_at = datetime.now(timezone.utc).isoformat()
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
        "AND local_path = ? LIMIT 1",
        ("odds3t", event["race_id"], raw_path),
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

    matching_rows = conn.execute(
        "SELECT snapshot_id, raw_json FROM odds_snapshots WHERE race_id = ? "
        "AND bet_type = 'trifecta' AND captured_at = ? AND odds_signature = ? "
        "ORDER BY snapshot_id",
        (event["race_id"], event["captured_at"], event["odds_signature"]),
    ).fetchall()
    row = None
    if event.get("target_offset_seconds") is None:
        row = matching_rows[0] if matching_rows else None
    else:
        for candidate in matching_rows:
            try:
                stored = json.loads(candidate[1] or "{}")
            except (TypeError, json.JSONDecodeError):
                continue
            collection = stored.get("_collection") or {}
            if collection.get("event_id") == event.get("event_id"):
                row = candidate
                break
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
        checkpoint_offsets: Iterable[int] = DEFAULT_CHECKPOINT_OFFSETS,
        retry_seconds: float = DEFAULT_RETRY_SECONDS,
        checkpoint_window_seconds: float = DEFAULT_CHECKPOINT_WINDOW_SECONDS,
        closing_window_seconds: float = DEFAULT_CLOSING_WINDOW_SECONDS,
        closing_cadence_seconds: float = DEFAULT_CLOSING_CADENCE_SECONDS,
        request_interval_seconds: float = 0.4,
        poll_seconds: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.spool = spool
        self.date_provider = date_provider
        self.fetch = fetch
        self.checkpoint_offsets = parse_checkpoint_offsets(checkpoint_offsets)
        self.retry_seconds = float(retry_seconds)
        if self.retry_seconds <= 0:
            raise ValueError("checkpoint retry seconds must be positive")
        self.checkpoint_window_seconds = float(checkpoint_window_seconds)
        self.closing_window_seconds = float(closing_window_seconds)
        self.closing_cadence_seconds = float(closing_cadence_seconds)
        if self.checkpoint_window_seconds < 0:
            raise ValueError("checkpoint window seconds must be non-negative")
        if self.closing_window_seconds < 0 or self.closing_cadence_seconds <= 0:
            raise ValueError(
                "closing window must be non-negative and cadence seconds positive"
            )
        self.request_interval_seconds = max(0.0, float(request_interval_seconds))
        self.poll_seconds = poll_seconds
        self._sleep = sleep
        self._monotonic = monotonic
        self._last_request_at: float | None = None
        self._capture_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.counters = {
            "captures": 0,
            "failed": 0,
            "capacity_rejected": 0,
            "attempts": 0,
            "late_checkpoint": 0,
        }

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="closing-checkpoint-durability",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.poll_seconds * 2))

    def capture_due_once(self, *, now: datetime) -> int:
        with self._capture_lock:
            return self._capture_due_once(now=now)

    def manages_row(self, row: Any, *, now: datetime) -> bool:
        """Return whether checkpoint collection owns odds HTTP for this race."""
        values = _row_dict(row)
        cutoff_at = estimated_deadline_from_start(
            stored_start_time(values.get("deadline_at"))
        )
        if cutoff_at is None:
            return False
        first_checkpoint = cutoff_at - timedelta(
            seconds=max(self.checkpoint_offsets) + self.checkpoint_window_seconds
        )
        return first_checkpoint <= now <= cutoff_at

    def _capture_due_once(self, *, now: datetime) -> int:
        race_date = self.date_provider()
        rows = self.spool.load_schedule(race_date)
        state = self.spool.load_checkpoint_state(race_date)
        race_state = state.get("races", {})
        candidates: list[
            tuple[
                tuple[int, float, str],
                dict[str, Any],
                datetime,
                int | None,
                dict[str, Any],
            ]
        ] = []

        for row in rows:
            cutoff_at = estimated_deadline_from_start(
                stored_start_time(row.get("deadline_at"))
            )
            if cutoff_at is None:
                continue
            race_id_value = str(row["race_id"])
            records = race_state.get(race_id_value, {})
            if now > cutoff_at:
                for offset in self.checkpoint_offsets:
                    record = records.get(str(offset), {})
                    if not record.get("success") and not record.get("expired"):
                        self.spool.mark_checkpoint_expired(
                            race_date=race_date,
                            race_id_value=race_id_value,
                            offset_seconds=offset,
                            expired_at=now,
                        )
                continue
            due: list[tuple[int, dict[str, Any], datetime]] = []
            for offset in self.checkpoint_offsets:
                target_at = cutoff_at - timedelta(seconds=offset)
                record = dict(records.get(str(offset), {}))
                if record.get("success") or record.get("expired"):
                    continue
                opens_at = target_at - timedelta(
                    seconds=self.checkpoint_window_seconds
                )
                if now > target_at:
                    self.spool.mark_checkpoint_expired(
                        race_date=race_date,
                        race_id_value=race_id_value,
                        offset_seconds=offset,
                        expired_at=now,
                    )
                    continue
                if now < opens_at:
                    continue
                next_retry = _parse_datetime(record.get("next_retry_at"))
                if next_retry is not None and now < next_retry:
                    continue
                due.append((offset, record, target_at))

            if due:
                offset, record, target_at = min(
                    due,
                    key=lambda item: abs((now - item[2]).total_seconds()),
                )
                candidates.append(
                    (
                        (0 if offset == 10 else 1, (cutoff_at - now).total_seconds(), race_id_value),
                        row,
                        cutoff_at,
                        offset,
                        record,
                    )
                )
                continue

            seconds_to_cutoff = (cutoff_at - now).total_seconds()
            if not (
                self.closing_window_seconds > 0
                and 0.0 <= seconds_to_cutoff <= self.closing_window_seconds
            ):
                continue
            closing_record = dict(records.get("_closing_cadence", {}))
            next_closing = _parse_datetime(closing_record.get("next_retry_at"))
            if next_closing is not None and now < next_closing:
                continue
            candidates.append(
                (
                    (2, seconds_to_cutoff, race_id_value),
                    row,
                    cutoff_at,
                    None,
                    closing_record,
                )
            )

        captured = 0
        for _priority, row, cutoff_at, offset, record in sorted(candidates):
            self._rate_limit()
            attempted_at = now
            attempt = int(record.get("attempts") or 0) + 1
            self.counters["attempts"] += 1
            event: dict[str, Any] | None = None
            raw_payload: bytes | None = None
            error: str | None = None
            late_checkpoint = False
            try:
                result = self.fetch(
                    race_date=race_date,
                    jcd=str(row["jcd"]),
                    rno=int(row["rno"]),
                )
                if result is None:
                    error = "no_valid_snapshot"
                else:
                    event, raw_payload = result
                    captured_at = _parse_datetime(event.get("captured_at"))
                    if offset is not None:
                        target_at = cutoff_at - timedelta(seconds=offset)
                        late_checkpoint = (
                            captured_at is None or captured_at > target_at
                        )
                    effective_offset = None if late_checkpoint else offset
                    event = decorate_checkpoint_capture(
                        event,
                        offset_seconds=effective_offset,
                        attempt=attempt,
                        deadline_at=cutoff_at,
                        observation_label=(
                            checkpoint_label(offset)
                            if offset is not None and not late_checkpoint
                            else "closing_cadence"
                        ),
                    )
                    if late_checkpoint:
                        event["requested_checkpoint_offset_seconds"] = offset
                        event["parsed"]["_collection"][
                            "requested_checkpoint_offset_seconds"
                        ] = offset
                        error = "late_checkpoint_capture"
                    self.spool.enqueue(event, raw_payload=raw_payload)
            except SpoolCapacityError:
                self.counters["capacity_rejected"] += 1
                error = "spool_capacity"
                event = None
            except Exception as exc:
                error = type(exc).__name__
                event = None

            capture_success = event is not None and raw_payload is not None
            checkpoint_success = capture_success and not late_checkpoint
            race_id_value = str(row["race_id"])
            if offset is not None:
                self.spool.record_checkpoint_attempt(
                    race_date=race_date,
                    race_id_value=race_id_value,
                    offset_seconds=offset,
                    attempted_at=attempted_at,
                    retry_seconds=self.retry_seconds,
                    success=checkpoint_success,
                    captured_at=str(event["captured_at"]) if event else None,
                    captured_age_seconds=(
                        float(event["captured_age_seconds"]) if event else None
                    ),
                    source_update_time=(
                        str(event["source_update_time"])
                        if event and event.get("source_update_time")
                        else None
                    ),
                    source_update_staleness_seconds=(
                        float(event["source_update_staleness_seconds"])
                        if event
                        and event.get("source_update_staleness_seconds") is not None
                        else None
                    ),
                    event_id=str(event["event_id"]) if event else None,
                    error=error,
                )
                if late_checkpoint and event is not None:
                    self.spool.mark_checkpoint_expired(
                        race_date=race_date,
                        race_id_value=race_id_value,
                        offset_seconds=offset,
                        expired_at=_parse_datetime(event["captured_at"]) or attempted_at,
                        status="late",
                        error="late_checkpoint_capture",
                        event_id=str(event["event_id"]),
                        captured_at=str(event["captured_at"]),
                        captured_age_seconds=float(event["captured_age_seconds"]),
                    )
                    self.counters["late_checkpoint"] += 1
            if (cutoff_at - attempted_at).total_seconds() <= self.closing_window_seconds:
                self.spool.record_closing_attempt(
                    race_date=race_date,
                    race_id_value=race_id_value,
                    attempted_at=attempted_at,
                    success=capture_success,
                    retry_seconds=self.closing_cadence_seconds,
                    event_id=str(event["event_id"]) if event else None,
                    error=error,
                )
            if capture_success:
                self.counters["captures"] += 1
                captured += 1
            else:
                self.counters["failed"] += 1
        return captured

    def next_poll_seconds(self, *, now: datetime, default: float) -> float:
        race_date = self.date_provider()
        rows = self.spool.load_schedule(race_date)
        state = self.spool.load_checkpoint_state(race_date).get("races", {})
        delays: list[float] = []
        for row in rows:
            cutoff_at = estimated_deadline_from_start(
                stored_start_time(row.get("deadline_at"))
            )
            if cutoff_at is None or now > cutoff_at:
                continue
            records = state.get(str(row["race_id"]), {})
            for offset in self.checkpoint_offsets:
                record = records.get(str(offset), {})
                if record.get("success") or record.get("expired"):
                    continue
                target_at = cutoff_at - timedelta(seconds=offset)
                opens_at = target_at - timedelta(
                    seconds=self.checkpoint_window_seconds
                )
                if now > target_at:
                    delays.append(0.1)
                    continue
                next_retry = _parse_datetime(record.get("next_retry_at"))
                due_at = max(opens_at, next_retry) if next_retry else opens_at
                due_at = min(target_at, due_at)
                delays.append(max(0.1, (due_at - now).total_seconds()))

            seconds_to_cutoff = (cutoff_at - now).total_seconds()
            if (
                self.closing_window_seconds > 0
                and 0.0 <= seconds_to_cutoff <= self.closing_window_seconds
            ):
                closing_record = records.get("_closing_cadence", {})
                next_closing = _parse_datetime(closing_record.get("next_retry_at"))
                if next_closing is None:
                    delays.append(0.1)
                else:
                    delays.append(max(0.1, (next_closing - now).total_seconds()))
            elif seconds_to_cutoff > self.closing_window_seconds:
                delays.append(
                    max(0.1, seconds_to_cutoff - self.closing_window_seconds)
                )
        return min(float(default), min(delays)) if delays else float(default)

    def status(self, *, now: datetime | None = None) -> dict[str, Any]:
        now = now or datetime.now(timezone.utc)
        race_date = self.date_provider()
        rows = self.spool.load_schedule(race_date)
        state = self.spool.load_checkpoint_state(race_date).get("races", {})
        pending = self.spool.pending_events()
        checkpoints: dict[str, dict[str, Any]] = {}
        for offset in self.checkpoint_offsets:
            eligible = 0
            attempts = 0
            successes = 0
            expired = 0
            late = 0
            ages: list[float] = []
            source_staleness: list[float] = []
            source_staleness_missing = 0
            for row in rows:
                cutoff_at = estimated_deadline_from_start(
                    stored_start_time(row.get("deadline_at"))
                )
                if cutoff_at is None or now < cutoff_at - timedelta(
                    seconds=offset + self.checkpoint_window_seconds
                ):
                    continue
                eligible += 1
                record = state.get(str(row["race_id"]), {}).get(str(offset), {})
                attempts += int(record.get("attempts") or 0)
                if record.get("expired"):
                    expired += 1
                if record.get("status") == "late":
                    late += 1
                if record.get("success"):
                    successes += 1
                    age = record.get("captured_age_seconds")
                    if age is not None:
                        ages.append(float(age))
                    stale = record.get("source_update_staleness_seconds")
                    if stale is None:
                        source_staleness_missing += 1
                    else:
                        source_staleness.append(float(stale))
            checkpoints[str(offset)] = {
                "eligible": eligible,
                "attempt": attempts,
                "success": successes,
                "expired": expired,
                "late": late,
                "missing": max(0, eligible - successes),
                "age_seconds_p50": _percentile(ages, 0.50),
                "age_seconds_p90": _percentile(ages, 0.90),
                "source_update_staleness_seconds_p50": _percentile(
                    source_staleness, 0.50
                ),
                "source_update_staleness_seconds_p90": _percentile(
                    source_staleness, 0.90
                ),
                "source_update_staleness_missing": source_staleness_missing,
                "pending": sum(
                    1
                    for event in pending
                    if event.get("target_offset_seconds") == offset
                ),
            }
        return {
            **self.spool.status(),
            **self.counters,
            "checkpoint_offsets": list(self.checkpoint_offsets),
            "checkpoint_window_seconds": self.checkpoint_window_seconds,
            "closing_window_seconds": self.closing_window_seconds,
            "closing_cadence_seconds": self.closing_cadence_seconds,
            "closing_cadence": {
                "attempt": sum(
                    int(records.get("_closing_cadence", {}).get("attempts") or 0)
                    for records in state.values()
                ),
                "success": sum(
                    int(records.get("_closing_cadence", {}).get("successes") or 0)
                    for records in state.values()
                ),
                "pending": sum(
                    1
                    for event in pending
                    if event.get("observation_label") == "closing_cadence"
                ),
            },
            "checkpoints": checkpoints,
        }

    def _rate_limit(self) -> None:
        current = self._monotonic()
        if self._last_request_at is not None:
            delay = self.request_interval_seconds - (current - self._last_request_at)
            if delay > 0:
                self._sleep(delay)
                current = self._monotonic()
        self._last_request_at = current

    def _run(self) -> None:
        while not self._stop.is_set():
            now = datetime.now(timezone.utc)
            self.capture_due_once(now=now)
            delay = self.next_poll_seconds(now=now, default=self.poll_seconds)
            self._stop.wait(delay)


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 3)


def replay_spool(spool: T5Spool, conn: Any) -> dict[str, int]:
    def persist(event: dict[str, Any]) -> None:
        try:
            persist_capture(conn, event)
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    return spool.replay(persist)
