from __future__ import annotations

import fcntl
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class VoteJournalError(RuntimeError):
    pass


_SENSITIVE_KEYS = {
    "application_token",
    "authorization",
    "authorization_number_of_mobile",
    "auth_secret",
    "bearer",
    "live_confirmation",
    "member_number",
    "password",
    "pin",
    "secret",
    "token",
}


def idempotency_fingerprint(value: str | None) -> str | None:
    if not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def request_snapshot(request) -> dict[str, Any]:
    return {
        "stadium_tel_code": request.stadium.formal_tel_code,
        "stadium_name": request.stadium.name,
        "race_number": request.race_number,
        "bet_type": request.bet_type.value,
        "bet_type_label": request.bet_type.label,
        "method": request.method.value,
        "source_positions": [list(position) for position in request.source_positions],
        "quantity": request.quantity,
        "expanded_ticket_count": request.expanded_ticket_count,
        "total_stake_yen": request.total_stake_yen,
        "tickets": [
            {
                "number": ticket.betting_number.value,
                "quantity": ticket.quantity,
                "stake_yen": ticket.stake_yen,
            }
            for ticket in request.tickets
        ],
    }


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): (
                "[REDACTED]"
                if str(key).lower() in _SENSITIVE_KEYS
                else _redact(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    return value


class VoteJournal:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, event: dict[str, Any]) -> dict[str, Any]:
        if not event.get("request_id") or not event.get("event"):
            raise VoteJournalError("journal event requires request_id and event")
        self._ensure_parent()
        safe_event = _redact(event)
        base = {
            "schema_version": 1,
            "recorded_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            **safe_event,
        }
        try:
            with self.path.open("a+", encoding="utf-8") as handle:
                os.chmod(self.path, 0o600)
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                handle.seek(0)
                previous_hash = ""
                for line in handle:
                    try:
                        previous_hash = str(json.loads(line).get("record_hash") or "")
                    except json.JSONDecodeError as exc:
                        raise VoteJournalError("journal contains an invalid JSONL record") from exc
                base["previous_hash"] = previous_hash or None
                canonical = json.dumps(
                    base,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                base["record_hash"] = hashlib.sha256(
                    (previous_hash + canonical).encode("utf-8")
                ).hexdigest()
                handle.seek(0, os.SEEK_END)
                handle.write(
                    json.dumps(
                        base,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except VoteJournalError:
            raise
        except OSError as exc:
            raise VoteJournalError("failed to append vote journal") from exc
        return base

    def _ensure_parent(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.parent != Path(".") and not self.path.parent.exists():
                raise VoteJournalError("journal parent was not created")
        except OSError as exc:
            raise VoteJournalError("failed to create vote journal directory") from exc


def verify_journal(path: str | Path) -> dict[str, Any]:
    journal_path = Path(path)
    previous_hash = ""
    records = 0
    events: dict[str, int] = {}
    try:
        lines = journal_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise VoteJournalError("failed to read vote journal") from exc
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise VoteJournalError(
                f"invalid JSON at journal line {line_number}"
            ) from exc
        record_hash = str(row.pop("record_hash", ""))
        if row.get("previous_hash") != (previous_hash or None):
            raise VoteJournalError(
                f"previous hash mismatch at journal line {line_number}"
            )
        canonical = json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        expected = hashlib.sha256(
            (previous_hash + canonical).encode("utf-8")
        ).hexdigest()
        if not record_hash or record_hash != expected:
            raise VoteJournalError(
                f"record hash mismatch at journal line {line_number}"
            )
        previous_hash = record_hash
        records += 1
        event = str(row.get("event") or "unknown")
        events[event] = events.get(event, 0) + 1
    return {
        "valid": True,
        "records": records,
        "last_record_hash": previous_hash or None,
        "events": events,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Verify Teleboat vote journal chain.")
    parser.add_argument("--path", default="data/teleboat_vote_journal.jsonl")
    args = parser.parse_args(argv)
    result = verify_journal(args.path)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
