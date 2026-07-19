from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .time_semantics import JST, stored_start_time


def result_interval(seconds_to_start: float) -> float | None:
    """Poll aggressively once official results are normally available."""
    if seconds_to_start > -5 * 60:
        return None
    elapsed = -seconds_to_start
    if elapsed <= 30 * 60:
        return 10.0
    if elapsed <= 90 * 60:
        return 30.0
    if elapsed <= 6 * 60 * 60:
        return 120.0
    if elapsed <= 24 * 60 * 60:
        return 600.0
    return None


def due_result_rows(rows: list[Any], *, now: datetime) -> list[Any]:
    """Return due result targets with the newest completed race first."""
    due: list[tuple[datetime, Any]] = []
    for row in rows:
        start_at = stored_start_time(row["deadline_at"])
        if start_at is None:
            continue
        interval = result_interval((start_at - now).total_seconds())
        if interval is None:
            continue
        attempted_at = _parse_attempt(row["latest_result_attempt_at"])
        if attempted_at is not None and (now - attempted_at).total_seconds() < interval:
            continue
        due.append((start_at, row))
    due.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in due]


def _parse_attempt(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(JST)
