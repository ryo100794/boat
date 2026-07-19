from __future__ import annotations

from datetime import date, datetime, timedelta, timezone


JST = timezone(timedelta(hours=9))


def now_jst() -> datetime:
    return datetime.now(timezone.utc).astimezone(JST)


def operational_race_date(fixed_date: date | None = None, *, at: datetime | None = None) -> date:
    if fixed_date is not None:
        return fixed_date
    current = at or now_jst()
    if current.tzinfo is None:
        current = current.replace(tzinfo=JST)
    return current.astimezone(JST).date()


def parse_jst(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=JST)
    return parsed.astimezone(JST)


def parse_any_time(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc).astimezone(JST)
    return parsed.astimezone(JST)


def minutes_between(start: datetime, end: datetime | None) -> int | None:
    if not end:
        return None
    return int((end - start).total_seconds() // 60)


def iso(value: datetime | None) -> str | None:
    return value.isoformat(timespec="seconds") if value else None


START_TO_DEADLINE_MINUTES = 5


def stored_start_time(value: str | None) -> datetime | None:
    return parse_jst(value)


def estimated_deadline_from_start(start: datetime | None) -> datetime | None:
    if start is None:
        return None
    return start - timedelta(minutes=START_TO_DEADLINE_MINUTES)


def time_fields_from_stored_start(
    stored_value: str | None,
    *,
    now: datetime,
    before_minutes: int = 5,
    result_rows: int = 0,
) -> dict[str, object]:
    start_at = stored_start_time(stored_value)
    deadline_at = estimated_deadline_from_start(start_at)
    buy_until_at = deadline_at - timedelta(minutes=before_minutes) if deadline_at else None
    if result_rows >= 3:
        status = "確定"
    elif not start_at:
        status = "時刻未取得"
    elif deadline_at and now >= deadline_at:
        status = "締切後"
    elif buy_until_at and now > buy_until_at:
        status = "T-5超過"
    else:
        status = "候補"
    return {
        "stored_schedule_at": iso(start_at),
        "deadline_at": iso(deadline_at),
        "race_time_at": iso(start_at),
        "buy_until_at": iso(buy_until_at),
        "minutes_to_deadline": minutes_between(now, deadline_at),
        "minutes_to_race_time": minutes_between(now, start_at),
        "minutes_to_buy_until": minutes_between(now, buy_until_at),
        "time_status": status,
        "time_basis": "stored_deadline_at_is_race_start",
    }

