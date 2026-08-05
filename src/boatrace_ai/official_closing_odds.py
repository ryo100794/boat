from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from .archive_closing_odds import (
    OFFICIAL_PARSER_VERSION,
    OFFICIAL_SOURCE_KEY,
    ensure_archive_schema,
    pending_races,
    record_attempt,
    store_archive_closing_odds,
    verify_winning_payout,
)
from .db import connection
from .http import FetchError, fetch_text
from .ingestion.parsers import parse_odds3t_html
from .odds_quality import TRIFECTA_COMBINATION_KEYS, plausible_trifecta_odds
from .official import race_page_url


class IncompleteOfficialTrifectaOdds(ValueError):
    def __init__(self, odds_count: int) -> None:
        self.odds_count = int(odds_count)
        super().__init__(
            f"official trifecta odds are incomplete: {self.odds_count}/120"
        )


def official_closing_url(race_date: str | date, jcd: str, rno: int) -> str:
    day = date.fromisoformat(str(race_date)[:10])
    return race_page_url("odds3t", day, str(jcd), int(rno))


def parse_official_closing_odds_html(html: str) -> dict[str, Any]:
    parsed = parse_odds3t_html(html)
    odds = {
        str(key): float(value)
        for key, value in (parsed.get("odds") or {}).items()
        if value is not None
    }
    if set(odds) != set(TRIFECTA_COMBINATION_KEYS):
        raise IncompleteOfficialTrifectaOdds(len(odds))
    if not plausible_trifecta_odds(odds):
        raise ValueError("official trifecta odds are implausible")
    return {
        "parser_version": str(parsed.get("parser_version") or OFFICIAL_PARSER_VERSION),
        "market_time": "closing",
        "source_key": OFFICIAL_SOURCE_KEY,
        "odds_count": len(odds),
        "unavailable_count": 0,
        "unavailable_combinations": [],
        "odds": odds,
    }


def _confirmed_result_boats(conn: Any, race_id: str) -> int:
    row = conn.execute(
        """
        SELECT COUNT(DISTINCT lane) AS boats
        FROM race_results
        WHERE race_id = ? AND rank IS NOT NULL
        """,
        (race_id,),
    ).fetchone()
    return int(row["boats"] if row is not None else 0)


def _increment_counter(counters: dict[str, Any], field: str, key: object) -> None:
    values = counters.setdefault(field, {})
    normalized = str(key)
    values[normalized] = int(values.get(normalized, 0)) + 1


def _record_failure_example(
    counters: dict[str, Any],
    row: Any,
    *,
    status: str,
    error: str,
    odds_count: int | None = None,
    confirmed_result_boats: int | None = None,
) -> None:
    examples = counters.setdefault("failure_examples", [])
    if len(examples) >= 25:
        return
    example = {
        "race_id": str(row["race_id"]),
        "race_date": str(row["race_date"]),
        "jcd": str(row["jcd"]),
        "rno": int(row["rno"]),
        "status": status,
        "error": error[:500],
    }
    if odds_count is not None:
        example["odds_count"] = int(odds_count)
    if confirmed_result_boats is not None:
        example["confirmed_result_boats"] = int(confirmed_result_boats)
    examples.append(example)


def reclassify_confirmed_non_six_boat_attempts(conn: Any) -> int:
    ensure_archive_schema(conn)
    cursor = conn.execute(
        """
        UPDATE archive_closing_odds_attempts
        SET status = 'excluded_non_six_boat'
        WHERE source_key = ? AND status = 'invalid'
          AND error = ?
          AND (
            SELECT COUNT(DISTINCT rr.lane)
            FROM race_results rr
            WHERE rr.race_id = archive_closing_odds_attempts.race_id
              AND rr.rank IS NOT NULL
          ) = 5
        """,
        (
            OFFICIAL_SOURCE_KEY,
            "ValueError: official trifecta odds are incomplete: 60/120",
        ),
    )
    return max(0, int(cursor.rowcount or 0))


def backfill_official_closing_odds(
    conn: Any,
    *,
    from_date: str,
    through_date: str,
    sleep_seconds: float = 0.5,
    max_pages: int | None = None,
) -> dict[str, Any]:
    reclassified = reclassify_confirmed_non_six_boat_attempts(conn)
    targets = pending_races(
        conn,
        from_date=from_date,
        through_date=through_date,
        source_key=OFFICIAL_SOURCE_KEY,
    )
    if max_pages is not None:
        targets = targets[: max(0, int(max_pages))]
    counters = {
        "status": "completed",
        "source_key": OFFICIAL_SOURCE_KEY,
        "source_role": "primary_official_historical_closing",
        "from_date": from_date,
        "through_date": through_date,
        "targets": len(targets),
        "stored": 0,
        "not_found": 0,
        "invalid": 0,
        "fetch_failed": 0,
        "excluded_non_six_boat": 0,
        "reclassified_non_six_boat": reclassified,
        "invalid_reason_counts": {},
        "incomplete_odds_count_counts": {},
        "invalid_confirmed_result_boats_counts": {},
        "fetch_failure_reason_counts": {},
        "failure_examples": [],
    }
    for index, row in enumerate(targets):
        url = official_closing_url(row["race_date"], row["jcd"], int(row["rno"]))
        try:
            status_code, html, payload = fetch_text(
                url,
                timeout=15.0,
                retries=1,
                sleep_seconds=sleep_seconds,
            )
            if status_code == 404:
                counters["not_found"] += 1
                record_attempt(
                    conn,
                    race_id=row["race_id"],
                    status="not_found",
                    http_status=404,
                    source_key=OFFICIAL_SOURCE_KEY,
                )
            elif status_code != 200:
                counters["fetch_failed"] += 1
                record_attempt(
                    conn,
                    race_id=row["race_id"],
                    status="http_error",
                    http_status=status_code,
                    source_key=OFFICIAL_SOURCE_KEY,
                )
            else:
                parsed = parse_official_closing_odds_html(html)
                verification = verify_winning_payout(
                    parsed["odds"],
                    combination=str(row["combination"]),
                    payout_yen=int(row["payout_yen"]),
                )
                verification["status"] = "official_primary_winner_payout_match"
                store_archive_closing_odds(
                    conn,
                    race_id=str(row["race_id"]),
                    source_url=url,
                    payload=payload,
                    parsed=parsed,
                    verification=verification,
                    source_key=OFFICIAL_SOURCE_KEY,
                    parser_version=str(parsed["parser_version"]),
                    source_kind="boatrace_official_historical_closing_display",
                )
                record_attempt(
                    conn,
                    race_id=row["race_id"],
                    status="stored",
                    source_key=OFFICIAL_SOURCE_KEY,
                )
                counters["stored"] += 1
        except (FetchError, OSError) as exc:
            counters["fetch_failed"] += 1
            reason = type(exc).__name__
            _increment_counter(counters, "fetch_failure_reason_counts", reason)
            _record_failure_example(
                counters,
                row,
                status="fetch_error",
                error=f"{reason}: {exc}",
            )
            record_attempt(
                conn,
                race_id=row["race_id"],
                status="fetch_error",
                error=f"{type(exc).__name__}: {exc}"[:500],
                source_key=OFFICIAL_SOURCE_KEY,
            )
        except (KeyError, TypeError, ValueError) as exc:
            confirmed_result_boats = _confirmed_result_boats(
                conn, str(row["race_id"])
            )
            confirmed_non_six = (
                isinstance(exc, IncompleteOfficialTrifectaOdds)
                and exc.odds_count == 60
                and confirmed_result_boats == 5
            )
            if confirmed_non_six:
                counters["excluded_non_six_boat"] += 1
                record_attempt(
                    conn,
                    race_id=row["race_id"],
                    status="excluded_non_six_boat",
                    error="confirmed five-boat race: official odds contain 60/120",
                    source_key=OFFICIAL_SOURCE_KEY,
                )
            else:
                counters["invalid"] += 1
                reason = type(exc).__name__
                _increment_counter(counters, "invalid_reason_counts", reason)
                _increment_counter(
                    counters,
                    "invalid_confirmed_result_boats_counts",
                    confirmed_result_boats,
                )
                odds_count = None
                if isinstance(exc, IncompleteOfficialTrifectaOdds):
                    odds_count = exc.odds_count
                    _increment_counter(
                        counters, "incomplete_odds_count_counts", odds_count
                    )
                _record_failure_example(
                    counters,
                    row,
                    status="invalid",
                    error=f"{reason}: {exc}",
                    odds_count=odds_count,
                    confirmed_result_boats=confirmed_result_boats,
                )
                record_attempt(
                    conn,
                    race_id=row["race_id"],
                    status="invalid",
                    error=f"{type(exc).__name__}: {exc}"[:500],
                    source_key=OFFICIAL_SOURCE_KEY,
                )
        conn.commit()
        if index + 1 < len(targets) and sleep_seconds > 0:
            time.sleep(sleep_seconds)
    counters["remaining"] = len(
        pending_races(
            conn,
            from_date=from_date,
            through_date=through_date,
            source_key=OFFICIAL_SOURCE_KEY,
        )
    )
    return counters


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rate-limited backfill of official historical closing odds"
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--from-date", required=True)
    parser.add_argument("--through-date", required=True)
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from_day = date.fromisoformat(args.from_date)
    through_day = date.fromisoformat(args.through_date)
    if from_day > through_day:
        raise ValueError("from-date must not be after through-date")
    if not 0.5 <= args.sleep_seconds <= 60.0:
        raise ValueError("sleep-seconds must be in [0.5, 60]")
    if args.max_pages is not None and args.max_pages < 1:
        raise ValueError("max-pages must be positive")
    with connection(args.db) as conn:
        result = backfill_official_closing_odds(
            conn,
            from_date=from_day.isoformat(),
            through_date=through_day.isoformat(),
            sleep_seconds=args.sleep_seconds,
            max_pages=args.max_pages,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
