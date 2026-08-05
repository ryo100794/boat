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
from .ingestion.backfill import backfill_historical
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


def _confirmed_dead_heat(conn: Any, race_id: str) -> bool:
    row = conn.execute(
        """
        SELECT COUNT(DISTINCT lane) AS boats,
               COUNT(DISTINCT rank) AS ranks
        FROM race_results
        WHERE race_id = ? AND rank IS NOT NULL
        """,
        (race_id,),
    ).fetchone()
    return bool(
        row is not None
        and int(row["boats"] or 0) == 6
        and int(row["ranks"] or 0) < 6
    )


def _trifecta_settlements(conn: Any, race_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT combination, payout_yen, popularity
        FROM payouts
        WHERE race_id = ? AND bet_type = '3連単' AND payout_yen IS NOT NULL
        ORDER BY combination, payout_yen
        """,
        (race_id,),
    ).fetchall()
    return [
        {
            "combination": str(row["combination"]),
            "payout_yen": int(row["payout_yen"]),
            "popularity": row["popularity"],
        }
        for row in rows
    ]


def _verify_special_settlement(
    conn: Any,
    *,
    race_id: str,
    odds: dict[str, float],
) -> dict[str, Any]:
    settlements = _trifecta_settlements(conn, race_id)
    if len(settlements) < 2 or not _confirmed_dead_heat(conn, race_id):
        raise ValueError("multiple payouts require a confirmed six-boat dead heat")
    missing = [
        row["combination"] for row in settlements if row["combination"] not in odds
    ]
    if missing:
        raise ValueError(
            "special-settlement combinations absent from odds: "
            f"{missing}"
        )
    return {
        "status": "official_primary_special_settlement",
        "payout_match_mode": "confirmed_dead_heat_multiple_payouts",
        "settlement_count": len(settlements),
        "settlements": settlements,
    }


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
          AND (
            SELECT COUNT(DISTINCT rr.lane)
            FROM race_results rr
            WHERE rr.race_id = archive_closing_odds_attempts.race_id
              AND rr.rank IS NOT NULL
          ) BETWEEN 1 AND 5
        """,
        (OFFICIAL_SOURCE_KEY,),
    )
    return max(0, int(cursor.rowcount or 0))


def _incomplete_dead_heat_dates(conn: Any) -> list[str]:
    rows = conn.execute(
        """
        SELECT DISTINCT r.race_date
        FROM races r
        WHERE (
            SELECT COUNT(DISTINCT rr.lane)
            FROM race_results rr
            WHERE rr.race_id = r.race_id AND rr.rank IS NOT NULL
        ) = 6
          AND EXISTS (
            SELECT 1
            FROM race_results tied
            WHERE tied.race_id = r.race_id AND tied.rank IS NOT NULL
            GROUP BY tied.rank
            HAVING COUNT(*) > 1
          )
          AND (
            SELECT COUNT(*)
            FROM payouts p
            WHERE p.race_id = r.race_id
              AND p.bet_type = '3連単'
              AND p.payout_yen IS NOT NULL
          ) = 1
        ORDER BY r.race_date
        """
    ).fetchall()
    return [str(row["race_date"])[:10] for row in rows]


def repair_incomplete_dead_heat_payouts(
    conn: Any,
    *,
    raw_dir: Path = Path("data/raw"),
    sleep_seconds: float = 0.5,
) -> dict[str, Any]:
    target_dates = _incomplete_dead_heat_dates(conn)
    before = len(target_dates)
    downloaded_dates = 0
    failed_dates = 0
    failure_examples: list[dict[str, str]] = []
    for value in target_dates:
        target_date = date.fromisoformat(value)
        try:
            stats = backfill_historical(
                conn,
                start=target_date,
                end=target_date,
                kind="result",
                raw_dir=raw_dir,
                sleep_seconds=sleep_seconds,
                skip_existing=False,
            )
            downloaded_dates += int(stats.downloaded > 0)
        except (FetchError, OSError, ValueError) as exc:
            failed_dates += 1
            if len(failure_examples) < 25:
                failure_examples.append(
                    {
                        "race_date": value,
                        "error": f"{type(exc).__name__}: {exc}"[:500],
                    }
                )
    remaining_dates = _incomplete_dead_heat_dates(conn)
    return {
        "target_dates": before,
        "downloaded_dates": downloaded_dates,
        "failed_dates": failed_dates,
        "repaired_dates": max(0, before - len(remaining_dates)),
        "remaining_dates": len(remaining_dates),
        "failure_examples": failure_examples,
    }


def reclassify_confirmed_special_settlement_attempts(conn: Any) -> int:
    ensure_archive_schema(conn)
    cursor = conn.execute(
        """
        UPDATE archive_closing_odds_attempts
        SET status = 'excluded_special_settlement'
        WHERE source_key = ? AND status = 'invalid'
          AND (
            SELECT COUNT(DISTINCT rr.lane)
            FROM race_results rr
            WHERE rr.race_id = archive_closing_odds_attempts.race_id
              AND rr.rank IS NOT NULL
          ) = 6
          AND EXISTS (
            SELECT 1
            FROM race_results tied
            WHERE tied.race_id = archive_closing_odds_attempts.race_id
              AND tied.rank IS NOT NULL
            GROUP BY tied.rank
            HAVING COUNT(*) > 1
          )
          AND (
            SELECT COUNT(*)
            FROM payouts p
            WHERE p.race_id = archive_closing_odds_attempts.race_id
              AND p.bet_type = '3連単'
              AND p.payout_yen IS NOT NULL
          ) > 1
        """,
        (OFFICIAL_SOURCE_KEY,),
    )
    return max(0, int(cursor.rowcount or 0))


def backfill_official_closing_odds(
    conn: Any,
    *,
    from_date: str,
    through_date: str,
    sleep_seconds: float = 0.5,
    max_pages: int | None = None,
    special_settlements_only: bool = False,
) -> dict[str, Any]:
    dead_heat_repair = repair_incomplete_dead_heat_payouts(
        conn, sleep_seconds=sleep_seconds
    )
    reclassified_special = reclassify_confirmed_special_settlement_attempts(conn)
    reclassified = reclassify_confirmed_non_six_boat_attempts(conn)
    targets = pending_races(
        conn,
        from_date=from_date,
        through_date=through_date,
        source_key=OFFICIAL_SOURCE_KEY,
        include_multi_payout=True,
    )
    if special_settlements_only:
        targets = [
            row for row in targets if int(row.get("payout_count") or 0) > 1
        ]
    if max_pages is not None:
        targets = targets[: max(0, int(max_pages))]
    counters = {
        "status": "completed",
        "source_key": OFFICIAL_SOURCE_KEY,
        "source_role": "primary_official_historical_closing",
        "from_date": from_date,
        "through_date": through_date,
        "targets": len(targets),
        "special_settlements_only": bool(special_settlements_only),
        "stored": 0,
        "stored_special_settlement": 0,
        "not_found": 0,
        "invalid": 0,
        "fetch_failed": 0,
        "excluded_non_six_boat": 0,
        "reclassified_non_six_boat": reclassified,
        "reclassified_special_settlement": reclassified_special,
        "dead_heat_payout_repair": dead_heat_repair,
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
                if int(row.get("payout_count") or 1) > 1:
                    verification = _verify_special_settlement(
                        conn,
                        race_id=str(row["race_id"]),
                        odds=parsed["odds"],
                    )
                else:
                    verification = verify_winning_payout(
                        parsed["odds"],
                        combination=str(row["combination"]),
                        payout_yen=int(row["payout_yen"]),
                    )
                    verification["status"] = (
                        "official_primary_winner_payout_match"
                    )
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
                counters["stored_special_settlement"] += int(
                    verification["status"]
                    == "official_primary_special_settlement"
                )
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
            confirmed_non_six = 0 < confirmed_result_boats < 6
            if confirmed_non_six:
                counters["excluded_non_six_boat"] += 1
                record_attempt(
                    conn,
                    race_id=row["race_id"],
                    status="excluded_non_six_boat",
                    error=(
                        "confirmed non-six-boat race: "
                        f"{confirmed_result_boats} result boats"
                    ),
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
            include_multi_payout=True,
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
    parser.add_argument("--special-settlements-only", action="store_true")
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
            special_settlements_only=args.special_settlements_only,
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
