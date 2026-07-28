from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from bs4 import BeautifulSoup

from .db import connection
from .http import FetchError, fetch_text
from .odds_quality import TRIFECTA_COMBINATION_KEYS, plausible_trifecta_odds


SOURCE_KEY = "kyotei_club_official_mirror_closing_v1"
PARSER_VERSION = "archive_closing_odds_dom_v2"
DEFAULT_BASE_URL = "https://odds.kyotei24.jp"


POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS archive_closing_odds_snapshots (
  race_id TEXT NOT NULL REFERENCES races(race_id) ON DELETE CASCADE,
  source_key TEXT NOT NULL,
  fetched_at TIMESTAMPTZ NOT NULL,
  source_url TEXT NOT NULL,
  payload_sha256 TEXT NOT NULL,
  parser_version TEXT NOT NULL,
  odds_count INTEGER NOT NULL,
  verification_status TEXT NOT NULL,
  raw_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  PRIMARY KEY (race_id, source_key)
);
CREATE TABLE IF NOT EXISTS archive_closing_odds (
  race_id TEXT NOT NULL,
  source_key TEXT NOT NULL,
  combination TEXT NOT NULL,
  odds DOUBLE PRECISION NOT NULL,
  PRIMARY KEY (race_id, source_key, combination),
  FOREIGN KEY (race_id, source_key)
    REFERENCES archive_closing_odds_snapshots(race_id, source_key)
    ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS archive_closing_odds_attempts (
  race_id TEXT NOT NULL REFERENCES races(race_id) ON DELETE CASCADE,
  source_key TEXT NOT NULL,
  attempted_at TIMESTAMPTZ NOT NULL,
  attempt_count INTEGER NOT NULL DEFAULT 1,
  status TEXT NOT NULL,
  http_status INTEGER,
  error TEXT,
  PRIMARY KEY (race_id, source_key)
);
CREATE INDEX IF NOT EXISTS idx_archive_closing_odds_source
  ON archive_closing_odds_snapshots(source_key, fetched_at);
"""


SQLITE_SCHEMA = POSTGRES_SCHEMA.replace("TIMESTAMPTZ", "TEXT").replace(
    " JSONB", " TEXT"
).replace("DOUBLE PRECISION", "REAL").replace("'{}'::jsonb", "'{}'")


def archive_url(race_date: str | date, jcd: str, rno: int, *, base_url: str = DEFAULT_BASE_URL) -> str:
    day = date.fromisoformat(str(race_date)[:10])
    return (
        f"{base_url.rstrip('/')}/od-{day:%Y%m%d}-{str(jcd).zfill(2)}-"
        f"{int(rno)}.html"
    )


def parse_archive_closing_odds_html(html: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    race_container = soup.find(id="brOddslist")
    header = (race_container or soup).find(
        lambda tag: tag.name in {"td", "th"}
        and "mainTopHeadline3t" in (tag.get("class") or [])
        and tag.get_text(" ", strip=True) == "3連単"
    )
    if header is not None:
        label = (race_container or soup).find(
            string=lambda value: value and "締切時オッズ" in value
        )
        if label is None:
            raise ValueError("closing odds label is missing")
        container = header.find_parent("table")
        if container is None:
            raise ValueError("trifecta odds table is malformed")
    else:
        archive_header = soup.find(
            lambda tag: tag.name in {"h2", "h3", "h4"}
            and tag.get_text(" ", strip=True) == "3連単"
        )
        container = archive_header.find_parent(id="oddsData") if archive_header else None
        if container is None:
            raise ValueError("trifecta odds table is missing")

    odds: dict[str, float] = {}
    for row in container.find_all("tr"):
        lanes = [node.get_text(strip=True) for node in row.select(".rgs3 .rb")]
        if len(lanes) != 3 or any(value not in "123456" for value in lanes):
            continue
        cells = [
            cell
            for cell in row.find_all("td", recursive=False)
            if "od_text" in (cell.get("class") or [])
            and str(cell.get("align") or "").lower() == "right"
        ]
        if len(cells) != 1:
            continue
        combination = "-".join(lanes)
        try:
            value = float(cells[0].get_text("", strip=True).replace(",", ""))
        except ValueError as exc:
            raise ValueError(f"invalid odds for {combination}") from exc
        if combination in odds:
            raise ValueError(f"duplicate odds combination: {combination}")
        odds[combination] = value

    if not plausible_trifecta_odds(odds):
        missing = sorted(set(TRIFECTA_COMBINATION_KEYS) - set(odds))
        raise ValueError(
            f"archive trifecta odds are incomplete or implausible: "
            f"count={len(odds)} missing={missing[:3]}"
        )
    return {
        "parser_version": PARSER_VERSION,
        "market_time": "closing",
        "source_key": SOURCE_KEY,
        "odds_count": len(odds),
        "odds": odds,
    }


def verify_winning_payout(
    odds: Mapping[str, float], *, combination: str, payout_yen: int
) -> dict[str, Any]:
    if combination not in odds:
        raise ValueError("winning combination is absent from archive odds")
    expected = int(round(float(odds[combination]) * 100.0))
    actual = int(payout_yen)
    if expected != actual:
        raise ValueError(
            f"winning payout mismatch: odds={odds[combination]} "
            f"expected={expected} actual={actual}"
        )
    return {
        "status": "winner_payout_match",
        "winning_combination": combination,
        "winning_odds": float(odds[combination]),
        "payout_yen": actual,
    }


def ensure_archive_schema(conn: Any) -> None:
    conn.executescript(
        POSTGRES_SCHEMA
        if getattr(conn, "dialect", None) == "postgresql"
        else SQLITE_SCHEMA
    )


def store_archive_closing_odds(
    conn: Any,
    *,
    race_id: str,
    source_url: str,
    payload: bytes,
    parsed: Mapping[str, Any],
    verification: Mapping[str, Any],
    fetched_at: str | None = None,
) -> None:
    odds = {str(key): float(value) for key, value in parsed["odds"].items()}
    if not plausible_trifecta_odds(odds):
        raise ValueError("refusing to store implausible archive odds")
    ensure_archive_schema(conn)
    fetched = fetched_at or datetime.now(timezone.utc).isoformat()
    raw = {
        "source_kind": "secondary_archive_of_official_closing_display",
        "market_time": "closing",
        "verification": dict(verification),
    }
    conn.execute(
        """
        INSERT INTO archive_closing_odds_snapshots(
          race_id, source_key, fetched_at, source_url, payload_sha256,
          parser_version, odds_count, verification_status, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(race_id, source_key) DO UPDATE SET
          fetched_at = excluded.fetched_at,
          source_url = excluded.source_url,
          payload_sha256 = excluded.payload_sha256,
          parser_version = excluded.parser_version,
          odds_count = excluded.odds_count,
          verification_status = excluded.verification_status,
          raw_json = excluded.raw_json
        """,
        (
            race_id,
            SOURCE_KEY,
            fetched,
            source_url,
            hashlib.sha256(payload).hexdigest(),
            PARSER_VERSION,
            len(odds),
            str(verification["status"]),
            json.dumps(raw, ensure_ascii=False, sort_keys=True),
        ),
    )
    conn.execute(
        "DELETE FROM archive_closing_odds WHERE race_id = ? AND source_key = ?",
        (race_id, SOURCE_KEY),
    )
    conn.executemany(
        """
        INSERT INTO archive_closing_odds(race_id, source_key, combination, odds)
        VALUES (?, ?, ?, ?)
        """,
        [(race_id, SOURCE_KEY, key, value) for key, value in sorted(odds.items())],
    )


def record_attempt(
    conn: Any,
    *,
    race_id: str,
    status: str,
    http_status: int | None = None,
    error: str | None = None,
) -> None:
    ensure_archive_schema(conn)
    conn.execute(
        """
        INSERT INTO archive_closing_odds_attempts(
          race_id, source_key, attempted_at, attempt_count, status,
          http_status, error
        ) VALUES (?, ?, CURRENT_TIMESTAMP, 1, ?, ?, ?)
        ON CONFLICT(race_id, source_key) DO UPDATE SET
          attempted_at = CURRENT_TIMESTAMP,
          attempt_count = archive_closing_odds_attempts.attempt_count + 1,
          status = excluded.status,
          http_status = excluded.http_status,
          error = excluded.error
        """,
        (race_id, SOURCE_KEY, status, http_status, error),
    )


def pending_races(
    conn: Any, *, from_date: str, through_date: str
) -> list[dict[str, Any]]:
    ensure_archive_schema(conn)
    rows = conn.execute(
        """
        SELECT r.race_id, r.race_date, r.jcd, r.rno,
               p.combination, p.payout_yen
        FROM races r
        JOIN payouts p ON p.race_id = r.race_id
        WHERE r.race_date BETWEEN ? AND ?
          AND p.bet_type = '3連単'
          AND p.payout_yen IS NOT NULL
          AND NOT EXISTS (
            SELECT 1 FROM archive_closing_odds_snapshots a
            WHERE a.race_id = r.race_id AND a.source_key = ?
          )
        ORDER BY r.race_date DESC, r.deadline_at DESC, r.jcd DESC, r.rno DESC
        """,
        (from_date, through_date, SOURCE_KEY),
    ).fetchall()
    return [{key: row[key] for key in row.keys()} for row in rows]


def backfill_archive_closing_odds(
    conn: Any,
    *,
    from_date: str,
    through_date: str,
    sleep_seconds: float = 1.0,
    max_pages: int | None = None,
    base_url: str = DEFAULT_BASE_URL,
) -> dict[str, Any]:
    targets = pending_races(conn, from_date=from_date, through_date=through_date)
    if max_pages is not None:
        targets = targets[: max(0, int(max_pages))]
    counters = {
        "status": "completed",
        "source_key": SOURCE_KEY,
        "source_role": "secondary_archive_research_only",
        "from_date": from_date,
        "through_date": through_date,
        "targets": len(targets),
        "stored": 0,
        "not_found": 0,
        "invalid": 0,
        "fetch_failed": 0,
    }
    for index, row in enumerate(targets):
        url = archive_url(
            row["race_date"], row["jcd"], int(row["rno"]), base_url=base_url
        )
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
                    conn, race_id=row["race_id"], status="not_found", http_status=404
                )
            elif status_code != 200:
                counters["fetch_failed"] += 1
                record_attempt(
                    conn,
                    race_id=row["race_id"],
                    status="http_error",
                    http_status=status_code,
                )
            else:
                parsed = parse_archive_closing_odds_html(html)
                verification = verify_winning_payout(
                    parsed["odds"],
                    combination=str(row["combination"]),
                    payout_yen=int(row["payout_yen"]),
                )
                store_archive_closing_odds(
                    conn,
                    race_id=str(row["race_id"]),
                    source_url=url,
                    payload=payload,
                    parsed=parsed,
                    verification=verification,
                )
                record_attempt(conn, race_id=row["race_id"], status="stored")
                counters["stored"] += 1
        except (FetchError, OSError) as exc:
            counters["fetch_failed"] += 1
            record_attempt(
                conn,
                race_id=row["race_id"],
                status="fetch_error",
                error=f"{type(exc).__name__}: {exc}"[:500],
            )
        except (KeyError, TypeError, ValueError) as exc:
            counters["invalid"] += 1
            record_attempt(
                conn,
                race_id=row["race_id"],
                status="invalid",
                error=f"{type(exc).__name__}: {exc}"[:500],
            )
        conn.commit()
        if index + 1 < len(targets) and sleep_seconds > 0:
            time.sleep(sleep_seconds)
    counters["remaining"] = len(
        pending_races(conn, from_date=from_date, through_date=through_date)
    )
    return counters


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rate-limited backfill of independently stored closing odds"
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--from-date", required=True)
    parser.add_argument("--through-date", required=True)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
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
        result = backfill_archive_closing_odds(
            conn,
            from_date=from_day.isoformat(),
            through_date=through_day.isoformat(),
            sleep_seconds=args.sleep_seconds,
            max_pages=args.max_pages,
            base_url=args.base_url,
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
