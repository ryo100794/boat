#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import psycopg


BASELINES = {
    "predictions": ("prediction_id", 18_886_410),
    "odds_snapshots": ("snapshot_id", 52_577),
    "raw_pages": ("raw_page_id", 76_008),
    "raw_files": ("raw_file_id", 7_308),
}
RECENT_TABLES = (
    "races",
    "entries",
    "beforeinfo",
    "race_results",
    "race_result_status",
    "payouts",
    "entry_series_features",
)


def columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')]


def primary_keys(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    return [str(row[1]) for row in sorted(rows, key=lambda row: int(row[5])) if row[5]]


def copy_rows(cursor, stage: str, names: list[str], rows: Iterable[Any]) -> int:
    count = 0
    quoted = ", ".join(f'"{name}"' for name in names)
    with cursor.copy(f'COPY "{stage}" ({quoted}) FROM STDIN') as copy:
        for row in rows:
            copy.write_row(tuple(row))
            count += 1
    return count


def recent_where(table: str) -> tuple[str, tuple[Any, ...]]:
    if table == "races":
        return "race_date >= ?", ("2026-07-19",)
    return "race_id IN (SELECT race_id FROM races WHERE race_date >= ?)", ("2026-07-19",)


def merge_recent(sqlite_conn, pg_conn, table: str) -> dict[str, Any]:
    names = columns(sqlite_conn, table)
    keys = primary_keys(sqlite_conn, table)
    if not names or not keys:
        return {"table": table, "staged": 0, "affected": 0, "skipped": "no schema or primary key"}
    where, params = recent_where(table)
    stage = f"sync_{table}"
    quoted = ", ".join(f'"{name}"' for name in names)
    key_sql = ", ".join(f'"{name}"' for name in keys)
    updates = [name for name in names if name not in keys]
    assignments = ", ".join(
        f'"{name}" = COALESCE(EXCLUDED."{name}", "{table}"."{name}")'
        for name in updates
    )
    with pg_conn.cursor() as cursor:
        cursor.execute(f'DROP TABLE IF EXISTS "{stage}"')
        cursor.execute(
            f'CREATE TEMP TABLE "{stage}" AS SELECT {quoted} FROM "{table}" WITH NO DATA'
        )
        rows = sqlite_conn.execute(f'SELECT {quoted} FROM "{table}" WHERE {where}', params)
        staged = copy_rows(cursor, stage, names, rows)
        if not staged:
            return {"table": table, "staged": 0, "affected": 0}
        conflict = f"DO UPDATE SET {assignments}" if assignments else "DO NOTHING"
        if "updated_at" in names:
            conflict += (
                f' WHERE "{table}"."updated_at" IS NULL OR '
                f'EXCLUDED."updated_at" >= "{table}"."updated_at"'
            )
        cursor.execute(
            f'INSERT INTO "{table}" ({quoted}) SELECT {quoted} FROM "{stage}" '
            f'ON CONFLICT ({key_sql}) {conflict}'
        )
        affected = cursor.rowcount
    return {"table": table, "staged": staged, "affected": affected}


def merge_raw_files(sqlite_conn, pg_conn) -> dict[str, Any]:
    table = "raw_files"
    id_name, baseline = BASELINES[table]
    names = [name for name in columns(sqlite_conn, table) if name != id_name]
    quoted = ", ".join(f'"{name}"' for name in names)
    stage = "sync_raw_files"
    with pg_conn.cursor() as cursor:
        cursor.execute(f'CREATE TEMP TABLE "{stage}" AS SELECT {quoted} FROM "{table}" WITH NO DATA')
        rows = sqlite_conn.execute(
            f'SELECT {quoted} FROM "{table}" WHERE "{id_name}" > ?', (baseline,)
        )
        staged = copy_rows(cursor, stage, names, rows)
        if staged:
            updates = ", ".join(
                f'"{name}"=EXCLUDED."{name}"'
                for name in names
                if name not in {"kind", "source_url"}
            )
            cursor.execute(
                f'INSERT INTO "{table}" ({quoted}) SELECT {quoted} FROM "{stage}" '
                f'ON CONFLICT (kind, source_url) DO UPDATE SET {updates}'
            )
            affected = cursor.rowcount
        else:
            affected = 0
    return {"table": table, "staged": staged, "affected": affected}


def merge_raw_pages(sqlite_conn, pg_conn) -> dict[str, Any]:
    table = "raw_pages"
    id_name, baseline = BASELINES[table]
    names = [name for name in columns(sqlite_conn, table) if name != id_name]
    quoted = ", ".join(f'"{name}"' for name in names)
    stage = "sync_raw_pages"
    with pg_conn.cursor() as cursor:
        cursor.execute(f'CREATE TEMP TABLE "{stage}" AS SELECT {quoted} FROM "{table}" WITH NO DATA')
        staged = copy_rows(
            cursor,
            stage,
            names,
            sqlite_conn.execute(
                f'SELECT {quoted} FROM "{table}" WHERE "{id_name}" > ?', (baseline,)
            ),
        )
        cursor.execute(
            f'INSERT INTO "{table}" ({quoted}) '
            f'SELECT {", ".join(f"s.\"{name}\"" for name in names)} FROM "{stage}" s '
            'WHERE NOT EXISTS (SELECT 1 FROM raw_pages t '
            'WHERE t.page_type=s.page_type AND t.source_url=s.source_url '
            'AND t.local_path=s.local_path AND t.fetched_at=s.fetched_at '
            'AND t.sha256 IS NOT DISTINCT FROM s.sha256)'
        )
        inserted = cursor.rowcount
    return {"table": table, "staged": staged, "inserted": inserted}


def merge_predictions(sqlite_conn, pg_conn) -> dict[str, Any]:
    table = "predictions"
    id_name, baseline = BASELINES[table]
    names = [name for name in columns(sqlite_conn, table) if name != id_name]
    quoted = ", ".join(f'"{name}"' for name in names)
    stage = "sync_predictions"
    with pg_conn.cursor() as cursor:
        cursor.execute(f'CREATE TEMP TABLE "{stage}" AS SELECT {quoted} FROM "{table}" WITH NO DATA')
        staged = copy_rows(
            cursor,
            stage,
            names,
            sqlite_conn.execute(
                f'SELECT {quoted} FROM "{table}" WHERE "{id_name}" > ?', (baseline,)
            ),
        )
        cursor.execute('CREATE INDEX ON sync_predictions (race_id, generated_at)')
        cursor.execute(
            f'INSERT INTO "{table}" ({quoted}) '
            f'SELECT {", ".join(f"s.\"{name}\"" for name in names)} FROM "{stage}" s '
            'WHERE NOT EXISTS (SELECT 1 FROM predictions t '
            'WHERE t.race_id=s.race_id AND t.generated_at=s.generated_at '
            'AND t.combination=s.combination '
            'AND t.model_path IS NOT DISTINCT FROM s.model_path)'
        )
        inserted = cursor.rowcount
        cursor.execute(
            'SELECT count(*) FROM sync_predictions s WHERE NOT EXISTS '
            '(SELECT 1 FROM predictions t WHERE t.race_id=s.race_id '
            'AND t.generated_at=s.generated_at AND t.combination=s.combination '
            'AND t.model_path IS NOT DISTINCT FROM s.model_path)'
        )
        missing = int(cursor.fetchone()[0])
    return {"table": table, "staged": staged, "inserted": inserted, "missing": missing}


def merge_odds(sqlite_conn, pg_conn) -> dict[str, Any]:
    id_name, baseline = BASELINES["odds_snapshots"]
    names = [name for name in columns(sqlite_conn, "odds_snapshots") if name != id_name]
    quoted = ", ".join(f'"{name}"' for name in names)
    with pg_conn.cursor() as cursor:
        cursor.execute(
            f'CREATE TEMP TABLE sync_odds_snapshots '
            f'(local_snapshot_id bigint, LIKE odds_snapshots INCLUDING DEFAULTS)'
        )
        snapshot_names = ["local_snapshot_id", *names]
        snapshot_rows = sqlite_conn.execute(
            f'SELECT "{id_name}", {quoted} FROM odds_snapshots WHERE "{id_name}" > ?',
            (baseline,),
        )
        staged_snapshots = copy_rows(cursor, "sync_odds_snapshots", snapshot_names, snapshot_rows)
        match = (
            't.race_id=s.race_id AND t.bet_type=s.bet_type AND t.captured_at=s.captured_at '
            'AND t.source_update_time IS NOT DISTINCT FROM s.source_update_time '
            'AND t.source_url IS NOT DISTINCT FROM s.source_url'
        )
        cursor.execute(
            f'INSERT INTO odds_snapshots ({quoted}) '
            f'SELECT {", ".join(f"s.\"{name}\"" for name in names)} '
            'FROM sync_odds_snapshots s WHERE NOT EXISTS '
            f'(SELECT 1 FROM odds_snapshots t WHERE {match})'
        )
        inserted_snapshots = cursor.rowcount
        cursor.execute(
            'CREATE TEMP TABLE sync_snapshot_map AS '
            'SELECT s.local_snapshot_id, min(t.snapshot_id) AS target_snapshot_id '
            'FROM sync_odds_snapshots s JOIN odds_snapshots t ON '
            f'{match} GROUP BY s.local_snapshot_id'
        )
        cursor.execute('SELECT count(*) FROM sync_snapshot_map')
        mapped = int(cursor.fetchone()[0])
        odds_names = columns(sqlite_conn, "odds_trifecta")
        cursor.execute(
            'CREATE TEMP TABLE sync_odds_trifecta '
            '(local_snapshot_id bigint, race_id text, combination text, odds double precision)'
        )
        staged_odds = copy_rows(
            cursor,
            "sync_odds_trifecta",
            ["local_snapshot_id", "race_id", "combination", "odds"],
            sqlite_conn.execute(
                'SELECT snapshot_id, race_id, combination, odds FROM odds_trifecta '
                'WHERE snapshot_id > ?',
                (baseline,),
            ),
        )
        cursor.execute(
            'INSERT INTO odds_trifecta (snapshot_id, race_id, combination, odds) '
            'SELECT m.target_snapshot_id, o.race_id, o.combination, o.odds '
            'FROM sync_odds_trifecta o JOIN sync_snapshot_map m '
            'ON m.local_snapshot_id=o.local_snapshot_id '
            'ON CONFLICT (snapshot_id, combination) DO UPDATE SET odds=EXCLUDED.odds'
        )
        affected_odds = cursor.rowcount
    return {
        "table": "odds_snapshots+odds_trifecta",
        "staged_snapshots": staged_snapshots,
        "inserted_snapshots": inserted_snapshots,
        "mapped_snapshots": mapped,
        "staged_odds": staged_odds,
        "affected_odds": affected_odds,
        "missing_snapshot_maps": staged_snapshots - mapped,
    }


def reset_sequences(pg_conn) -> None:
    with pg_conn.cursor() as cursor:
        for table, column in (
            ("odds_snapshots", "snapshot_id"),
            ("raw_pages", "raw_page_id"),
            ("raw_files", "raw_file_id"),
            ("predictions", "prediction_id"),
        ):
            cursor.execute(
                "SELECT setval(pg_get_serial_sequence(%s, %s), "
                f'COALESCE((SELECT max("{column}") FROM "{table}"), 1), true)',
                (table, column),
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge a frozen SQLite collector delta into PostgreSQL.")
    parser.add_argument("--sqlite", default="data/boatrace.sqlite", type=Path)
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    sqlite_conn = sqlite3.connect(f"file:{args.sqlite.resolve()}?mode=ro", uri=True)
    sqlite_conn.row_factory = sqlite3.Row
    sqlite_conn.execute("PRAGMA query_only=ON")
    operations: list[dict[str, Any]] = []
    try:
        with psycopg.connect(args.dsn, connect_timeout=30) as pg_conn:
            pg_conn.execute("SET synchronous_commit=off")
            for table in RECENT_TABLES:
                if columns(sqlite_conn, table):
                    operations.append(merge_recent(sqlite_conn, pg_conn, table))
                    pg_conn.commit()
            operations.append(merge_raw_files(sqlite_conn, pg_conn))
            pg_conn.commit()
            operations.append(merge_raw_pages(sqlite_conn, pg_conn))
            pg_conn.commit()
            operations.append(merge_odds(sqlite_conn, pg_conn))
            pg_conn.commit()
            operations.append(merge_predictions(sqlite_conn, pg_conn))
            pg_conn.commit()
            reset_sequences(pg_conn)
            pg_conn.commit()
    finally:
        sqlite_conn.close()
    passed = all(
        not item.get("missing") and not item.get("missing_snapshot_maps")
        for item in operations
    )
    report = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source": str(args.sqlite.resolve()),
        "target": args.dsn,
        "baselines": BASELINES,
        "operations": operations,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "passed": passed,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
