from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .db import connection, init_db
from .series_form import entry_series_features


SERIES_CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS entry_series_features (
  race_id TEXT NOT NULL,
  lane INTEGER NOT NULL,
  series_starts REAL,
  series_avg_finish REAL,
  series_latest_finish REAL,
  series_best_finish REAL,
  series_worst_finish REAL,
  series_win_rate REAL,
  series_top2_rate REAL,
  series_top3_rate REAL,
  series_finish_trend REAL,
  series_accident_count REAL,
  series_has_f INTEGER,
  series_has_l INTEGER,
  series_has_s INTEGER,
  series_has_accident INTEGER,
  series_has_results INTEGER,
  has_early_look INTEGER,
  early_look_rno REAL,
  early_look_gap REAL,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (race_id, lane)
);
CREATE INDEX IF NOT EXISTS idx_entry_series_features_race ON entry_series_features(race_id);
"""


CACHE_FIELDS = (
    "series_starts",
    "series_avg_finish",
    "series_latest_finish",
    "series_best_finish",
    "series_worst_finish",
    "series_win_rate",
    "series_top2_rate",
    "series_top3_rate",
    "series_finish_trend",
    "series_accident_count",
    "series_has_f",
    "series_has_l",
    "series_has_s",
    "series_has_accident",
    "series_has_results",
    "has_early_look",
    "early_look_rno",
    "early_look_gap",
)


def ensure_series_cache_table(conn) -> None:
    if getattr(conn, "dialect", None) == "postgresql":
        conn.execute("SELECT 1 FROM entry_series_features LIMIT 0")
        return
    conn.executescript(SERIES_CACHE_SCHEMA)


def populate_series_cache(
    conn,
    *,
    batch_size: int = 5000,
    limit: int | None = None,
    from_date: str | None = None,
    refresh_all: bool = False,
) -> dict[str, Any]:
    ensure_series_cache_table(conn)
    is_postgresql = getattr(conn, "dialect", None) == "postgresql"
    filters = [
        "jsonb_extract_path(CAST(e.raw_json AS jsonb), 'series_results') IS NOT NULL"
        if is_postgresql
        else "e.raw_json LIKE ?"
    ]
    params: list[Any] = [] if is_postgresql else ["%series_results%"]
    if not refresh_all:
        filters.append(
            "(sf.race_id IS NULL OR CAST(sf.updated_at AS timestamp) < CAST(e.updated_at AS timestamp))"
            if is_postgresql
            else "(sf.race_id IS NULL OR datetime(sf.updated_at) < datetime(e.updated_at))"
        )
    if from_date:
        filters.append("r.race_date >= ?")
        params.append(from_date)
    limit_sql = ""
    if limit is not None:
        limit_sql = "LIMIT ?"
        params.append(int(limit))
    rows = conn.execute(
        f"""
        SELECT r.rno, e.race_id, e.lane, e.raw_json
        FROM entries e
        JOIN races r ON r.race_id = e.race_id
        LEFT JOIN entry_series_features sf
          ON sf.race_id = e.race_id AND sf.lane = e.lane
        WHERE {" AND ".join(filters)}
        ORDER BY r.race_date DESC, r.jcd DESC, r.rno DESC, e.lane
        {limit_sql}
        """,
        params,
    )
    total = 0
    buffer: list[dict[str, Any]] = []
    for row in rows:
        features = entry_series_features(row)
        item = {"race_id": row["race_id"], "lane": int(row["lane"])}
        item.update({field: features.get(field) for field in CACHE_FIELDS})
        buffer.append(item)
        if len(buffer) >= batch_size:
            _flush(conn, buffer)
            total += len(buffer)
            print(json.dumps({"cached": total}, ensure_ascii=False), flush=True)
            buffer.clear()
    if buffer:
        _flush(conn, buffer)
        total += len(buffer)
    conn.commit()
    return {
        "cached": total,
        "from_date": from_date,
        "refresh_all": refresh_all,
    }


def _flush(conn, rows: list[dict[str, Any]]) -> None:
    cols = ("race_id", "lane", *CACHE_FIELDS)
    placeholders = ", ".join(f":{col}" for col in cols)
    update_cols = ", ".join(f"{col}=excluded.{col}" for col in CACHE_FIELDS)
    conn.executemany(
        f"""
        INSERT INTO entry_series_features ({", ".join(cols)})
        VALUES ({placeholders})
        ON CONFLICT(race_id, lane) DO UPDATE SET
          {update_cols},
          updated_at=CURRENT_TIMESTAMP
        """,
        rows,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cache official series-form features into a compact table.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--from-date")
    parser.add_argument("--refresh-all", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    init_db(args.db)
    with connection(args.db) as conn:
        result = populate_series_cache(
            conn,
            batch_size=args.batch_size,
            limit=args.limit,
            from_date=args.from_date,
            refresh_all=args.refresh_all,
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
