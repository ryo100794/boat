from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .constants import VENUES


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS venues (
  code TEXT PRIMARY KEY,
  name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS races (
  race_id TEXT PRIMARY KEY,
  race_date TEXT NOT NULL,
  jcd TEXT NOT NULL,
  venue_name TEXT NOT NULL,
  rno INTEGER NOT NULL,
  title TEXT,
  race_type TEXT,
  distance_m INTEGER,
  deadline_at TEXT,
  status TEXT NOT NULL DEFAULT 'scheduled',
  source_url TEXT,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE (race_date, jcd, rno)
);

CREATE TABLE IF NOT EXISTS entries (
  race_id TEXT NOT NULL,
  lane INTEGER NOT NULL,
  racer_no INTEGER,
  racer_name TEXT,
  racer_class TEXT,
  branch TEXT,
  origin TEXT,
  age INTEGER,
  weight_kg REAL,
  f_count INTEGER,
  l_count INTEGER,
  avg_st REAL,
  national_win_rate REAL,
  national_2_rate REAL,
  national_3_rate REAL,
  local_win_rate REAL,
  local_2_rate REAL,
  local_3_rate REAL,
  motor_no INTEGER,
  motor_2_rate REAL,
  motor_3_rate REAL,
  boat_no INTEGER,
  boat_2_rate REAL,
  boat_3_rate REAL,
  raw_json TEXT,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (race_id, lane),
  FOREIGN KEY (race_id) REFERENCES races(race_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS odds_snapshots (
  snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
  race_id TEXT NOT NULL,
  bet_type TEXT NOT NULL,
  captured_at TEXT NOT NULL,
  source_update_time TEXT,
  raw_json TEXT,
  source_url TEXT,
  FOREIGN KEY (race_id) REFERENCES races(race_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS odds_trifecta (
  snapshot_id INTEGER NOT NULL,
  race_id TEXT NOT NULL,
  combination TEXT NOT NULL,
  odds REAL,
  PRIMARY KEY (snapshot_id, combination),
  FOREIGN KEY (snapshot_id) REFERENCES odds_snapshots(snapshot_id) ON DELETE CASCADE,
  FOREIGN KEY (race_id) REFERENCES races(race_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS beforeinfo (
  race_id TEXT NOT NULL,
  captured_at TEXT NOT NULL,
  lane INTEGER NOT NULL,
  weight_kg REAL,
  exhibition_time REAL,
  tilt REAL,
  adjusted_weight REAL,
  propeller TEXT,
  parts_exchange TEXT,
  course INTEGER,
  start_timing REAL,
  weather TEXT,
  wind_direction TEXT,
  wind_speed_m REAL,
  air_temp_c REAL,
  water_temp_c REAL,
  wave_cm REAL,
  raw_json TEXT,
  PRIMARY KEY (race_id, captured_at, lane),
  FOREIGN KEY (race_id) REFERENCES races(race_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS race_results (
  race_id TEXT NOT NULL,
  lane INTEGER NOT NULL,
  rank INTEGER,
  course INTEGER,
  start_timing REAL,
  race_time TEXT,
  decision TEXT,
  raw_json TEXT,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (race_id, lane),
  FOREIGN KEY (race_id) REFERENCES races(race_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS race_result_status (
  race_id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  trifecta_evaluable INTEGER NOT NULL DEFAULT 1,
  reason TEXT,
  finish_rows INTEGER NOT NULL DEFAULT 0,
  payout_rows INTEGER NOT NULL DEFAULT 0,
  raw_json TEXT,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (race_id) REFERENCES races(race_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS payouts (
  race_id TEXT NOT NULL,
  bet_type TEXT NOT NULL,
  combination TEXT NOT NULL,
  payout_yen INTEGER,
  popularity INTEGER,
  raw_json TEXT,
  PRIMARY KEY (race_id, bet_type, combination),
  FOREIGN KEY (race_id) REFERENCES races(race_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS racer_period_stats (
  year INTEGER NOT NULL,
  half INTEGER NOT NULL,
  racer_no INTEGER NOT NULL,
  racer_name TEXT,
  racer_class TEXT,
  raw_json TEXT NOT NULL,
  PRIMARY KEY (year, half, racer_no)
);

CREATE TABLE IF NOT EXISTS raw_files (
  raw_file_id INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT NOT NULL,
  source_url TEXT NOT NULL,
  local_path TEXT NOT NULL,
  race_date TEXT,
  year INTEGER,
  half INTEGER,
  status_code INTEGER,
  sha256 TEXT,
  bytes INTEGER,
  fetched_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE (kind, source_url)
);

CREATE TABLE IF NOT EXISTS raw_pages (
  raw_page_id INTEGER PRIMARY KEY AUTOINCREMENT,
  page_type TEXT NOT NULL,
  race_id TEXT,
  source_url TEXT NOT NULL,
  local_path TEXT NOT NULL,
  sha256 TEXT,
  bytes INTEGER,
  fetched_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
  prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
  race_id TEXT NOT NULL,
  generated_at TEXT NOT NULL,
  model_path TEXT,
  combination TEXT NOT NULL,
  probability REAL NOT NULL,
  odds REAL,
  expected_value REAL,
  raw_json TEXT,
  FOREIGN KEY (race_id) REFERENCES races(race_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_entries_racer_no ON entries(racer_no);
CREATE INDEX IF NOT EXISTS idx_odds_race_snapshot ON odds_snapshots(race_id, captured_at);
CREATE INDEX IF NOT EXISTS idx_results_rank ON race_results(race_id, rank);
CREATE INDEX IF NOT EXISTS idx_result_status_eval ON race_result_status(status, trifecta_evaluable);
CREATE INDEX IF NOT EXISTS idx_predictions_race ON predictions(race_id, generated_at);
"""


def race_id(race_date: str, jcd: str, rno: int) -> str:
    return f"{race_date}-{jcd.zfill(2)}-{int(rno):02d}"


def connect(path: str | Path) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def connection(path: str | Path) -> Iterator[sqlite3.Connection]:
    conn = connect(path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(path: str | Path, *, attempts: int = 6, retry_seconds: float = 5.0) -> None:
    for attempt in range(attempts):
        try:
            with connection(path) as conn:
                conn.executescript(SCHEMA)
                conn.executemany(
                    "INSERT OR IGNORE INTO venues(code, name) VALUES (?, ?)",
                    [(venue.code, venue.name) for venue in VENUES],
                )
            return
        except sqlite3.OperationalError as exc:
            locked = "locked" in str(exc).lower() or "busy" in str(exc).lower()
            if not locked or attempt + 1 >= attempts:
                raise
            time.sleep(retry_seconds)


def upsert_race(conn: sqlite3.Connection, payload: dict[str, Any]) -> str:
    rid = payload.get("race_id") or race_id(
        payload["race_date"], payload["jcd"], payload["rno"]
    )
    values = {
        "race_id": rid,
        "race_date": payload["race_date"],
        "jcd": payload["jcd"].zfill(2),
        "venue_name": payload["venue_name"],
        "rno": int(payload["rno"]),
        "title": payload.get("title"),
        "race_type": payload.get("race_type"),
        "distance_m": payload.get("distance_m"),
        "deadline_at": payload.get("deadline_at"),
        "status": payload.get("status", "scheduled"),
        "source_url": payload.get("source_url"),
    }
    conn.execute(
        """
        INSERT INTO races (
          race_id, race_date, jcd, venue_name, rno, title, race_type,
          distance_m, deadline_at, status, source_url
        )
        VALUES (
          :race_id, :race_date, :jcd, :venue_name, :rno, :title, :race_type,
          :distance_m, :deadline_at, :status, :source_url
        )
        ON CONFLICT(race_id) DO UPDATE SET
          title=COALESCE(excluded.title, races.title),
          race_type=COALESCE(excluded.race_type, races.race_type),
          distance_m=COALESCE(excluded.distance_m, races.distance_m),
          deadline_at=COALESCE(excluded.deadline_at, races.deadline_at),
          status=CASE WHEN races.status = "final" THEN races.status ELSE COALESCE(excluded.status, races.status) END,
          source_url=COALESCE(excluded.source_url, races.source_url),
          updated_at=CURRENT_TIMESTAMP
        """,
        values,
    )
    return rid


def upsert_entry(conn: sqlite3.Connection, race_id_value: str, entry: dict[str, Any]) -> None:
    payload = dict(entry)
    payload["race_id"] = race_id_value
    payload["raw_json"] = json.dumps(entry, ensure_ascii=False, sort_keys=True)
    conn.execute(
        """
        INSERT INTO entries (
          race_id, lane, racer_no, racer_name, racer_class, branch, origin, age,
          weight_kg, f_count, l_count, avg_st, national_win_rate,
          national_2_rate, national_3_rate, local_win_rate, local_2_rate,
          local_3_rate, motor_no, motor_2_rate, motor_3_rate, boat_no,
          boat_2_rate, boat_3_rate, raw_json
        )
        VALUES (
          :race_id, :lane, :racer_no, :racer_name, :racer_class, :branch,
          :origin, :age, :weight_kg, :f_count, :l_count, :avg_st,
          :national_win_rate, :national_2_rate, :national_3_rate,
          :local_win_rate, :local_2_rate, :local_3_rate, :motor_no,
          :motor_2_rate, :motor_3_rate, :boat_no, :boat_2_rate,
          :boat_3_rate, :raw_json
        )
        ON CONFLICT(race_id, lane) DO UPDATE SET
          racer_no=COALESCE(excluded.racer_no, entries.racer_no),
          racer_name=COALESCE(excluded.racer_name, entries.racer_name),
          racer_class=COALESCE(excluded.racer_class, entries.racer_class),
          branch=COALESCE(excluded.branch, entries.branch),
          origin=COALESCE(excluded.origin, entries.origin),
          age=COALESCE(excluded.age, entries.age),
          weight_kg=COALESCE(excluded.weight_kg, entries.weight_kg),
          f_count=COALESCE(excluded.f_count, entries.f_count),
          l_count=COALESCE(excluded.l_count, entries.l_count),
          avg_st=COALESCE(excluded.avg_st, entries.avg_st),
          national_win_rate=COALESCE(excluded.national_win_rate, entries.national_win_rate),
          national_2_rate=COALESCE(excluded.national_2_rate, entries.national_2_rate),
          national_3_rate=COALESCE(excluded.national_3_rate, entries.national_3_rate),
          local_win_rate=COALESCE(excluded.local_win_rate, entries.local_win_rate),
          local_2_rate=COALESCE(excluded.local_2_rate, entries.local_2_rate),
          local_3_rate=COALESCE(excluded.local_3_rate, entries.local_3_rate),
          motor_no=COALESCE(excluded.motor_no, entries.motor_no),
          motor_2_rate=COALESCE(excluded.motor_2_rate, entries.motor_2_rate),
          motor_3_rate=COALESCE(excluded.motor_3_rate, entries.motor_3_rate),
          boat_no=COALESCE(excluded.boat_no, entries.boat_no),
          boat_2_rate=COALESCE(excluded.boat_2_rate, entries.boat_2_rate),
          boat_3_rate=COALESCE(excluded.boat_3_rate, entries.boat_3_rate),
          raw_json=excluded.raw_json,
          updated_at=CURRENT_TIMESTAMP
        """,
        payload,
    )


def insert_odds_snapshot(
    conn: sqlite3.Connection,
    race_id_value: str,
    captured_at: str,
    source_update_time: str | None,
    odds: dict[str, float | None],
    source_url: str,
    raw: dict[str, Any],
) -> int:
    conn.execute(
        """
        INSERT INTO odds_snapshots (
          race_id, bet_type, captured_at, source_update_time, raw_json, source_url
        )
        VALUES (?, 'trifecta', ?, ?, ?, ?)
        """,
        (
            race_id_value,
            captured_at,
            source_update_time,
            json.dumps(raw, ensure_ascii=False, sort_keys=True),
            source_url,
        ),
    )
    snapshot_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
    conn.executemany(
        """
        INSERT OR REPLACE INTO odds_trifecta(
          snapshot_id, race_id, combination, odds
        )
        VALUES (?, ?, ?, ?)
        """,
        [
            (snapshot_id, race_id_value, combination, value)
            for combination, value in odds.items()
        ],
    )
    return snapshot_id


def insert_prediction_rows(
    conn: sqlite3.Connection,
    race_id_value: str,
    generated_at: str,
    model_path: str | None,
    rows: list[dict[str, Any]],
) -> None:
    conn.executemany(
        """
        INSERT INTO predictions (
          race_id, generated_at, model_path, combination, probability,
          odds, expected_value, raw_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                race_id_value,
                generated_at,
                model_path,
                row["combination"],
                row["probability"],
                row.get("odds"),
                row.get("expected_value"),
                json.dumps(row, ensure_ascii=False, sort_keys=True),
            )
            for row in rows
        ],
    )
