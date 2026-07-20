#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import psycopg


QUERIES = {
    "race_range_count": """
        SELECT count(*) FROM races
        WHERE race_date BETWEEN :start_date AND :end_date
    """,
    "daily_dashboard": """
        SELECT r.race_id, r.jcd, r.rno, r.deadline_at,
               count(DISTINCT e.lane) AS entry_count,
               count(DISTINCT rr.lane) AS result_count
        FROM races r
        LEFT JOIN entries e ON e.race_id = r.race_id
        LEFT JOIN race_results rr
          ON rr.race_id = r.race_id AND rr.rank IS NOT NULL
        WHERE r.race_date = :end_date
        GROUP BY r.race_id, r.jcd, r.rno, r.deadline_at
        ORDER BY r.jcd, r.rno
    """,
    "training_join": """
        SELECT count(*) AS sample_count,
               avg(e.national_win_rate) AS mean_win_rate,
               sum(CASE WHEN rr.rank = 1 THEN 1 ELSE 0 END) AS wins
        FROM entries e
        JOIN races r ON r.race_id = e.race_id
        JOIN race_results rr
          ON rr.race_id = e.race_id AND rr.lane = e.lane
        WHERE r.race_date BETWEEN :start_date AND :end_date
    """,
    "racer_aggregate": """
        SELECT e.racer_no, count(*) AS sample_count,
               avg(e.national_win_rate) AS mean_win_rate,
               sum(CASE WHEN rr.rank = 1 THEN 1 ELSE 0 END) AS wins
        FROM entries e
        JOIN races r ON r.race_id = e.race_id
        JOIN race_results rr
          ON rr.race_id = e.race_id AND rr.lane = e.lane
        WHERE r.race_date BETWEEN :start_date AND :end_date
        GROUP BY e.racer_no
        ORDER BY sample_count DESC
        LIMIT 100
    """,
    "payout_aggregate": """
        SELECT r.jcd, p.bet_type, count(*) AS payout_count,
               avg(p.payout_yen) AS mean_payout
        FROM payouts p
        JOIN races r ON r.race_id = p.race_id
        WHERE r.race_date BETWEEN :start_date AND :end_date
        GROUP BY r.jcd, p.bet_type
        ORDER BY r.jcd, p.bet_type
    """,
    "prediction_daily": """
        SELECT p.race_id, max(p.generated_at) AS latest_prediction,
               max(p.probability) AS highest_probability
        FROM predictions p
        JOIN races r ON r.race_id = p.race_id
        WHERE r.race_date = :end_date
        GROUP BY p.race_id
        ORDER BY p.race_id
    """,
}


def read_password(path: Path, database: str, role: str) -> str:
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        _host, _port, dbname, username, password = line.split(":", 4)
        if dbname in {"*", database} and username in {"*", role}:
            return password
    raise RuntimeError(f"no credential for {database}/{role} in {path}")


def summarize(samples: list[float], rows: int) -> dict:
    return {
        "runs": len(samples),
        "rows": rows,
        "median_ms": round(statistics.median(samples), 3),
        "min_ms": round(min(samples), 3),
        "max_ms": round(max(samples), 3),
        "samples_ms": [round(value, 3) for value in samples],
    }


def measure(execute, params: dict[str, str], runs: int) -> dict:
    output = {}
    for name, query in QUERIES.items():
        execute(query, params)
        samples = []
        rows = 0
        for _ in range(runs):
            started = time.perf_counter()
            result = execute(query, params)
            samples.append((time.perf_counter() - started) * 1000)
            rows = len(result)
        output[name] = summarize(samples, rows)
    return output


def sqlite_results(path: Path, params: dict[str, str], runs: int) -> dict:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True, timeout=60)
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA busy_timeout=60000")
    connection.execute("PRAGMA mmap_size=1073741824")

    def execute(query: str, values: dict[str, str]):
        return connection.execute(query, values).fetchall()

    try:
        return measure(execute, params, runs)
    finally:
        connection.close()


def postgresql_results(args, params: dict[str, str], runs: int) -> dict:
    password = read_password(args.pgpass, args.database, args.role)
    connection = psycopg.connect(
        host=args.host,
        port=args.port,
        dbname=args.database,
        user=args.role,
        password=password,
        connect_timeout=30,
        application_name="database_backend_benchmark",
    )
    connection.execute("SET default_transaction_read_only=on")
    connection.execute("SET statement_timeout=0")
    connection.commit()

    def execute(query: str, values: dict[str, str]):
        pg_query = query.replace(":start_date", "%(start_date)s").replace(
            ":end_date", "%(end_date)s"
        )
        return connection.execute(pg_query, values).fetchall()

    try:
        return measure(execute, params, runs)
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", required=True, type=Path)
    parser.add_argument("--pgpass", required=True, type=Path)
    parser.add_argument("--database", default="boatrace")
    parser.add_argument("--role", default="boatrace_app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=5432, type=int)
    parser.add_argument("--start-date", default="2016-07-18")
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--runs", default=3, type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    params = {"start_date": args.start_date, "end_date": args.end_date}
    sqlite = sqlite_results(args.sqlite, params, args.runs)
    postgresql = postgresql_results(args, params, args.runs)
    payload = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "range": params,
        "runs_after_warmup": args.runs,
        "measurement": "client_elapsed_including_ssh_tunnel",
        "sqlite": {"path": str(args.sqlite.resolve()), "results": sqlite},
        "postgresql": {
            "target": f"{args.host}:{args.port}/{args.database}",
            "results": postgresql,
        },
        "median_speedup_postgresql_over_sqlite": {
            name: round(sqlite[name]["median_ms"] / postgresql[name]["median_ms"], 3)
            for name in QUERIES
        },
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(args.output)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
