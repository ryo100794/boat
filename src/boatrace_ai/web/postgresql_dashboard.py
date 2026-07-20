from __future__ import annotations

import argparse
import os
from pathlib import Path

from .. import postgresql
from . import dashboard


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Serve the BOAT RACE dashboard from PostgreSQL."
    )
    parser.add_argument(
        "--postgres-dsn",
        default=os.environ.get(
            "BOATRACE_POSTGRES_DSN",
            "host=127.0.0.1 port=5432 dbname=boatrace user=boatrace_app",
        ),
    )
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=10001, type=int)
    parser.add_argument(
        "--backtest",
        default="data/models/backtest_no_odds_v8.json",
        type=Path,
    )
    args = parser.parse_args(argv)

    # Dashboard functions accept a path for locating report artifacts. Database
    # calls are routed to PostgreSQL while that path remains rooted in data-dir.
    dashboard.connect = lambda _path: postgresql.connection(args.postgres_dsn)
    dashboard.init_db = lambda _path: None
    artifact_anchor = args.data_dir / "boatrace.sqlite"
    print(
        f"Serving BOAT RACE AI Dashboard on http://{args.host}:{args.port} "
        "using PostgreSQL",
        flush=True,
    )
    dashboard.serve(
        db_path=artifact_anchor,
        host=args.host,
        port=args.port,
        backtest_path=args.backtest,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
