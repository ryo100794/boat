from __future__ import annotations

import argparse
import sys
import time
from contextlib import contextmanager

import psycopg

from . import collector
from ..postgresql import connection


@contextmanager
def retrying_connection(dsn: str, *, retry_seconds: float = 5.0):
    """Keep the collector process and T-5 worker alive during DB outages."""
    manager = None
    conn = None
    while conn is None:
        try:
            manager = connection(dsn)
            conn = manager.__enter__()
        except (psycopg.OperationalError, ConnectionError, OSError):
            time.sleep(retry_seconds)
    try:
        yield conn
    except BaseException:
        assert manager is not None
        manager.__exit__(*sys.exc_info())
        raise
    else:
        assert manager is not None
        manager.__exit__(None, None, None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--postgres-dsn", required=True)
    args, collector_args = parser.parse_known_args(argv)

    collector.connection = lambda _path: retrying_connection(args.postgres_dsn)
    collector.init_db = lambda _path: None
    return collector.main(["--db", "postgresql-direct", *collector_args])


if __name__ == "__main__":
    raise SystemExit(main())
