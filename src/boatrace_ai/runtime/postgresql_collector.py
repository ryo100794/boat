from __future__ import annotations

import argparse

from . import collector
from ..postgresql import connection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--postgres-dsn", required=True)
    args, collector_args = parser.parse_known_args(argv)

    collector.connection = lambda _path: connection(args.postgres_dsn)
    collector.init_db = lambda _path: None
    return collector.main(["--db", "postgresql-direct", *collector_args])


if __name__ == "__main__":
    raise SystemExit(main())
