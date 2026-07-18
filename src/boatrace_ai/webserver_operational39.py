from __future__ import annotations

from . import webserver_operational38 as base


HTML = base.HTML

base._DEFAULT_HISTORY_DAYS = 180


def main(argv: list[str] | None = None) -> int:
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
