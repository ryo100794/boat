from __future__ import annotations

from . import webserver_operational40 as prev


HTML = prev.HTML

prev._history_base._DEFAULT_HISTORY_DAYS = 90


def main(argv: list[str] | None = None) -> int:
    return prev.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
