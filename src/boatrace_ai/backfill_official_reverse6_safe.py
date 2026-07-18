from __future__ import annotations

from . import backfill_official_reverse5_safe as base
from .historical_official5 import parse_official_archive_v5


base.parse_official_archive = parse_official_archive_v5


if __name__ == "__main__":
    raise SystemExit(base.main())
