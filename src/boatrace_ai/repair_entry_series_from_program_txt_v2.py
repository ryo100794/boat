from __future__ import annotations

from . import repair_entry_series_from_program_txt as base
from .historical_official6 import parse_program_entry


base.parse_program_entry = parse_program_entry


if __name__ == "__main__":
    raise SystemExit(base.main())
