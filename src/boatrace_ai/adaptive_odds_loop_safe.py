from __future__ import annotations

from .live_safe_patch import install

install()

from .adaptive_odds_loop import main


if __name__ == "__main__":
    raise SystemExit(main())
