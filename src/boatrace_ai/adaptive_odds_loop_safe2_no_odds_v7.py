from __future__ import annotations

from .live_safe_patch2 import install
from .modeling_no_odds_v7 import predict_open_races

install()

from . import adaptive_odds_loop

adaptive_odds_loop.predict_open_races = predict_open_races
main = adaptive_odds_loop.main


if __name__ == "__main__":
    raise SystemExit(main())
