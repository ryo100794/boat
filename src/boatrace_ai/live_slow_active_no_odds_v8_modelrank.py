from __future__ import annotations

from . import live_slow_active_no_odds_v7 as base
from .modeling_no_odds_v8_modelrank import predict_open_races

base.predict_open_races = predict_open_races
main = base.main


if __name__ == "__main__":
    raise SystemExit(main())

