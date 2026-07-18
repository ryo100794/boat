from __future__ import annotations

from . import live
from . import live_slow_active_no_odds_v7 as base
from .modeling_no_odds_v8_modelrank import predict_open_races
from .racelist_parser_dom import parse_racelist_html

live.parse_racelist_html = parse_racelist_html
base.predict_open_races = predict_open_races
main = base.main


if __name__ == "__main__":
    raise SystemExit(main())

