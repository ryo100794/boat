from __future__ import annotations

from . import predict_loop_no_odds_v8_modelrank as base
from .modeling_pastlog_v4_modelrank import predict_open_races


base.predict_open_races = predict_open_races


if __name__ == "__main__":
    raise SystemExit(base.main())
