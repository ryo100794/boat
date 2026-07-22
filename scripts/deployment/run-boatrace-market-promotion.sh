#!/usr/bin/env bash
set -Eeuo pipefail

APP_ROOT="${BOATRACE_APP_ROOT:-/workspace/boat}"
PYTHON="${BOATRACE_PYTHON:-$APP_ROOT/.venv/bin/python}"
LOCK="${BOATRACE_MARKET_PROMOTION_LOCK:-$APP_ROOT/run/market-promotion.lock}"

install -d -m 0750 "$(dirname "$LOCK")" "$APP_ROOT/data/models" "$APP_ROOT/logs/runtime"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "Market promotion cycle is already running" >&2
  exit 75
fi

exec env \
  PYTHONPATH="$APP_ROOT/src" \
  "$PYTHON" -m boatrace_ai.runtime.market_promotion_cycle \
  --candidate "$APP_ROOT/data/models/listwise_market_calibrated_shadow.json" \
  --candidate "$APP_ROOT/data/models/listwise_market_calibrated_cutoff_shadow.json" \
  --candidate "$APP_ROOT/data/models/listwise_market_residual_shadow.json" \
  --candidate "$APP_ROOT/data/models/stagewise_blend_market_shadow.json" \
  --output "$APP_ROOT/data/models/active_market_model.json" \
  --state "$APP_ROOT/data/models/market_promotion_cycle_state.json" \
  --interval 3600
