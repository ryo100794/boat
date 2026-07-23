#!/usr/bin/env bash
set -Eeuo pipefail

CONF=/workspace/shared-services/conf/boatrace-collector.env
[[ -f "$CONF" ]] && source "$CONF"

APP_ROOT="${BOATRACE_APP_ROOT:-/workspace/boat}"
PYTHON="${BOATRACE_PYTHON:-$APP_ROOT/.venv/bin/python}"
LOCK="${BOATRACE_MARKET_SHADOW_LOCK:-$APP_ROOT/run/market-calibrated-shadow.lock}"
DSN="${BOATRACE_POSTGRES_DSN:-host=127.0.0.1 port=5432 dbname=boatrace user=boatrace_app}"
PG_BIN=/workspace/postgresql/runtime/bin
export LD_LIBRARY_PATH="$PG_BIN/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

install -d -m 0750 "$(dirname "$LOCK")" "$APP_ROOT/data/models" "$APP_ROOT/logs/runtime"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "Market-calibrated shadow cycle is already running" >&2
  exit 75
fi

for _attempt in $(seq 1 120); do
  if "$PG_BIN/pg_isready" -h 127.0.0.1 -p 5432 -d boatrace -q; then
    break
  fi
  sleep 1
done
"$PG_BIN/pg_isready" -h 127.0.0.1 -p 5432 -d boatrace -t 5

exec env \
  PGPASSFILE="${BOATRACE_PGPASSFILE:-/workspace/postgresql/conf/databases/boatrace.pgpass}" \
  PYTHONPATH="$APP_ROOT/src" \
  "$PYTHON" -m boatrace_ai.runtime.market_shadow_cycle \
  --db "$DSN" \
  --model "$APP_ROOT/data/models/listwise_newton_cg_v1.joblib" \
  --output "$APP_ROOT/data/models/listwise_market_calibrated_shadow.json" \
  --state "$APP_ROOT/data/models/listwise_market_calibrated_shadow_state.json" \
  --from-date 2026-07-18 \
  --daily-budget-yen 10000 \
  --min-calibration-days 2 \
  --max-snapshot-age-seconds 65 \
  --interval 3600 \
  --timeout 3600
