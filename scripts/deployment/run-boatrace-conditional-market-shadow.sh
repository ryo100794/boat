#!/usr/bin/env bash
set -Eeuo pipefail

CONF=/workspace/shared-services/conf/boatrace-collector.env
[[ -f "$CONF" ]] && source "$CONF"

APP_ROOT="${BOATRACE_APP_ROOT:-/workspace/boat}"
PYTHON="${BOATRACE_PYTHON:-$APP_ROOT/.venv/bin/python}"
LOCK="${BOATRACE_CONDITIONAL_MARKET_SHADOW_LOCK:-$APP_ROOT/run/conditional-market-shadow.lock}"
DSN="${BOATRACE_POSTGRES_DSN:-host=127.0.0.1 port=5432 dbname=boatrace user=boatrace_app}"
PG_BIN=/workspace/postgresql/runtime/bin
export LD_LIBRARY_PATH="$PG_BIN/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

install -d -m 0750 "$(dirname "$LOCK")" "$APP_ROOT/data/models" "$APP_ROOT/logs/runtime"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "Conditional stagewise market shadow cycle is already running" >&2
  exit 75
fi

model="$APP_ROOT/data/models/conditional_stagewise_holdout.joblib"
[[ -f "$model" ]] || { echo "conditional stagewise model not found: $model" >&2; exit 2; }

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
  --model "$model" \
  --output "$APP_ROOT/data/models/conditional_stagewise_market_shadow.json" \
  --state "$APP_ROOT/data/models/conditional_stagewise_market_shadow_state.json" \
  --scored-cache "$APP_ROOT/data/models/conditional_stagewise_market_residual.races.joblib" \
  --from-date 2026-07-18 \
  --daily-budget-yen 10000 \
  --min-calibration-days 2 \
  --calibrator-strategy newton_residual \
  --max-snapshot-age-seconds 60 \
  --interval 3600 \
  --timeout 3600
