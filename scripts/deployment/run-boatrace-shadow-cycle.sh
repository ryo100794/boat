#!/usr/bin/env bash
set -Eeuo pipefail

CONF=/workspace/shared-services/conf/boatrace-collector.env
[[ -f "$CONF" ]] && source "$CONF"

APP_ROOT="${BOATRACE_APP_ROOT:-/workspace/boat}"
PYTHON="${BOATRACE_PYTHON:-$APP_ROOT/.venv/bin/python}"
LOCK="${BOATRACE_SHADOW_LOCK:-$APP_ROOT/run/realtime-odds-shadow.lock}"
DSN="${BOATRACE_POSTGRES_DSN:-host=127.0.0.1 port=5432 dbname=boatrace user=boatrace_app}"
SHADOW_MODEL="${BOATRACE_SHADOW_MODEL:-$APP_ROOT/data/models/realtime_odds_shadow.joblib}"
SHADOW_BACKTEST="${BOATRACE_SHADOW_BACKTEST:-$APP_ROOT/data/models/realtime_odds_shadow_backtest.json}"
SHADOW_STATE="${BOATRACE_SHADOW_STATE:-$APP_ROOT/data/models/realtime_odds_shadow_state.json}"
SHADOW_FROM_DATE="${BOATRACE_SHADOW_FROM_DATE:-2026-07-18}"
SHADOW_TARGET_RACES="${BOATRACE_SHADOW_TARGET_RACES:-1000}"
SHADOW_MIN_TRAIN_RACES="${BOATRACE_SHADOW_MIN_TRAIN_RACES:-500}"
SHADOW_MIN_NEW_RACES="${BOATRACE_SHADOW_MIN_NEW_RACES:-50}"
SHADOW_FOLDS="${BOATRACE_SHADOW_FOLDS:-5}"
SHADOW_INTERVAL="${BOATRACE_SHADOW_INTERVAL:-900}"
PG_BIN=/workspace/postgresql/runtime/bin
export LD_LIBRARY_PATH="$PG_BIN/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

install -d -m 0750 "$(dirname "$LOCK")" "$APP_ROOT/data/models"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "Realtime odds shadow cycle is already running" >&2
  exit 75
fi

for _attempt in $(seq 1 120); do
  if "$PG_BIN/pg_isready" -h 127.0.0.1 -p 5432 -d boatrace -q; then
    break
  fi
  sleep 1
done
"$PG_BIN/pg_isready" -h 127.0.0.1 -p 5432 -d boatrace -t 5

echo "$(date -u +%FT%TZ) Realtime odds shadow cycle start"
exec env \
  PGPASSFILE="${BOATRACE_PGPASSFILE:-/workspace/postgresql/conf/databases/boatrace.pgpass}" \
  PYTHONPATH="$APP_ROOT/src" \
  "$PYTHON" -m boatrace_ai.runtime.model_cycle \
  --db "$DSN" \
  --model "$SHADOW_MODEL" \
  --backtest "$SHADOW_BACKTEST" \
  --state "$SHADOW_STATE" \
  --from-date "$SHADOW_FROM_DATE" \
  --include-odds \
  --min-odds-snapshots 10 \
  --min-eligible-races "$SHADOW_TARGET_RACES" \
  --min-train-races "$SHADOW_MIN_TRAIN_RACES" \
  --min-new-races "$SHADOW_MIN_NEW_RACES" \
  --folds "$SHADOW_FOLDS" \
  --interval "$SHADOW_INTERVAL"
