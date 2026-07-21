#!/usr/bin/env bash
set -Eeuo pipefail

CONF=/workspace/shared-services/conf/boatrace-collector.env
[[ -f "$CONF" ]] && source "$CONF"

APP_ROOT="${BOATRACE_APP_ROOT:-/workspace/boat}"
PYTHON="${BOATRACE_PYTHON:-$APP_ROOT/.venv/bin/python}"
LOCK="${BOATRACE_SHADOW_LOCK:-$APP_ROOT/run/realtime-odds-shadow.lock}"
DSN="${BOATRACE_POSTGRES_DSN:-host=127.0.0.1 port=5432 dbname=boatrace user=boatrace_app}"
PG_BIN=/workspace/postgresql/runtime/bin

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
  --model "$APP_ROOT/data/models/realtime_odds_shadow.joblib" \
  --backtest "$APP_ROOT/data/models/realtime_odds_shadow_backtest.json" \
  --state "$APP_ROOT/data/models/realtime_odds_shadow_state.json" \
  --from-date 2026-07-18 \
  --include-odds \
  --min-odds-snapshots 10 \
  --min-eligible-races 1000 \
  --min-train-races 500 \
  --min-new-races 50 \
  --folds 5 \
  --interval 900
