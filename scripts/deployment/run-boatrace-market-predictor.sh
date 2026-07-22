#!/usr/bin/env bash
set -Eeuo pipefail

CONF=/workspace/shared-services/conf/boatrace-collector.env
[[ -f "$CONF" ]] && source "$CONF"

APP_ROOT="${BOATRACE_APP_ROOT:-/workspace/boat}"
PYTHON="${BOATRACE_PYTHON:-$APP_ROOT/.venv/bin/python}"
LOCK="${BOATRACE_MARKET_PREDICTOR_LOCK:-$APP_ROOT/run/market-predictor.lock}"
DSN="${BOATRACE_POSTGRES_DSN:-host=127.0.0.1 port=5432 dbname=boatrace user=boatrace_app}"
PG_BIN=/workspace/postgresql/runtime/bin
export LD_LIBRARY_PATH="$PG_BIN/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

install -d -m 0750 "$(dirname "$LOCK")" "$APP_ROOT/logs/runtime"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "Promoted market predictor is already running" >&2
  exit 75
fi

exec env \
  PGPASSFILE="${BOATRACE_PGPASSFILE:-/workspace/postgresql/conf/databases/boatrace.pgpass}" \
  PYTHONPATH="$APP_ROOT/src" \
  "$PYTHON" -m boatrace_ai.runtime.market_predictor \
  --db "$DSN" \
  --manifest "$APP_ROOT/data/models/active_market_model.json" \
  --interval 30
