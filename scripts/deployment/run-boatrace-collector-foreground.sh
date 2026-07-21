#!/usr/bin/env bash
set -Eeuo pipefail

CONF=/workspace/shared-services/conf/boatrace-collector.env
[[ -f "$CONF" ]] && source "$CONF"

APP_ROOT="${BOATRACE_APP_ROOT:-/workspace/boat}"
PYTHON="${BOATRACE_PYTHON:-$APP_ROOT/.venv/bin/python}"
RAW="${BOATRACE_RAW_DIR:-$APP_ROOT/data/raw}"
LOCK="${BOATRACE_COLLECTOR_LOCK:-$APP_ROOT/run/postgresql-collector.lock}"
DSN="${BOATRACE_POSTGRES_DSN:-host=127.0.0.1 port=5432 dbname=boatrace user=boatrace_app}"
PG_BIN=/workspace/postgresql/runtime/bin

install -d -m 0750 "$RAW" "$(dirname "$LOCK")"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "PostgreSQL collector is already running" >&2
  exit 75
fi

for _attempt in $(seq 1 120); do
  if "$PG_BIN/pg_isready" -h 127.0.0.1 -p 5432 -d boatrace -q; then
    break
  fi
  sleep 1
done
"$PG_BIN/pg_isready" -h 127.0.0.1 -p 5432 -d boatrace -t 5

echo "$(date -u +%FT%TZ) PostgreSQL collector start"
exec env \
  PGPASSFILE="${BOATRACE_PGPASSFILE:-/workspace/postgresql/conf/databases/boatrace.pgpass}" \
  PYTHONPATH="$APP_ROOT/src" \
  "$PYTHON" -m boatrace_ai.runtime.postgresql_collector \
  --postgres-dsn "$DSN" \
  --model "$APP_ROOT/data/models/win_model_no_odds_v8.joblib" \
  --raw-dir "$RAW" \
  --sleep-loop 10 \
  --sleep-page 0.4 \
  --predict \
  --collect-results
