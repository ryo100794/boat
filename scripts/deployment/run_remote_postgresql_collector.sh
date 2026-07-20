#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/workspace/shared-services/boatrace-runtime
CONF="$ROOT/conf/postgresql-runtime.env"
[[ -f "$CONF" ]] && source "$CONF"

APP_ROOT="${BOATRACE_APP_ROOT:-/workspace/boat}"
PYTHON="${BOATRACE_PYTHON:-$APP_ROOT/.venv/bin/python}"
RAW="${BOATRACE_RAW_DIR:-$ROOT/data/raw}"
LOG="${BOATRACE_COLLECTOR_LOG:-$ROOT/log/collector.log}"
LOCK="${BOATRACE_COLLECTOR_LOCK:-$ROOT/run/collector.lock}"
RESTART_DELAY="${BOATRACE_RESTART_DELAY:-10}"
DSN="${BOATRACE_POSTGRES_DSN:-host=127.0.0.1 port=5432 dbname=boatrace user=boatrace_app}"

install -d -m 0750 "$RAW" "$ROOT/log" "$ROOT/run"
touch "$LOG"
exec 9>"$LOCK"
flock -n 9 || exit 0

child_pid=""
stopping=0
shutdown() {
  [[ "$stopping" == 1 ]] && return
  stopping=1
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill -TERM "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
}
trap shutdown TERM INT EXIT

while [[ "$stopping" == 0 ]]; do
  echo "$(date -u +%FT%TZ) PostgreSQL collector start" >>"$LOG"
  PGPASSFILE="${BOATRACE_PGPASSFILE:-/workspace/shared-services/postgresql/conf/databases/boatrace.pgpass}" \
  PYTHONPATH="$APP_ROOT/src" "$PYTHON" -m boatrace_ai.runtime.postgresql_collector \
    --postgres-dsn "$DSN" \
    --model "$APP_ROOT/data/models/win_model_no_odds_v8.joblib" \
    --raw-dir "$RAW" \
    --sleep-loop 10 \
    --sleep-page 0.4 \
    --collect-results >>"$LOG" 2>&1 &
  child_pid=$!
  status=0
  wait "$child_pid" || status=$?
  child_pid=""
  [[ "$stopping" == 1 ]] && break
  echo "$(date -u +%FT%TZ) collector exited status=$status; restart in ${RESTART_DELAY}s" >>"$LOG"
  sleep "$RESTART_DELAY"
done
