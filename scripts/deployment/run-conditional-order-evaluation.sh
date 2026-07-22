#!/usr/bin/env bash
set -Eeuo pipefail

CONF=/workspace/shared-services/conf/boatrace-collector.env
[[ -f "$CONF" ]] && source "$CONF"

APP_ROOT="${BOATRACE_APP_ROOT:-/workspace/boat}"
PYTHON="${BOATRACE_PYTHON:-$APP_ROOT/.venv/bin/python}"
LOCK="${BOATRACE_CONDITIONAL_ORDER_LOCK:-$APP_ROOT/run/conditional-order-evaluation.lock}"
DSN="${BOATRACE_POSTGRES_DSN:-host=127.0.0.1 port=5432 dbname=boatrace user=boatrace_app}"
PG_BIN=/workspace/postgresql/runtime/bin
export LD_LIBRARY_PATH="$PG_BIN/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

install -d -m 0750 "$(dirname "$LOCK")" "$APP_ROOT/data/models" "$APP_ROOT/logs/runtime"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "Conditional-order evaluation is already running" >&2
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
  "$PYTHON" -m boatrace_ai.listwise.conditional_order \
  --db "$DSN" \
  --cache-prefix "$APP_ROOT/data/models/standardized_365d_v2/listwise_search_cache/listwise_search_4096_drop_research_correlates" \
  --baseline-model "$APP_ROOT/data/models/standardized_365d_v2/listwise_newton.joblib" \
  --training-through 2025-07-19 \
  --evaluation-from 2025-07-20 \
  --evaluation-through 2026-07-19 \
  --model-output "$APP_ROOT/data/models/conditional_order_365d.joblib" \
  --output "$APP_ROOT/data/models/conditional_order_365d.json" \
  --validation-days 90 \
  --batch-races 4000 \
  --promote-legacy-cache
