#!/usr/bin/env bash
set -Eeuo pipefail

APP_ROOT="${BOATRACE_APP_ROOT:-/workspace/boat}"
PYTHON="${BOATRACE_PYTHON:-$APP_ROOT/.venv/bin/python}"
DSN="${BOATRACE_POSTGRES_DSN:-host=127.0.0.1 port=5432 dbname=boatrace user=boatrace_app}"
BASE_MODEL="${BOATRACE_T300_SHADOW_BASE_MODEL:?set immutable base model joblib path}"
PG_BIN=/workspace/postgresql/runtime/bin
export LD_LIBRARY_PATH="$PG_BIN/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec env PGPASSFILE="${BOATRACE_PGPASSFILE:-/workspace/postgresql/conf/databases/boatrace.pgpass}" \
  PYTHONPATH="$APP_ROOT/src" "$PYTHON" \
  -m boatrace_ai.runtime.daily_shadow_bundle_update \
  --postgres-dsn "$DSN" --app-root "$APP_ROOT" \
  --output-root "${BOATRACE_DAILY_BUNDLE_ROOT:-$APP_ROOT/data/models/daily-shadow-bundles}" \
  --state-root "${BOATRACE_DAILY_MODEL_STATE_ROOT:-$APP_ROOT/data/runtime/daily-shadow-models}" \
  --base-model "$BASE_MODEL" \
  --interval-seconds "${BOATRACE_DAILY_BUNDLE_INTERVAL_SECONDS:-300}"
