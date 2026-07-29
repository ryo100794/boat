#!/usr/bin/env bash
set -Eeuo pipefail

APP_ROOT="${BOATRACE_APP_ROOT:-/workspace/boat}"
PYTHON="${BOATRACE_PYTHON:-$APP_ROOT/.venv/bin/python}"
DSN="${BOATRACE_POSTGRES_DSN:-host=127.0.0.1 port=5432 dbname=boatrace user=boatrace_app}"
MODEL_SPEC="${BOATRACE_T300_SHADOW_MODEL_SPEC:?set MODEL_KEY:STRATEGY:BUNDLE_JOBLIB:BASE_MODEL_JOBLIB}"
PG_BIN=/workspace/postgresql/runtime/bin
export LD_LIBRARY_PATH="$PG_BIN/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

install -d -m 0750 "$APP_ROOT/logs/runtime"
args=(
  --db "$DSN"
  --model-spec "$MODEL_SPEC"
  --interval "${BOATRACE_T300_SHADOW_INTERVAL_SECONDS:-5}"
  --max-checkpoint-age-seconds "${BOATRACE_T300_MAX_CHECKPOINT_AGE_SECONDS:-90}"
  --max-source-update-staleness-seconds "${BOATRACE_T300_MAX_SOURCE_UPDATE_STALENESS_SECONDS:-120}"
  --starting-bankroll-yen "${BOATRACE_T300_SHADOW_STARTING_BANKROLL_YEN:-10000}"
)
if [[ -n "${BOATRACE_T300_SHADOW_DATE:-}" ]]; then
  args+=(--date "$BOATRACE_T300_SHADOW_DATE")
fi
for extra_spec in ${BOATRACE_T300_SHADOW_EXTRA_MODEL_SPECS:-}; do
  args+=(--model-spec "$extra_spec")
done

exec env \
  PGPASSFILE="${BOATRACE_PGPASSFILE:-/workspace/postgresql/conf/databases/boatrace.pgpass}" \
  PYTHONPATH="$APP_ROOT/src" \
  "$PYTHON" -m boatrace_ai.runtime.intraday_t300_shadow "${args[@]}"
