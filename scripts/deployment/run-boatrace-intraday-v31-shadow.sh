#!/usr/bin/env bash
set -Eeuo pipefail

APP_ROOT="${BOATRACE_APP_ROOT:-/workspace/boat}"
STATE_ROOT="${BOATRACE_DAILY_MODEL_STATE_ROOT:-$APP_ROOT/data/runtime/daily-shadow-models}"
SPEC_ENV="$STATE_ROOT/active/model-spec.env"
POLL_SECONDS="${BOATRACE_V31_SHADOW_SPEC_POLL_SECONDS:-10}"
export PYTHONPATH="$APP_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

while true; do
  if [[ -r "$SPEC_ENV" ]]; then
    unset BOATRACE_T300_SHADOW_MODEL_SPEC BOATRACE_T300_SHADOW_EXTRA_MODEL_SPECS
    unset BOATRACE_T300_SHADOW_REAL_BETTING_ENABLED
    source "$SPEC_ENV"
    for spec in ${BOATRACE_T300_SHADOW_MODEL_SPEC:-} ${BOATRACE_T300_SHADOW_EXTRA_MODEL_SPECS:-}; do
      if [[ "$spec" == v21_daily:v21_triple_head_t300:* ]]; then
        export BOATRACE_T300_SHADOW_MODEL_SPEC="v31_daily:v31_uncertainty_adjusted_top5_t300:${spec#v21_daily:v21_triple_head_t300:}"
        export BOATRACE_T300_SHADOW_EXTRA_MODEL_SPECS=""
        export BOATRACE_T300_SHADOW_REAL_BETTING_ENABLED=0
        exec "$APP_ROOT/.venv/bin/python" -m boatrace_ai.runtime.v31_uncertainty_adjusted_shadow
      fi
    done
  fi
  sleep "$POLL_SECONDS"
done
