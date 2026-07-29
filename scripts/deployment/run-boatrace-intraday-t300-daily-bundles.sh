#!/usr/bin/env bash
set -Eeuo pipefail

APP_ROOT="${BOATRACE_APP_ROOT:-/workspace/boat}"
STATE_ROOT="${BOATRACE_DAILY_MODEL_STATE_ROOT:-$APP_ROOT/data/runtime/daily-shadow-models}"
SPEC_ENV="$STATE_ROOT/active/model-spec.env"
[[ -r "$SPEC_ENV" ]] || { echo "verified model spec unavailable: $SPEC_ENV" >&2; exit 1; }
source "$SPEC_ENV"
[[ "${BOATRACE_T300_SHADOW_REAL_BETTING_ENABLED:-1}" == 0 ]] || {
  echo "daily bundle wrapper permits shadow operation only" >&2; exit 1;
}
export BOATRACE_T300_SHADOW_MODEL_SPEC BOATRACE_T300_SHADOW_EXTRA_MODEL_SPECS
export BOATRACE_T300_SHADOW_DATE
exec "$APP_ROOT/scripts/deployment/run-boatrace-intraday-t300-shadow.sh"
