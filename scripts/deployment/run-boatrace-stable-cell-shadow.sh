#!/usr/bin/env bash
set -Eeuo pipefail

APP_ROOT="${BOATRACE_APP_ROOT:-/workspace/boat}"
STATE_ROOT="${BOATRACE_DAILY_MODEL_STATE_ROOT:-$APP_ROOT/data/runtime/daily-shadow-models}"
SPEC_ENV="$STATE_ROOT/active/model-spec.env"
POLL_SECONDS="${BOATRACE_STABLE_CELL_SPEC_POLL_SECONDS:-10}"
BUNDLE="${BOATRACE_STABLE_CELL_BUNDLE:-$APP_ROOT/data/models/daily-shadow-bundles/2026-08-02/stable-cell-job-10730.joblib}"
SHADOW_RUNNER="$APP_ROOT/scripts/deployment/run-boatrace-intraday-t300-shadow.sh"

child_pid=""
active_identity=""

stop_child() {
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill -TERM "$child_pid"
    wait "$child_pid" || true
  fi
  child_pid=""
}

shutdown() {
  stop_child
  exit 0
}
trap shutdown TERM INT

while true; do
  if [[ ! -r "$SPEC_ENV" || ! -r "$BUNDLE" ]]; then
    stop_child
    sleep "$POLL_SECONDS" & wait $!
    continue
  fi

  read -r spec_hash _ < <(sha256sum "$SPEC_ENV")
  unset BOATRACE_T300_SHADOW_MODEL_SPEC BOATRACE_T300_SHADOW_EXTRA_MODEL_SPECS
  unset BOATRACE_T300_SHADOW_DATE BOATRACE_T300_SHADOW_REAL_BETTING_ENABLED
  source "$SPEC_ENV"
  v21_spec=""
  for spec in ${BOATRACE_T300_SHADOW_MODEL_SPEC:-} ${BOATRACE_T300_SHADOW_EXTRA_MODEL_SPECS:-}; do
    if [[ "$spec" == v21_daily:v21_triple_head_t300:* ]]; then
      v21_spec="$spec"
      break
    fi
  done
  prediction_date="${BOATRACE_T300_SHADOW_DATE:-}"
  if [[ -z "$v21_spec" || "$prediction_date" < "2026-08-02" || "${BOATRACE_T300_SHADOW_REAL_BETTING_ENABLED:-1}" != 0 ]]; then
    stop_child
    sleep "$POLL_SECONDS" & wait $!
    continue
  fi

  identity="${prediction_date}:${spec_hash}:$(sha256sum "$BUNDLE" | cut -d' ' -f1)"
  if [[ -n "$child_pid" ]] && ! kill -0 "$child_pid" 2>/dev/null; then
    wait "$child_pid" || true
    child_pid=""
    active_identity=""
  fi
  if [[ "$identity" != "$active_identity" ]]; then
    stop_child
    base_model="${v21_spec##*:}"
    export BOATRACE_T300_SHADOW_MODEL_SPEC="stable_cell_daily:stable_cell_top5_lt20_t300:$BUNDLE:$base_model"
    export BOATRACE_T300_SHADOW_EXTRA_MODEL_SPECS=""
    export BOATRACE_T300_SHADOW_DATE="$prediction_date"
    export BOATRACE_T300_SHADOW_REAL_BETTING_ENABLED=0
    export BOATRACE_T300_SHADOW_MODULE="boatrace_ai.runtime.stable_cell_shadow"
    "$SHADOW_RUNNER" &
    child_pid=$!
    active_identity="$identity"
  fi

  sleep "$POLL_SECONDS" & wait $!
done
