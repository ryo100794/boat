#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."

cycle_seconds="${BOATRACE_EVAL_CYCLE_SECONDS:-21600}"
retry_seconds="${BOATRACE_EVAL_RETRY_SECONDS:-900}"

while true; do
  printf 'CYCLE START %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if scripts/run_standardized_365d_evaluations.sh; then
    printf 'CYCLE COMPLETE %s next=%ss\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$cycle_seconds"
    sleep "$cycle_seconds"
  else
    status=$?
    printf 'CYCLE FAILED %s exit=%s retry=%ss\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" "$retry_seconds" >&2
    sleep "$retry_seconds"
  fi
done
