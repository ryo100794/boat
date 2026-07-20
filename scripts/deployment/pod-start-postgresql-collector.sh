#!/usr/bin/env bash
set -Eeuo pipefail

base_pid=""
collector_pid=""
stopping=0
shutdown() {
  [[ "$stopping" == 1 ]] && return
  stopping=1
  if [[ -n "$collector_pid" ]] && kill -0 "$collector_pid" 2>/dev/null; then
    kill -TERM "$collector_pid" 2>/dev/null || true
    wait "$collector_pid" 2>/dev/null || true
  fi
  /workspace/shared-services/bin/stop-postgresql.sh || true
  if [[ -n "$base_pid" ]] && kill -0 "$base_pid" 2>/dev/null; then
    kill -TERM "$base_pid" 2>/dev/null || true
    wait "$base_pid" 2>/dev/null || true
  fi
}
trap shutdown TERM INT EXIT

if [[ -x /start.sh ]]; then
  /start.sh &
  base_pid=$!
else
  sleep infinity &
  base_pid=$!
fi

/workspace/shared-services/bin/start-postgresql.sh
/workspace/shared-services/bin/run-boatrace-collector.sh &
collector_pid=$!
wait "$base_pid"
