#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/workspace/service-manager
install -d -m 0750 "$ROOT/log" "$ROOT/run"
install -d -m 0750 /workspace/postgresql/log /workspace/boat/logs/runtime

if [[ ! -x "$ROOT/.venv/bin/supervisord" ]]; then
  echo "Supervisor runtime is missing: $ROOT/.venv" >&2
  exit 1
fi

exec "$ROOT/.venv/bin/supervisord" \
  -c /workspace/shared-services/conf/supervisord.conf
