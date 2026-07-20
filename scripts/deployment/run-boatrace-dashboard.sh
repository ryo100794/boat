#!/usr/bin/env bash
set -euo pipefail

conf="${BOATRACE_SERVICE_CONF:-/workspace/shared-services/conf/boatrace-collector.env}"
[[ -f "$conf" ]] && source "$conf"
app_dir="${BOATRACE_APP_DIR:-/workspace/boat}"
python="${BOATRACE_PYTHON:-$app_dir/.venv/bin/python}"
dsn="${BOATRACE_POSTGRES_DSN:-host=127.0.0.1 port=5432 dbname=boatrace user=boatrace_app}"

cd "$app_dir"
export PYTHONPATH="$app_dir/src"
export PGPASSFILE="${BOATRACE_PGPASSFILE:-/workspace/postgresql/conf/databases/boatrace.pgpass}"
exec "$python" -m boatrace_ai.web.postgresql_dashboard \
  --postgres-dsn "$dsn" \
  --data-dir "$app_dir/data" \
  --backtest "$app_dir/data/models/backtest_no_odds_v8.json" \
  --host 0.0.0.0 \
  --port 10001
