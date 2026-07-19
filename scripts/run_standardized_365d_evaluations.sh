#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=src

db="${BOATRACE_DB:-data/boatrace.sqlite}"
model_dir="${BOATRACE_MODEL_DIR:-data/models}"
log_dir="${BOATRACE_LOG_DIR:-logs}"
wait_pid="${1:-}"
mkdir -p "$model_dir" "$log_dir"

if [[ -n "$wait_pid" ]]; then
  while kill -0 "$wait_pid" 2>/dev/null; do
    sleep 30
  done
fi

read -r holdout_start holdout_end total_races min_train selection_fraction train_fraction < <(
  .venv/bin/python -c '
import datetime
import sqlite3
import sys

conn = sqlite3.connect(sys.argv[1])
complete = """
  (SELECT COUNT(*) FROM entries e WHERE e.race_id = r.race_id) = 6
  AND (SELECT COUNT(*) FROM race_results rr
       WHERE rr.race_id = r.race_id AND rr.rank IS NOT NULL) = 6
"""
end = conn.execute(
    f"SELECT MAX(r.race_date) FROM races r WHERE {complete}"
).fetchone()[0]
start = (datetime.date.fromisoformat(end) - datetime.timedelta(days=364)).isoformat()
total = conn.execute(f"SELECT COUNT(*) FROM races r WHERE {complete}").fetchone()[0]
before = conn.execute(
    f"SELECT COUNT(*) FROM races r WHERE r.race_date < ? AND {complete}",
    (start,),
).fetchone()[0]
selection_fraction = (before - 1) / total
train_fraction = (before * 0.8) / total
print(start, end, total, before, selection_fraction, train_fraction)
' "$db"
)

manifest="$model_dir/standardized_365d_manifest.json"
.venv/bin/python -c '
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
path.write_text(json.dumps({
    "protocol": "fixed final 365 calendar days, full-day boundary",
    "holdout_start": sys.argv[2],
    "holdout_end": sys.argv[3],
    "total_complete_races": int(sys.argv[4]),
    "training_races": int(sys.argv[5]),
    "holdout_races": int(sys.argv[4]) - int(sys.argv[5]),
    "daily_budget_yen": 10000,
    "selection_rule": "all model/feature/teacher selection precedes holdout",
}, ensure_ascii=False, indent=2), encoding="utf-8")
' "$manifest" "$holdout_start" "$holdout_end" "$total_races" "$min_train"

run_job() {
  local name="$1"
  shift
  printf 'START %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$name" | tee -a "$log_dir/standardized_365d_queue.log"
  "$@" >"$log_dir/${name}.log" 2>&1
  printf 'DONE  %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$name" | tee -a "$log_dir/standardized_365d_queue.log"
}

run_job standardized_365d_no_odds_v8_backtest \
  .venv/bin/python -m boatrace_ai.historical_model backtest \
  --db "$db" --output "$model_dir/standardized_365d_no_odds_v8_backtest.json" \
  --folds 1 --min-train-races "$min_train"

run_job standardized_365d_no_odds_v8_bankroll \
  .venv/bin/python -m boatrace_ai.operational_bankroll \
  --db "$db" --output "$model_dir/standardized_365d_no_odds_v8_bankroll.json" \
  --folds 1 --min-train-races "$min_train" --daily-budget-yen 10000

run_job standardized_365d_pastlog_v7_backtest \
  .venv/bin/python -m boatrace_ai.feature_tuning backtest \
  --db "$db" --output "$model_dir/standardized_365d_pastlog_v7_backtest.json" \
  --folds 1 --min-train-races "$min_train"

run_job standardized_365d_pastlog_v7_bankroll \
  .venv/bin/python -m boatrace_ai.bankroll_optimizer \
  --db "$db" --output "$model_dir/standardized_365d_pastlog_v7_bankroll.json" \
  --folds 1 --min-train-races "$min_train" --daily-budget-yen 10000 \
  --ev-threshold 1.20 --fractional-kelly 0.25 \
  --max-daily-exposure-fraction 0.60 --min-daily-exposure-fraction 0.40 \
  --race-cap-fraction 0.10 --ticket-cap-fraction 0.03 \
  --max-daily-tickets 30 --allocation-mode normalized_kelly

for kind in linear mlp; do
  run_job "standardized_365d_calibrated_${kind}" \
    .venv/bin/python -m boatrace_ai.calibrated_shadow_model backtest \
    --db "$db" --model-kind "$kind" \
    --output "$model_dir/standardized_365d_calibrated_${kind}.json" \
    --folds 1 --min-train-races "$min_train"
done

run_job standardized_365d_listwise_feature_teacher \
  .venv/bin/python -m boatrace_ai.listwise.feature_search \
  --db "$db" \
  --output "$model_dir/standardized_365d_listwise_feature_teacher.json" \
  --cache-dir "$model_dir/listwise_search_cache" \
  --train-fraction "$train_fraction" --selection-fraction "$selection_fraction" \
  --daily-budget-yen 10000 --ev-threshold 1.20

run_job standardized_365d_listwise_newton \
  .venv/bin/python -m boatrace_ai.listwise.newton_refine \
  --db "$db" \
  --search-result "$model_dir/standardized_365d_listwise_feature_teacher.json" \
  --output "$model_dir/standardized_365d_listwise_newton.json" \
  --model-output "$model_dir/standardized_365d_listwise_newton.joblib" \
  --cache-dir "$model_dir/listwise_search_cache" \
  --daily-budget-yen 10000 --ev-threshold 1.20

printf 'COMPLETE %s standardized_365d\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$log_dir/standardized_365d_queue.log"
