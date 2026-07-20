#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=src

db="${BOATRACE_DB:-data/boatrace.sqlite}"
model_dir="${BOATRACE_MODEL_DIR:-data/models}"
eval_dir="$model_dir/standardized_365d_v2"
raw_dir="$eval_dir/raw"
log_dir="${BOATRACE_LOG_DIR:-logs}/standardized_365d_v2"
wait_pid="${1:-}"
mkdir -p "$raw_dir" "$log_dir"

if [[ -n "$wait_pid" ]]; then
  while kill -0 "$wait_pid" 2>/dev/null; do
    sleep 30
  done
fi

read -r holdout_start holdout_end total_races min_train selection_fraction train_fraction < <(
  .venv/bin/python -c '
import sys

from boatrace_ai.db import connection
from boatrace_ai.standard_evaluation import build_protocol

with connection(sys.argv[1]) as conn:
    protocol = build_protocol(conn)
total = protocol["training_races"] + protocol["prediction_races"]
before = protocol["training_races"]
selection_fraction = (before - 1) / total
train_fraction = (before * 0.8) / total
print(protocol["holdout_start"], protocol["holdout_end"], total, before, selection_fraction, train_fraction)
' "$db"
)
export BOATRACE_EVAL_MAX_RACE_DATE="$holdout_end"

manifest="$eval_dir/protocol_draft.json"
.venv/bin/python -c '
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
path.write_text(json.dumps({
    "protocol_id": "standard_365d_v2",
    "protocol": "fixed final 365 calendar days, full-day boundary and one bankroll policy",
    "holdout_start": sys.argv[2],
    "holdout_end": sys.argv[3],
    "total_complete_races": int(sys.argv[4]),
    "training_races": int(sys.argv[5]),
    "holdout_races": int(sys.argv[4]) - int(sys.argv[5]),
    "daily_budget_yen": 10000,
    "selection_rule": "all model/feature/teacher/parameter selection precedes holdout",
    "bankroll_policy": {
        "daily_budget_yen": 10000,
        "unit_yen": 100,
        "ev_threshold": 1.20,
        "payout_prior_weight": 30.0,
        "fractional_kelly": 0.25,
        "max_daily_exposure_fraction": 0.60,
        "min_daily_exposure_fraction": 0.40,
        "race_cap_fraction": 0.10,
        "ticket_cap_fraction": 0.03,
        "max_daily_tickets": 30,
        "allocation_mode": "normalized_kelly"
    },
}, ensure_ascii=False, indent=2), encoding="utf-8")
' "$manifest" "$holdout_start" "$holdout_end" "$total_races" "$min_train"

run_job() {
  local name="$1"
  shift
  printf 'START %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$name" | tee -a "$log_dir/standardized_365d_queue.log"
  "$@" >"$log_dir/${name}.log" 2>&1
  printf 'DONE  %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$name" | tee -a "$log_dir/standardized_365d_queue.log"
}

run_job standardized_365d_v2_no_odds_v8_prediction \
  .venv/bin/python -m boatrace_ai.historical_model backtest \
  --db "$db" --output "$raw_dir/no_odds_v8_prediction.json" \
  --folds 1 --min-train-races "$min_train"

run_job standardized_365d_v2_no_odds_v8_bankroll \
  .venv/bin/python -m boatrace_ai.operational_bankroll \
  --db "$db" --output "$raw_dir/no_odds_v8_bankroll.json" \
  --folds 1 --min-train-races "$min_train" --daily-budget-yen 10000

run_job standardized_365d_v2_pastlog_v7_prediction \
  .venv/bin/python -m boatrace_ai.feature_tuning backtest \
  --db "$db" --output "$raw_dir/pastlog_v7_prediction.json" \
  --folds 1 --min-train-races "$min_train"

run_job standardized_365d_v2_pastlog_v7_bankroll \
  .venv/bin/python -m boatrace_ai.bankroll_optimizer \
  --db "$db" --output "$raw_dir/pastlog_v7_bankroll.json" \
  --folds 1 --min-train-races "$min_train" --daily-budget-yen 10000 \
  --ev-threshold 1.20 --fractional-kelly 0.25 \
  --max-daily-exposure-fraction 0.60 --min-daily-exposure-fraction 0.40 \
  --race-cap-fraction 0.10 --ticket-cap-fraction 0.03 \
  --max-daily-tickets 30 --allocation-mode normalized_kelly

for kind in linear mlp; do
  run_job "standardized_365d_v2_calibrated_${kind}" \
    .venv/bin/python -m boatrace_ai.calibrated_shadow_model backtest \
    --db "$db" --model-kind "$kind" \
    --output "$raw_dir/calibrated_${kind}.json" \
    --folds 1 --min-train-races "$min_train" \
    --daily-budget-yen 10000 --ev-threshold 1.20
done

run_job standardized_365d_v2_listwise_feature_teacher \
  .venv/bin/python -m boatrace_ai.listwise.feature_search \
  --db "$db" \
  --output "$raw_dir/listwise_feature_teacher.json" \
  --cache-dir "$eval_dir/listwise_search_cache" \
  --train-fraction "$train_fraction" --selection-fraction "$selection_fraction" \
  --daily-budget-yen 10000 --ev-threshold 1.20

run_job standardized_365d_v2_listwise_newton \
  .venv/bin/python -m boatrace_ai.listwise.newton_refine \
  --db "$db" \
  --search-result "$raw_dir/listwise_feature_teacher.json" \
  --output "$raw_dir/listwise_newton.json" \
  --model-output "$eval_dir/listwise_newton.joblib" \
  --cache-dir "$eval_dir/listwise_search_cache" \
  --daily-budget-yen 10000 --ev-threshold 1.20

run_job standardized_365d_v2_consolidate \
  .venv/bin/python -m boatrace_ai.standard_evaluation \
  --db "$db" --raw-dir "$raw_dir" --output-dir "$eval_dir"

printf 'COMPLETE %s standardized_365d_v2\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$log_dir/standardized_365d_queue.log"
