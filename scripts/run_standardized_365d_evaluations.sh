#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=src

db="${BOATRACE_DB:-data/boatrace.sqlite}"
model_dir="${BOATRACE_MODEL_DIR:-data/models}"
eval_dir="$model_dir/standardized_365d_v2"
raw_dir="$eval_dir/raw"
log_dir="${BOATRACE_LOG_DIR:-logs}/standardized_365d_v2"
protocol="$eval_dir/protocol.json"
as_of_date="${BOATRACE_EVAL_AS_OF_DATE:-}"
wait_pid="${1:-}"
resume_completed="${BOATRACE_EVAL_RESUME_COMPLETED:-1}"
eval_nice="${BOATRACE_EVAL_NICE:-10}"
mkdir -p "$raw_dir" "$log_dir"

if [[ -n "$wait_pid" ]]; then
  while kill -0 "$wait_pid" 2>/dev/null; do
    sleep 30
  done
fi

prepare=(
  .venv/bin/python -m boatrace_ai.standard_evaluation
  --db "$db" --protocol-file "$protocol" --prepare-only
)
if [[ -n "$as_of_date" ]]; then
  prepare+=(--as-of-date "$as_of_date")
fi
"${prepare[@]}" >"$log_dir/protocol.json.log" 2>&1

read -r holdout_start holdout_end total_races min_train selection_fraction train_fraction < <(
  .venv/bin/python -c '
import json
import sys

protocol = json.load(open(sys.argv[1], encoding="utf-8"))
total = int(protocol["training_races"]) + int(protocol["prediction_races"])
before = int(protocol["training_races"])
selection_fraction = (before - 1) / total
train_fraction = (before * 0.8) / total
print(protocol["holdout_start"], protocol["holdout_end"], total, before, selection_fraction, train_fraction)
' "$protocol"
)
export BOATRACE_EVAL_MAX_RACE_DATE="$holdout_end"

run_job() {
  local name="$1"
  shift
  printf 'START %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$name" | tee -a "$log_dir/standardized_365d_queue.log"
  nice -n "$eval_nice" "$@" >"$log_dir/${name}.log" 2>&1
  printf 'DONE  %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$name" | tee -a "$log_dir/standardized_365d_queue.log"
}

source_needs_run() {
  local model_id="$1"
  if [[ "$resume_completed" == "1" ]] && \
    .venv/bin/python -m boatrace_ai.standard_evaluation \
      --db "$db" --raw-dir "$raw_dir" --protocol-file "$protocol" \
      --validate-source "$model_id" >/dev/null 2>&1; then
    printf 'SKIP  %s %s (validated)\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$model_id" \
      | tee -a "$log_dir/standardized_365d_queue.log"
    return 1
  fi
  return 0
}

if source_needs_run no_odds_v8; then
run_job standardized_365d_v2_no_odds_v8_prediction \
  .venv/bin/python -m boatrace_ai.historical_model backtest \
  --db "$db" --output "$raw_dir/no_odds_v8_prediction.json" \
  --folds 1 --min-train-races "$min_train"

run_job standardized_365d_v2_no_odds_v8_bankroll \
  .venv/bin/python -m boatrace_ai.operational_bankroll \
  --db "$db" --output "$raw_dir/no_odds_v8_bankroll.json" \
  --folds 1 --min-train-races "$min_train" --daily-budget-yen 10000

fi

if source_needs_run pastlog_v7; then
run_job standardized_365d_v2_pastlog_v7_prediction \
  .venv/bin/python -m boatrace_ai.feature_tuning backtest \
  --db "$db" --output "$raw_dir/pastlog_v7_prediction.json" \
  --model-output "$eval_dir/pastlog_v7.joblib" \
  --drop-feature-groups research_correlates \
  --folds 1 --min-train-races "$min_train"

run_job standardized_365d_v2_pastlog_v7_bankroll \
  .venv/bin/python -m boatrace_ai.bankroll_optimizer \
  --db "$db" --output "$raw_dir/pastlog_v7_bankroll.json" \
  --model-input "$eval_dir/pastlog_v7.joblib" \
  --drop-feature-groups research_correlates \
  --folds 1 --min-train-races "$min_train" --daily-budget-yen 10000 \
  --ev-threshold 1.20 --fractional-kelly 0.25 \
  --max-daily-exposure-fraction 0.60 --min-daily-exposure-fraction 0.40 \
  --race-cap-fraction 0.10 --ticket-cap-fraction 0.03 \
  --max-daily-tickets 30 --allocation-mode normalized_kelly

fi

if source_needs_run pastlog_v9_research; then
run_job standardized_365d_v2_pastlog_v9_research_prediction \
  .venv/bin/python -m boatrace_ai.feature_tuning backtest \
  --db "$db" --output "$raw_dir/pastlog_v9_research_prediction.json" \
  --model-output "$eval_dir/pastlog_v9_research.joblib" \
  --folds 1 --min-train-races "$min_train"

run_job standardized_365d_v2_pastlog_v9_research_bankroll \
  .venv/bin/python -m boatrace_ai.bankroll_optimizer \
  --db "$db" --output "$raw_dir/pastlog_v9_research_bankroll.json" \
  --model-input "$eval_dir/pastlog_v9_research.joblib" \
  --folds 1 --min-train-races "$min_train" --daily-budget-yen 10000 \
  --ev-threshold 1.20 --fractional-kelly 0.25 \
  --max-daily-exposure-fraction 0.60 --min-daily-exposure-fraction 0.40 \
  --race-cap-fraction 0.10 --ticket-cap-fraction 0.03 \
  --max-daily-tickets 30 --allocation-mode normalized_kelly

fi

for kind in linear mlp; do
  if source_needs_run "calibrated_${kind}"; then
  run_job "standardized_365d_v2_calibrated_${kind}" \
    .venv/bin/python -m boatrace_ai.calibrated_shadow_model backtest \
    --db "$db" --model-kind "$kind" \
    --output "$raw_dir/calibrated_${kind}.json" \
    --folds 1 --min-train-races "$min_train" \
    --drop-feature-groups research_correlates \
    --daily-budget-yen 10000 --ev-threshold 1.20
  fi
done

if source_needs_run listwise_feature_teacher; then
run_job standardized_365d_v2_listwise_feature_teacher \
  .venv/bin/python -m boatrace_ai.listwise.feature_search \
  --db "$db" \
  --output "$raw_dir/listwise_feature_teacher.json" \
  --cache-dir "$eval_dir/listwise_search_cache" \
  --train-fraction "$train_fraction" --selection-fraction "$selection_fraction" \
  --daily-budget-yen 10000 --ev-threshold 1.20

fi

if source_needs_run listwise_newton; then
run_job standardized_365d_v2_listwise_newton \
  .venv/bin/python -m boatrace_ai.listwise.newton_refine \
  --db "$db" \
  --search-result "$raw_dir/listwise_feature_teacher.json" \
  --output "$raw_dir/listwise_newton.json" \
  --model-output "$eval_dir/listwise_newton.joblib" \
  --cache-dir "$eval_dir/listwise_search_cache" \
  --daily-budget-yen 10000 --ev-threshold 1.20

fi

run_job standardized_365d_v2_consolidate \
  .venv/bin/python -m boatrace_ai.standard_evaluation \
  --db "$db" --raw-dir "$raw_dir" --output-dir "$eval_dir" \
  --protocol-file "$protocol"

run_job standardized_365d_v2_audit \
  .venv/bin/python scripts/audit_standardized_evaluation.py "$eval_dir"

printf 'COMPLETE %s standardized_365d_v2\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$log_dir/standardized_365d_queue.log"
