#!/usr/bin/env bash
set -euo pipefail

ROOT="${BOATRACE_ROOT:-/workspace/boat}"
PYTHON="${BOATRACE_PYTHON:-${ROOT}/.venv/bin/python}"

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 EVALUATION_JSON OUTPUT_JOBLIB [SCORED_CACHE]" >&2
  exit 2
fi

evaluation_json="$1"
output_joblib="$2"
prediction_date="${PREDICTION_DATE:-$(TZ=Asia/Tokyo date -d tomorrow +%F)}"

args=(
  -m boatrace_ai.runtime.v12_shadow_bundle
  --evaluation-json "$evaluation_json"
  --output "$output_joblib"
  --prediction-date "$prediction_date"
)
if [[ $# -eq 3 ]]; then
  args+=(--scored-cache "$3")
fi

cd "$ROOT"
exec "$PYTHON" "${args[@]}"
