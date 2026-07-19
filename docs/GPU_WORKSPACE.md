# GPU workspace plan

GPU worker runs must stay inside the shared project directory:

```text
/workspace/boat-no-odds-v4
```

Do not install packages globally. Use a venv below the shared project or a sibling venv path that is mounted with the workspace.

```text
/workspace/boat-no-odds-v4/.venv
/workspace/boat-no-odds-v4/data
/workspace/boat-no-odds-v4/data/models
/workspace/boat-no-odds-v4/logs
```

Current CPU-safe sequence:

```bash
cd /workspace/boat-no-odds-v4/src
../.venv/bin/python -m boatrace_ai.feature_diagnostics_stream \
  --db ../data/boatrace.sqlite \
  --model ../data/models/win_model_no_odds_v8.joblib \
  --output ../data/models/feature_diagnostics_stream.json

../.venv/bin/python -m boatrace_ai.modeling_pastlog_v2 train \
  --db ../data/boatrace.sqlite \
  --model ../data/models/win_model_pastlog_v2.joblib

../.venv/bin/python -m boatrace_ai.modeling_pastlog_v2 backtest \
  --db ../data/boatrace.sqlite \
  --output ../data/models/backtest_pastlog_v2.json \
  --folds 5 \
  --min-train-races 500

../.venv/bin/python -m boatrace_ai.bankroll_backtest_pastlog_v2 \
  --db ../data/boatrace.sqlite \
  --output ../data/models/bankroll_backtest_pastlog_v2_10000.json \
  --daily-budget-yen 10000
```

GPU is useful after the past-log feature set is stable, mainly for tree boosting or neural ranking that can learn feature interactions. It is not expected to speed up the current sparse logistic regression much.

GPU candidate requirements:

```text
NVIDIA driver visible through nvidia-smi
one of: xgboost, lightgbm, catboost, torch
all caches and outputs under /workspace/boat-no-odds-v4
```

Promotion rule:

```text
primary model: pastlog_v2, stored in predictions table
shadow model: realtime_hybrid_v2 or GPU boosting candidate, written to JSON/log first
promote only after shadow top1/top5 hit rate and bankroll ROI beat primary on accumulated live results
```
