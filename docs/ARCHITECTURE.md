# Source architecture

`src/boatrace_ai` keeps stable application entry points at the package root and
groups implementation modules by responsibility.

## Packages

- `boatrace_ai.ingestion`: official archives, daily programs, race pages, odds,
  and result parsing.
- `boatrace_ai.runtime`: long-running collection, prediction, result polling,
  and shadow-model cycles.
- `boatrace_ai.listwise`: race-wise ranking models, Newton refinement, temporal
  validation, feature search, and backtesting.
- `boatrace_ai.web`: the dashboard, reports, and prediction-summary queries.
- `teleboat_agent`: audited browser automation, validation, and journals.

## Canonical commands

```bash
python -m boatrace_ai.web.dashboard
python -m boatrace_ai.runtime.collector
python -m boatrace_ai.runtime.predictor
python -m boatrace_ai.runtime.model_cycle
python -m boatrace_ai.listwise.feature_search
```

`boatrace_ai.web_dashboard`, `boatrace_ai.realtime_collector`,
`boatrace_ai.realtime_predictor`, and `boatrace_ai.model_cycle` are thin CLI
compatibility entry points. New code must import the canonical package modules.

## Maintenance rules

- Do not add numbered or `vN` Python module names. Record versions in model
  metadata and Git history.
- Put new code in the package that owns its responsibility; do not add a new
  root module unless it is a stable public entry point.
- Before removing compatibility code, verify imports with
  `scripts/versioned_module_inventory.py` and run the full test suite.
- Keep serialized-model compatibility aliases in `legacy_model_aliases.py` only.
