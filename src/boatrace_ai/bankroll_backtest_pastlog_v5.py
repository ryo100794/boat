from __future__ import annotations

from . import bankroll_backtest_pastlog_v4 as base_cli
from .features_pastlog_v5 import load_training_examples
from .modeling_pastlog_v5 import FEATURE_SET, make_pipeline, positive_probs


base_cli.FEATURE_SET = FEATURE_SET
base_cli.load_training_examples = load_training_examples
base_cli.make_pipeline = make_pipeline
base_cli.positive_probs = positive_probs


if __name__ == "__main__":
    raise SystemExit(base_cli.main())
