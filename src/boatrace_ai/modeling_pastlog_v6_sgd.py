from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

from . import modeling_pastlog_v4 as base_cli
from . import modeling_no_odds_v4 as base
from .features_pastlog_v5 import load_training_examples, prediction_features


FEATURE_SET = "pastlog_v6_cached_series_form_prevday_rolling_history_sgd_elasticnet"


def make_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("vectorizer", DictVectorizer(sparse=True)),
            (
                "classifier",
                SGDClassifier(
                    loss="log_loss",
                    penalty="elasticnet",
                    alpha=0.00005,
                    l1_ratio=0.05,
                    max_iter=80,
                    tol=1e-3,
                    class_weight="balanced",
                    random_state=42,
                    average=True,
                ),
            ),
        ]
    )


base.FEATURE_SET = FEATURE_SET
base.make_pipeline = make_pipeline
base.load_training_examples = load_training_examples
base.prediction_features = prediction_features


def train_model(conn, *, model_path: Path, min_examples: int = 100) -> dict[str, Any]:
    X, y, meta = load_training_examples(conn, include_odds=False)
    if len(X) < min_examples:
        raise ValueError(f"training examples are too few: {len(X)} < {min_examples}")
    if len(set(y)) < 2:
        raise ValueError("training labels need both winners and non-winners")
    pipeline = make_pipeline()
    pipeline.fit(X, y)
    metadata = {
        "trained_at": base._now(),
        "examples": len(X),
        "races": len({row["race_id"] for row in meta}),
        "include_odds": False,
        "target": "lane_win_probability",
        "vectorizer": "sparse",
        "classifier": "SGDClassifier(log_loss, elasticnet, balanced, average=True)",
        "feature_set": FEATURE_SET,
        "role": "primary_pastlog_memory_safe",
        "design_note": "cached official series-form features with SGD to avoid liblinear memory spikes.",
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, model_path)
    return metadata


backtest_model = base.backtest_model
predict_race = base.predict_race
predict_open_races = base.predict_open_races
positive_probs = base.positive_probs


base_cli.FEATURE_SET = FEATURE_SET
base_cli.load_training_examples = load_training_examples
base_cli.prediction_features = prediction_features
base_cli.make_pipeline = make_pipeline
base_cli.train_model = train_model


if __name__ == "__main__":
    raise SystemExit(base_cli.main())
