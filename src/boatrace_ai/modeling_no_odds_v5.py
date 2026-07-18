from __future__ import annotations

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

from . import modeling_no_odds_v4 as base


def make_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("vectorizer", DictVectorizer(sparse=False)),
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


base.FEATURE_SET = "no_odds_v5_relative_racer_motor_boat_weather_dense_sgd"
base.make_pipeline = make_pipeline


train_model = base.train_model
backtest_model = base.backtest_model
predict_race = base.predict_race
predict_open_races = base.predict_open_races
positive_probs = base.positive_probs
main = base.main


if __name__ == "__main__":
    raise SystemExit(main())
