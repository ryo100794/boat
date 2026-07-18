from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

from . import modeling_no_odds_v4 as base


class SparseIndex32(BaseEstimator, TransformerMixin):
    """Normalize scipy sparse index arrays for sklearn builds requiring int32."""

    def fit(self, X: Any, y: Any = None) -> "SparseIndex32":
        return self

    def transform(self, X: Any) -> Any:
        if not sparse.issparse(X):
            return X
        matrix = X.tocsr(copy=False)
        if matrix.indices.dtype != np.int32:
            matrix.indices = matrix.indices.astype(np.int32, copy=False)
        if matrix.indptr.dtype != np.int32:
            matrix.indptr = matrix.indptr.astype(np.int32, copy=False)
        return matrix


def make_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("vectorizer", DictVectorizer(sparse=True)),
            ("sparse_index_32", SparseIndex32()),
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


base.FEATURE_SET = "no_odds_v6_relative_racer_motor_boat_weather_sparse32_sgd"
base.make_pipeline = make_pipeline


train_model = base.train_model
backtest_model = base.backtest_model
predict_race = base.predict_race
predict_open_races = base.predict_open_races
positive_probs = base.positive_probs
main = base.main


if __name__ == "__main__":
    raise SystemExit(main())
