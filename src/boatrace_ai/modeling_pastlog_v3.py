from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

from . import modeling_no_odds_v4 as base
from .db import connection, init_db
from .features_pastlog_v3 import load_training_examples, prediction_features
from .modeling_no_odds_v6 import SparseIndex32


FEATURE_SET = "pastlog_v3_pruned_prevday_rolling_history_logreg_C0.12"


def make_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("vectorizer", DictVectorizer(sparse=True)),
            ("sparse_index_32_a", SparseIndex32()),
            ("scaler", MaxAbsScaler(copy=False)),
            ("sparse_index_32_b", SparseIndex32()),
            (
                "classifier",
                LogisticRegression(
                    solver="liblinear",
                    C=0.12,
                    max_iter=1000,
                    class_weight=None,
                    random_state=42,
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
        "scaler": "MaxAbsScaler",
        "classifier": "LogisticRegression(liblinear, C=0.12, class_weight=None)",
        "feature_set": FEATURE_SET,
        "role": "primary_pastlog",
        "design_note": (
            "prior-date rolling history plus card fields; prunes low-coverage avg_st, F/L, "
            "3-rate, realtime beforeinfo/weather/odds features."
        ),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, model_path)
    return metadata


backtest_model = base.backtest_model
predict_race = base.predict_race
predict_open_races = base.predict_open_races
positive_probs = base.positive_probs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train/backtest pruned prior-date past-log v3 model.")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    add_common(train)
    train.add_argument("--model", default="data/models/win_model_pastlog_v3.joblib")
    train.add_argument("--min-examples", type=int, default=100)
    train.set_defaults(func=_cmd_train)
    backtest = sub.add_parser("backtest")
    add_common(backtest)
    backtest.add_argument("--output", default="data/models/backtest_pastlog_v3.json")
    backtest.add_argument("--folds", type=int, default=5)
    backtest.add_argument("--min-train-races", type=int, default=500)
    backtest.set_defaults(func=_cmd_backtest)
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", default="data/boatrace.sqlite")


def _cmd_train(args: argparse.Namespace) -> int:
    init_db(args.db)
    with connection(args.db) as conn:
        result = train_model(conn, model_path=Path(args.model), min_examples=args.min_examples)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


def _cmd_backtest(args: argparse.Namespace) -> int:
    init_db(args.db)
    with connection(args.db) as conn:
        result = backtest_model(
            conn,
            output_path=Path(args.output),
            folds=args.folds,
            min_train_races=args.min_train_races,
        )
    result["feature_set"] = FEATURE_SET
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
