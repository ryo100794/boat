from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

from . import modeling_no_odds_v4 as base
from .db import connection, init_db
from .features_pastlog_v2 import load_training_examples, prediction_features
from .modeling_no_odds_v6 import SparseIndex32


FEATURE_SET = "pastlog_v2_xgboost_interactions_gpu_candidate"


def make_pipeline(*, device: str = "cuda") -> Pipeline:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError(
            "xgboost is not installed in this venv. Install it only inside the project venv "
            "on the GPU worker, not globally."
        ) from exc
    return Pipeline(
        [
            ("vectorizer", DictVectorizer(sparse=True)),
            ("sparse_index_32", SparseIndex32()),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=700,
                    max_depth=4,
                    learning_rate=0.035,
                    subsample=0.85,
                    colsample_bytree=0.75,
                    min_child_weight=20,
                    reg_alpha=0.05,
                    reg_lambda=3.0,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    device=device,
                    random_state=42,
                    n_jobs=4,
                ),
            ),
        ]
    )


base.FEATURE_SET = FEATURE_SET
base.load_training_examples = load_training_examples
base.prediction_features = prediction_features


def train_model(conn, *, model_path: Path, min_examples: int = 100, device: str = "cuda") -> dict[str, Any]:
    X, y, meta = load_training_examples(conn, include_odds=False)
    if len(X) < min_examples:
        raise ValueError(f"training examples are too few: {len(X)} < {min_examples}")
    if len(set(y)) < 2:
        raise ValueError("training labels need both winners and non-winners")
    pipeline = make_pipeline(device=device)
    pipeline.fit(X, y)
    metadata = {
        "trained_at": base._now(),
        "examples": len(X),
        "races": len({row["race_id"] for row in meta}),
        "include_odds": False,
        "target": "lane_win_probability",
        "vectorizer": "sparse",
        "classifier": f"XGBClassifier(hist, device={device})",
        "feature_set": FEATURE_SET,
        "role": "gpu_interaction_candidate",
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, model_path)
    return metadata


def configure_for_backtest(device: str) -> None:
    base.make_pipeline = lambda: make_pipeline(device=device)


predict_race = base.predict_race
predict_open_races = base.predict_open_races
positive_probs = base.positive_probs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train/backtest XGBoost GPU candidate on pastlog v2 features.")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    add_common(train)
    train.add_argument("--model", default="data/models/win_model_gpu_xgb_pastlog_v2.joblib")
    train.add_argument("--min-examples", type=int, default=100)
    train.set_defaults(func=_cmd_train)
    backtest = sub.add_parser("backtest")
    add_common(backtest)
    backtest.add_argument("--output", default="data/models/backtest_gpu_xgb_pastlog_v2.json")
    backtest.add_argument("--folds", type=int, default=3)
    backtest.add_argument("--min-train-races", type=int, default=500)
    backtest.set_defaults(func=_cmd_backtest)
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])


def _cmd_train(args: argparse.Namespace) -> int:
    init_db(args.db)
    with connection(args.db) as conn:
        result = train_model(
            conn,
            model_path=Path(args.model),
            min_examples=args.min_examples,
            device=args.device,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


def _cmd_backtest(args: argparse.Namespace) -> int:
    configure_for_backtest(args.device)
    init_db(args.db)
    with connection(args.db) as conn:
        result = base.backtest_model(
            conn,
            output_path=Path(args.output),
            folds=args.folds,
            min_train_races=args.min_train_races,
        )
    result["feature_set"] = FEATURE_SET
    result["device"] = args.device
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
