from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

from . import modeling_no_odds_v4 as base
from .db import connection, init_db
from .features import latest_trifecta_odds
from .features_realtime_hybrid_v1 import load_training_examples, prediction_features
from .modeling import _normalize_lane_probs, trifecta_predictions
from .modeling_no_odds_v6 import SparseIndex32


FEATURE_SET = "realtime_hybrid_v1_pastlog_beforeinfo_odds_logreg_C0.08_shadow"


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
                    C=0.08,
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
    X, y, meta = load_training_examples(conn, include_odds=True)
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
        "include_odds": True,
        "target": "lane_win_probability",
        "vectorizer": "sparse",
        "scaler": "MaxAbsScaler",
        "classifier": "LogisticRegression(liblinear, C=0.08, class_weight=None)",
        "feature_set": FEATURE_SET,
        "role": "shadow_realtime_hybrid",
        "design_note": (
            "past-log features plus latest beforeinfo/weather and odds aggregate features; "
            "kept separate from production predictions until enough realtime-labeled data accumulates."
        ),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, model_path)
    return metadata


def backtest_model(
    conn,
    *,
    output_path: Path,
    folds: int = 5,
    min_train_races: int = 500,
) -> dict[str, Any]:
    result = base.backtest_model(
        conn,
        output_path=output_path,
        folds=folds,
        min_train_races=min_train_races,
    )
    result["feature_set"] = FEATURE_SET
    result["shadow_note"] = (
        "historical realtime beforeinfo/odds coverage is sparse, so this backtest is mainly a guardrail; "
        "live shadow accuracy should decide promotion."
    )
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def predict_race(
    conn,
    *,
    model_path: Path,
    race_id_value: str,
    top_n: int = 120,
) -> list[dict[str, Any]]:
    bundle = joblib.load(model_path)
    pipeline = bundle["pipeline"]
    X = prediction_features(conn, race_id=race_id_value, include_odds=True)
    if len(X) != 6:
        raise ValueError(f"race needs six entries before prediction: {race_id_value}")
    raw = positive_probs(pipeline, X)
    lane_probs = _normalize_lane_probs({lane: raw[lane - 1] for lane in range(1, 7)})
    rows = trifecta_predictions(lane_probs, latest_odds=latest_trifecta_odds(conn, race_id_value))
    rows = sorted(
        rows,
        key=lambda row: (float(row["probability"]), float(row.get("expected_value") or 0.0)),
        reverse=True,
    )[:top_n]
    for row in rows:
        row["rank_basis"] = "model_probability"
        row["feature_set"] = FEATURE_SET
        row["model_role"] = "shadow_realtime_hybrid"
    return rows


def predict_open_races(
    conn,
    *,
    model_path: Path,
    race_date: date,
) -> dict[str, Any]:
    rows = conn.execute(
        """
        SELECT r.race_id, r.race_date, r.jcd, r.venue_name, r.rno, r.title, r.deadline_at
        FROM races r
        WHERE r.race_date = ?
          AND (r.status IS NULL OR r.status != 'final')
          AND (SELECT COUNT(*) FROM entries e WHERE e.race_id = r.race_id) = 6
        ORDER BY r.jcd, r.rno
        """,
        (race_date.isoformat(),),
    ).fetchall()
    predicted = []
    failed = []
    for row in rows:
        try:
            predictions = predict_race(conn, model_path=model_path, race_id_value=row["race_id"])
            predicted.append(
                {
                    "race_id": row["race_id"],
                    "race_date": row["race_date"],
                    "jcd": row["jcd"],
                    "venue_name": row["venue_name"],
                    "rno": row["rno"],
                    "title": row["title"],
                    "deadline_at": row["deadline_at"],
                    "top_prediction": predictions[0] if predictions else None,
                    "top5": predictions[:5],
                }
            )
        except Exception as exc:
            failed.append({"race_id": row["race_id"], "error": str(exc)})
    return {
        "generated_at": _now(),
        "model": str(model_path),
        "feature_set": FEATURE_SET,
        "role": "shadow_realtime_hybrid",
        "predicted": len(predicted),
        "failed": len(failed),
        "races": predicted,
        "errors": failed[:20],
    }


def positive_probs(pipeline: Pipeline, X: list[dict[str, Any]]) -> list[float]:
    classifier = pipeline.named_steps["classifier"]
    classes = list(classifier.classes_)
    positive_index = classes.index(1)
    return [float(row[positive_index]) for row in pipeline.predict_proba(X)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train/backtest/predict realtime hybrid shadow model.")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    add_common(train)
    train.add_argument("--model", default="data/models/win_model_realtime_hybrid_v1.joblib")
    train.add_argument("--min-examples", type=int, default=100)
    train.set_defaults(func=_cmd_train)
    backtest = sub.add_parser("backtest")
    add_common(backtest)
    backtest.add_argument("--output", default="data/models/backtest_realtime_hybrid_v1.json")
    backtest.add_argument("--folds", type=int, default=5)
    backtest.add_argument("--min-train-races", type=int, default=500)
    backtest.set_defaults(func=_cmd_backtest)
    predict = sub.add_parser("predict")
    add_common(predict)
    predict.add_argument("--model", default="data/models/win_model_realtime_hybrid_v1.joblib")
    predict.add_argument("--date", default=date.today().isoformat())
    predict.add_argument("--output", default="data/models/shadow_realtime_hybrid_latest.json")
    predict.set_defaults(func=_cmd_predict)
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
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    init_db(args.db)
    with connection(args.db) as conn:
        result = predict_open_races(
            conn,
            model_path=Path(args.model),
            race_date=date.fromisoformat(args.date),
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "races"}, ensure_ascii=False), flush=True)
    return 0


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
