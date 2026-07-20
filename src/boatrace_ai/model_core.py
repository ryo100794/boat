from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from .legacy_model_aliases import load_model_bundle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline

from .db import connection, init_db, insert_prediction_rows
from .features import latest_trifecta_odds
from .base_features import load_training_examples, prediction_features
from .modeling import _normalize_lane_probs, _race_level_metrics, trifecta_predictions
from .standard_evaluation import race_set_sha256


FEATURE_SET = "no_odds_v4_relative_racer_motor_boat_weather_sgd"


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
                    max_iter=60,
                    tol=1e-3,
                    class_weight="balanced",
                    random_state=42,
                    average=True,
                ),
            ),
        ]
    )


def train_model(conn, *, model_path: Path, min_examples: int = 100) -> dict[str, Any]:
    X, y, meta = load_training_examples(conn, include_odds=False)
    if len(X) < min_examples:
        raise ValueError(f"training examples are too few: {len(X)} < {min_examples}")
    if len(set(y)) < 2:
        raise ValueError("training labels need both winners and non-winners")
    pipeline = make_pipeline()
    pipeline.fit(X, y)
    metadata = {
        "trained_at": _now(),
        "examples": len(X),
        "races": len({row["race_id"] for row in meta}),
        "include_odds": False,
        "target": "lane_win_probability",
        "vectorizer": "sparse",
        "classifier": "SGDClassifier(log_loss, elasticnet)",
        "feature_set": FEATURE_SET,
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
    include_research: bool = True,
) -> dict[str, Any]:
    X, y, meta = load_training_examples(
        conn,
        include_odds=False,
        include_research=include_research,
    )
    races = sorted({row["race_id"] for row in meta})
    if len(X) < 100:
        raise ValueError(f"not enough parsed examples: {len(X)}")
    if len(races) < min_train_races + folds:
        raise ValueError(f"not enough parsed races: {len(races)}")

    race_index = {race: idx for idx, race in enumerate(races)}
    test_window = max(1, (len(races) - min_train_races) // folds)
    all_probs: list[float] = []
    all_labels: list[int] = []
    race_predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fold_rows = []

    for fold in range(folds):
        test_start = min_train_races + fold * test_window
        test_end = len(races) if fold == folds - 1 else min(len(races), test_start + test_window)
        train_idx = [i for i, row in enumerate(meta) if race_index[row["race_id"]] < test_start]
        test_idx = [
            i
            for i, row in enumerate(meta)
            if test_start <= race_index[row["race_id"]] < test_end
        ]
        if not train_idx or not test_idx:
            continue
        pipeline = make_pipeline()
        pipeline.fit([X[i] for i in train_idx], [y[i] for i in train_idx])
        probs = positive_probs(pipeline, [X[i] for i in test_idx])
        labels = [y[i] for i in test_idx]
        all_probs.extend(probs)
        all_labels.extend(labels)
        for local_i, global_i in enumerate(test_idx):
            row = meta[global_i]
            race_predictions[row["race_id"]].append(
                {"lane": row["lane"], "rank": row["rank"], "probability": probs[local_i]}
            )
        fold_rows.append(
            {
                "fold": fold + 1,
                "train_races": test_start,
                "test_races": test_end - test_start,
                "entry_log_loss": _safe_log_loss(labels, probs),
                "entry_brier": brier_score_loss(labels, probs),
            }
        )

    result = {
        "generated_at": _now(),
        "folds": fold_rows,
        "examples": len(X),
        "races": len(races),
        "include_odds": False,
        "feature_set": FEATURE_SET,
        "evaluation_race_set_sha256": race_set_sha256(race_predictions),
        "entry_log_loss": _safe_log_loss(all_labels, all_probs),
        "entry_brier": brier_score_loss(all_labels, all_probs),
        **_race_level_metrics(race_predictions),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def predict_race(
    conn,
    *,
    model_path: Path,
    race_id_value: str,
    top_n: int = 30,
    store: bool = True,
    include_research: bool = True,
) -> list[dict[str, Any]]:
    bundle = load_model_bundle(model_path)
    pipeline = bundle["pipeline"]
    X = prediction_features(
        conn,
        race_id=race_id_value,
        include_odds=False,
        include_research=include_research,
    )
    if len(X) != 6:
        raise ValueError(f"race needs six entries before prediction: {race_id_value}")
    raw = positive_probs(pipeline, X)
    lane_probs = _normalize_lane_probs({lane: raw[lane - 1] for lane in range(1, 7)})
    rows = trifecta_predictions(lane_probs, latest_odds=latest_trifecta_odds(conn, race_id_value))[:top_n]
    if store:
        insert_prediction_rows(conn, race_id_value, _now(), str(model_path), rows)
    return rows


def predict_open_races(
    conn,
    *,
    model_path: Path,
    race_date: date,
    jcd: str | None = None,
    rno: int | None = None,
    include_research: bool = True,
) -> dict[str, int]:
    params: list[Any] = [race_date.isoformat()]
    filters = ["r.race_date = ?"]
    if jcd:
        filters.append("r.jcd = ?")
        params.append(jcd.zfill(2))
    if rno:
        filters.append("r.rno = ?")
        params.append(int(rno))
    race_rows = conn.execute(
        f"""
        SELECT r.race_id
        FROM races r
        WHERE {" AND ".join(filters)}
          AND (r.status IS NULL OR r.status != 'final')
          AND (SELECT COUNT(*) FROM entries e WHERE e.race_id = r.race_id) = 6
        ORDER BY r.jcd, r.rno
        """,
        params,
    ).fetchall()
    ok = 0
    failed = 0
    for row in race_rows:
        try:
            predict_race(
                conn,
                model_path=model_path,
                race_id_value=row["race_id"],
                include_research=include_research,
            )
            ok += 1
        except Exception:
            failed += 1
    return {"predicted": ok, "failed": failed}


def positive_probs(pipeline: Pipeline, X: list[dict[str, Any]]) -> list[float]:
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        classes = list(classifier.classes_)
        positive_index = classes.index(1)
        return [float(row[positive_index]) for row in pipeline.predict_proba(X)]
    scores = pipeline.decision_function(X)
    return [1.0 / (1.0 + pow(2.718281828459045, -float(score))) for score in scores]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train/backtest no-odds v4 model.")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    add_common(train)
    train.add_argument("--model", default="data/models/win_model_no_odds_v4.joblib")
    train.add_argument("--min-examples", type=int, default=100)
    train.set_defaults(func=_cmd_train)
    backtest = sub.add_parser("backtest")
    add_common(backtest)
    backtest.add_argument("--output", default="data/models/backtest_no_odds_v4.json")
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
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


def _safe_log_loss(labels: list[int], probs: list[float]) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    return float(log_loss(labels, probs, labels=[0, 1]))


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
