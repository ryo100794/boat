from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline

from .db import connection, init_db
from .features import load_training_examples
from .modeling import _race_level_metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dense sklearn training/backtest tasks.")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train")
    add_common(train)
    train.add_argument("--model", default="data/models/win_model.joblib")
    train.add_argument("--min-examples", type=int, default=100)
    train.set_defaults(func=cmd_train)

    backtest = sub.add_parser("backtest")
    add_common(backtest)
    backtest.add_argument("--output", default="data/models/backtest.json")
    backtest.add_argument("--folds", type=int, default=5)
    backtest.add_argument("--min-train-races", type=int, default=500)
    backtest.set_defaults(func=cmd_backtest)

    cycle = sub.add_parser("cycle")
    add_common(cycle)
    cycle.add_argument("--model", default="data/models/win_model.joblib")
    cycle.add_argument("--backtest", default="data/models/backtest.json")
    cycle.add_argument("--interval", type=float, default=300.0)
    cycle.add_argument("--min-examples", type=int, default=100)
    cycle.add_argument("--min-train-races", type=int, default=500)
    cycle.add_argument("--folds", type=int, default=5)
    cycle.add_argument("--max-loops", type=int)
    cycle.set_defaults(func=cmd_cycle)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", default="data/boatrace.sqlite")


def cmd_train(args: argparse.Namespace) -> int:
    init_db(args.db)
    with connection(args.db) as conn:
        result = train_dense(conn, model_path=Path(args.model), min_examples=args.min_examples)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    init_db(args.db)
    with connection(args.db) as conn:
        result = backtest_dense(
            conn,
            output_path=Path(args.output),
            folds=args.folds,
            min_train_races=args.min_train_races,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


def cmd_cycle(args: argparse.Namespace) -> int:
    init_db(args.db)
    loop = 0
    while True:
        event = {"loop": loop, "trained": False, "backtested": False}
        try:
            with connection(args.db) as conn:
                counts = counts_for_training(conn)
                event["counts"] = counts
                if counts["examples"] >= args.min_examples:
                    event["train"] = train_dense(
                        conn,
                        model_path=Path(args.model),
                        min_examples=args.min_examples,
                    )
                    event["trained"] = True
                if counts["races"] >= args.min_train_races + args.folds:
                    bt = backtest_dense(
                        conn,
                        output_path=Path(args.backtest),
                        folds=args.folds,
                        min_train_races=args.min_train_races,
                    )
                    event["backtest"] = {
                        "evaluated_races": bt.get("evaluated_races"),
                        "winner_top1_accuracy": bt.get("winner_top1_accuracy"),
                        "trifecta_top5_hit_rate": bt.get("trifecta_top5_hit_rate"),
                    }
                    event["backtested"] = True
        except Exception as exc:
            event["error"] = str(exc)
        print(json.dumps(event, ensure_ascii=False), flush=True)
        loop += 1
        if args.max_loops is not None and loop >= args.max_loops:
            return 0
        time.sleep(args.interval)


def train_dense(conn, *, model_path: Path, min_examples: int) -> dict[str, Any]:
    X, y, meta = load_training_examples(conn, include_odds=False)
    if len(X) < min_examples:
        raise ValueError(f"training examples are too few: {len(X)} < {min_examples}")
    if len(set(y)) < 2:
        raise ValueError("training labels need both winners and non-winners")
    pipeline = make_pipeline()
    pipeline.fit(X, y)
    result = {
        "trained_at": now(),
        "examples": len(X),
        "races": len({row["race_id"] for row in meta}),
        "include_odds": False,
        "target": "lane_win_probability",
        "vectorizer": "dense",
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metadata": result}, model_path)
    return result


def backtest_dense(
    conn,
    *,
    output_path: Path,
    folds: int,
    min_train_races: int,
) -> dict[str, Any]:
    X, y, meta = load_training_examples(conn, include_odds=False)
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
                "entry_log_loss": safe_log_loss(labels, probs),
                "entry_brier": brier_score_loss(labels, probs),
            }
        )
    result = {
        "generated_at": now(),
        "folds": fold_rows,
        "examples": len(X),
        "races": len(races),
        "include_odds": False,
        "entry_log_loss": safe_log_loss(all_labels, all_probs),
        "entry_brier": brier_score_loss(all_labels, all_probs),
        **_race_level_metrics(race_predictions),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def counts_for_training(conn) -> dict[str, int]:
    examples = conn.execute(
        """
        SELECT COUNT(*)
        FROM entries e
        JOIN race_results rr ON rr.race_id = e.race_id AND rr.lane = e.lane
        WHERE rr.rank IS NOT NULL
        """
    ).fetchone()[0]
    races = conn.execute(
        """
        SELECT COUNT(DISTINCT e.race_id)
        FROM entries e
        JOIN race_results rr ON rr.race_id = e.race_id AND rr.lane = e.lane
        WHERE rr.rank IS NOT NULL
        """
    ).fetchone()[0]
    return {"examples": int(examples), "races": int(races)}


def make_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("vectorizer", DictVectorizer(sparse=False)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=600,
                    class_weight="balanced",
                    solver="liblinear",
                ),
            ),
        ]
    )


def positive_probs(pipeline: Pipeline, X: list[dict[str, Any]]) -> list[float]:
    classifier = pipeline.named_steps["classifier"]
    positive_index = list(classifier.classes_).index(1)
    return [float(row[positive_index]) for row in pipeline.predict_proba(X)]


def safe_log_loss(labels: list[int], probs: list[float]) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    return float(log_loss(labels, probs, labels=[0, 1]))


def now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
