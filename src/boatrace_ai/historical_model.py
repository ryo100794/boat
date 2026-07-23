from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

from . import model_core as base
from .db import connection, init_db
from .base_features import iter_training_examples
from .feature_tuning import load_complete_race_ids
from .modeling import _race_level_metrics
from .standard_evaluation import race_set_sha256


FEATURE_SET = (
    "no_odds_v8_historical_only_beforeinfo_excluded_"
    "sparse32_scaled_logreg_C0.20_unweighted"
)


SparseIndex32 = base.SparseIndex32


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
                    C=0.20,
                    max_iter=1000,
                    class_weight=None,
                    random_state=42,
                ),
            ),
        ]
    )


base.FEATURE_SET = FEATURE_SET
base.make_pipeline = make_pipeline


def train_model(
    conn,
    *,
    model_path: Path,
    min_examples: int = 100,
    batch_size: int = 24_000,
) -> dict[str, Any]:
    race_keys = load_complete_race_ids(conn)
    races = {race_id for race_id, *_rest in race_keys}
    if len(races) * 6 < min_examples:
        raise ValueError(
            f"training examples are too few: {len(races) * 6} < {min_examples}"
        )
    bundle = fit_streaming_pipeline(
        conn,
        train_races=races,
        batch_size=batch_size,
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    return dict(bundle["metadata"])


def fit_streaming_pipeline(
    conn,
    *,
    train_races: set[str],
    batch_size: int = 24_000,
) -> dict[str, Any]:
    if not train_races:
        raise ValueError("no training races")
    pipeline = make_pipeline()
    vectorizer: DictVectorizer = pipeline.named_steps["vectorizer"]
    vectorizer.fit(
        item
        for item, _label, _meta in iter_training_examples(
            conn,
            include_odds=False,
            include_research=False,
            include_beforeinfo=False,
            include_races=train_races,
        )
    )
    matrix, labels = _training_matrix(
        conn,
        vectorizer=vectorizer,
        train_races=train_races,
        batch_size=batch_size,
    )
    if len(set(labels.tolist())) < 2:
        raise ValueError("training labels need both winners and non-winners")
    scaler: MaxAbsScaler = pipeline.named_steps["scaler"]
    matrix = scaler.fit_transform(matrix)
    matrix = pipeline.named_steps["sparse_index_32_b"].transform(matrix)
    classifier: LogisticRegression = pipeline.named_steps["classifier"]
    classifier.fit(matrix, labels)
    metadata = {
        "trained_at": base._now(),
        "examples": int(labels.size),
        "races": len(train_races),
        "include_odds": False,
        "include_beforeinfo": False,
        "target": "lane_win_probability",
        "vectorizer": "DictVectorizer(sparse=True, streamed vocabulary and CSR batches)",
        "scaler": "MaxAbsScaler",
        "classifier": "LogisticRegression(liblinear, C=0.20, class_weight=None)",
        "feature_set": FEATURE_SET,
        "train_race_set_sha256": race_set_sha256(train_races),
    }
    del matrix, labels
    gc.collect()
    return {"pipeline": pipeline, "metadata": metadata}


def _training_matrix(
    conn,
    *,
    vectorizer: DictVectorizer,
    train_races: set[str],
    batch_size: int,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    matrices: list[sparse.csr_matrix] = []
    labels: list[int] = []
    batch: list[dict[str, Any]] = []
    for item, label, meta in iter_training_examples(
        conn,
        include_odds=False,
        include_research=False,
        include_beforeinfo=False,
        include_races=train_races,
    ):
        batch.append(item)
        labels.append(label)
        if len(batch) >= batch_size:
            matrices.append(_vectorize_batch(vectorizer, batch))
            batch.clear()
    if batch:
        matrices.append(_vectorize_batch(vectorizer, batch))
    if not matrices:
        raise ValueError("no training examples")
    matrix = sparse.vstack(matrices, format="csr")
    del matrices
    gc.collect()
    return matrix, np.asarray(labels, dtype=np.int8)


def _vectorize_batch(
    vectorizer: DictVectorizer,
    batch: list[dict[str, Any]],
) -> sparse.csr_matrix:
    matrix = vectorizer.transform(batch)
    return SparseIndex32().transform(matrix)


def iter_scored_entries(
    conn,
    *,
    pipeline: Pipeline,
    include_races: set[str],
    batch_size: int = 24_000,
) -> Iterable[tuple[float, dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for item, _label, meta in iter_training_examples(
        conn,
        include_odds=False,
        include_research=False,
        include_beforeinfo=False,
        include_races=include_races,
    ):
        batch.append(item)
        metadata.append(meta)
        if len(batch) >= batch_size:
            probabilities = base.positive_probs(pipeline, batch)
            yield from zip(probabilities, metadata)
            batch.clear()
            metadata.clear()
    if batch:
        probabilities = base.positive_probs(pipeline, batch)
        yield from zip(probabilities, metadata)


def backtest_model(
    conn,
    *,
    output_path: Path,
    folds: int = 5,
    min_train_races: int = 500,
    batch_size: int = 24_000,
    model_output_path: Path | None = None,
) -> dict[str, Any]:
    if model_output_path is not None and folds != 1:
        raise ValueError("model_output_path requires folds=1")
    race_keys = load_complete_race_ids(conn)
    races = [race_id for race_id, *_rest in race_keys]
    if len(races) < min_train_races + folds:
        raise ValueError(f"not enough parsed races: {len(races)}")
    test_window = max(1, (len(races) - min_train_races) // folds)
    all_probs: list[float] = []
    all_labels: list[int] = []
    race_predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fold_rows: list[dict[str, Any]] = []

    for fold in range(folds):
        test_start = min_train_races + fold * test_window
        test_end = (
            len(races)
            if fold == folds - 1
            else min(len(races), test_start + test_window)
        )
        train_races = set(races[:test_start])
        test_races = set(races[test_start:test_end])
        bundle = fit_streaming_pipeline(
            conn,
            train_races=train_races,
            batch_size=batch_size,
        )
        bundle["metadata"].update(
            {
                "folds": folds,
                "train_races": len(train_races),
                "train_race_set_sha256": race_set_sha256(train_races),
            }
        )
        if model_output_path is not None:
            model_output_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(bundle, model_output_path)
        labels: list[int] = []
        probabilities: list[float] = []
        for probability, meta in iter_scored_entries(
            conn,
            pipeline=bundle["pipeline"],
            include_races=test_races,
            batch_size=batch_size,
        ):
            label = 1 if int(meta["rank"]) == 1 else 0
            labels.append(label)
            probabilities.append(float(probability))
            race_predictions[str(meta["race_id"])].append(
                {
                    "lane": int(meta["lane"]),
                    "rank": int(meta["rank"]),
                    "probability": float(probability),
                }
            )
        all_probs.extend(probabilities)
        all_labels.extend(labels)
        fold_rows.append(
            {
                "fold": fold + 1,
                "train_races": len(train_races),
                "test_races": len(test_races),
                "entry_log_loss": base._safe_log_loss(labels, probabilities),
                "entry_brier": float(brier_score_loss(labels, probabilities)),
            }
        )
        print(json.dumps(fold_rows[-1], ensure_ascii=False), flush=True)

    result = {
        "generated_at": base._now(),
        "folds": fold_rows,
        "examples": len(races) * 6,
        "races": len(races),
        "include_odds": False,
        "include_beforeinfo": False,
        "feature_set": FEATURE_SET,
        "evaluation_race_set_sha256": race_set_sha256(race_predictions),
        "entry_log_loss": base._safe_log_loss(all_labels, all_probs),
        "entry_brier": float(brier_score_loss(all_labels, all_probs)),
        **_race_level_metrics(race_predictions),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def predict_race(conn, **kwargs):
    kwargs.pop("include_beforeinfo", None)
    return base.predict_race(
        conn,
        include_research=False,
        include_beforeinfo=False,
        **kwargs,
    )


def predict_open_races(conn, **kwargs):
    kwargs.pop("include_beforeinfo", None)
    return base.predict_open_races(
        conn,
        include_research=False,
        include_beforeinfo=False,
        **kwargs,
    )


positive_probs = base.positive_probs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train/backtest no-odds v8 model.")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    add_common(train)
    train.add_argument("--model", default="data/models/win_model_no_odds_v8.joblib")
    train.add_argument("--min-examples", type=int, default=100)
    train.set_defaults(func=_cmd_train)
    backtest = sub.add_parser("backtest")
    add_common(backtest)
    backtest.add_argument("--output", default="data/models/backtest_no_odds_v8.json")
    backtest.add_argument("--folds", type=int, default=5)
    backtest.add_argument("--min-train-races", type=int, default=500)
    backtest.add_argument("--batch-size", type=int, default=24_000)
    backtest.add_argument("--model-output")
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
            batch_size=args.batch_size,
            model_output_path=Path(args.model_output) if args.model_output else None,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
