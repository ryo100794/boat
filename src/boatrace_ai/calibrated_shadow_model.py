from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .db import connection, init_db
try:
    from . import feature_tuning as _feature_source
except ImportError:
    from . import modeling_pastlog_v7_stream_hash as _feature_source

_ensure_sparse_index32 = _feature_source._ensure_sparse_index32
iter_race_feature_rows = _feature_source.iter_race_feature_rows
iter_training_entries = _feature_source.iter_training_entries
load_complete_race_ids = _feature_source.load_complete_race_ids
normalize_drop_feature_groups = _feature_source.normalize_drop_feature_groups
to_hashable = _feature_source.to_hashable

from .modeling import _race_level_metrics


FEATURE_SET = "pastlog_calibrated_hash_shadow"
MODEL_KINDS = ("linear", "mlp")


def train_bundle(
    conn,
    *,
    include_races: set[str],
    model_kind: str,
    drop_feature_groups: Iterable[str] | str | None = None,
    n_features: int = 1 << 14,
    batch_size: int = 12_000,
    epochs: int = 2,
    alpha: float = 0.0001,
) -> dict[str, Any]:
    model_kind = normalize_model_kind(model_kind)
    drop_feature_groups = normalize_drop_feature_groups(drop_feature_groups)
    hasher = FeatureHasher(
        n_features=max(256, int(n_features)),
        input_type="dict",
        alternate_sign=False,
    )
    scaler = StandardScaler(with_mean=False)
    scaler_batches = 0
    for features, _labels in iter_feature_batches(
        conn,
        include_races=include_races,
        drop_feature_groups=drop_feature_groups,
        batch_size=batch_size,
    ):
        scaler.partial_fit(hash_features(hasher, features))
        scaler_batches += 1
    if not scaler_batches:
        raise ValueError("no training examples for scaler")

    classifier = make_classifier(model_kind, alpha=alpha, batch_size=batch_size)
    first = True
    examples = 0
    for epoch in range(max(1, int(epochs))):
        for features, labels in iter_feature_batches(
            conn,
            include_races=include_races,
            drop_feature_groups=drop_feature_groups,
            batch_size=batch_size,
        ):
            matrix = scaler.transform(hash_features(hasher, features))
            kwargs = {"classes": np.asarray([0, 1], dtype=np.int64)} if first else {}
            classifier.partial_fit(matrix, np.asarray(labels, dtype=np.int64), **kwargs)
            first = False
            if epoch == 0:
                examples += len(labels)
    if first:
        raise ValueError("no training examples for classifier")
    return {
        "hasher": hasher,
        "scaler": scaler,
        "classifier": classifier,
        "model_kind": model_kind,
        "drop_feature_groups": list(drop_feature_groups),
        "examples": examples,
        "n_features": int(hasher.n_features),
        "epochs": max(1, int(epochs)),
        "alpha": float(alpha),
    }


def iter_feature_batches(
    conn,
    *,
    include_races: set[str],
    drop_feature_groups: Iterable[str] | str | None,
    batch_size: int,
) -> Iterable[tuple[list[dict[str, float]], list[int]]]:
    features: list[dict[str, float]] = []
    labels: list[int] = []
    for feature, label, _meta in iter_training_entries(
        conn,
        include_races=include_races,
        drop_feature_groups=drop_feature_groups,
    ):
        features.append(to_hashable(feature))
        labels.append(int(label))
        if len(features) >= max(1, int(batch_size)):
            yield features, labels
            features = []
            labels = []
    if features:
        yield features, labels


def hash_features(hasher: FeatureHasher, features: list[dict[str, float]]):
    return _ensure_sparse_index32(hasher.transform(features))


def make_classifier(model_kind: str, *, alpha: float, batch_size: int):
    if model_kind == "linear":
        return SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=float(alpha),
            max_iter=1,
            tol=None,
            random_state=42,
            average=True,
        )
    return MLPClassifier(
        hidden_layer_sizes=(64, 16),
        activation="relu",
        solver="adam",
        alpha=float(alpha),
        batch_size="auto",
        learning_rate_init=0.001,
        max_iter=1,
        random_state=42,
    )


def normalize_model_kind(value: str) -> str:
    model_kind = str(value).strip().lower()
    if model_kind not in MODEL_KINDS:
        raise ValueError(f"unknown model kind: {value}; choices: {', '.join(MODEL_KINDS)}")
    return model_kind


def predict_probabilities(bundle: dict[str, Any], features: list[dict[str, Any]]) -> list[float]:
    matrix = hash_features(
        bundle["hasher"],
        [to_hashable(feature) for feature in features],
    )
    matrix = bundle["scaler"].transform(matrix)
    probabilities = bundle["classifier"].predict_proba(matrix)[:, 1]
    return [float(value) for value in probabilities]


def iter_scored_races(
    conn,
    *,
    bundle: dict[str, Any],
    include_races: set[str],
) -> Iterable[list[dict[str, Any]]]:
    for race_features in iter_race_feature_rows(
        conn,
        include_races=include_races,
        drop_feature_groups=bundle.get("drop_feature_groups", ()),
    ):
        raw = predict_probabilities(
            bundle,
            [item["features"] for item in race_features],
        )
        total = sum(raw) or 1.0
        rows = []
        for item, probability in zip(race_features, raw):
            meta = item["meta"]
            rows.append(
                {
                    "race_id": str(meta["race_id"]),
                    "race_date": str(meta["race_date"]),
                    "jcd": str(meta["jcd"]),
                    "rno": int(meta["rno"]),
                    "lane": int(meta["lane"]),
                    "rank": int(meta["rank"]),
                    "label": int(meta["label"]),
                    "probability": float(probability) / total,
                }
            )
        yield rows


def backtest_model(
    conn,
    *,
    output_path: Path,
    model_kind: str,
    drop_feature_groups: Iterable[str] | str | None = None,
    folds: int = 5,
    min_train_races: int = 500,
    n_features: int = 1 << 14,
    batch_size: int = 12_000,
    epochs: int = 2,
    alpha: float = 0.0001,
) -> dict[str, Any]:
    model_kind = normalize_model_kind(model_kind)
    drop_feature_groups = normalize_drop_feature_groups(drop_feature_groups)
    race_keys = load_complete_race_ids(conn)
    races = [race_id for race_id, *_ in race_keys]
    if len(races) < min_train_races + folds:
        raise ValueError(f"not enough parsed races: {len(races)}")
    test_window = max(1, (len(races) - min_train_races) // folds)
    all_labels: list[int] = []
    all_probs: list[float] = []
    race_predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fold_rows = []

    for fold in range(folds):
        test_start = min_train_races + fold * test_window
        test_end = len(races) if fold == folds - 1 else min(len(races), test_start + test_window)
        train_races = set(races[:test_start])
        test_races = set(races[test_start:test_end])
        bundle = train_bundle(
            conn,
            include_races=train_races,
            model_kind=model_kind,
            drop_feature_groups=drop_feature_groups,
            n_features=n_features,
            batch_size=batch_size,
            epochs=epochs,
            alpha=alpha,
        )
        labels: list[int] = []
        probs: list[float] = []
        for rows in iter_scored_races(conn, bundle=bundle, include_races=test_races):
            for row in rows:
                label = int(row["label"])
                probability = float(row["probability"])
                labels.append(label)
                probs.append(probability)
                race_predictions[row["race_id"]].append(
                    {
                        "lane": row["lane"],
                        "rank": row["rank"],
                        "probability": probability,
                    }
                )
        all_labels.extend(labels)
        all_probs.extend(probs)
        fold_row = {
            "fold": fold + 1,
            "train_races": test_start,
            "test_races": test_end - test_start,
            "entry_log_loss": safe_log_loss(labels, probs),
            "entry_brier": float(brier_score_loss(labels, probs)),
        }
        fold_rows.append(fold_row)
        print(json.dumps({"model_kind": model_kind, **fold_row}, ensure_ascii=False), flush=True)

    result = {
        "generated_at": now_iso(),
        "model": f"calibrated_{model_kind}_shadow",
        "model_kind": model_kind,
        "role": "shadow",
        "feature_set": FEATURE_SET,
        "drop_feature_groups": list(drop_feature_groups),
        "include_odds": False,
        "scaler": "StandardScaler(with_mean=False, streaming partial_fit)",
        "class_weight": None,
        "n_features": int(n_features),
        "epochs": max(1, int(epochs)),
        "alpha": float(alpha),
        "folds": fold_rows,
        "examples": len(all_labels),
        "races": len(races),
        "evaluated_races": len(race_predictions),
        "entry_log_loss": safe_log_loss(all_labels, all_probs),
        "entry_brier": float(brier_score_loss(all_labels, all_probs)),
        **_race_level_metrics(race_predictions),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def train_model(
    conn,
    *,
    model_path: Path,
    model_kind: str,
    n_features: int,
    batch_size: int,
    epochs: int,
    alpha: float,
) -> dict[str, Any]:
    races = {race_id for race_id, *_ in load_complete_race_ids(conn)}
    bundle = train_bundle(
        conn,
        include_races=races,
        model_kind=model_kind,
        n_features=n_features,
        batch_size=batch_size,
        epochs=epochs,
        alpha=alpha,
    )
    metadata = {
        "trained_at": now_iso(),
        "model": f"calibrated_{normalize_model_kind(model_kind)}_shadow",
        "role": "shadow",
        "feature_set": FEATURE_SET,
        "races": len(races),
        "examples": bundle["examples"],
        "n_features": bundle["n_features"],
        "epochs": bundle["epochs"],
        "alpha": bundle["alpha"],
        "include_odds": False,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({**bundle, "metadata": metadata}, model_path)
    return metadata


def safe_log_loss(labels: list[int], probs: list[float]) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    return float(log_loss(labels, probs, labels=[0, 1]))


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrated linear/MLP past-log shadow model.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    backtest = subparsers.add_parser("backtest")
    add_common(backtest)
    backtest.add_argument("--output", required=True)
    backtest.add_argument("--folds", type=int, default=5)
    backtest.add_argument("--min-train-races", type=int, default=500)
    backtest.set_defaults(handler=run_backtest)
    train = subparsers.add_parser("train")
    add_common(train)
    train.add_argument("--model", required=True)
    train.set_defaults(handler=run_train)
    args = parser.parse_args(argv)
    return int(args.handler(args) or 0)


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--model-kind", choices=MODEL_KINDS, required=True)
    parser.add_argument("--n-features", type=int, default=1 << 14)
    parser.add_argument("--batch-size", type=int, default=12_000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.0001)


def run_backtest(args: argparse.Namespace) -> int:
    init_db(args.db)
    with connection(args.db) as conn:
        result = backtest_model(
            conn,
            output_path=Path(args.output),
            model_kind=args.model_kind,
            folds=args.folds,
            min_train_races=args.min_train_races,
            n_features=args.n_features,
            batch_size=args.batch_size,
            epochs=args.epochs,
            alpha=args.alpha,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


def run_train(args: argparse.Namespace) -> int:
    init_db(args.db)
    with connection(args.db) as conn:
        result = train_model(
            conn,
            model_path=Path(args.model),
            model_kind=args.model_kind,
            n_features=args.n_features,
            batch_size=args.batch_size,
            epochs=args.epochs,
            alpha=args.alpha,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
