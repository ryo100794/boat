from __future__ import annotations

import argparse
import json
import time
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

from .adaptive_allocation import zero_totals
from .bankroll_backtest import _load_trifecta_payouts
from .db import connection, init_db
from .standard_evaluation import race_set_sha256
from .hashed_feature_dataset import (
    HashedRaceDataset,
    load_or_build_hashed_dataset,
)
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


def train_bundle_from_dataset(
    dataset: HashedRaceDataset,
    *,
    train_race_count: int,
    model_kind: str,
    batch_size: int = 12_000,
    epochs: int = 2,
    alpha: float = 0.0001,
) -> dict[str, Any]:
    model_kind = normalize_model_kind(model_kind)
    train_count = min(dataset.race_count, int(train_race_count))
    train_end = train_count * 6
    if train_end <= 0:
        raise ValueError("no cached training examples")
    labels = (dataset.ranks[:train_count].reshape(-1) == 1).astype(np.int64)
    if len(labels) != train_end:
        raise ValueError("cached label shape mismatch")
    scaler = StandardScaler(with_mean=False)
    for start, end in matrix_batch_ranges(train_end, batch_size):
        scaler.partial_fit(dataset.matrix[start:end])

    classifier = make_classifier(model_kind, alpha=alpha, batch_size=batch_size)
    first = True
    for _epoch in range(max(1, int(epochs))):
        for start, end in matrix_batch_ranges(train_end, batch_size):
            matrix = scaler.transform(dataset.matrix[start:end])
            kwargs = {"classes": np.asarray([0, 1], dtype=np.int64)} if first else {}
            classifier.partial_fit(matrix, labels[start:end], **kwargs)
            first = False
    if first:
        raise ValueError("no cached classifier examples")
    return {
        "scaler": scaler,
        "classifier": classifier,
        "model_kind": model_kind,
        "drop_feature_groups": list(dataset.drop_feature_groups),
        "examples": train_end,
        "n_features": dataset.n_features,
        "epochs": max(1, int(epochs)),
        "alpha": float(alpha),
        "matrix_cached": True,
    }


def score_dataset_fold(
    dataset: HashedRaceDataset,
    *,
    bundle: dict[str, Any],
    race_start: int,
    race_end: int,
    batch_size: int,
) -> Iterable[list[dict[str, Any]]]:
    row_slice = dataset.row_slice(race_start, race_end)
    matrix = dataset.matrix[row_slice]
    raw_parts = []
    for start, end in matrix_batch_ranges(matrix.shape[0], batch_size):
        transformed = bundle["scaler"].transform(matrix[start:end])
        raw_parts.append(bundle["classifier"].predict_proba(transformed)[:, 1])
    if not raw_parts:
        return
    raw = np.concatenate(raw_parts).reshape(-1, 6)
    totals = raw.sum(axis=1)
    totals[totals == 0.0] = 1.0
    normalized = raw / totals[:, None]
    for local_race, race_index in enumerate(range(race_start, race_end)):
        race_id, race_date, jcd, rno = dataset.race_keys[race_index]
        yield [
            {
                "race_id": race_id,
                "race_date": race_date,
                "jcd": jcd,
                "rno": int(rno),
                "lane": lane,
                "rank": int(dataset.ranks[race_index, lane - 1]),
                "label": 1 if int(dataset.ranks[race_index, lane - 1]) == 1 else 0,
                "probability": float(normalized[local_race, lane - 1]),
            }
            for lane in range(1, 7)
        ]


def matrix_batch_ranges(row_count: int, batch_size: int) -> Iterable[tuple[int, int]]:
    size = max(1, int(batch_size))
    for start in range(0, int(row_count), size):
        yield start, min(int(row_count), start + size)


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
    feature_cache: Path | None = None,
    daily_budget_yen: int = 10_000,
    ev_threshold: float = 1.20,
) -> dict[str, Any]:
    # Imported lazily because listwise.model reuses the matrix batching helper here.
    from .listwise.validation import default_policy, evaluate_bankroll_fold

    model_kind = normalize_model_kind(model_kind)
    drop_feature_groups = normalize_drop_feature_groups(drop_feature_groups)
    race_keys = load_complete_race_ids(conn)
    races = [race_id for race_id, *_ in race_keys]
    if len(races) < min_train_races + folds:
        raise ValueError(f"not enough parsed races: {len(races)}")

    hasher = FeatureHasher(
        n_features=max(256, int(n_features)),
        input_type="dict",
        alternate_sign=False,
    )
    cache_started = time.perf_counter()
    dataset, cache_source = load_or_build_hashed_dataset(
        cache_prefix=feature_cache,
        race_keys=race_keys,
        race_rows=lambda: iter_race_feature_rows(
            conn,
            include_races=set(races),
            drop_feature_groups=drop_feature_groups,
        ),
        hasher=hasher,
        to_hashable=to_hashable,
        ensure_sparse_index32=_ensure_sparse_index32,
        drop_feature_groups=drop_feature_groups,
        batch_size=batch_size,
    )
    cache_elapsed = time.perf_counter() - cache_started
    test_window = max(1, (len(races) - min_train_races) // folds)
    all_labels: list[int] = []
    all_probs: list[float] = []
    race_predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fold_rows = []

    for fold in range(folds):
        fold_started = time.perf_counter()
        test_start = min_train_races + fold * test_window
        test_end = (
            len(races)
            if fold == folds - 1
            else min(len(races), test_start + test_window)
        )
        bundle = train_bundle_from_dataset(
            dataset,
            train_race_count=test_start,
            model_kind=model_kind,
            batch_size=batch_size,
            epochs=epochs,
            alpha=alpha,
        )
        labels: list[int] = []
        probs: list[float] = []
        for rows in score_dataset_fold(
            dataset,
            bundle=bundle,
            race_start=test_start,
            race_end=test_end,
            batch_size=batch_size,
        ):
            for row in rows:
                label = int(row["label"])
                probability = float(row["probability"])
                labels.append(label)
                probs.append(probability)
                race_predictions[row["race_id"]].append(
                    {
                        "race_id": row["race_id"],
                        "race_date": row["race_date"],
                        "jcd": row["jcd"],
                        "rno": row["rno"],
                        "lane": row["lane"],
                        "rank": row["rank"],
                        "label": label,
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
            "elapsed_seconds": round(time.perf_counter() - fold_started, 3),
        }
        fold_rows.append(fold_row)
        print(
            json.dumps({"model_kind": model_kind, **fold_row}, ensure_ascii=False),
            flush=True,
        )

    policy = default_policy(
        daily_budget_yen=daily_budget_yen,
        ev_threshold=ev_threshold,
    )
    policy.update(
        {
            "model": f"calibrated_{model_kind}_shadow",
            "feature_set": FEATURE_SET,
        }
    )
    totals = zero_totals()
    daily_rows: list[dict[str, Any]] = []
    bankroll, profit_state = evaluate_bankroll_fold(
        rows_by_race=race_predictions,
        train_races=set(races[:min_train_races]),
        test_dates={
            race_date
            for _race_id, race_date, _jcd, _rno in race_keys[min_train_races:]
        },
        payouts=_load_trifecta_payouts(conn),
        policy=policy,
        totals=totals,
        daily_rows=daily_rows,
        profit_state=(0, 0, 0),
    )
    tickets = int(totals["tickets"])
    selected_races = int(totals["races_bet"])
    stake_yen = int(bankroll["stake_yen"])
    bankroll_summary = {
        **bankroll,
        "evaluated_races": int(totals["evaluated_races"]),
        "race_days": len(daily_rows),
        "selected_races": selected_races,
        "tickets": tickets,
        "hit_tickets": int(totals["hit_tickets"]),
        "ticket_hit_rate": (
            float(totals["hit_tickets"]) / tickets if tickets else 0.0
        ),
        "race_hit_rate": (
            float(totals["hit_races"]) / selected_races
            if selected_races
            else 0.0
        ),
        "winning_days": int(totals["winning_days"]),
        "losing_days": int(totals["losing_days"]),
        "budget_utilization": (
            stake_yen / (daily_budget_yen * len(daily_rows))
            if daily_rows
            else 0.0
        ),
        "max_drawdown_yen": int(profit_state[2]),
    }
    evaluation_hash = race_set_sha256(race_predictions)
    bankroll_summary["evaluation_race_set_sha256"] = evaluation_hash
    bankroll_flat = {
        key: value
        for key, value in bankroll_summary.items()
        if key != "evaluated_races"
    }
    result = {
        "generated_at": now_iso(),
        "model": f"calibrated_{model_kind}_shadow",
        "model_kind": model_kind,
        "role": "shadow",
        "feature_set": FEATURE_SET,
        "drop_feature_groups": list(drop_feature_groups),
        "include_odds": False,
        "scaler": "StandardScaler(with_mean=False, cached CSR partial_fit)",
        "class_weight": None,
        "n_features": int(n_features),
        "epochs": max(1, int(epochs)),
        "alpha": float(alpha),
        "feature_cache_source": cache_source,
        "feature_cache_prefix": str(feature_cache) if feature_cache else None,
        "feature_cache_seconds": round(cache_elapsed, 3),
        "matrix_shape": list(dataset.matrix.shape),
        "matrix_nnz": int(dataset.matrix.nnz),
        "folds": fold_rows,
        "examples": len(all_labels),
        "races": len(races),
        "evaluated_races": len(race_predictions),
        "evaluation_race_set_sha256": evaluation_hash,
        "entry_log_loss": safe_log_loss(all_labels, all_probs),
        "entry_brier": float(brier_score_loss(all_labels, all_probs)),
        **_race_level_metrics(race_predictions),
        "policy": policy,
        "bankroll": bankroll_summary,
        "bankroll_evaluated_races": bankroll_summary["evaluated_races"],
        "daily": daily_rows,
        **bankroll_flat,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
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
    backtest.add_argument("--daily-budget-yen", type=int, default=10_000)
    backtest.add_argument("--ev-threshold", type=float, default=1.20)
    backtest.add_argument(
        "--feature-cache",
        default="data/models/calibrated_shadow_features_16384",
    )
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
            feature_cache=Path(args.feature_cache) if args.feature_cache else None,
            daily_budget_yen=args.daily_budget_yen,
            ev_threshold=args.ev_threshold,
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
