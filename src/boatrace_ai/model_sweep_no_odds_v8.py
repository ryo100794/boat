from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

from .db import connection, init_db
from .features_no_odds_v3 import load_training_examples
from .modeling import _race_level_metrics
from .modeling_no_odds_v6 import SparseIndex32
from .modeling_no_odds_v7 import FEATURE_SET, positive_probs


VARIANTS = {
    "logreg_liblinear_C0.20_unweighted": lambda: _logreg_pipeline(C=0.20, solver="liblinear", class_weight=None),
    "logreg_liblinear_C0.50_unweighted": lambda: _logreg_pipeline(C=0.50, solver="liblinear", class_weight=None),
    "logreg_liblinear_C1.00_unweighted": lambda: _logreg_pipeline(C=1.00, solver="liblinear", class_weight=None),
    "logreg_liblinear_C0.50_winner2": lambda: _logreg_pipeline(C=0.50, solver="liblinear", class_weight={0: 1.0, 1: 2.0}),
    "logreg_lbfgs_C0.50_unweighted": lambda: _logreg_pipeline(C=0.50, solver="lbfgs", class_weight=None),
}


def _logreg_pipeline(*, C: float, solver: str, class_weight: dict[int, float] | None) -> Pipeline:
    return Pipeline(
        [
            ("vectorizer", DictVectorizer(sparse=True)),
            ("sparse_index_32_a", SparseIndex32()),
            ("scaler", MaxAbsScaler(copy=False)),
            ("sparse_index_32_b", SparseIndex32()),
            (
                "classifier",
                LogisticRegression(
                    solver=solver,
                    C=C,
                    max_iter=1000,
                    class_weight=class_weight,
                    random_state=42,
                ),
            ),
        ]
    )


def sweep(
    conn,
    *,
    output_path: Path,
    folds: int,
    min_train_races: int,
    variant_names: list[str],
) -> dict[str, Any]:
    X, y, meta = load_training_examples(conn, include_odds=False)
    races = sorted({row["race_id"] for row in meta})
    if len(X) < 100:
        raise ValueError(f"not enough parsed examples: {len(X)}")
    if len(races) < min_train_races + folds:
        raise ValueError(f"not enough parsed races: {len(races)}")
    race_index = {race: idx for idx, race in enumerate(races)}
    test_window = max(1, (len(races) - min_train_races) // folds)

    results = []
    for name in variant_names:
        if name not in VARIANTS:
            raise ValueError(f"unknown variant: {name}")
        result = evaluate_variant(
            name,
            X=X,
            y=y,
            meta=meta,
            races=races,
            race_index=race_index,
            test_window=test_window,
            folds=folds,
            min_train_races=min_train_races,
        )
        results.append(result)
        print(json.dumps({"event": "variant_done", **result}, ensure_ascii=False), flush=True)

    payload = {
        "generated_at": _now(),
        "base_feature_set": FEATURE_SET,
        "examples": len(X),
        "races": len(races),
        "folds": folds,
        "min_train_races": min_train_races,
        "results": sorted(
            results,
            key=lambda row: (
                row["winner_top1_accuracy"],
                row["trifecta_top5_hit_rate"],
                -row["entry_log_loss"],
            ),
            reverse=True,
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def evaluate_variant(
    name: str,
    *,
    X: list[dict[str, Any]],
    y: list[int],
    meta: list[dict[str, Any]],
    races: list[str],
    race_index: dict[str, int],
    test_window: int,
    folds: int,
    min_train_races: int,
) -> dict[str, Any]:
    all_probs: list[float] = []
    all_labels: list[int] = []
    race_predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fold_rows = []
    for fold in range(folds):
        test_start = min_train_races + fold * test_window
        test_end = len(races) if fold == folds - 1 else min(len(races), test_start + test_window)
        train_idx = [i for i, row in enumerate(meta) if race_index[row["race_id"]] < test_start]
        test_idx = [i for i, row in enumerate(meta) if test_start <= race_index[row["race_id"]] < test_end]
        if not train_idx or not test_idx:
            continue
        pipeline = VARIANTS[name]()
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
    return {
        "variant": name,
        "entry_log_loss": _safe_log_loss(all_labels, all_probs),
        "entry_brier": brier_score_loss(all_labels, all_probs),
        **_race_level_metrics(race_predictions),
        "folds": fold_rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sweep no-odds v8 candidate models.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--output", default="data/models/model_sweep_no_odds_v8.json")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--min-train-races", type=int, default=500)
    parser.add_argument("--variants", default=",".join(VARIANTS.keys()))
    args = parser.parse_args(argv)
    variant_names = [item.strip() for item in args.variants.split(",") if item.strip()]

    init_db(args.db)
    with connection(args.db) as conn:
        payload = sweep(
            conn,
            output_path=Path(args.output),
            folds=args.folds,
            min_train_races=args.min_train_races,
            variant_names=variant_names,
        )
    print(json.dumps({"event": "sweep_done", **{k: v for k, v in payload.items() if k != "results"}, "best": payload["results"][0]}, ensure_ascii=False), flush=True)
    return 0


def _safe_log_loss(labels: list[int], probs: list[float]) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    return float(log_loss(labels, probs, labels=[0, 1]))


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
