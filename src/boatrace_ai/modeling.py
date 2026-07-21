from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from .legacy_model_aliases import load_model_bundle
from .db import insert_prediction_rows, race_id
from .fast_math import TRIFECTA_COMBINATIONS, plackett_luce_probabilities
from .features import (
    entry_features,
    latest_trifecta_odds,
    load_race_entries,
    load_training_examples,
    odds_lane_features,
)


def train_model(
    conn,
    *,
    model_path: Path,
    include_odds: bool = False,
    through_date: str | None = None,
    from_date: str | None = None,
    min_odds_snapshots: int = 0,
    complete_results_only: bool = False,
    min_examples: int = 100,
) -> dict[str, Any]:
    X, y, meta = load_training_examples(
        conn,
        through_date=through_date,
        from_date=from_date,
        include_odds=include_odds,
        min_odds_snapshots=min_odds_snapshots,
        complete_results_only=complete_results_only,
    )
    if len(X) < min_examples:
        raise ValueError(f"training examples are too few: {len(X)} < {min_examples}")
    if len(set(y)) < 2:
        raise ValueError("training labels need both winners and non-winners")
    pipeline = _make_pipeline()
    pipeline.fit(X, y)
    metadata = {
        "trained_at": _now(),
        "examples": len(X),
        "races": len({row["race_id"] for row in meta}),
        "through_date": through_date,
        "from_date": from_date,
        "include_odds": include_odds,
        "target": "lane_win_probability",
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, model_path)
    return metadata


def backtest_model(
    conn,
    *,
    output_path: Path,
    folds: int = 5,
    include_odds: bool = False,
    from_date: str | None = None,
    min_odds_snapshots: int = 0,
    complete_results_only: bool = False,
    min_train_races: int = 500,
) -> dict[str, Any]:
    X, y, meta = load_training_examples(
        conn,
        from_date=from_date,
        include_odds=include_odds,
        min_odds_snapshots=min_odds_snapshots,
        complete_results_only=complete_results_only,
    )
    if len(X) < 100:
        raise ValueError(f"not enough parsed historical examples: {len(X)}")
    races = sorted({row["race_id"] for row in meta})
    if len(races) < max(10, min_train_races + folds):
        raise ValueError(
            f"not enough parsed historical races: {len(races)} "
            f"(need at least {min_train_races + folds})"
        )
    race_index = {race: idx for idx, race in enumerate(races)}
    test_window = max(1, (len(races) - min_train_races) // folds)
    fold_results = []
    all_entry_probs: list[float] = []
    all_entry_labels: list[int] = []
    race_predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for fold in range(folds):
        test_start = min_train_races + fold * test_window
        test_end = len(races) if fold == folds - 1 else min(len(races), test_start + test_window)
        train_idx = [
            i for i, row in enumerate(meta) if race_index[row["race_id"]] < test_start
        ]
        test_idx = [
            i
            for i, row in enumerate(meta)
            if test_start <= race_index[row["race_id"]] < test_end
        ]
        if not train_idx or not test_idx:
            continue
        pipeline = _make_pipeline()
        pipeline.fit([X[i] for i in train_idx], [y[i] for i in train_idx])
        probs = _positive_probs(pipeline, [X[i] for i in test_idx])
        fold_labels = [y[i] for i in test_idx]
        all_entry_probs.extend(probs)
        all_entry_labels.extend(fold_labels)
        for local_i, global_i in enumerate(test_idx):
            row = meta[global_i]
            race_predictions[row["race_id"]].append(
                {
                    "lane": row["lane"],
                    "rank": row["rank"],
                    "probability": probs[local_i],
                }
            )
        fold_results.append(
            {
                "fold": fold + 1,
                "train_races": test_start,
                "test_races": test_end - test_start,
                "entry_log_loss": _safe_log_loss(fold_labels, probs),
                "entry_brier": brier_score_loss(fold_labels, probs),
            }
        )

    race_metrics = _race_level_metrics(race_predictions)
    result = {
        "generated_at": _now(),
        "folds": fold_results,
        "examples": len(X),
        "races": len(races),
        "include_odds": include_odds,
        "from_date": from_date,
        "entry_log_loss": _safe_log_loss(all_entry_labels, all_entry_probs),
        "entry_brier": brier_score_loss(all_entry_labels, all_entry_probs),
        **race_metrics,
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
) -> list[dict[str, Any]]:
    bundle = load_model_bundle(model_path)
    pipeline = bundle["pipeline"]
    entries = load_race_entries(conn, race_id=race_id_value)
    if len(entries) != 6:
        raise ValueError(f"race needs six entries before prediction: {race_id_value}")
    odds_features = odds_lane_features(conn, race_id_value)
    X = [
        entry_features(row, odds_features=odds_features.get(int(row["lane"]), {}))
        for row in entries
    ]
    lane_probs_raw = _positive_probs(pipeline, X)
    lane_probs = _normalize_lane_probs(
        {int(entry["lane"]): lane_probs_raw[index] for index, entry in enumerate(entries)}
    )
    latest_odds = latest_trifecta_odds(conn, race_id_value)
    rows = trifecta_predictions(lane_probs, latest_odds=latest_odds)
    rows = rows[:top_n]
    if store:
        insert_prediction_rows(
            conn,
            race_id_value,
            _now(),
            str(model_path),
            rows,
        )
    return rows


def predict_open_races(
    conn,
    *,
    model_path: Path,
    race_date: date,
    jcd: str | None = None,
    rno: int | None = None,
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
            predict_race(conn, model_path=model_path, race_id_value=row["race_id"])
            ok += 1
        except Exception:
            failed += 1
    return {"predicted": ok, "failed": failed}


def trifecta_predictions(
    lane_probs: dict[int, float],
    *,
    latest_odds: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    latest_odds = latest_odds or {}
    rows = []
    probabilities = plackett_luce_probabilities(
        lane_probs[lane] for lane in range(1, 7)
    )
    for (first, second, third), probability in zip(
        TRIFECTA_COMBINATIONS, probabilities
    ):
        combination = f"{first}-{second}-{third}"
        odds = latest_odds.get(combination)
        rows.append(
            {
                "combination": combination,
                "probability": probability,
                "odds": odds,
                "expected_value": probability * odds if odds else None,
                "lane_probabilities": lane_probs,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["expected_value"] is not None,
            row["expected_value"] or row["probability"],
            row["probability"],
        ),
        reverse=True,
    )


def latest_predictions(conn, *, race_id_value: str, limit: int = 30) -> list[dict[str, Any]]:
    stamp = conn.execute(
        """
        SELECT generated_at
        FROM predictions
        WHERE race_id = ?
        ORDER BY generated_at DESC
        LIMIT 1
        """,
        (race_id_value,),
    ).fetchone()
    if not stamp:
        return []
    return [
        {
            "combination": row["combination"],
            "probability": row["probability"],
            "odds": row["odds"],
            "expected_value": row["expected_value"],
            "generated_at": row["generated_at"],
        }
        for row in conn.execute(
            """
            SELECT combination, probability, odds, expected_value, generated_at
            FROM predictions
            WHERE race_id = ? AND generated_at = ?
            ORDER BY COALESCE(expected_value, probability) DESC, probability DESC
            LIMIT ?
            """,
            (race_id_value, stamp["generated_at"], limit),
        ).fetchall()
    ]


def _make_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("vectorizer", DictVectorizer(sparse=True)),
            (
                "sparse_index_compat",
                FunctionTransformer(_ensure_int32_sparse_indices, accept_sparse=True),
            ),
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


def _ensure_int32_sparse_indices(matrix):
    """Keep SciPy sparse output compatible with sklearn 32-bit solvers."""
    if hasattr(matrix, "indices") and matrix.indices.dtype != np.int32:
        matrix = matrix.copy()
        matrix.indices = matrix.indices.astype(np.int32, copy=False)
        matrix.indptr = matrix.indptr.astype(np.int32, copy=False)
    return matrix


def _positive_probs(pipeline: Pipeline, X: list[dict[str, Any]]) -> list[float]:
    classifier = pipeline.named_steps["classifier"]
    classes = list(classifier.classes_)
    positive_index = classes.index(1)
    return [float(row[positive_index]) for row in pipeline.predict_proba(X)]


def _normalize_lane_probs(probs: dict[int, float]) -> dict[int, float]:
    clipped = {lane: max(1e-6, float(value)) for lane, value in probs.items()}
    total = sum(clipped.values())
    if total <= 0:
        return {lane: 1.0 / 6.0 for lane in range(1, 7)}
    return {lane: clipped.get(lane, 1e-6) / total for lane in range(1, 7)}


def _pl_probability(probs: dict[int, float], first: int, second: int, third: int) -> float:
    p_first = probs[first]
    remaining_after_first = max(1e-9, 1.0 - p_first)
    p_second = probs[second] / remaining_after_first
    remaining_after_second = max(1e-9, 1.0 - p_first - probs[second])
    p_third = probs[third] / remaining_after_second
    value = p_first * p_second * p_third
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _safe_log_loss(labels: list[int], probs: list[float]) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    return float(log_loss(labels, probs, labels=[0, 1]))


def _race_level_metrics(race_predictions: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    top1_hits = 0
    trifecta_top1_hits = 0
    trifecta_top5_hits = 0
    eligible = 0
    for rows in race_predictions.values():
        if len(rows) != 6:
            continue
        winner = next((row["lane"] for row in rows if row["rank"] == 1), None)
        actual_order = [
            row["lane"] for row in sorted(rows, key=lambda row: row["rank"] or 99)[:3]
        ]
        if winner is None or len(actual_order) < 3:
            continue
        eligible += 1
        lane_probs = _normalize_lane_probs(
            {int(row["lane"]): float(row["probability"]) for row in rows}
        )
        if max(lane_probs.items(), key=lambda item: item[1])[0] == winner:
            top1_hits += 1
        trifecta = trifecta_predictions(lane_probs)
        actual_combo = "-".join(str(lane) for lane in actual_order)
        if trifecta and trifecta[0]["combination"] == actual_combo:
            trifecta_top1_hits += 1
        if any(row["combination"] == actual_combo for row in trifecta[:5]):
            trifecta_top5_hits += 1
    if eligible == 0:
        return {
            "winner_top1_accuracy": 0.0,
            "trifecta_top1_hit_rate": 0.0,
            "trifecta_top5_hit_rate": 0.0,
            "evaluated_races": 0,
        }
    return {
        "winner_top1_accuracy": top1_hits / eligible,
        "trifecta_top1_hit_rate": trifecta_top1_hits / eligible,
        "trifecta_top5_hit_rate": trifecta_top5_hits / eligible,
        "evaluated_races": eligible,
    }


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
