from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import brier_score_loss, log_loss

from .bankroll_backtest import (
    _allocate_daily_budget,
    _build_payout_model,
    _candidate_tickets,
    _load_trifecta_payouts,
)
from .cache_entry_series_features import CACHE_FIELDS, ensure_series_cache_table
from .db import connection, init_db
from .features_no_odds_v3 import _group_by_race, race_relative_features
from .features_no_odds_v9 import RollingState, _race_sort_key
from .features_pastlog_v3 import base_pastlog_features
from .features_pastlog_v5 import cached_series_features, series_relative_features
from .modeling import _race_level_metrics


FEATURE_SET = "pastlog_v7_stream_hash_cached_series_sgd"
FEATURE_GROUPS = ("base_pastlog", "series_cached", "series_relative", "rolling_history")
HASH_FEATURES = 1 << 20

ROI_DIAGNOSTIC_FEATURES = (
    "racer_class",
    "origin",
    "race_month",
    "race_weekday",
    "race_rno_bucket",
    "class_rank",
    "national_win_rate_rank",
    "local_win_rate_rank",
    "motor_2_rate_rank",
    "boat_2_rate_rank",
    "hist_racer_win_rate_s",
    "hist_racer_venue_win_rate_s",
    "hist_motor_win_rate_s",
    "hist_boat_win_rate_s",
    "series_win_rate",
    "series_avg_finish",
)


SERIES_SELECT = ", ".join(f"sf.{field} AS {field}" for field in CACHE_FIELDS)


def normalize_drop_feature_groups(drop_feature_groups: Iterable[str] | str | None = None) -> tuple[str, ...]:
    if drop_feature_groups is None:
        return ()
    if isinstance(drop_feature_groups, str):
        requested = [group.strip() for group in drop_feature_groups.split(",")]
    else:
        requested = []
        for group in drop_feature_groups:
            requested.extend(str(group).split(","))
        requested = [group.strip() for group in requested]
    selected = {group for group in requested if group}
    unknown = sorted(selected.difference(FEATURE_GROUPS))
    if unknown:
        choices = ", ".join(FEATURE_GROUPS)
        raise ValueError(f"unknown feature group(s): {', '.join(unknown)}; choices: {choices}")
    return tuple(group for group in FEATURE_GROUPS if group in selected)


def _parse_drop_feature_groups(value: str) -> tuple[str, ...]:
    try:
        return normalize_drop_feature_groups(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def make_hasher(n_features: int = HASH_FEATURES) -> FeatureHasher:
    return FeatureHasher(n_features=n_features, input_type="dict", alternate_sign=False)


def make_classifier() -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        alpha=0.00005,
        l1_ratio=0.05,
        max_iter=1,
        tol=None,
        random_state=42,
        average=True,
    )


def train_streaming_model(
    conn,
    *,
    model_path: Path | None = None,
    include_races: set[str] | None = None,
    drop_feature_groups: Iterable[str] | str | None = None,
    batch_size: int = 24000,
    epochs: int = 1,
    n_features: int = HASH_FEATURES,
) -> dict[str, Any]:
    drop_feature_groups = normalize_drop_feature_groups(drop_feature_groups)
    hasher = make_hasher(n_features)
    classifier = make_classifier()
    first = True
    examples = 0
    races_seen: set[str] = set()
    for epoch in range(max(1, epochs)):
        batch_x: list[dict[str, float]] = []
        batch_y: list[int] = []
        batch_weight: list[float] = []
        for feature, label, meta in iter_training_entries(
            conn,
            include_races=include_races,
            drop_feature_groups=drop_feature_groups,
        ):
            batch_x.append(to_hashable(feature))
            batch_y.append(label)
            batch_weight.append(3.0 if label else 0.6)
            races_seen.add(str(meta["race_id"]))
            if len(batch_x) >= batch_size:
                first = _partial_fit(classifier, hasher, batch_x, batch_y, batch_weight, first=first)
                if epoch == 0:
                    examples += len(batch_x)
                batch_x.clear()
                batch_y.clear()
                batch_weight.clear()
        if batch_x:
            first = _partial_fit(classifier, hasher, batch_x, batch_y, batch_weight, first=first)
            if epoch == 0:
                examples += len(batch_x)
    if first:
        raise ValueError("no training examples")
    metadata = {
        "trained_at": _now(),
        "examples": examples,
        "races": len(races_seen),
        "include_odds": False,
        "target": "lane_win_probability",
        "feature_set": FEATURE_SET,
        "drop_feature_groups": list(drop_feature_groups),
        "vectorizer": f"FeatureHasher(n_features={n_features}, alternate_sign=False)",
        "classifier": "SGDClassifier(log_loss, elasticnet, partial_fit, sample_weight 3.0/0.6)",
        "epochs": max(1, epochs),
        "role": "primary_pastlog_streaming_memory_safe",
    }
    if model_path:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"hasher": hasher, "classifier": classifier, "metadata": metadata}, model_path)
    return metadata


def backtest_streaming(
    conn,
    *,
    output_path: Path,
    drop_feature_groups: Iterable[str] | str | None = None,
    folds: int = 5,
    min_train_races: int = 500,
    batch_size: int = 24000,
    epochs: int = 1,
    log_folds: bool = True,
) -> dict[str, Any]:
    drop_feature_groups = normalize_drop_feature_groups(drop_feature_groups)
    race_keys = load_complete_race_ids(conn)
    races = [race_id for race_id, *_ in race_keys]
    if len(races) < min_train_races + folds:
        raise ValueError(f"not enough parsed races: {len(races)}")
    race_index = {race_id: idx for idx, race_id in enumerate(races)}
    test_window = max(1, (len(races) - min_train_races) // folds)
    all_probs: list[float] = []
    all_labels: list[int] = []
    race_predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fold_rows = []

    for fold in range(folds):
        test_start = min_train_races + fold * test_window
        test_end = len(races) if fold == folds - 1 else min(len(races), test_start + test_window)
        train_races = set(races[:test_start])
        test_races = set(races[test_start:test_end])
        if not train_races or not test_races:
            continue
        bundle = train_bundle(
            conn,
            include_races=train_races,
            drop_feature_groups=drop_feature_groups,
            batch_size=batch_size,
            epochs=epochs,
        )
        labels: list[int] = []
        probs: list[float] = []
        for rows in iter_scored_races(
            conn,
            bundle=bundle,
            include_races=test_races,
            drop_feature_groups=drop_feature_groups,
        ):
            for row in rows:
                labels.append(int(row["label"]))
                probs.append(float(row["probability"]))
                race_predictions[row["race_id"]].append(
                    {
                        "lane": row["lane"],
                        "rank": row["rank"],
                        "probability": row["probability"],
                    }
                )
        all_labels.extend(labels)
        all_probs.extend(probs)
        fold_rows.append(
            {
                "fold": fold + 1,
                "train_races": test_start,
                "test_races": test_end - test_start,
                "entry_log_loss": safe_log_loss(labels, probs),
                "entry_brier": float(brier_score_loss(labels, probs)),
            }
        )
        if log_folds:
            print(json.dumps(fold_rows[-1], ensure_ascii=False), flush=True)

    result = {
        "generated_at": _now(),
        "folds": fold_rows,
        "examples": len(all_labels),
        "races": len(races),
        "include_odds": False,
        "feature_set": FEATURE_SET,
        "drop_feature_groups": list(drop_feature_groups),
        "entry_log_loss": safe_log_loss(all_labels, all_probs),
        "entry_brier": float(brier_score_loss(all_labels, all_probs)),
        **_race_level_metrics(race_predictions),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def bankroll_streaming(
    conn,
    *,
    output_path: Path,
    drop_feature_groups: Iterable[str] | str | None = None,
    daily_budget_yen: int = 10_000,
    unit_yen: int = 100,
    folds: int = 5,
    min_train_races: int = 500,
    ev_threshold: float = 1.0,
    max_tickets_per_race: int = 5,
    payout_prior_weight: float = 30.0,
    batch_size: int = 24000,
    epochs: int = 1,
) -> dict[str, Any]:
    drop_feature_groups = normalize_drop_feature_groups(drop_feature_groups)
    race_keys = load_complete_race_ids(conn)
    races = [race_id for race_id, *_ in race_keys]
    payouts = _load_trifecta_payouts(conn)
    test_window = max(1, (len(races) - min_train_races) // folds)
    candidates: list[dict[str, Any]] = []
    evaluated_races: set[str] = set()
    fold_rows = []
    for fold in range(folds):
        test_start = min_train_races + fold * test_window
        test_end = len(races) if fold == folds - 1 else min(len(races), test_start + test_window)
        train_races = set(races[:test_start])
        test_races = set(races[test_start:test_end])
        if not train_races or not test_races:
            continue
        payout_model = _build_payout_model(
            payouts,
            train_races=train_races,
            prior_weight=payout_prior_weight,
        )
        bundle = train_bundle(
            conn,
            include_races=train_races,
            drop_feature_groups=drop_feature_groups,
            batch_size=batch_size,
            epochs=epochs,
        )
        fold_candidates = 0
        fold_evaluated = 0
        for rows in iter_scored_races(
            conn,
            bundle=bundle,
            include_races=test_races,
            drop_feature_groups=drop_feature_groups,
        ):
            race_id_value = str(rows[0]["race_id"])
            payout = payouts.get(race_id_value)
            if len(rows) != 6 or not payout:
                continue
            evaluated_races.add(race_id_value)
            fold_evaluated += 1
            race_candidates = _candidate_tickets(
                rows,
                actual=payout,
                payout_model=payout_model,
                ev_threshold=ev_threshold,
            )[:max_tickets_per_race]
            fold_candidates += len(race_candidates)
            candidates.extend(race_candidates)
        fold_rows.append(
            {
                "fold": fold + 1,
                "train_races": test_start,
                "test_races": test_end - test_start,
                "evaluated_races": fold_evaluated,
                "candidate_tickets": fold_candidates,
            }
        )
        print(json.dumps(fold_rows[-1], ensure_ascii=False), flush=True)

    allocated = _allocate_daily_budget(
        candidates,
        evaluated_races=evaluated_races,
        daily_budget_yen=daily_budget_yen,
        unit_yen=unit_yen,
    )
    result = {
        "generated_at": _now(),
        "policy": {
            "daily_budget_yen": daily_budget_yen,
            "unit_yen": unit_yen,
            "bet_type": "3連単",
            "include_odds": False,
            "ev_threshold": ev_threshold,
            "max_tickets_per_race": max_tickets_per_race,
            "payout_estimator": "train-fold average payout by trifecta combination, blended with train-fold global average",
            "payout_prior_weight": payout_prior_weight,
            "allocation": "each day, rank positive-EV tickets by estimated EV; buy within daily budget and split stake in 100-yen units",
            "feature_set": FEATURE_SET,
            "drop_feature_groups": list(drop_feature_groups),
            "model": "win_model_pastlog_v7_stream_hash",
        },
        "folds": fold_rows,
        "examples": 0,
        "races": len(races),
        "evaluated_races": len(evaluated_races),
        "candidate_tickets": len(candidates),
        "feature_set": FEATURE_SET,
        "drop_feature_groups": list(drop_feature_groups),
        "model": "win_model_pastlog_v7_stream_hash",
        **allocated,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def ablation_streaming(
    conn,
    *,
    output_path: Path,
    folds: int = 5,
    min_train_races: int = 500,
    batch_size: int = 24000,
    epochs: int = 1,
) -> dict[str, Any]:
    variants: list[tuple[str, tuple[str, ...]]] = [("baseline", ())]
    variants.extend((f"drop_{group}", (group,)) for group in FEATURE_GROUPS)
    results: list[dict[str, Any]] = []
    for variant, drop_groups in variants:
        detail_path = _ablation_detail_path(output_path, variant)
        result = backtest_streaming(
            conn,
            output_path=detail_path,
            drop_feature_groups=drop_groups,
            folds=folds,
            min_train_races=min_train_races,
            batch_size=batch_size,
            epochs=epochs,
            log_folds=False,
        )
        results.append({"variant": variant, "output": str(detail_path), **result})
    summary = {
        "generated_at": _now(),
        "feature_set": FEATURE_SET,
        "feature_groups": list(FEATURE_GROUPS),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _ablation_detail_path(output_path: Path, variant: str) -> Path:
    suffix = output_path.suffix or ".json"
    return output_path.with_name(f"{output_path.stem}_{variant}{suffix}")


def train_bundle(
    conn,
    *,
    include_races: set[str],
    batch_size: int,
    epochs: int,
    drop_feature_groups: Iterable[str] | str | None = None,
) -> dict[str, Any]:
    drop_feature_groups = normalize_drop_feature_groups(drop_feature_groups)
    hasher = make_hasher()
    classifier = make_classifier()
    first = True
    for _ in range(max(1, epochs)):
        batch_x: list[dict[str, float]] = []
        batch_y: list[int] = []
        batch_weight: list[float] = []
        for feature, label, _meta in iter_training_entries(
            conn,
            include_races=include_races,
            drop_feature_groups=drop_feature_groups,
        ):
            batch_x.append(to_hashable(feature))
            batch_y.append(label)
            batch_weight.append(3.0 if label else 0.6)
            if len(batch_x) >= batch_size:
                first = _partial_fit(classifier, hasher, batch_x, batch_y, batch_weight, first=first)
                batch_x.clear()
                batch_y.clear()
                batch_weight.clear()
        if batch_x:
            first = _partial_fit(classifier, hasher, batch_x, batch_y, batch_weight, first=first)
    if first:
        raise ValueError("no training examples in fold")
    return {"hasher": hasher, "classifier": classifier, "drop_feature_groups": list(drop_feature_groups)}


def _partial_fit(
    classifier: SGDClassifier,
    hasher: FeatureHasher,
    batch_x: list[dict[str, float]],
    batch_y: list[int],
    batch_weight: list[float],
    *,
    first: bool,
) -> bool:
    matrix = _ensure_sparse_index32(hasher.transform(batch_x))
    kwargs = {"classes": [0, 1]} if first else {}
    classifier.partial_fit(matrix, batch_y, sample_weight=batch_weight, **kwargs)
    return False


def _ensure_sparse_index32(matrix: Any) -> Any:
    if not sparse.issparse(matrix):
        return matrix
    matrix = matrix.tocsr(copy=False)
    if matrix.indices.dtype != np.int32:
        matrix.indices = matrix.indices.astype(np.int32, copy=False)
    if matrix.indptr.dtype != np.int32:
        matrix.indptr = matrix.indptr.astype(np.int32, copy=False)
    return matrix


def iter_scored_races(
    conn,
    *,
    bundle: dict[str, Any],
    include_races: set[str],
    drop_feature_groups: Iterable[str] | str | None = None,
) -> Iterable[list[dict[str, Any]]]:
    if drop_feature_groups is None:
        drop_feature_groups = bundle.get("drop_feature_groups", ())
    drop_feature_groups = normalize_drop_feature_groups(drop_feature_groups)
    hasher: FeatureHasher = bundle["hasher"]
    classifier: SGDClassifier = bundle["classifier"]
    for race_features in iter_race_feature_rows(
        conn,
        include_races=include_races,
        drop_feature_groups=drop_feature_groups,
    ):
        features = [to_hashable(item["features"]) for item in race_features]
        matrix = _ensure_sparse_index32(hasher.transform(features))
        probabilities = classifier.predict_proba(matrix)[:, 1].tolist()
        total = sum(float(value) for value in probabilities) or 1.0
        rows = []
        for item, probability in zip(race_features, probabilities):
            meta = item["meta"]
            rows.append(
                {
                    "race_id": meta["race_id"],
                    "race_date": meta["race_date"],
                    "jcd": meta["jcd"],
                    "rno": meta["rno"],
                    "lane": meta["lane"],
                    "rank": meta["rank"],
                    "label": meta["label"],
                    "probability": float(probability) / total,
                    "diagnostic_features": diagnostic_feature_snapshot(item["features"]),
                }
            )
        yield rows


def iter_training_entries(
    conn,
    *,
    include_races: set[str] | None = None,
    drop_feature_groups: Iterable[str] | str | None = None,
) -> Iterable[tuple[dict[str, Any], int, dict[str, Any]]]:
    drop_feature_groups = normalize_drop_feature_groups(drop_feature_groups)
    for race_features in iter_race_feature_rows(
        conn,
        include_races=include_races,
        drop_feature_groups=drop_feature_groups,
    ):
        for item in race_features:
            meta = item["meta"]
            yield item["features"], int(meta["label"]), meta


def iter_race_feature_rows(
    conn,
    *,
    include_races: set[str] | None = None,
    drop_feature_groups: Iterable[str] | str | None = None,
) -> Iterable[list[dict[str, Any]]]:
    drop_feature_groups = normalize_drop_feature_groups(drop_feature_groups)
    state = RollingState()
    current_date: str | None = None
    day_updates: list[list[Any]] = []
    for race_rows in iter_complete_races(conn):
        race_id_value = str(race_rows[0]["race_id"])
        race_date_value = str(race_rows[0]["race_date"])
        if current_date is None:
            current_date = race_date_value
        if race_date_value != current_date:
            for rows in day_updates:
                state.update_race(rows)
            day_updates = []
            current_date = race_date_value
        use_race = include_races is None or race_id_value in include_races
        if use_race:
            yield build_race_features(race_rows, state, drop_feature_groups=drop_feature_groups)
        day_updates.append(race_rows)
    for rows in day_updates:
        state.update_race(rows)


def build_race_features(
    race_rows: list[Any],
    state: RollingState,
    *,
    drop_feature_groups: Iterable[str] | str | None = None,
) -> list[dict[str, Any]]:
    drop_feature_groups = normalize_drop_feature_groups(drop_feature_groups)
    dropped = set(drop_feature_groups)
    relatives = (
        race_relative_features(race_rows, {lane: {} for lane in range(1, 7)})
        if "base_pastlog" not in dropped
        else {}
    )
    series_relatives = series_relative_features(race_rows) if "series_relative" not in dropped else {}
    out = []
    for row in race_rows:
        lane = int(row["lane"])
        item: dict[str, Any] = {}
        if "base_pastlog" not in dropped:
            item.update(base_pastlog_features(row, relatives[lane]))
        if "series_cached" not in dropped:
            item.update(cached_series_features(row))
        if "series_relative" not in dropped:
            item.update(series_relatives[lane])
        if "rolling_history" not in dropped:
            item.update(state.features_for(row))
        out.append(
            {
                "features": item,
                "meta": {
                    "race_id": row["race_id"],
                    "race_date": row["race_date"],
                    "jcd": row["jcd"],
                    "rno": int(row["rno"]),
                    "lane": lane,
                    "rank": int(row["rank"]),
                    "label": 1 if int(row["rank"]) == 1 else 0,
                },
            }
        )
    return out


def iter_complete_races(conn) -> Iterable[list[Any]]:
    ensure_series_cache_table(conn)
    rows = conn.execute(
        f"""
        SELECT
          r.race_id, r.race_date, r.jcd, r.rno, r.race_type, r.distance_m,
          e.lane, e.racer_no, e.racer_name, e.racer_class, e.branch, e.origin,
          e.age, e.weight_kg, e.f_count, e.l_count, e.avg_st,
          e.national_win_rate, e.national_2_rate, e.national_3_rate,
          e.local_win_rate, e.local_2_rate, e.local_3_rate,
          e.motor_no, e.motor_2_rate, e.motor_3_rate,
          e.boat_no, e.boat_2_rate, e.boat_3_rate,
          {SERIES_SELECT},
          rr.rank, rr.course AS result_course, rr.start_timing AS result_start_timing
        FROM entries e
        JOIN races r ON r.race_id = e.race_id
        JOIN race_results rr ON rr.race_id = e.race_id AND rr.lane = e.lane
        LEFT JOIN entry_series_features sf ON sf.race_id = e.race_id AND sf.lane = e.lane
        WHERE rr.rank IS NOT NULL
        ORDER BY r.race_date, r.jcd, r.rno, e.lane
        """
    )
    current_id: str | None = None
    current_rows: list[Any] = []
    for row in rows:
        race_id_value = str(row["race_id"])
        if current_id is None:
            current_id = race_id_value
        if race_id_value != current_id:
            if len(current_rows) == 6:
                yield current_rows
            current_id = race_id_value
            current_rows = []
        current_rows.append(row)
    if len(current_rows) == 6:
        yield current_rows


def load_complete_race_ids(conn) -> list[tuple[str, str, str, int]]:
    rows = conn.execute(
        """
        SELECT r.race_id, r.race_date, r.jcd, r.rno
        FROM races r
        JOIN race_results rr ON rr.race_id = r.race_id AND rr.rank IS NOT NULL
        WHERE (SELECT COUNT(*) FROM entries e WHERE e.race_id = r.race_id) = 6
          AND (SELECT COUNT(*) FROM race_results x WHERE x.race_id = r.race_id AND x.rank IS NOT NULL) = 6
        GROUP BY r.race_id
        ORDER BY r.race_date, r.jcd, r.rno
        """
    ).fetchall()
    return [(row["race_id"], row["race_date"], row["jcd"], int(row["rno"])) for row in rows]


def to_hashable(item: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in item.items():
        if value is None:
            continue
        if isinstance(value, str):
            if value:
                out[f"{key}={value}"] = 1.0
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            text = str(value)
            if text:
                out[f"{key}={text}"] = 1.0
            continue
        if math.isfinite(number):
            out[key] = number
    return out


def diagnostic_feature_snapshot(item: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ROI_DIAGNOSTIC_FEATURES:
        value = item.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, str):
            out[key] = value
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out[key] = number
    return out


def safe_log_loss(labels: list[int], probs: list[float]) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    return float(log_loss(labels, probs, labels=[0, 1]))


def positive_probs(bundle: dict[str, Any], features: list[dict[str, Any]]) -> list[float]:
    matrix = bundle["hasher"].transform([to_hashable(item) for item in features])
    return [float(value) for value in bundle["classifier"].predict_proba(matrix)[:, 1]]


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Streaming hashed past-log v7 model.")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    add_common(train)
    add_drop_feature_groups(train)
    train.add_argument("--model", default="data/models/win_model_pastlog_v7_stream_hash.joblib")
    train.add_argument("--batch-size", type=int, default=24000)
    train.add_argument("--epochs", type=int, default=1)
    train.set_defaults(func=_cmd_train)
    backtest = sub.add_parser("backtest")
    add_common(backtest)
    add_drop_feature_groups(backtest)
    backtest.add_argument("--output", default="data/models/backtest_pastlog_v7_stream_hash.json")
    backtest.add_argument("--folds", type=int, default=5)
    backtest.add_argument("--min-train-races", type=int, default=500)
    backtest.add_argument("--batch-size", type=int, default=24000)
    backtest.add_argument("--epochs", type=int, default=1)
    backtest.set_defaults(func=_cmd_backtest)
    bankroll = sub.add_parser("bankroll")
    add_common(bankroll)
    add_drop_feature_groups(bankroll)
    bankroll.add_argument("--output", default="data/models/bankroll_backtest_pastlog_v7_stream_hash_10000.json")
    bankroll.add_argument("--daily-budget-yen", type=int, default=10000)
    bankroll.add_argument("--unit-yen", type=int, default=100)
    bankroll.add_argument("--folds", type=int, default=5)
    bankroll.add_argument("--min-train-races", type=int, default=500)
    bankroll.add_argument("--ev-threshold", type=float, default=1.0)
    bankroll.add_argument("--max-tickets-per-race", type=int, default=5)
    bankroll.add_argument("--payout-prior-weight", type=float, default=30.0)
    bankroll.add_argument("--batch-size", type=int, default=24000)
    bankroll.add_argument("--epochs", type=int, default=1)
    bankroll.set_defaults(func=_cmd_bankroll)
    ablation = sub.add_parser("ablation")
    add_common(ablation)
    ablation.add_argument("--output", default="data/models/ablation_pastlog_v7_stream_hash.json")
    ablation.add_argument("--folds", type=int, default=5)
    ablation.add_argument("--min-train-races", type=int, default=500)
    ablation.add_argument("--batch-size", type=int, default=24000)
    ablation.add_argument("--epochs", type=int, default=1)
    ablation.set_defaults(func=_cmd_ablation)
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", default="data/boatrace.sqlite")


def add_drop_feature_groups(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--drop-feature-groups",
        default=(),
        type=_parse_drop_feature_groups,
        metavar="GROUPS",
        help=f"Comma-separated feature groups to drop: {', '.join(FEATURE_GROUPS)}",
    )


def _cmd_train(args: argparse.Namespace) -> int:
    init_db(args.db)
    with connection(args.db) as conn:
        result = train_streaming_model(
            conn,
            model_path=Path(args.model),
            drop_feature_groups=args.drop_feature_groups,
            batch_size=args.batch_size,
            epochs=args.epochs,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


def _cmd_backtest(args: argparse.Namespace) -> int:
    init_db(args.db)
    with connection(args.db) as conn:
        result = backtest_streaming(
            conn,
            output_path=Path(args.output),
            drop_feature_groups=args.drop_feature_groups,
            folds=args.folds,
            min_train_races=args.min_train_races,
            batch_size=args.batch_size,
            epochs=args.epochs,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


def _cmd_bankroll(args: argparse.Namespace) -> int:
    init_db(args.db)
    with connection(args.db) as conn:
        result = bankroll_streaming(
            conn,
            output_path=Path(args.output),
            drop_feature_groups=args.drop_feature_groups,
            daily_budget_yen=args.daily_budget_yen,
            unit_yen=args.unit_yen,
            folds=args.folds,
            min_train_races=args.min_train_races,
            ev_threshold=args.ev_threshold,
            max_tickets_per_race=args.max_tickets_per_race,
            payout_prior_weight=args.payout_prior_weight,
            batch_size=args.batch_size,
            epochs=args.epochs,
        )
    print(json.dumps({key: value for key, value in result.items() if key != "daily"} | {"daily_rows": len(result.get("daily", []))}, ensure_ascii=False, indent=2), flush=True)
    return 0


def _cmd_ablation(args: argparse.Namespace) -> int:
    init_db(args.db)
    with connection(args.db) as conn:
        result = ablation_streaming(
            conn,
            output_path=Path(args.output),
            folds=args.folds,
            min_train_races=args.min_train_races,
            batch_size=args.batch_size,
            epochs=args.epochs,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
