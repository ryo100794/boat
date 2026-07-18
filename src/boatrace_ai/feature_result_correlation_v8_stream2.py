from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from .db import connect, init_db
from .feature_result_correlation_v8 import (
    _advanced_diagnostics,
    _category_summaries,
    _compact_backtest,
    _compact_bankroll,
    _coverage,
    _diagnosis,
    _empty_numeric,
    _is_present_numeric,
    _model_coefficients,
    _now,
    _numeric_summary,
    _read_json,
)
from .features import entry_features
from .features_no_odds_v3 import _latest_beforeinfo, before_features, race_relative_features
from .modeling_no_odds_v8 import FEATURE_SET


def analyze_stream(
    conn: sqlite3.Connection,
    *,
    model_path: Path | None,
    output_path: Path,
    min_category_count: int,
    top_n: int,
) -> dict[str, Any]:
    numeric_stats: dict[str, dict[str, float]] = {}
    numeric_present_stats: dict[str, dict[str, float]] = {}
    category_stats: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: {"count": 0.0, "wins": 0.0})
    )
    labels_seen = 0
    positive_labels = 0
    races_seen = 0
    beforeinfo = _latest_beforeinfo(conn)
    for race_rows in _iter_training_races(conn):
        races_seen += 1
        race_id_value = str(race_rows[0]["race_id"])
        before_rows = {lane: beforeinfo.get((race_id_value, lane), {}) for lane in range(1, 7)}
        relatives = race_relative_features(race_rows, before_rows)
        for row in race_rows:
            label = 1 if int(row["rank"]) == 1 else 0
            item = entry_features(row, odds_features={})
            item.update(before_features(before_rows.get(int(row["lane"]), {})))
            item.update(relatives[int(row["lane"])])
            labels_seen += 1
            positive_labels += label
            _update_feature_stats(
                item,
                label=label,
                numeric_stats=numeric_stats,
                numeric_present_stats=numeric_present_stats,
                category_stats=category_stats,
            )

    if labels_seen <= 0:
        raise ValueError("no training examples were loaded")
    numeric_rows = [
        _numeric_summary(key, stats, numeric_present_stats.get(key))
        for key, stats in numeric_stats.items()
    ]
    numeric_rows.sort(key=lambda row: (abs(row["pearson"]), row["present_count"]), reverse=True)
    category_rows = _category_summaries(
        category_stats,
        total_examples=labels_seen,
        global_win_rate=positive_labels / labels_seen,
        min_count=min_category_count,
    )
    category_rows.sort(key=lambda row: (row["max_abs_gap"], row["covered_count"]), reverse=True)
    coefficients = _model_coefficients(model_path, top_n=top_n) if model_path else {}
    coverage = _coverage(conn)
    backtest = _read_json(output_path.parent / "backtest_no_odds_v8.json")
    bankroll = _read_json(output_path.parent / "bankroll_backtest_no_odds_v8_10000.json")
    advanced = _advanced_diagnostics(
        coverage,
        numeric_rows,
        category_rows,
        coefficients,
        bankroll,
        top_n=top_n,
    )
    payload = {
        "generated_at": _now(),
        "feature_set": FEATURE_SET,
        "streaming": True,
        "examples": labels_seen,
        "races": races_seen,
        "positive_labels": positive_labels,
        "global_win_rate": positive_labels / labels_seen,
        "top_numeric_abs_correlation": numeric_rows[:top_n],
        "top_numeric_low_coverage": sorted(
            numeric_rows,
            key=lambda row: (row["present_rate"], -abs(row["pearson"])),
        )[:top_n],
        "top_categorical_gap": category_rows[:top_n],
        **advanced,
        "model_coefficients": coefficients,
        "coverage": coverage,
        "backtest": _compact_backtest(backtest),
        "bankroll_backtest": _compact_bankroll(bankroll),
        "diagnosis": _diagnosis(coverage, numeric_rows, category_rows, coefficients) + advanced.get("diagnosis", []),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _iter_training_races(conn: sqlite3.Connection):
    rows = conn.execute(
        """
        SELECT
          r.race_id, r.race_date, r.jcd, r.rno, r.race_type, r.distance_m,
          e.lane, e.racer_no, e.racer_name, e.racer_class, e.branch, e.origin,
          e.age, e.weight_kg, e.f_count, e.l_count, e.avg_st,
          e.national_win_rate, e.national_2_rate, e.national_3_rate,
          e.local_win_rate, e.local_2_rate, e.local_3_rate,
          e.motor_no, e.motor_2_rate, e.motor_3_rate,
          e.boat_no, e.boat_2_rate, e.boat_3_rate,
          rr.rank
        FROM entries e
        JOIN races r ON r.race_id = e.race_id
        JOIN race_results rr ON rr.race_id = e.race_id AND rr.lane = e.lane
        WHERE rr.rank IS NOT NULL
        ORDER BY r.race_date, r.jcd, r.rno, e.lane
        """
    )
    current_race_id = None
    bucket = []
    for row in rows:
        if current_race_id is not None and row["race_id"] != current_race_id:
            if len(bucket) == 6:
                yield sorted(bucket, key=lambda item: int(item["lane"]))
            bucket = []
        current_race_id = row["race_id"]
        bucket.append(row)
    if len(bucket) == 6:
        yield sorted(bucket, key=lambda item: int(item["lane"]))


def _update_feature_stats(
    item: dict[str, Any],
    *,
    label: int,
    numeric_stats: dict[str, dict[str, float]],
    numeric_present_stats: dict[str, dict[str, float]],
    category_stats: dict[str, dict[str, dict[str, float]]],
) -> None:
    for key, value in item.items():
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, (int, float)):
            number = float(value)
            _add_numeric(numeric_stats.setdefault(key, _empty_numeric()), number, label)
            if _is_present_numeric(key, number):
                _add_numeric(numeric_present_stats.setdefault(key, _empty_numeric()), number, label)
        else:
            text = str(value or "")
            row = category_stats[key][text]
            row["count"] += 1.0
            row["wins"] += float(label)


def _add_numeric(stats: dict[str, float], value: float, label: int) -> None:
    stats["count"] += 1.0
    stats["wins"] += float(label)
    stats["sum"] += value
    stats["sum_sq"] += value * value
    stats["sum_y"] += float(label)
    stats["sum_y_sq"] += float(label) * float(label)
    stats["sum_xy"] += value * float(label)
    if label:
        stats["win_sum"] += value
    else:
        stats["loss_sum"] += value
        stats["losses"] += 1.0
    if value == -1.0:
        stats["sentinel_minus_one"] += 1.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Streaming v8 feature/result correlation analysis.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--model", default="data/models/win_model_no_odds_v8.joblib")
    parser.add_argument("--output", default="data/models/feature_result_correlation_v8_stream.json")
    parser.add_argument("--min-category-count", type=int, default=200)
    parser.add_argument("--top-n", type=int, default=40)
    args = parser.parse_args(argv)

    init_db(args.db)
    with connect(args.db) as conn:
        payload = analyze_stream(
            conn,
            model_path=Path(args.model) if args.model else None,
            output_path=Path(args.output),
            min_category_count=args.min_category_count,
            top_n=args.top_n,
        )
    print(
        json.dumps(
            {
                "generated_at": payload["generated_at"],
                "examples": payload["examples"],
                "races": payload["races"],
                "top_numeric": payload["top_numeric_abs_correlation"][:5],
                "diagnosis": payload["diagnosis"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
