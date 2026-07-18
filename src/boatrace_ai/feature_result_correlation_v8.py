from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from .db import connect, init_db
from .features_no_odds_v3 import load_training_examples
from .modeling_no_odds_v8 import FEATURE_SET


DERIVED_NUMERIC_SUFFIXES = (
    "_vs_mean",
    "_z",
    "_best_gap",
    "_scaled",
)


def analyze(
    conn: sqlite3.Connection,
    *,
    model_path: Path | None,
    output_path: Path,
    min_category_count: int,
    top_n: int,
) -> dict[str, Any]:
    features, labels, meta = load_training_examples(conn, include_odds=False)
    if not features:
        raise ValueError("no training examples were loaded")

    numeric_stats: dict[str, dict[str, float]] = {}
    numeric_present_stats: dict[str, dict[str, float]] = {}
    category_stats: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: {"count": 0.0, "wins": 0.0})
    )
    category_totals: dict[str, float] = defaultdict(float)

    for x, y in zip(features, labels):
        for key, value in x.items():
            if isinstance(value, bool):
                value = int(value)
            if isinstance(value, (int, float)):
                _add_numeric(numeric_stats.setdefault(key, _empty_numeric()), float(value), y)
                if _is_present_numeric(key, float(value)):
                    _add_numeric(numeric_present_stats.setdefault(key, _empty_numeric()), float(value), y)
            else:
                text = str(value or "")
                row = category_stats[key][text]
                row["count"] += 1.0
                row["wins"] += float(y)
                category_totals[key] += 1.0

    numeric_rows = [
        _numeric_summary(key, stats, numeric_present_stats.get(key))
        for key, stats in numeric_stats.items()
    ]
    numeric_rows.sort(key=lambda row: (abs(row["pearson"]), row["present_count"]), reverse=True)

    category_rows = _category_summaries(
        category_stats,
        total_examples=len(labels),
        global_win_rate=sum(labels) / len(labels),
        min_count=min_category_count,
    )
    category_rows.sort(key=lambda row: (row["max_abs_gap"], row["covered_count"]), reverse=True)

    coefficient_rows = _model_coefficients(model_path, top_n=top_n) if model_path else {}
    coverage = _coverage(conn)
    backtest = _read_json(output_path.parent / "backtest_no_odds_v8.json")
    bankroll = _read_json(output_path.parent / "bankroll_backtest_no_odds_v8_10000.json")

    payload = {
        "generated_at": _now(),
        "feature_set": FEATURE_SET,
        "examples": len(features),
        "races": len({row["race_id"] for row in meta}),
        "positive_labels": int(sum(labels)),
        "global_win_rate": sum(labels) / len(labels),
        "top_numeric_abs_correlation": numeric_rows[:top_n],
        "top_numeric_low_coverage": sorted(
            numeric_rows,
            key=lambda row: (row["present_rate"], -abs(row["pearson"])),
        )[:top_n],
        "top_categorical_gap": category_rows[:top_n],
        "model_coefficients": coefficient_rows,
        "coverage": coverage,
        "backtest": _compact_backtest(backtest),
        "bankroll_backtest": _compact_bankroll(bankroll),
        "diagnosis": _diagnosis(coverage, numeric_rows, category_rows, coefficient_rows),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _empty_numeric() -> dict[str, float]:
    return {
        "count": 0.0,
        "wins": 0.0,
        "sum": 0.0,
        "sum_sq": 0.0,
        "sum_y": 0.0,
        "sum_y_sq": 0.0,
        "sum_xy": 0.0,
        "win_sum": 0.0,
        "loss_sum": 0.0,
        "losses": 0.0,
        "sentinel_minus_one": 0.0,
    }


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


def _is_present_numeric(key: str, value: float) -> bool:
    if any(key.endswith(suffix) for suffix in DERIVED_NUMERIC_SUFFIXES):
        return True
    return value != -1.0


def _numeric_summary(
    key: str,
    stats: dict[str, float],
    present_stats: dict[str, float] | None,
) -> dict[str, Any]:
    count = stats["count"]
    wins = stats["wins"]
    losses = stats["losses"]
    present_count = present_stats["count"] if present_stats else 0.0
    present_corr = _pearson(present_stats) if present_stats and present_count > 1 else 0.0
    return {
        "feature": key,
        "count": int(count),
        "present_count": int(present_count),
        "present_rate": present_count / count if count else 0.0,
        "sentinel_minus_one_rate": stats["sentinel_minus_one"] / count if count else 0.0,
        "pearson": _pearson(stats),
        "present_only_pearson": present_corr,
        "mean": stats["sum"] / count if count else 0.0,
        "mean_when_win": stats["win_sum"] / wins if wins else 0.0,
        "mean_when_loss": stats["loss_sum"] / losses if losses else 0.0,
    }


def _pearson(stats: dict[str, float] | None) -> float:
    if not stats:
        return 0.0
    n = stats["count"]
    if n <= 1:
        return 0.0
    numerator = n * stats["sum_xy"] - stats["sum"] * stats["sum_y"]
    x_var = n * stats["sum_sq"] - stats["sum"] * stats["sum"]
    y_var = n * stats["sum_y_sq"] - stats["sum_y"] * stats["sum_y"]
    denominator = math.sqrt(max(0.0, x_var) * max(0.0, y_var))
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _category_summaries(
    category_stats: dict[str, dict[str, dict[str, float]]],
    *,
    total_examples: int,
    global_win_rate: float,
    min_count: int,
) -> list[dict[str, Any]]:
    rows = []
    for key, values in category_stats.items():
        covered = sum(row["count"] for row in values.values())
        candidates = []
        for value, stats in values.items():
            count = stats["count"]
            if count < min_count:
                continue
            rate = stats["wins"] / count if count else 0.0
            gap = rate - global_win_rate
            candidates.append(
                {
                    "value": value,
                    "count": int(count),
                    "win_rate": rate,
                    "gap_vs_global": gap,
                }
            )
        if not candidates:
            continue
        candidates.sort(key=lambda row: abs(row["gap_vs_global"]), reverse=True)
        rows.append(
            {
                "feature": key,
                "distinct_values": len(values),
                "covered_count": int(covered),
                "covered_rate": covered / total_examples if total_examples else 0.0,
                "max_abs_gap": abs(candidates[0]["gap_vs_global"]),
                "top_values": candidates[:8],
            }
        )
    return rows


def _model_coefficients(model_path: Path | None, *, top_n: int) -> dict[str, Any]:
    if not model_path or not model_path.exists():
        return {"error": f"model file not found: {model_path}"}
    bundle = joblib.load(model_path)
    pipeline = bundle["pipeline"]
    vectorizer = pipeline.named_steps.get("vectorizer")
    classifier = pipeline.named_steps.get("classifier")
    if not vectorizer or not classifier:
        return {"error": "pipeline lacks vectorizer or classifier"}
    names = list(vectorizer.get_feature_names_out())
    coefs = list(classifier.coef_[0])
    rows = [
        {"feature": str(name), "coefficient": float(coef), "abs_coefficient": abs(float(coef))}
        for name, coef in zip(names, coefs)
    ]
    rows.sort(key=lambda row: row["coefficient"], reverse=True)
    positive = rows[:top_n]
    negative = sorted(rows, key=lambda row: row["coefficient"])[:top_n]
    by_abs = sorted(rows, key=lambda row: row["abs_coefficient"], reverse=True)[:top_n]
    return {
        "model_path": str(model_path),
        "metadata": bundle.get("metadata", {}),
        "feature_dimensions": len(rows),
        "top_positive": positive,
        "top_negative": negative,
        "top_abs": by_abs,
    }


def _coverage(conn: sqlite3.Connection) -> dict[str, Any]:
    overview = dict(
        conn.execute(
            """
            SELECT
              (SELECT COUNT(*) FROM races) AS races_total,
              (SELECT COUNT(*) FROM races WHERE race_date = (SELECT MAX(race_date) FROM races)) AS races_on_latest_date,
              (SELECT MIN(race_date) FROM races) AS min_race_date,
              (SELECT MAX(race_date) FROM races) AS max_race_date,
              (SELECT COUNT(*) FROM entries) AS entries_total,
              (SELECT COUNT(*) FROM race_results WHERE rank IS NOT NULL) AS result_rows_total,
              (SELECT COUNT(DISTINCT race_id) FROM race_results WHERE rank IS NOT NULL) AS races_with_results,
              (SELECT COUNT(DISTINCT race_id) FROM beforeinfo) AS races_with_beforeinfo,
              (SELECT COUNT(DISTINCT race_id) FROM odds_snapshots) AS races_with_odds,
              (SELECT COUNT(*) FROM beforeinfo) AS beforeinfo_rows,
              (SELECT COUNT(*) FROM odds_snapshots) AS odds_snapshots,
              (SELECT COUNT(*) FROM racer_period_stats) AS racer_period_stats_rows
            """
        ).fetchone()
    )
    entry = dict(
        conn.execute(
            """
            SELECT
              COUNT(*) AS training_entries,
              SUM(CASE WHEN e.racer_no IS NULL THEN 1 ELSE 0 END) AS missing_racer_no,
              SUM(CASE WHEN e.racer_name IS NULL OR TRIM(e.racer_name) = '' OR e.racer_name GLOB '[0-9]*' THEN 1 ELSE 0 END) AS missing_or_number_name,
              SUM(CASE WHEN e.racer_class IS NULL OR TRIM(e.racer_class) = '' THEN 1 ELSE 0 END) AS missing_class,
              SUM(CASE WHEN e.branch IS NULL OR TRIM(e.branch) = '' THEN 1 ELSE 0 END) AS missing_branch,
              SUM(CASE WHEN e.origin IS NULL OR TRIM(e.origin) = '' THEN 1 ELSE 0 END) AS missing_origin,
              SUM(CASE WHEN e.avg_st IS NULL THEN 1 ELSE 0 END) AS missing_avg_st,
              SUM(CASE WHEN e.national_win_rate IS NULL THEN 1 ELSE 0 END) AS missing_national_win_rate,
              SUM(CASE WHEN e.local_win_rate IS NULL THEN 1 ELSE 0 END) AS missing_local_win_rate,
              SUM(CASE WHEN e.motor_no IS NULL THEN 1 ELSE 0 END) AS missing_motor_no,
              SUM(CASE WHEN e.motor_2_rate IS NULL THEN 1 ELSE 0 END) AS missing_motor_2_rate,
              SUM(CASE WHEN e.motor_3_rate IS NULL THEN 1 ELSE 0 END) AS missing_motor_3_rate,
              SUM(CASE WHEN e.boat_no IS NULL THEN 1 ELSE 0 END) AS missing_boat_no,
              SUM(CASE WHEN e.boat_2_rate IS NULL THEN 1 ELSE 0 END) AS missing_boat_2_rate,
              SUM(CASE WHEN e.boat_3_rate IS NULL THEN 1 ELSE 0 END) AS missing_boat_3_rate
            FROM entries e
            JOIN race_results rr ON rr.race_id = e.race_id AND rr.lane = e.lane
            WHERE rr.rank IS NOT NULL
            """
        ).fetchone()
    )
    latest_date = overview.get("max_race_date")
    same_day = {}
    historical = {}
    if latest_date:
        same_day = dict(
            conn.execute(
                """
                SELECT
                  COUNT(DISTINCT b.race_id) AS latest_date_beforeinfo_races,
                  COUNT(DISTINCT os.race_id) AS latest_date_odds_races
                FROM races r
                LEFT JOIN beforeinfo b ON b.race_id = r.race_id
                LEFT JOIN odds_snapshots os ON os.race_id = r.race_id
                WHERE r.race_date = ?
                """,
                (latest_date,),
            ).fetchone()
        )
        historical = dict(
            conn.execute(
                """
                SELECT
                  COUNT(DISTINCT b.race_id) AS historical_beforeinfo_races,
                  COUNT(DISTINCT os.race_id) AS historical_odds_races
                FROM races r
                LEFT JOIN beforeinfo b ON b.race_id = r.race_id
                LEFT JOIN odds_snapshots os ON os.race_id = r.race_id
                WHERE r.race_date < ?
                """,
                (latest_date,),
            ).fetchone()
        )
    return {"overview": overview, "entry_missing": entry, "latest_date": same_day, "historical": historical}


def _diagnosis(
    coverage: dict[str, Any],
    numeric_rows: list[dict[str, Any]],
    category_rows: list[dict[str, Any]],
    coefficients: dict[str, Any],
) -> list[str]:
    notes = []
    overview = coverage.get("overview", {})
    entry = coverage.get("entry_missing", {})
    training_entries = max(1, int(entry.get("training_entries") or 1))
    before_races = int(overview.get("races_with_beforeinfo") or 0)
    odds_races = int(overview.get("races_with_odds") or 0)
    result_races = max(1, int(overview.get("races_with_results") or 1))
    if before_races / result_races < 0.05:
        notes.append("展示・気象 beforeinfo は特徴量に存在するが、過去学習データでの充足率が低く、現行モデルでは主力特徴量にできない。")
    if odds_races / result_races < 0.05:
        notes.append("オッズ時系列は過去側の充足率が低く、バックチェックモデルでは検証可能な特徴量になっていない。")
    for field in ("motor_2_rate", "motor_3_rate", "boat_2_rate", "boat_3_rate"):
        missing = int(entry.get(f"missing_{field}") or 0)
        if missing / training_entries > 0.80:
            notes.append(f"{field} は学習エントリの欠損率が高く、番号そのものではなく場・年度別のローリング成績へ置き換えるべき。")
            break
    abs_coefs = coefficients.get("top_abs") if isinstance(coefficients, dict) else None
    if abs_coefs:
        strong_numeric_ids = [
            row["feature"]
            for row in abs_coefs
            if row["feature"] in {"motor_no", "boat_no", "racer_no"}
        ]
        if strong_numeric_ids:
            notes.append(f"{', '.join(strong_numeric_ids)} が係数上位に入っている場合、番号を順序尺度として扱っているリスクがある。")
    low_coverage = [row["feature"] for row in numeric_rows if row["present_rate"] < 0.05 and abs(row["pearson"]) > 0.01]
    if low_coverage:
        notes.append(f"低充足だが見かけ相関がある特徴量がある: {', '.join(low_coverage[:6])}。欠損フラグによる疑似相関を疑う。")
    if category_rows:
        notes.append("カテゴリ特徴量はレーン、場、レース番号、支部/出身の勝率差を示す。高カードinalityはローリング統計へ圧縮して使う。")
    return notes


def _compact_backtest(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    keys = (
        "generated_at",
        "examples",
        "races",
        "entry_log_loss",
        "entry_brier",
        "winner_top1_accuracy",
        "winner_top2_accuracy",
        "trifecta_top5_hit_rate",
    )
    return {key: payload.get(key) for key in keys if key in payload}


def _compact_bankroll(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    keys = (
        "generated_at",
        "race_days",
        "days_with_bets",
        "winning_days",
        "losing_days",
        "selected_races",
        "hit_races",
        "tickets",
        "hit_tickets",
        "ticket_hit_rate",
        "race_hit_rate",
        "stake_yen",
        "return_yen",
        "profit_yen",
        "roi",
        "max_drawdown_yen",
    )
    return {key: payload.get(key) for key in keys if key in payload}


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze v8 feature/result correlation and learned coefficients.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--model", default="data/models/win_model_no_odds_v8.joblib")
    parser.add_argument("--output", default="data/models/feature_result_correlation_v8.json")
    parser.add_argument("--min-category-count", type=int, default=200)
    parser.add_argument("--top-n", type=int, default=40)
    args = parser.parse_args(argv)

    init_db(args.db)
    with connect(args.db) as conn:
        payload = analyze(
            conn,
            model_path=Path(args.model) if args.model else None,
            output_path=Path(args.output),
            min_category_count=args.min_category_count,
            top_n=args.top_n,
        )
    summary = {
        "generated_at": payload["generated_at"],
        "examples": payload["examples"],
        "races": payload["races"],
        "top_numeric": payload["top_numeric_abs_correlation"][:5],
        "top_categorical": payload["top_categorical_gap"][:5],
        "diagnosis": payload["diagnosis"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
