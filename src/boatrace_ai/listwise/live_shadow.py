from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

from ..cache_entry_series_features import ensure_series_cache_table
from ..contextual_features import RollingState
from ..db import connection, init_db
from ..feature_tuning import (
    SERIES_SELECT,
    _ensure_sparse_index32,
    build_race_features,
    iter_complete_races,
    to_hashable,
)
from ..modeling import trifecta_predictions
from .model import stable_softmax


JST = timezone(timedelta(hours=9))
MODEL_NAME = "pastlog_listwise_newton_cg_v1_today_shadow"


def score_date(conn, *, artifact: dict[str, Any], race_date: str) -> dict[str, Any]:
    model = artifact["model"]
    hasher = artifact["hasher"]
    dropped = tuple(artifact.get("drop_feature_groups") or ())
    state = historical_state(conn, race_date=race_date)
    rows_by_race = load_date_races(conn, race_date=race_date)
    actual_by_race = load_actual_orders(conn, race_date=race_date)
    race_predictions = []

    for race_id, race_rows in rows_by_race.items():
        feature_rows = build_race_features(
            race_rows,
            state,
            drop_feature_groups=dropped,
            feature_schema_version=artifact.get("feature_schema_version"),
        )
        matrix = _ensure_sparse_index32(
            hasher.transform([to_hashable(item["features"]) for item in feature_rows])
        )
        scores = np.asarray(model.scaler.transform(matrix).dot(model.weights)).reshape(6)
        lane_probabilities = stable_softmax(scores)
        lane_rows = [
            {
                "lane": lane,
                "probability": float(lane_probabilities[lane - 1]),
            }
            for lane in range(1, 7)
        ]
        trifectas = trifecta_predictions(
            {row["lane"]: row["probability"] for row in lane_rows}
        )
        actual = actual_by_race.get(race_id)
        top_lane = max(lane_rows, key=lambda row: row["probability"])["lane"]
        top5 = [row["combination"] for row in trifectas[:5]]
        race_predictions.append(
            {
                "race_id": race_id,
                "race_date": race_date,
                "jcd": str(race_rows[0]["jcd"]),
                "rno": int(race_rows[0]["rno"]),
                "lane_probabilities": lane_rows,
                "winner_prediction": top_lane,
                "trifecta_top5": top5,
                "actual_order": actual,
                "winner_hit": bool(actual and top_lane == actual[0]),
                "trifecta_top5_hit": bool(
                    actual and "-".join(str(value) for value in actual[:3]) in top5
                ),
            }
        )

    metrics = evaluate_predictions(race_predictions)
    return {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "model": MODEL_NAME,
        "source_model": "pastlog_listwise_newton_cg_v1",
        "comparison_role": "today_pre_race_shadow_evaluation",
        "race_date": race_date,
        "artifact_trained_races": artifact.get("trained_races"),
        "artifact_trained_through": artifact.get("trained_through"),
        "predicted_races": len(race_predictions),
        **metrics,
        "predictions": race_predictions,
    }


def evaluate_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("actual_order")]
    labels = []
    probabilities = []
    for row in completed:
        actual_winner = int(row["actual_order"][0])
        for lane_row in row.get("lane_probabilities") or []:
            labels.append(int(lane_row["lane"] == actual_winner))
            probabilities.append(float(lane_row["probability"]))
    return {
        "evaluated_races": len(completed),
        "entry_log_loss": (
            float(log_loss(labels, probabilities, labels=[0, 1])) if labels else None
        ),
        "entry_brier": float(brier_score_loss(labels, probabilities)) if labels else None,
        "winner_top1_accuracy": (
            sum(bool(row.get("winner_hit")) for row in completed) / len(completed)
            if completed
            else None
        ),
        "trifecta_top5_hit_rate": (
            sum(bool(row.get("trifecta_top5_hit")) for row in completed) / len(completed)
            if completed
            else None
        ),
    }


def historical_state(conn, *, race_date: str) -> RollingState:
    state = RollingState()
    for rows in iter_complete_races(conn):
        if str(rows[0]["race_date"]) >= race_date:
            break
        state.update_race(rows)
    return state


def load_date_races(conn, *, race_date: str) -> dict[str, list[Any]]:
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
          0 AS rank, NULL AS result_course, NULL AS result_start_timing
        FROM races r
        JOIN entries e ON e.race_id = r.race_id
        LEFT JOIN entry_series_features sf
          ON sf.race_id = e.race_id AND sf.lane = e.lane
        WHERE r.race_date = ?
        ORDER BY r.jcd, r.rno, e.lane
        """,
        (race_date,),
    ).fetchall()
    grouped: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        grouped[str(row["race_id"])].append(row)
    return {
        race_id: race_rows
        for race_id, race_rows in grouped.items()
        if len(race_rows) == 6
    }


def load_actual_orders(conn, *, race_date: str) -> dict[str, list[int]]:
    rows = conn.execute(
        """
        SELECT rr.race_id, rr.lane, rr.rank
        FROM race_results rr
        JOIN races r ON r.race_id = rr.race_id
        WHERE r.race_date = ? AND rr.rank IS NOT NULL
        ORDER BY rr.race_id, rr.rank
        """,
        (race_date,),
    ).fetchall()
    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["race_id"])].append((int(row["rank"]), int(row["lane"])))
    return {
        race_id: [lane for _rank, lane in sorted(values)]
        for race_id, values in grouped.items()
        if len(values) == 6
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Today's listwise Newton shadow scorer.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument(
        "--model",
        default="data/models/listwise_newton_cg_v1.joblib",
    )
    parser.add_argument(
        "--output",
        default="data/models/listwise_newton_today_shadow.json",
    )
    parser.add_argument("--date")
    parser.add_argument("--interval", type=float, default=120.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)

    init_db(args.db)
    artifact = joblib.load(args.model)
    while True:
        race_date = args.date or datetime.now(JST).date().isoformat()
        with connection(args.db) as conn:
            result = score_date(conn, artifact=artifact, race_date=race_date)
        write_json_atomic(Path(args.output), result)
        print(
            json.dumps(
                {
                    key: result.get(key)
                    for key in (
                        "generated_at",
                        "race_date",
                        "predicted_races",
                        "evaluated_races",
                        "winner_top1_accuracy",
                        "trifecta_top5_hit_rate",
                    )
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if args.once:
            return 0
        time.sleep(max(30.0, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
