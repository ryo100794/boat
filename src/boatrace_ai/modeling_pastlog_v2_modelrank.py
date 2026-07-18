from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from .db import connection, init_db, insert_prediction_rows
from .features import latest_trifecta_odds
from .features_no_odds_v3 import race_relative_features
from .features_no_odds_v9 import RollingState, load_race_entries
from .features_pastlog_v1 import base_pastlog_features
from .features_pastlog_v2 import history_groups_prior_dates, prediction_features
from .modeling import _normalize_lane_probs, trifecta_predictions
from .modeling_pastlog_v2 import FEATURE_SET, positive_probs


MODEL_RANK_FEATURE_SET = f"{FEATURE_SET}_model_probability_rank"


def predict_race(
    conn,
    *,
    model_path: Path,
    race_id_value: str,
    top_n: int = 120,
    store: bool = True,
) -> list[dict[str, Any]]:
    bundle = joblib.load(model_path)
    pipeline = bundle["pipeline"]
    X = prediction_features(conn, race_id=race_id_value, include_odds=False)
    if len(X) != 6:
        raise ValueError(f"race needs six entries before prediction: {race_id_value}")
    rows = _predictions_from_features(conn, pipeline, X, race_id_value, top_n=top_n)
    if store:
        insert_prediction_rows(conn, race_id_value, _now(), str(model_path), rows)
    return rows


def predict_open_races(
    conn,
    *,
    model_path: Path,
    race_date: date,
    jcd: str | None = None,
    rno: int | None = None,
) -> dict[str, int]:
    bundle = joblib.load(model_path)
    pipeline = bundle["pipeline"]
    grouped = _open_race_rows(conn, race_date=race_date, jcd=jcd, rno=rno)
    if not grouped:
        return {"predicted": 0, "failed": 0}
    first_row = next(iter(grouped.values()))[0]
    state = RollingState()
    for history_rows in history_groups_prior_dates(conn, first_row):
        state.update_race(history_rows)

    ok = 0
    failed = 0
    for race_id_value, rows in grouped.items():
        try:
            X = _features_for_rows(rows, state)
            predictions = _predictions_from_features(conn, pipeline, X, race_id_value, top_n=120)
            insert_prediction_rows(conn, race_id_value, _now(), str(model_path), predictions)
            ok += 1
        except Exception:
            failed += 1
    return {"predicted": ok, "failed": failed}


def _open_race_rows(
    conn,
    *,
    race_date: date,
    jcd: str | None = None,
    rno: int | None = None,
) -> dict[str, list[Any]]:
    params: list[Any] = [race_date.isoformat()]
    filters = ["r.race_date = ?"]
    if jcd:
        filters.append("r.jcd = ?")
        params.append(jcd.zfill(2))
    if rno:
        filters.append("r.rno = ?")
        params.append(int(rno))
    rows = conn.execute(
        f"""
        SELECT
          r.race_id, r.race_date, r.jcd, r.rno, r.race_type, r.distance_m,
          e.lane, e.racer_no, e.racer_name, e.racer_class, e.branch, e.origin,
          e.age, e.weight_kg, e.f_count, e.l_count, e.avg_st,
          e.national_win_rate, e.national_2_rate, e.national_3_rate,
          e.local_win_rate, e.local_2_rate, e.local_3_rate,
          e.motor_no, e.motor_2_rate, e.motor_3_rate,
          e.boat_no, e.boat_2_rate, e.boat_3_rate
        FROM entries e
        JOIN races r ON r.race_id = e.race_id
        WHERE {" AND ".join(filters)}
          AND (r.status IS NULL OR r.status != 'final')
        ORDER BY r.jcd, r.rno, e.lane
        """,
        params,
    ).fetchall()
    grouped: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        grouped[row["race_id"]].append(row)
    return {race_id_value: sorted(items, key=lambda row: int(row["lane"])) for race_id_value, items in grouped.items() if len(items) == 6}


def _features_for_rows(rows: list[Any], state: RollingState) -> list[dict[str, Any]]:
    relatives = race_relative_features(rows, {lane: {} for lane in range(1, 7)})
    result = []
    for row in rows:
        lane = int(row["lane"])
        item = base_pastlog_features(row, relatives[lane])
        item.update(state.features_for(row))
        result.append(item)
    return result


def _predictions_from_features(conn, pipeline, X: list[dict[str, Any]], race_id_value: str, *, top_n: int) -> list[dict[str, Any]]:
    raw = positive_probs(pipeline, X)
    lane_probs = _normalize_lane_probs({lane: raw[lane - 1] for lane in range(1, 7)})
    rows = trifecta_predictions(lane_probs, latest_odds=latest_trifecta_odds(conn, race_id_value))
    rows = sorted(
        rows,
        key=lambda row: (float(row["probability"]), float(row.get("expected_value") or 0.0)),
        reverse=True,
    )[:top_n]
    for row in rows:
        row["rank_basis"] = "model_probability"
        row["feature_set"] = MODEL_RANK_FEATURE_SET
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Store past-log v2 predictions ranked by model probability.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--model", default="data/models/win_model_pastlog_v2.joblib")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--jcd")
    parser.add_argument("--rno", type=int)
    args = parser.parse_args(argv)

    init_db(args.db)
    with connection(args.db) as conn:
        result = predict_open_races(
            conn,
            model_path=Path(args.model),
            race_date=date.fromisoformat(args.date),
            jcd=args.jcd,
            rno=args.rno,
        )
    print(json.dumps({"model": str(args.model), "rank_basis": "model_probability", **result}, ensure_ascii=False), flush=True)
    return 0


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
