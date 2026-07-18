from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from .db import connection, init_db, insert_prediction_rows
from .features import latest_trifecta_odds
from .features_pastlog_v4 import prediction_features
from .modeling import _normalize_lane_probs, trifecta_predictions
from .modeling_pastlog_v4 import FEATURE_SET, positive_probs


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Store past-log v4 predictions ranked by model probability.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--model", default="data/models/win_model_pastlog_v4.joblib")
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
