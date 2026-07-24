from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .legacy_model_aliases import load_model_bundle

from .db import connection, init_db, insert_prediction_rows
from .features import latest_trifecta_odds
from .base_features import prediction_features
from .calibrated_shadow_model import (
    predict_probabilities as calibrated_predict_probabilities,
)
from .operational_features import (
    prediction_features as calibrated_prediction_features,
)
from .modeling import _normalize_lane_probs, trifecta_predictions
from .listwise.conditional_order import (
    ConditionalOrderModel,
    conditional_probabilities,
)
from .historical_model import FEATURE_SET, positive_probs


MODEL_RANK_FEATURE_SET = f"{FEATURE_SET}_model_probability_rank"


def predict_race(
    conn,
    *,
    model_path: Path,
    race_id_value: str,
    top_n: int = 120,
    store: bool = True,
) -> list[dict[str, Any]]:
    bundle = load_model_bundle(model_path)
    if "pipeline" in bundle:
        X = prediction_features(conn, race_id=race_id_value, include_odds=False)
        raw = positive_probs(bundle["pipeline"], X)
        feature_set = MODEL_RANK_FEATURE_SET
    elif all(key in bundle for key in ("hasher", "scaler", "classifier")):
        feature_schema_version = str(bundle.get("feature_schema_version") or "")
        if not feature_schema_version:
            raise ValueError("calibrated model artifact lacks feature schema")
        X = calibrated_prediction_features(
            conn,
            race_id=race_id_value,
            include_odds=False,
            feature_schema_version=feature_schema_version,
            drop_feature_groups=tuple(bundle.get("drop_feature_groups") or ()),
        )
        raw = calibrated_predict_probabilities(bundle, X)
        artifact_feature_set = str(
            (bundle.get("metadata") or {}).get("feature_set")
            or "pastlog_calibrated_hash_shadow"
        )
        feature_set = f"{artifact_feature_set}_model_probability_rank"
    else:
        raise ValueError("unsupported operational model artifact")
    if len(X) != 6 or len(raw) != 6:
        raise ValueError(f"race needs six entries before prediction: {race_id_value}")
    lane_probs = _normalize_lane_probs({lane: raw[lane - 1] for lane in range(1, 7)})
    order_model = bundle.get("conditional_order_model")
    trifecta_values = None
    rank_basis = "model_probability"
    if order_model is not None:
        if not isinstance(order_model, ConditionalOrderModel):
            raise ValueError(
                "calibrated model artifact has an invalid conditional order model"
            )
        lane_values = np.asarray(
            [[lane_probs[lane] for lane in range(1, 7)]],
            dtype=np.float64,
        )
        trifecta_values = conditional_probabilities(
            np.log(np.clip(lane_values, 1e-15, 1.0)),
            order_model,
        )[0]
        rank_basis = "conditional_order_probability"
    rows = trifecta_predictions(
        lane_probs,
        latest_odds=latest_trifecta_odds(conn, race_id_value),
        trifecta_probabilities=trifecta_values,
    )
    rows = sorted(
        rows,
        key=lambda row: (float(row["probability"]), float(row.get("expected_value") or 0.0)),
        reverse=True,
    )[:top_n]
    for row in rows:
        row["rank_basis"] = rank_basis
        row["feature_set"] = feature_set
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
    parser = argparse.ArgumentParser(description="Store no-odds v8 predictions ranked by model probability.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--model", default="data/models/win_model_no_odds_v8.joblib")
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

