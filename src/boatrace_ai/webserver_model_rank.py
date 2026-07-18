from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

from .db import connect
from .webserver_all import required, rowdict
from .webserver_operational2 import race_payload


def race_payload_model_rank(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    now,
    before_minutes: int = 5,
) -> dict[str, Any]:
    item = race_payload(conn, row, now=now, before_minutes=before_minutes)
    model_top, model_top5 = latest_prediction_rows_by_probability(conn, row["race_id"], limit=5)
    buy_top, buy_top5 = latest_prediction_rows_by_ev(conn, row["race_id"], limit=5)
    item["top_prediction"] = model_top
    item["top5"] = model_top5
    item["buy_prediction"] = buy_top
    item["buy_top5"] = buy_top5
    item["prediction_rank_basis"] = "model_probability"
    return item


def latest_prediction_rows_by_probability(
    conn: sqlite3.Connection,
    race_id: str,
    *,
    limit: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    return _latest_prediction_rows(
        conn,
        race_id,
        limit=limit,
        order_sql="probability DESC, COALESCE(expected_value, 0) DESC, combination",
    )


def latest_prediction_rows_by_ev(
    conn: sqlite3.Connection,
    race_id: str,
    *,
    limit: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    return _latest_prediction_rows(
        conn,
        race_id,
        limit=limit,
        order_sql="expected_value IS NOT NULL DESC, expected_value DESC, probability DESC, combination",
    )


def _latest_prediction_rows(
    conn: sqlite3.Connection,
    race_id: str,
    *,
    limit: int,
    order_sql: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    latest = conn.execute(
        "SELECT generated_at FROM predictions WHERE race_id = ? ORDER BY generated_at DESC LIMIT 1",
        (race_id,),
    ).fetchone()
    if not latest:
        return None, []
    rows = conn.execute(
        f"""
        SELECT combination, probability, odds, expected_value, generated_at
        FROM predictions
        WHERE race_id = ? AND generated_at = ?
        ORDER BY {order_sql}
        LIMIT ?
        """,
        (race_id, latest["generated_at"], limit),
    ).fetchall()
    mapped = [rowdict(row) for row in rows]
    return (mapped[0] if mapped else None), mapped


def predictions_model_rank(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    race_id = required(query, "race_id")
    with connect(db_path) as conn:
        race = conn.execute("SELECT * FROM races WHERE race_id = ?", (race_id,)).fetchone()
        entries = conn.execute(
            """
            SELECT lane, racer_no, racer_name, racer_class, motor_no, boat_no
            FROM entries
            WHERE race_id = ?
            ORDER BY lane
            """,
            (race_id,),
        ).fetchall()
        latest = conn.execute(
            """
            SELECT generated_at
            FROM predictions
            WHERE race_id = ?
            ORDER BY generated_at DESC
            LIMIT 1
            """,
            (race_id,),
        ).fetchone()
        pred_rows = []
        if latest:
            pred_rows = conn.execute(
                """
                SELECT combination, probability, odds, expected_value, generated_at
                FROM predictions
                WHERE race_id = ? AND generated_at = ?
                ORDER BY probability DESC, COALESCE(expected_value, 0) DESC, combination
                LIMIT 120
                """,
                (race_id, latest["generated_at"]),
            ).fetchall()
    return {
        "race": rowdict(race) if race else None,
        "entries": [rowdict(row) for row in entries],
        "predictions": [rowdict(row) for row in pred_rows],
        "prediction_rank_basis": "model_probability",
    }


def accuracy_model_rank(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    race_date = query.get("date", [date.today().isoformat()])[0]
    with connect(db_path) as conn:
        race_rows = conn.execute(
            """
            SELECT r.race_id
            FROM races r
            WHERE r.race_date = ?
              AND (SELECT COUNT(*) FROM race_results rr WHERE rr.race_id = r.race_id AND rr.rank IS NOT NULL) >= 3
              AND EXISTS (SELECT 1 FROM predictions p WHERE p.race_id = r.race_id)
            """,
            (race_date,),
        ).fetchall()
        evaluated = 0
        winner_hits = 0
        trifecta_top1_hits = 0
        trifecta_top5_hits = 0
        for race in race_rows:
            rid = race["race_id"]
            actual_rows = conn.execute(
                """
                SELECT lane, rank
                FROM race_results
                WHERE race_id = ? AND rank IS NOT NULL
                ORDER BY rank
                LIMIT 3
                """,
                (rid,),
            ).fetchall()
            if len(actual_rows) < 3:
                continue
            actual_combo = "-".join(str(row["lane"]) for row in actual_rows)
            actual_winner = actual_combo.split("-")[0]
            _, pred_rows = latest_prediction_rows_by_probability(conn, rid, limit=5)
            if not pred_rows:
                continue
            top = pred_rows[0]["combination"]
            top5 = [row["combination"] for row in pred_rows]
            evaluated += 1
            winner_hits += 1 if top.split("-")[0] == actual_winner else 0
            trifecta_top1_hits += 1 if top == actual_combo else 0
            trifecta_top5_hits += 1 if actual_combo in top5 else 0
    return {
        "date": race_date,
        "evaluated": evaluated,
        "winner_top1_accuracy": winner_hits / evaluated if evaluated else None,
        "trifecta_top1_hit_rate": trifecta_top1_hits / evaluated if evaluated else None,
        "trifecta_top5_hit_rate": trifecta_top5_hits / evaluated if evaluated else None,
        "prediction_rank_basis": "model_probability",
    }


def buy_score(item: dict[str, Any]) -> float:
    top = item.get("buy_prediction") or item.get("top_prediction") or {}
    if top.get("expected_value") is not None:
        return float(top["expected_value"])
    return float(top.get("probability") or 0.0)
