from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from typing import Any


def attach_latest_prediction_summaries(
    conn: sqlite3.Connection,
    items: Iterable[dict[str, Any]],
) -> None:
    """Attach model-ranked and EV-ranked predictions using one indexed batch read."""
    by_race = {str(item["race_id"]): item for item in items if item.get("race_id")}
    if not by_race:
        return

    predictions: dict[str, list[sqlite3.Row]] = {}
    race_ids = list(by_race)
    for offset in range(0, len(race_ids), 800):
        chunk = race_ids[offset : offset + 800]
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            f"""
            WITH latest AS (
              SELECT race_id, MAX(generated_at) AS generated_at
              FROM predictions
              WHERE race_id IN ({placeholders})
              GROUP BY race_id
            )
            SELECT p.race_id, p.combination, p.probability, p.odds,
                   p.expected_value, p.generated_at
            FROM predictions p
            JOIN latest l
              ON l.race_id = p.race_id AND l.generated_at = p.generated_at
            """,
            chunk,
        ).fetchall()
        for row in rows:
            predictions.setdefault(str(row["race_id"]), []).append(row)

    for race_id, rows in predictions.items():
        item = by_race[race_id]
        top = max(
            rows,
            key=lambda row: (
                float(row["probability"] or 0.0),
                float(row["expected_value"] or 0.0),
                str(row["combination"]),
            ),
        )
        buy = max(
            rows,
            key=lambda row: (
                row["expected_value"] is not None,
                float(row["expected_value"] or 0.0),
                float(row["probability"] or 0.0),
                str(row["combination"]),
            ),
        )
        item["top_prediction"] = _payload(top)
        item["buy_prediction"] = _payload(buy)
        item["top5"] = [item["top_prediction"]]
        item["buy_top5"] = [item["buy_prediction"]]


def _payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "combination": row["combination"],
        "probability": row["probability"],
        "odds": row["odds"],
        "expected_value": row["expected_value"],
        "generated_at": row["generated_at"],
    }
