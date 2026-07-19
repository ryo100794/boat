from __future__ import annotations

import sqlite3

from boatrace_ai.web.prediction_summary import attach_latest_prediction_summaries


def test_attaches_latest_model_and_ev_ranked_predictions() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE predictions (
          race_id TEXT, generated_at TEXT, combination TEXT,
          probability REAL, odds REAL, expected_value REAL
        )
        """
    )
    conn.executemany(
        "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("r1", "2026-07-19T00:00:00+00:00", "1-2-3", 0.90, 2.0, 1.8),
            ("r1", "2026-07-19T00:01:00+00:00", "2-1-3", 0.30, 4.0, 1.2),
            ("r1", "2026-07-19T00:01:00+00:00", "3-1-2", 0.20, 9.0, 1.8),
        ],
    )
    items = [{"race_id": "r1", "top_prediction": None, "buy_prediction": None}]

    attach_latest_prediction_summaries(conn, items)

    assert items[0]["top_prediction"]["combination"] == "2-1-3"
    assert items[0]["top_prediction"]["odds"] == 4.0
    assert items[0]["buy_prediction"]["combination"] == "3-1-2"
