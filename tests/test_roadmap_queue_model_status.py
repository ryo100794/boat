from __future__ import annotations

import json
import sqlite3

from boatrace_ai.web.dashboard import _roadmap_milestones
from boatrace_ai.web.roadmap_model_status import queue_model_roadmap_status


def test_queue_model_status_drives_current_m4_and_m6_text(tmp_path) -> None:
    db_path = tmp_path / "roadmap.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE model_evaluation_jobs(
          job_id INTEGER PRIMARY KEY,
          task_type TEXT NOT NULL,
          model_key TEXT NOT NULL,
          status TEXT NOT NULL,
          parameters TEXT,
          result_summary TEXT,
          decision TEXT,
          updated_at TEXT
        )
        """
    )
    rows = [
        (
            9132,
            "listwise_newton_refine",
            "history_centered_decayed:newton",
            "completed",
            {},
            {
                "winner_top1_accuracy": 0.5748,
                "trifecta_top5_hit_rate": 0.3284,
                "entry_log_loss": 0.31859,
                "roi": 0.8103,
            },
        ),
        (
            8666,
            "market_residual_walk_forward",
            "production-market-v21",
            "completed",
            {},
            {
                "model": "odds_path_observed_closing_return_schedule_quota_triple_head_v21",
                "evaluated_races": 918,
                "evaluation_days": 6,
                "roi": 1.4756,
                "profit_yen": 3900,
            },
        ),
        (
            9160,
            "market_residual_walk_forward",
            "decayed-market-v21",
            "completed",
            {},
            {
                "model": "odds_path_observed_closing_return_schedule_quota_triple_head_v21",
                "evaluated_races": 918,
                "evaluation_days": 6,
                "roi": 0.7561,
                "profit_yen": -2000,
            },
        ),
        (
            9205,
            "archive_closing_backfill",
            "official-closing",
            "running",
            {"source": "official"},
            {},
        ),
    ]
    conn.executemany(
        "INSERT INTO model_evaluation_jobs VALUES (?, ?, ?, ?, ?, ?, NULL, 'now')",
        [
            (job_id, task_type, model_key, status, json.dumps(parameters), json.dumps(metrics))
            for job_id, task_type, model_key, status, parameters, metrics in rows
        ],
    )
    conn.commit()
    conn.close()

    status = queue_model_roadmap_status(db_path)
    assert status["latest_newton"]["job_id"] == 9132
    assert status["best_market_v21"]["job_id"] == 8666
    assert status["latest_market_v21"]["job_id"] == 9160
    assert status["official_closing_collection"]["job_id"] == 9205

    milestones = {
        row["id"]: row
        for row in _roadmap_milestones({"queue_model_status": status}, [], {})
    }
    assert "job 9132" in milestones["M4"]["next"]
    assert "57.48%" in milestones["M4"]["next"]
    assert "job 8666" in milestones["M6"]["next"]
    assert "ROI 1.4756" in milestones["M6"]["next"]
    assert "job 9160" in milestones["M6"]["next"]
