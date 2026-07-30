from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..db import connect


def _object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def queue_model_roadmap_status(db_path: Path) -> dict[str, Any]:
    """Read only recent model/collection jobs needed by the roadmap header."""
    try:
        with connect(db_path) as conn:
            rows = conn.execute(
                """
                SELECT job_id, task_type, model_key, status, parameters,
                       result_summary, decision, updated_at
                FROM model_evaluation_jobs
                WHERE task_type IN (
                  'listwise_newton_refine',
                  'bankroll_policy_search',
                  'market_residual_walk_forward',
                  'archive_closing_backfill'
                )
                ORDER BY job_id DESC
                LIMIT 200
                """
            ).fetchall()
    except Exception:
        return {}

    jobs = []
    for row in rows:
        jobs.append(
            {
                "job_id": int(row["job_id"]),
                "task_type": str(row["task_type"]),
                "model_key": str(row["model_key"]),
                "status": str(row["status"]),
                "parameters": _object(row["parameters"]),
                "metrics": _object(row["result_summary"]),
                "decision": row["decision"],
                "updated_at": str(row["updated_at"]),
            }
        )

    completed = [row for row in jobs if row["status"] == "completed"]
    latest_newton = next(
        (row for row in completed if row["task_type"] == "listwise_newton_refine"),
        None,
    )
    latest_lcb = next(
        (
            row
            for row in completed
            if row["task_type"] == "bankroll_policy_search"
            and "contextual_lcb95" in row["model_key"]
        ),
        None,
    )
    market = [
        row
        for row in completed
        if row["task_type"] == "market_residual_walk_forward"
        and "triple_head_v21" in str(row["metrics"].get("model") or "")
    ]
    best_market = max(
        market,
        key=lambda row: float(row["metrics"].get("roi") or 0.0),
        default=None,
    )
    latest_market = market[0] if market else None
    official_collection = next(
        (
            row
            for row in jobs
            if row["task_type"] == "archive_closing_backfill"
            and row["parameters"].get("source") == "official"
        ),
        None,
    )
    return {
        "latest_newton": latest_newton,
        "latest_contextual_lcb95": latest_lcb,
        "best_market_v21": best_market,
        "latest_market_v21": latest_market,
        "official_closing_collection": official_collection,
    }
