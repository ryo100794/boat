from __future__ import annotations

import hashlib
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


def archive_oracle_queue_status(
    db_path: Path, *, connector: Any = None
) -> dict[str, Any]:
    """Read only the archive jobs needed by the three-page audit header."""
    try:
        with (connector or connect)(db_path) as conn:
            rows = conn.execute(
                """
                WITH ranked AS (
                  SELECT job_id, task_type, model_key, status, parameters,
                         result_summary, decision, updated_at,
                         ROW_NUMBER() OVER (
                           PARTITION BY status ORDER BY job_id DESC
                         ) AS status_rank
                  FROM model_evaluation_jobs
                  WHERE task_type = 'archive_market_oracle'
                )
                SELECT job_id, task_type, model_key, status, parameters,
                       result_summary, decision, updated_at
                FROM ranked
                WHERE status_rank = 1
                ORDER BY job_id DESC
                """
            ).fetchall()
    except Exception:
        return {}

    jobs = [
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
        for row in rows
    ]
    latest = jobs[0] if jobs else None
    running = next(
        (row for row in jobs if row["status"] == "running"), None
    )
    queued = next(
        (row for row in jobs if row["status"] == "queued"), None
    )
    completed = next(
        (row for row in jobs if row["status"] == "completed"), None
    )
    return {
        "latest_archive_oracle": latest,
        "active_archive_oracle": running or queued,
        "running_archive_oracle": running,
        "queued_archive_oracle": queued,
        "latest_completed_archive_oracle": completed,
        "recent_archive_oracles": jobs,
    }


def queue_model_roadmap_status(
    db_path: Path, *, connector: Any = None
) -> dict[str, Any]:
    """Read only recent model/collection jobs needed by the roadmap header."""
    try:
        with (connector or connect)(db_path) as conn:
            rows = conn.execute(
                """
                SELECT job_id, task_type, model_key, status, parameters,
                       result_summary, decision, updated_at
                FROM model_evaluation_jobs
                WHERE task_type IN (
                  'listwise_newton_refine',
                  'bankroll_policy_search',
                  'market_residual_walk_forward',
                  'archive_market_oracle',
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
    archive_oracles = [
        row for row in jobs if row["task_type"] == "archive_market_oracle"
    ]
    latest_archive_oracle = archive_oracles[0] if archive_oracles else None
    running_archive_oracle = next(
        (row for row in archive_oracles if row["status"] == "running"),
        None,
    )
    queued_archive_oracle = next(
        (row for row in archive_oracles if row["status"] == "queued"),
        None,
    )
    active_archive_oracle = running_archive_oracle or queued_archive_oracle
    latest_completed_archive_oracle = next(
        (row for row in archive_oracles if row["status"] == "completed"),
        None,
    )
    return {
        "latest_newton": latest_newton,
        "latest_contextual_lcb95": latest_lcb,
        "best_market_v21": best_market,
        "latest_market_v21": latest_market,
        "official_closing_collection": official_collection,
        "latest_archive_oracle": latest_archive_oracle,
        "active_archive_oracle": active_archive_oracle,
        "running_archive_oracle": running_archive_oracle,
        "queued_archive_oracle": queued_archive_oracle,
        "latest_completed_archive_oracle": latest_completed_archive_oracle,
        "recent_archive_oracles": archive_oracles[:5],
    }


def archive_oracle_audit_status(
    queue_status: dict[str, Any],
) -> dict[str, Any]:
    """Build one small audit state shared by the operations pages."""
    running = queue_status.get("running_archive_oracle")
    queued = queue_status.get("queued_archive_oracle")
    latest = queue_status.get("latest_archive_oracle")
    completed = queue_status.get("latest_completed_archive_oracle")
    if running:
        status = "評価実行中"
        reason = "実行中の評価成果物が確定していない"
    elif queued:
        status = "評価待機中"
        reason = "後段の評価成果物が確定していない"
    elif isinstance(latest, dict) and latest.get("status") in {
        "failed",
        "cancelled",
    }:
        status = "評価失敗"
        reason = "最新評価が正常完了していない"
    elif completed:
        status = "外部監査可能"
        reason = "最新評価が完了し、3画面のDB正本を確定できる"
    else:
        status = "評価未登録"
        reason = "監査対象となる評価成果物がない"

    def compact(job: Any) -> dict[str, Any] | None:
        if not isinstance(job, dict):
            return None
        metrics = job.get("metrics")
        metrics = metrics if isinstance(metrics, dict) else {}
        return {
            "job_id": job.get("job_id"),
            "model_key": job.get("model_key"),
            "status": job.get("status"),
            "decision": job.get("decision"),
            "updated_at": job.get("updated_at"),
            "model": metrics.get("model"),
            "nested_value_model": metrics.get("nested_value_model"),
            "roi": metrics.get("roi"),
            "roi_status": metrics.get("roi_status"),
            "profit_yen": metrics.get("profit_yen"),
            "promotion_eligible": metrics.get(
                "nested_value_promotion_eligible",
                metrics.get("promotion_eligible"),
            ),
        }

    completed_public = compact(completed)
    public_state = {
        "scope": ["dashboard", "model-performance", "roadmap"],
        "status": status,
        "audit_ready": bool(
            completed
            and not running
            and not queued
            and isinstance(latest, dict)
            and latest.get("job_id") == completed.get("job_id")
        ),
        "reason": reason,
        "promotion_status": (
            "昇格ゲート合格"
            if (completed_public or {}).get("promotion_eligible") is True
            else "未承認"
        ),
        "running": compact(running),
        "queued": compact(queued),
        "latest_completed": completed_public,
    }
    logical_state = {
        **public_state,
        "running": (
            {key: value for key, value in public_state["running"].items()
             if key != "updated_at"}
            if public_state["running"] else None
        ),
        "queued": (
            {key: value for key, value in public_state["queued"].items()
             if key != "updated_at"}
            if public_state["queued"] else None
        ),
        "latest_completed": (
            {
                key: value
                for key, value in public_state["latest_completed"].items()
                if key != "updated_at"
            }
            if public_state["latest_completed"] else None
        ),
    }
    canonical = json.dumps(
        logical_state,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    public_state["audit_snapshot_id"] = hashlib.sha256(canonical).hexdigest()
    public_state["audit_snapshot_basis"] = (
        "archive evaluation job IDs, states, decisions, model identities, "
        "ROI state, profit and promotion result; heartbeat timestamps excluded"
    )
    return public_state
