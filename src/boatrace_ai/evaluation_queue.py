from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
import resource
import shutil
import socket
import subprocess
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .db import connection


JST = ZoneInfo("Asia/Tokyo")
DEFAULT_DSN = "host=127.0.0.1 port=5432 dbname=boatrace user=boatrace_app"
SCHEMA_LOCK_ID = 71234001
CLAIM_LOCK_ID = 71234002


@dataclass(frozen=True)
class ResourceSnapshot:
    available_memory_mb: int
    available_disk_mb: int
    idle_cpu_percent: float
    cpu_count: int
    load_1m: float
    memory_limit_mb: int | None = None
    memory_usage_mb: int | None = None


TASK_PROFILES: dict[str, dict[str, Any]] = {
    "standardized_365d": {"category": "evaluation", "memory_mb": 16384, "idle_cpu": 15.0, "max_parallel": 1, "disk_mb": 8192},
    "market_curvature": {"category": "evaluation", "memory_mb": 2048, "idle_cpu": 5.0, "max_parallel": 4, "disk_mb": 1024},
    "listwise_feature_search": {"category": "evaluation", "memory_mb": 8192, "idle_cpu": 15.0, "max_parallel": 2, "disk_mb": 4096},
    "listwise_newton_refine": {"category": "evaluation", "memory_mb": 8192, "idle_cpu": 15.0, "max_parallel": 2, "disk_mb": 4096},
    "venue_conditional_order": {"category": "evaluation", "memory_mb": 12288, "idle_cpu": 15.0, "max_parallel": 1, "disk_mb": 2048},
    "evaluation_aggregate": {"category": "aggregation", "memory_mb": 512, "idle_cpu": 3.0, "max_parallel": 1, "disk_mb": 256},
    "gdrive_raw_archive": {"category": "backup", "memory_mb": 512, "idle_cpu": 3.0, "max_parallel": 1, "disk_mb": 256},
}


SCHEMA = """
CREATE TABLE IF NOT EXISTS model_evaluation_jobs (
  job_id BIGSERIAL PRIMARY KEY,
  task_type TEXT NOT NULL,
  category TEXT NOT NULL DEFAULT 'evaluation',
  model_key TEXT NOT NULL,
  parameters JSONB NOT NULL DEFAULT '{}'::jsonb,
  dedupe_key TEXT NOT NULL UNIQUE,
  priority INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'queued'
    CHECK (status IN ('queued', 'running', 'completed', 'failed', 'cancelled')),
  attempt INTEGER NOT NULL DEFAULT 0,
  max_attempts INTEGER NOT NULL DEFAULT 2,
  min_free_memory_mb INTEGER NOT NULL DEFAULT 0,
  min_free_disk_mb INTEGER NOT NULL DEFAULT 0,
  min_idle_cpu_percent DOUBLE PRECISION NOT NULL DEFAULT 0,
  max_parallel INTEGER NOT NULL DEFAULT 4,
  available_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  worker_id TEXT,
  locked_at TIMESTAMPTZ,
  started_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  result_path TEXT,
  result_summary JSONB,
  last_resource_snapshot JSONB,
  decision TEXT,
  error TEXT,
  parent_job_id BIGINT REFERENCES model_evaluation_jobs(job_id),
  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
ALTER TABLE model_evaluation_jobs ADD COLUMN IF NOT EXISTS category TEXT NOT NULL DEFAULT 'evaluation';
ALTER TABLE model_evaluation_jobs ADD COLUMN IF NOT EXISTS min_free_memory_mb INTEGER NOT NULL DEFAULT 0;
ALTER TABLE model_evaluation_jobs ADD COLUMN IF NOT EXISTS min_free_disk_mb INTEGER NOT NULL DEFAULT 0;
ALTER TABLE model_evaluation_jobs ADD COLUMN IF NOT EXISTS min_idle_cpu_percent DOUBLE PRECISION NOT NULL DEFAULT 0;
ALTER TABLE model_evaluation_jobs ADD COLUMN IF NOT EXISTS max_parallel INTEGER NOT NULL DEFAULT 4;
ALTER TABLE model_evaluation_jobs ADD COLUMN IF NOT EXISTS last_resource_snapshot JSONB;
CREATE INDEX IF NOT EXISTS idx_model_evaluation_jobs_claim
  ON model_evaluation_jobs(status, available_at, priority DESC, job_id);
CREATE INDEX IF NOT EXISTS idx_model_evaluation_jobs_model
  ON model_evaluation_jobs(model_key, completed_at DESC);
CREATE TABLE IF NOT EXISTS model_improvement_candidates (
  candidate_id BIGSERIAL PRIMARY KEY,
  job_id BIGINT NOT NULL UNIQUE REFERENCES model_evaluation_jobs(job_id),
  model_key TEXT NOT NULL,
  task_type TEXT NOT NULL,
  decision TEXT NOT NULL,
  metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
  parameters JSONB NOT NULL DEFAULT '{}'::jsonb,
  result_path TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  reviewed_at TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS model_evaluation_job_runs (
  run_id BIGSERIAL PRIMARY KEY,
  job_id BIGINT NOT NULL REFERENCES model_evaluation_jobs(job_id),
  attempt INTEGER NOT NULL,
  worker_id TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
  resource_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
  started_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  completed_at TIMESTAMPTZ,
  result_path TEXT,
  error TEXT,
  UNIQUE(job_id, attempt)
);
CREATE TABLE IF NOT EXISTS work_tickets (
  ticket_key TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  area TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  acceptance_criteria TEXT NOT NULL DEFAULT '',
  owner TEXT NOT NULL DEFAULT 'codex',
  priority INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'queued'
    CHECK (status IN ('queued', 'in_progress', 'blocked', 'completed', 'cancelled')),
  progress INTEGER NOT NULL DEFAULT 0 CHECK (progress BETWEEN 0 AND 100),
  related_job_id BIGINT REFERENCES model_evaluation_jobs(job_id),
  source TEXT NOT NULL DEFAULT 'user',
  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  completed_at TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS work_ticket_events (
  event_id BIGSERIAL PRIMARY KEY,
  ticket_key TEXT NOT NULL REFERENCES work_tickets(ticket_key),
  status TEXT NOT NULL,
  progress INTEGER NOT NULL,
  note TEXT NOT NULL DEFAULT '',
  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_model_evaluation_job_runs_job
  ON model_evaluation_job_runs(job_id, attempt DESC);
CREATE INDEX IF NOT EXISTS idx_work_tickets_status
  ON work_tickets(status, priority DESC, updated_at DESC);
"""


def ensure_schema(conn: Any) -> None:
    if getattr(conn, "dialect", None) != "postgresql":
        raise RuntimeError("model evaluation queue requires PostgreSQL")
    conn.execute("SELECT pg_advisory_xact_lock(?)", (SCHEMA_LOCK_ID,))
    conn.executescript(SCHEMA)
    for task_type, profile in TASK_PROFILES.items():
        conn.execute(
            """
            UPDATE model_evaluation_jobs
            SET category = ?, min_free_memory_mb = ?, min_free_disk_mb = ?,
                min_idle_cpu_percent = ?, max_parallel = ?
            WHERE task_type = ?
              AND min_free_memory_mb = 0 AND min_free_disk_mb = 0
            """,
            (
                profile["category"], profile["memory_mb"], profile["disk_mb"],
                profile["idle_cpu"], profile["max_parallel"], task_type,
            ),
        )


def _read_cpu_times() -> tuple[int, int]:
    fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()[1:]
    values = [int(value) for value in fields]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return idle, sum(values)


def _cgroup_memory(root: Path = Path("/sys/fs/cgroup")) -> tuple[int, int] | None:
    candidates = (
        (root / "memory.max", root / "memory.current"),
        (
            root / "memory" / "memory.limit_in_bytes",
            root / "memory" / "memory.usage_in_bytes",
        ),
    )
    for limit_path, usage_path in candidates:
        try:
            limit_text = limit_path.read_text(encoding="utf-8").strip()
            usage = int(usage_path.read_text(encoding="utf-8").strip())
            if limit_text == "max":
                continue
            limit = int(limit_text)
        except (OSError, ValueError):
            continue
        if 0 < limit < 1 << 60 and usage >= 0:
            return limit, usage
    return None


def system_resources(sample_seconds: float = 0.15) -> ResourceSnapshot:
    meminfo = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        meminfo[key] = int(value.strip().split()[0])
    host_available_mb = int(meminfo.get("MemAvailable", 0) // 1024)
    memory_limit_mb = None
    memory_usage_mb = None
    cgroup = _cgroup_memory()
    if cgroup is not None:
        memory_limit_mb = int(cgroup[0] // 1024**2)
        memory_usage_mb = int(cgroup[1] // 1024**2)
        quota_available_mb = max(0, memory_limit_mb - memory_usage_mb - 4096)
        host_available_mb = min(host_available_mb, quota_available_mb)
    idle_before, total_before = _read_cpu_times()
    time.sleep(max(0.0, sample_seconds))
    idle_after, total_after = _read_cpu_times()
    total_delta = max(1, total_after - total_before)
    idle_percent = max(0.0, min(100.0, (idle_after - idle_before) * 100.0 / total_delta))
    return ResourceSnapshot(
        available_memory_mb=host_available_mb,
        available_disk_mb=int(shutil.disk_usage("/tmp").free // 1024**2),
        idle_cpu_percent=idle_percent,
        cpu_count=os.cpu_count() or 1,
        load_1m=float(os.getloadavg()[0]),
        memory_limit_mb=memory_limit_mb,
        memory_usage_mb=memory_usage_mb,
    )


def resource_snapshot_dict(snapshot: ResourceSnapshot) -> dict[str, Any]:
    return {
        "available_memory_mb": snapshot.available_memory_mb,
        "available_disk_mb": snapshot.available_disk_mb,
        "idle_cpu_percent": round(snapshot.idle_cpu_percent, 3),
        "cpu_count": snapshot.cpu_count,
        "load_1m": round(snapshot.load_1m, 3),
        "memory_limit_mb": snapshot.memory_limit_mb,
        "memory_usage_mb": snapshot.memory_usage_mb,
    }


def resources_allow(
    snapshot: ResourceSnapshot,
    *,
    min_free_memory_mb: int,
    min_free_disk_mb: int,
    min_idle_cpu_percent: float,
) -> bool:
    return (
        snapshot.available_memory_mb >= int(min_free_memory_mb)
        and snapshot.available_disk_mb >= int(min_free_disk_mb)
        and snapshot.idle_cpu_percent >= float(min_idle_cpu_percent)
    )


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def dedupe_key(task_type: str, model_key: str, parameters: dict[str, Any]) -> str:
    payload = _json([task_type, model_key, parameters]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def enqueue_job(
    conn: Any,
    *,
    task_type: str,
    model_key: str,
    parameters: dict[str, Any],
    priority: int = 0,
    max_attempts: int = 2,
    parent_job_id: int | None = None,
    category: str | None = None,
    min_free_memory_mb: int | None = None,
    min_free_disk_mb: int | None = None,
    min_idle_cpu_percent: float | None = None,
    max_parallel: int | None = None,
) -> int | None:
    profile = TASK_PROFILES.get(task_type)
    if profile is None:
        raise ValueError(f"unsupported task_type: {task_type}")
    category = category or str(profile["category"])
    min_free_memory_mb = int(profile["memory_mb"] if min_free_memory_mb is None else min_free_memory_mb)
    min_free_disk_mb = int(profile["disk_mb"] if min_free_disk_mb is None else min_free_disk_mb)
    min_idle_cpu_percent = float(profile["idle_cpu"] if min_idle_cpu_percent is None else min_idle_cpu_percent)
    max_parallel = int(profile["max_parallel"] if max_parallel is None else max_parallel)
    key = dedupe_key(task_type, model_key, parameters)
    row = conn.execute(
        """
        INSERT INTO model_evaluation_jobs(
          task_type, category, model_key, parameters, dedupe_key, priority,
          max_attempts, parent_job_id, min_free_memory_mb, min_free_disk_mb,
          min_idle_cpu_percent, max_parallel
        ) VALUES (?, ?, ?, CAST(? AS JSONB), ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(dedupe_key) DO NOTHING
        RETURNING job_id
        """,
        (
            task_type,
            category,
            model_key,
            _json(parameters),
            key,
            int(priority),
            int(max_attempts),
            parent_job_id,
            min_free_memory_mb,
            min_free_disk_mb,
            min_idle_cpu_percent,
            max_parallel,
        ),
    ).fetchone()
    return int(row["job_id"]) if row else None


def claim_job(
    conn: Any,
    *,
    worker_id: str,
    resources: ResourceSnapshot,
) -> dict[str, Any] | None:
    snapshot = _json(resource_snapshot_dict(resources))
    conn.execute("SELECT pg_advisory_xact_lock(?)", (CLAIM_LOCK_ID,))
    row = conn.execute(
        """
        WITH candidate AS (
          SELECT job_id
          FROM model_evaluation_jobs
          WHERE status = 'queued'
            AND available_at <= CURRENT_TIMESTAMP
            AND attempt < max_attempts
            AND min_free_memory_mb <= ?
            AND min_free_disk_mb <= ?
            AND min_idle_cpu_percent <= ?
            AND (
              SELECT COUNT(*) FROM model_evaluation_jobs running
              WHERE running.status = 'running'
                AND running.task_type = model_evaluation_jobs.task_type
            ) < max_parallel
            AND (
              category <> 'evaluation'
              OR min_free_memory_mb < 8192
              OR (
                CASE WHEN min_free_memory_mb >= 16384 THEN 2 ELSE 1 END
                + COALESCE((
                  SELECT SUM(
                    CASE WHEN running.min_free_memory_mb >= 16384 THEN 2 ELSE 1 END
                  )
                  FROM model_evaluation_jobs running
                  WHERE running.status = 'running'
                    AND running.category = 'evaluation'
                    AND running.min_free_memory_mb >= 8192
                ), 0)
              ) <= 2
            )
          ORDER BY priority DESC, job_id
          FOR UPDATE SKIP LOCKED
          LIMIT 1
        )
        UPDATE model_evaluation_jobs AS jobs
        SET status = 'running', worker_id = ?, locked_at = CURRENT_TIMESTAMP,
            started_at = COALESCE(started_at, CURRENT_TIMESTAMP),
            attempt = attempt + 1, updated_at = CURRENT_TIMESTAMP,
            error = NULL, last_resource_snapshot = CAST(? AS JSONB)
        FROM candidate
        WHERE jobs.job_id = candidate.job_id
        RETURNING jobs.*
        """,
        (resources.available_memory_mb, resources.available_disk_mb, resources.idle_cpu_percent, worker_id, snapshot),
    ).fetchone()
    if row is None:
        return None
    result = {key: row[key] for key in row.keys()}
    params = result.get("parameters")
    result["parameters"] = params if isinstance(params, dict) else json.loads(params or "{}")
    conn.execute(
        """
        INSERT INTO model_evaluation_job_runs(
          job_id, attempt, worker_id, status, resource_snapshot
        ) VALUES (?, ?, ?, 'running', CAST(? AS JSONB))
        ON CONFLICT(job_id, attempt) DO UPDATE SET
          worker_id = excluded.worker_id, status = 'running',
          resource_snapshot = excluded.resource_snapshot,
          started_at = CURRENT_TIMESTAMP, completed_at = NULL,
          result_path = NULL, error = NULL
        """,
        (result["job_id"], result["attempt"], worker_id, snapshot),
    )
    return result


def requeue_stale_jobs(conn: Any, *, stale_minutes: int = 180) -> int:
    cursor = conn.execute(
        """
        UPDATE model_evaluation_jobs
        SET status = CASE WHEN attempt < max_attempts THEN 'queued' ELSE 'failed' END,
            available_at = CURRENT_TIMESTAMP,
            worker_id = NULL,
            locked_at = NULL,
            error = COALESCE(error, 'worker lease expired'),
            updated_at = CURRENT_TIMESTAMP
        WHERE status = 'running'
          AND locked_at < CURRENT_TIMESTAMP - (? * INTERVAL '1 minute')
        RETURNING job_id
        """,
        (max(1, int(stale_minutes)),),
    )
    return len(cursor.fetchall())


def retry_pending_jobs(
    conn: Any,
    *,
    include_failed: bool = False,
    include_running: bool = False,
) -> int:
    statuses = ["queued"]
    if include_failed:
        statuses.append("failed")
    if include_running:
        statuses.append("running")
    status_sql = "(" + ",".join(f"'{value}'" for value in statuses) + ")"
    reset = "attempt = 0," if include_failed else ""
    attempt_filter = "" if include_failed else "AND attempt < max_attempts"
    rows = conn.execute(
        f"""
        UPDATE model_evaluation_jobs
        SET status = 'queued', {reset}
            available_at = CURRENT_TIMESTAMP, completed_at = NULL,
            worker_id = NULL, locked_at = NULL, error = NULL,
            updated_at = CURRENT_TIMESTAMP
        WHERE status IN {status_sql} {attempt_filter}
        RETURNING job_id
        """
    ).fetchall()
    return len(rows)


def recover_worker_job(conn: Any, *, worker_id: str) -> int:
    rows = conn.execute(
        """
        UPDATE model_evaluation_jobs
        SET status = 'queued', available_at = CURRENT_TIMESTAMP,
            worker_id = NULL, locked_at = NULL, updated_at = CURRENT_TIMESTAMP,
            error = COALESCE(error, 'worker restarted before completion update')
        WHERE status = 'running' AND worker_id = ?
        RETURNING job_id
        """,
        (worker_id,),
    ).fetchall()
    return len(rows)


def _number(
    params: dict[str, Any],
    key: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    value = float(params.get(key, default))
    if not minimum <= value <= maximum:
        raise ValueError(f"{key} must be in [{minimum}, {maximum}]")
    return value


def _integer(
    params: dict[str, Any],
    key: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = int(params.get(key, default))
    if not minimum <= value <= maximum:
        raise ValueError(f"{key} must be in [{minimum}, {maximum}]")
    return value


def _date(params: dict[str, Any], key: str) -> str:
    value = str(params[key])
    datetime.strptime(value, "%Y-%m-%d")
    return value


def build_command(
    job: dict[str, Any],
    *,
    app_root: Path,
    python: Path,
    db: str,
) -> tuple[list[str], Path]:
    job_id = int(job["job_id"])
    task_type = str(job["task_type"])
    params = dict(job.get("parameters") or {})
    output_dir = app_root / "data" / "models" / "evaluation_queue"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"job-{job_id:08d}.json"
    if task_type == "standardized_365d":
        return [str(app_root / "scripts" / "run_standardized_365d_evaluations.sh")], (
            app_root / "data" / "models" / "standardized_365d_v2" / "manifest.json"
        )
    if task_type == "market_curvature":
        cache = app_root / "data" / "models" / "stagewise_blend_market_shadow.races.joblib"
        clip = _number(params, "disagreement_clip", 4.0, 0.1, 12.0)
        return [
            str(python),
            str(app_root / "scripts" / "analyze_market_curvature.py"),
            str(cache),
            "--evaluation-date",
            _date(params, "evaluation_date"),
            "--disagreement-clip",
            str(clip),
            "--output",
            str(output),
        ], output
    if task_type == "listwise_feature_search":
        n_features = _integer(params, "n_features", 4096, 1024, 32768)
        epochs = _integer(params, "epochs", 2, 1, 6)
        batch_races = _integer(params, "batch_races", 1000, 250, 5000)
        learning_rate = _number(params, "learning_rate", 0.02, 0.001, 0.2)
        targets = str(params.get("targets", "winner,top3_pl"))
        if targets not in {"winner", "top3_pl", "winner,top3_pl"}:
            raise ValueError("unsupported targets")
        alphas = str(params.get("alphas", "0.00001,0.0001"))
        if alphas not in {
            "0.000001,0.00001",
            "0.00001,0.0001",
            "0.0001,0.001",
        }:
            raise ValueError("unsupported alphas")
        cache_root = Path("/tmp/boatrace-evaluation") / f"job-{job_id:08d}"
        search_cache = cache_root / "search"
        selected_cache = (
            app_root / "data" / "models" / "evaluation_cache"
            / f"job-{job_id:08d}"
        )
        return [
            str(python), "-m", "boatrace_ai.listwise.feature_search",
            "--db", db,
            "--output", str(output),
            "--cache-dir", str(search_cache),
            "--cache-write-mode", "never",
            "--selected-cache-dir", str(selected_cache),
            "--n-features", str(n_features),
            "--batch-races", str(batch_races),
            "--epochs", str(epochs),
            "--learning-rate", str(learning_rate),
            "--targets", targets,
            "--alphas", alphas,
            "--daily-budget-yen", "10000",
            "--ev-threshold", str(_number(params, "ev_threshold", 1.2, 1.0, 3.0)),
        ], output
    if task_type == "venue_conditional_order":
        training_through = _date(params, "training_through")
        evaluation_from = _date(params, "evaluation_from")
        evaluation_through = _date(params, "evaluation_through")
        if not training_through < evaluation_from <= evaluation_through:
            raise ValueError("venue evaluation dates must be adjacent chronological ranges")
        baseline_model = (
            app_root / "data" / "models" / "standardized_365d_v2"
            / "listwise_newton.joblib"
        )
        legacy_evaluation = app_root / "data" / "models" / "conditional_order_365d.json"
        cache_dir = Path("/tmp/boatrace-evaluation") / f"job-{job_id:08d}" / "venue"
        return [
            str(python), "-m", "boatrace_ai.listwise.venue_conditional_order",
            "--db", db,
            "--baseline-model", str(baseline_model),
            "--legacy-evaluation", str(legacy_evaluation),
            "--cache-dir", str(cache_dir),
            "--training-through", training_through,
            "--evaluation-from", evaluation_from,
            "--evaluation-through", evaluation_through,
            "--global-regularization", str(
                _number(params, "global_regularization", 0.0001, 0.000001, 1.0)
            ),
            "--venue-regularizations", "0.0001", "0.001", "0.01", "0.1",
            "--max-iterations", str(
                _integer(params, "max_iterations", 100, 20, 300)
            ),
            "--model-output", str(output.with_suffix(".joblib")),
            "--output", str(output),
        ], output
    if task_type == "evaluation_aggregate":
        return [
            str(python), "-m", "boatrace_ai.maintenance_tasks", "aggregate-evaluations",
            "--db", db, "--output", str(output),
        ], output
    if task_type == "gdrive_raw_archive":
        return [
            str(python), "-m", "boatrace_ai.maintenance_tasks", "backup-raw",
            "--app-root", str(app_root), "--output", str(output),
        ], output
    if task_type == "listwise_newton_refine":
        search_result = app_root / str(params["search_result"])
        if app_root not in search_result.resolve().parents:
            raise ValueError("search_result must be inside app root")
        model_output = output.with_suffix(".joblib")
        cache = Path(str(params.get("cache_dir") or "/tmp/boatrace-evaluation/newton"))
        return [
            str(python), "-m", "boatrace_ai.listwise.newton_refine",
            "--db", db,
            "--search-result", str(search_result),
            "--output", str(output),
            "--model-output", str(model_output),
            "--cache-dir", str(cache),
            "--cache-write-mode", "never",
            "--max-newton-iterations", str(_integer(params, "max_newton_iterations", 10, 3, 30)),
            "--max-cg-iterations", str(_integer(params, "max_cg_iterations", 50, 10, 200)),
            "--gradient-tolerance", str(_number(params, "gradient_tolerance", 1e-4, 1e-7, 1e-2)),
            "--cg-tolerance", str(_number(params, "cg_tolerance", 1e-3, 1e-6, 1e-1)),
            "--daily-budget-yen", "10000",
            "--ev-threshold", str(_number(params, "ev_threshold", 1.2, 1.0, 3.0)),
        ], output
    raise ValueError(f"unsupported task_type: {task_type}")


METRIC_KEYS = (
    "evaluated_races", "evaluation_races", "evaluation_days", "entry_log_loss",
    "trifecta_log_loss", "calibrated_trifecta_log_loss", "winner_top1_accuracy",
    "trifecta_top5_hit_rate", "roi", "profit_yen", "stake_yen",
    "promotion_eligible", "incremental_confidence_pass", "converged",
    "gradient_norm", "elapsed_seconds",
)


def summarize_result(payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    def visit(value: Any, depth: int = 0) -> None:
        if depth > 5 or not isinstance(value, dict):
            return
        for key in METRIC_KEYS:
            if key in value and key not in summary and not isinstance(value[key], (dict, list)):
                summary[key] = value[key]
        for key in (
            "metrics", "holdout", "holdout_after_newton", "bankroll",
            "conditional_order", "momentum_newton_residual",
        ):
            if key in value:
                visit(value[key], depth + 1)

    visit(payload)
    summary["model"] = payload.get("model")
    summary["status"] = payload.get("status")
    return {key: value for key, value in summary.items() if value is not None}


def result_decision(task_type: str, summary: dict[str, Any]) -> str:
    if summary.get("promotion_eligible") is True:
        return "promotion_candidate"
    if summary.get("incremental_confidence_pass") is True:
        return "confirm_on_new_holdout"
    roi = summary.get("roi")
    profit = summary.get("profit_yen")
    if roi is not None and float(roi) >= 1.0 and float(profit or 0) > 0:
        return "bankroll_gate_pass"
    if task_type == "listwise_feature_search":
        return "refine_selected_candidate"
    if task_type == "evaluation_aggregate":
        return "aggregation_complete"
    if task_type == "gdrive_raw_archive":
        return "backup_complete"
    return "reject_or_research_only"


def _load_result(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("evaluation result must be a JSON object")
    return payload, summarize_result(payload)


def complete_job(
    conn: Any,
    *,
    job: dict[str, Any],
    result_path: Path,
    summary: dict[str, Any],
    decision: str,
) -> None:
    conn.execute(
        """
        UPDATE model_evaluation_jobs
        SET status = 'completed', completed_at = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP, result_path = ?,
            result_summary = CAST(? AS JSONB), decision = ?, worker_id = NULL,
            locked_at = NULL, error = NULL
        WHERE job_id = ?
        """,
        (str(result_path), _json(summary), decision, int(job["job_id"])),
    )
    conn.execute(
        """
        UPDATE model_evaluation_job_runs
        SET status = 'completed', completed_at = CURRENT_TIMESTAMP,
            result_path = ?, error = NULL
        WHERE job_id = ? AND attempt = ?
        """,
        (str(result_path), int(job["job_id"]), int(job["attempt"])),
    )
    if str(job.get("category") or "evaluation") != "evaluation":
        return
    conn.execute(
        """
        INSERT INTO model_improvement_candidates(
          job_id, model_key, task_type, decision, metrics, parameters, result_path
        ) VALUES (?, ?, ?, ?, CAST(? AS JSONB), CAST(? AS JSONB), ?)
        ON CONFLICT(job_id) DO UPDATE SET
          decision = excluded.decision, metrics = excluded.metrics,
          result_path = excluded.result_path
        """,
        (
            int(job["job_id"]), job["model_key"], job["task_type"], decision,
            _json(summary), _json(job.get("parameters") or {}), str(result_path),
        ),
    )


def fail_job(conn: Any, *, job: dict[str, Any], error: str) -> None:
    terminal = int(job["attempt"]) >= int(job["max_attempts"])
    conn.execute(
        """
        UPDATE model_evaluation_jobs
        SET status = ?, available_at = CURRENT_TIMESTAMP + INTERVAL '15 minutes',
            worker_id = NULL, locked_at = NULL, updated_at = CURRENT_TIMESTAMP,
            completed_at = CASE WHEN ? = 'failed' THEN CURRENT_TIMESTAMP ELSE completed_at END,
            error = ?
        WHERE job_id = ?
        """,
        ("failed" if terminal else "queued", "failed" if terminal else "queued", error[-8000:], int(job["job_id"])),
    )
    conn.execute(
        """
        UPDATE model_evaluation_job_runs
        SET status = 'failed', completed_at = CURRENT_TIMESTAMP, error = ?
        WHERE job_id = ? AND attempt = ?
        """,
        (error[-8000:], int(job["job_id"]), int(job["attempt"])),
    )


def _limit_resources(vm_limit_gib: int, nice: int) -> None:
    if vm_limit_gib > 0:
        limit = int(vm_limit_gib) * 1024**3
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    if nice:
        os.nice(int(nice))


def heartbeat_job(db: str, *, job_id: int, worker_id: str) -> None:
    with connection(db) as conn:
        conn.execute(
            """
            UPDATE model_evaluation_jobs
            SET locked_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
            WHERE job_id = ? AND status = 'running' AND worker_id = ?
            """,
            (int(job_id), worker_id),
        )


def _heartbeat_loop(
    stop: threading.Event,
    *,
    db: str,
    job_id: int,
    worker_id: str,
) -> None:
    while not stop.wait(30.0):
        try:
            heartbeat_job(db, job_id=job_id, worker_id=worker_id)
        except Exception as exc:
            print(
                f"evaluation heartbeat error job={job_id}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )


def prepare_standardized_workspace(
    app_root: Path,
    *,
    evaluation_date: str,
) -> None:
    evaluation_dir = app_root / "data" / "models" / "standardized_365d_v2"
    protocol_path = evaluation_dir / "protocol.json"
    if not protocol_path.is_file():
        return
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    requested_as_of = (
        datetime.strptime(evaluation_date, "%Y-%m-%d").date()
        + timedelta(days=1)
    ).isoformat()
    existing_as_of = str(protocol.get("as_of_date_jst") or "")
    if existing_as_of == requested_as_of:
        return
    archive = (
        app_root
        / "data"
        / "models"
        / "evaluation_queue"
        / "standardized_history"
        / (existing_as_of or "unknown")
    )
    archive.mkdir(parents=True, exist_ok=True)
    for source in evaluation_dir.glob("*.json"):
        target = archive / source.name
        target.write_bytes(source.read_bytes())
    protocol_path.unlink()


def execute_job(
    job: dict[str, Any],
    *,
    app_root: Path,
    python: Path,
    db: str,
    vm_limit_gib: int,
    nice: int,
) -> tuple[Path, dict[str, Any], str]:
    if job["task_type"] == "standardized_365d":
        prepare_standardized_workspace(
            app_root,
            evaluation_date=str((job.get("parameters") or {})["evaluation_date"]),
        )
    command, result_path = build_command(
        job, app_root=app_root, python=python, db=db
    )
    log_dir = app_root / "logs" / "evaluation_queue"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"job-{int(job['job_id']):08d}.log"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(app_root / "src")
    if job["task_type"] == "standardized_365d":
        evaluation_date = datetime.strptime(
            str((job.get("parameters") or {})["evaluation_date"]), "%Y-%m-%d"
        ).date()
        env["BOATRACE_EVAL_AS_OF_DATE"] = (
            evaluation_date + timedelta(days=1)
        ).isoformat()
        env["BOATRACE_EVAL_RESUME_COMPLETED"] = "1"
        env["BOATRACE_EVAL_VM_LIMIT_KB"] = "0"
        env["BOATRACE_DB"] = db
    timeout = _integer(job.get("parameters") or {}, "timeout_seconds", 21600, 300, 86400)
    stop_heartbeat = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat_loop,
        kwargs={
            "stop": stop_heartbeat,
            "db": db,
            "job_id": int(job["job_id"]),
            "worker_id": str(job["worker_id"]),
        },
        daemon=True,
    )
    heartbeat.start()
    try:
        with log_path.open("ab") as log:
            completed = subprocess.run(
                command,
                cwd=app_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
                preexec_fn=lambda: _limit_resources(vm_limit_gib, nice),
            )
    finally:
        stop_heartbeat.set()
        heartbeat.join(timeout=2.0)
    if completed.returncode != 0:
        tail = log_path.read_text(errors="replace").splitlines()[-20:]
        raise RuntimeError(
            f"exit={completed.returncode}; " + " | ".join(tail)
        )
    payload, summary = _load_result(result_path)
    decision = result_decision(str(job["task_type"]), summary)
    return result_path, summary, decision


def enqueue_refinement(
    conn: Any,
    job: dict[str, Any],
    decision: str,
    *,
    app_root: Path,
) -> int | None:
    if job["task_type"] != "listwise_feature_search" or decision != "refine_selected_candidate":
        return None
    relative = f"data/models/evaluation_queue/job-{int(job['job_id']):08d}.json"
    parent_cache = str(
        app_root / "data" / "models" / "evaluation_cache"
        / f"job-{int(job['job_id']):08d}"
    )
    params = {
        "search_result": relative,
        "cache_dir": parent_cache,
        "max_newton_iterations": 10,
        "max_cg_iterations": 75,
        "gradient_tolerance": 0.0001,
        "cg_tolerance": 0.001,
        "ev_threshold": float((job.get("parameters") or {}).get("ev_threshold", 1.2)),
        "timeout_seconds": 21600,
    }
    return enqueue_job(
        conn,
        task_type="listwise_newton_refine",
        model_key=f"{job['model_key']}:newton",
        parameters=params,
        priority=int(job["priority"]) + 1,
        parent_job_id=int(job["job_id"]),
    )


def seed_periodic_jobs(conn: Any, *, now: datetime | None = None) -> list[int]:
    now = now or datetime.now(timezone.utc)
    inserted: list[int] = []

    def active(task_type: str) -> bool:
        row = conn.execute(
            "SELECT COUNT(*) AS count FROM model_evaluation_jobs WHERE task_type = ? AND status IN ('queued','running')",
            (task_type,),
        ).fetchone()
        return bool(row and int(row["count"]))

    schedules = (
        ("gdrive_raw_archive", "raw-data", 600, 90, 1800),
        ("evaluation_aggregate", "all-models", 900, 30, 900),
    )
    epoch = int(now.timestamp())
    for task_type, model_key, interval, priority, timeout in schedules:
        if active(task_type):
            continue
        bucket = epoch - epoch % interval
        job_id = enqueue_job(
            conn,
            task_type=task_type,
            model_key=model_key,
            parameters={
                "schedule_bucket": datetime.fromtimestamp(bucket, timezone.utc).isoformat(),
                "timeout_seconds": timeout,
            },
            priority=priority,
            max_attempts=3,
        )
        if job_id is not None:
            inserted.append(job_id)
    return inserted


DEFAULT_WORK_TICKETS = (
    ("OPS-QUEUE-001", "DBジョブ基盤と資源監視", "運用基盤", "評価・集計・バックアップをDBキューから実行する", "4ランナーが資源条件付きで取得し完了履歴をDBへ残す", 100, "in_progress", 70),
    ("OPS-BACKUP-001", "GDriveバックアップのキュー移行", "バックアップ", "生データ転送を定期DBジョブとして管理する", "排他付き転送が完了し元データ削除と結果記録を確認する", 90, "in_progress", 65),
    ("MODEL-OPT-001", "モデル再設計と収益ゲート収束", "モデル", "特徴量・教師・構造を同一評価軸で反復検証する", "未使用holdoutでROI・損益・確率指標の昇格基準を満たす", 100, "in_progress", 55),
    ("UI-MODEL-001", "モデル性能ページの評価表現統一", "WebUI", "評価母集団と指標表現を統一する", "全モデルが同じ列定義と評価群で比較できる", 70, "queued", 20),
    ("UI-PRED-001", "タイムラインとGantt的中判定の統一", "WebUI", "主系予測と購入的中を別指標として表示する", "同一レースで各表示の意味と判定が一致する", 80, "queued", 25),
)


def seed_work_tickets(conn: Any) -> int:
    changed = 0
    for key, title, area, description, acceptance, priority, status, progress in DEFAULT_WORK_TICKETS:
        row = conn.execute(
            """
            INSERT INTO work_tickets(
              ticket_key, title, area, description, acceptance_criteria,
              priority, status, progress, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'codex')
            ON CONFLICT(ticket_key) DO NOTHING
            RETURNING ticket_key
            """,
            (key, title, area, description, acceptance, priority, status, progress),
        ).fetchone()
        changed += int(row is not None)
    return changed


def update_work_ticket(
    conn: Any, *, ticket_key: str, status: str, progress: int, note: str = ""
) -> None:
    if status not in {"queued", "in_progress", "blocked", "completed", "cancelled"}:
        raise ValueError("invalid ticket status")
    progress = max(0, min(100, int(progress)))
    row = conn.execute(
        """
        UPDATE work_tickets
        SET status = ?, progress = ?, updated_at = CURRENT_TIMESTAMP,
            completed_at = CASE WHEN ? = 'completed' THEN CURRENT_TIMESTAMP ELSE NULL END
        WHERE ticket_key = ? RETURNING ticket_key
        """,
        (status, progress, status, ticket_key),
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown ticket: {ticket_key}")
    conn.execute(
        "INSERT INTO work_ticket_events(ticket_key, status, progress, note) VALUES (?, ?, ?, ?)",
        (ticket_key, status, progress, note),
    )


def seed_default_jobs(conn: Any, *, evaluation_date: str) -> list[int]:
    inserted: list[int] = []

    def add(**kwargs: Any) -> None:
        job_id = enqueue_job(conn, **kwargs)
        if job_id is not None:
            inserted.append(job_id)

    add(
        task_type="standardized_365d",
        model_key="all_registered_models",
        parameters={"evaluation_date": evaluation_date, "timeout_seconds": 21600},
        priority=100,
        max_attempts=2,
    )
    for clip in (0.5, 1.0, 2.0, 3.0, 4.0, 6.0):
        add(
            task_type="market_curvature",
            model_key="stagewise_blend_market_curvature",
            parameters={
                "evaluation_date": evaluation_date,
                "disagreement_clip": clip,
                "timeout_seconds": 1800,
            },
            priority=60,
            max_attempts=2,
        )
    feature_variants = (
        (4096, "winner,top3_pl", "0.00001,0.0001", 0.02),
        (8192, "winner,top3_pl", "0.00001,0.0001", 0.02),
        (4096, "top3_pl", "0.000001,0.00001", 0.01),
        (8192, "winner", "0.0001,0.001", 0.03),
    )
    for n_features, targets, alphas, learning_rate in feature_variants:
        add(
            task_type="listwise_feature_search",
            model_key=f"listwise_{n_features}_{targets}_{alphas}",
            parameters={
                "evaluation_date": evaluation_date,
                "n_features": n_features,
                "targets": targets,
                "alphas": alphas,
                "learning_rate": learning_rate,
                "epochs": 2,
                "batch_races": 1000,
                "ev_threshold": 1.2,
                "timeout_seconds": 21600,
            },
            priority=40,
            max_attempts=2,
        )
    return inserted


def run_worker(args: argparse.Namespace) -> int:
    worker_id = args.worker_id or f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
    app_root = Path(args.app_root).resolve()
    python = Path(args.python)
    if not python.is_absolute():
        python = (app_root / python).absolute()
    last_seed = 0.0
    last_schedule = 0.0
    is_leader = args.worker_id is None or str(args.worker_id).endswith("-00")
    with connection(args.db) as conn:
        ensure_schema(conn)
        recover_worker_job(conn, worker_id=worker_id)
        if is_leader:
            seed_work_tickets(conn)
    while True:
        try:
            with connection(args.db) as conn:
                now = time.monotonic()
                if is_leader:
                    requeue_stale_jobs(conn, stale_minutes=args.stale_minutes)
                if (
                    is_leader
                    and args.seed_defaults
                    and now - last_seed >= args.seed_interval
                ):
                    evaluation_date = (
                        datetime.now(JST).date() - timedelta(days=1)
                    ).isoformat()
                    seed_default_jobs(conn, evaluation_date=evaluation_date)
                    last_seed = now
                if is_leader and args.schedule_periodic and now - last_schedule >= 60.0:
                    seed_periodic_jobs(conn)
                    last_schedule = now
                resources = system_resources()
                job = claim_job(conn, worker_id=worker_id, resources=resources)
            if job is None:
                if args.once:
                    return 0
                time.sleep(args.poll_seconds)
                continue
            try:
                result_path, summary, decision = execute_job(
                    job,
                    app_root=app_root,
                    python=python,
                    db=args.db,
                    vm_limit_gib=args.vm_limit_gib,
                    nice=args.nice,
                )
                with connection(args.db) as conn:
                    complete_job(
                        conn,
                        job=job,
                        result_path=result_path,
                        summary=summary,
                        decision=decision,
                    )
                    enqueue_refinement(
                        conn,
                        job,
                        decision,
                        app_root=app_root,
                    )
            except Exception as exc:
                with connection(args.db) as conn:
                    fail_job(conn, job=job, error=f"{type(exc).__name__}: {exc}")
            if args.once:
                return 0
        except KeyboardInterrupt:
            return 130
        except Exception as exc:
            print(f"evaluation worker error: {type(exc).__name__}: {exc}", flush=True)
            if args.once:
                raise
            time.sleep(args.poll_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PostgreSQL-backed model evaluation queue")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "seed", "retry", "status", "run"):
        command = sub.add_parser(name)
        command.add_argument("--db", default=DEFAULT_DSN)
        if name == "seed":
            command.add_argument("--evaluation-date", required=True)
        if name == "retry":
            command.add_argument("--include-failed", action="store_true")
            command.add_argument("--include-running", action="store_true")
        if name == "run":
            command.add_argument("--app-root", default="/workspace/boat")
            command.add_argument("--python", default="/workspace/boat/.venv/bin/python")
            command.add_argument("--worker-id")
            command.add_argument("--poll-seconds", type=float, default=5.0)
            command.add_argument("--seed-interval", type=float, default=3600.0)
            command.add_argument("--stale-minutes", type=int, default=180)
            command.add_argument("--vm-limit-gib", type=int, default=20)
            command.add_argument("--nice", type=int, default=10)
            command.add_argument("--seed-defaults", action="store_true")
            command.add_argument("--schedule-periodic", action="store_true")
            command.add_argument("--once", action="store_true")
    return parser


def status_rows(conn: Any) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT job_id, task_type, model_key, status, priority, attempt,
               max_attempts, worker_id, started_at, completed_at,
               result_path, decision, error, category,
               min_free_memory_mb, min_free_disk_mb, min_idle_cpu_percent, max_parallel,
               last_resource_snapshot
        FROM model_evaluation_jobs
        ORDER BY CASE status WHEN 'running' THEN 0 WHEN 'queued' THEN 1 ELSE 2 END,
                 priority DESC, job_id DESC
        LIMIT 200
        """
    ).fetchall()
    return [{key: row[key] for key in row.keys()} for row in rows]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return run_worker(args)
    with connection(args.db) as conn:
        ensure_schema(conn)
        if args.command == "seed":
            print(_json({"inserted": seed_default_jobs(conn, evaluation_date=args.evaluation_date)}))
        elif args.command == "retry":
            print(
                _json(
                    {
                        "requeued": retry_pending_jobs(
                            conn,
                            include_failed=args.include_failed,
                            include_running=args.include_running,
                        )
                    }
                )
            )
        elif args.command == "status":
            print(json.dumps(status_rows(conn), ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
