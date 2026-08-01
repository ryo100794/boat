from __future__ import annotations

import argparse
from dataclasses import dataclass
import errno
import hashlib
import json
import math
import os
import resource
import re
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

from .archive_residual_summary import apply_archive_residual_summary
from .db import connection
from .evaluation_probability_summary import canonicalize_probability_metrics
from .evaluation_result_summary import canonicalize_primary_bankroll
from .feature_schema import (
    FEATURE_SCHEMA_VERSION,
    SUPPORTED_LISTWISE_FEATURE_SCHEMA_VERSIONS,
)
from .listwise.tail_portfolio_diagnostics import diagnose_tail_portfolio


JST = ZoneInfo("Asia/Tokyo")
DEFAULT_DSN = "host=127.0.0.1 port=5432 dbname=boatrace user=boatrace_app"
STANDARDIZED_SELECTED_CACHE_DIR = Path(
    "/workspace/boat/data/models/standardized_365d_v2/selected_cache"
)
SCHEMA_LOCK_ID = 71234001
CLAIM_LOCK_ID = 71234002
EVALUATION_MEMORY_SAFETY_MB = 6144
CHECKPOINT_RECOVERABLE_TASKS = frozenset({
    "listwise_feature_search",
    "combined_feature_search",
})


@dataclass(frozen=True)
class ResourceSnapshot:
    available_memory_mb: int
    available_disk_mb: int
    idle_cpu_percent: float
    cpu_count: int
    load_1m: float
    memory_limit_mb: int | None = None
    memory_usage_mb: int | None = None
    reclaimable_file_mb: int | None = None


TASK_PROFILES: dict[str, dict[str, Any]] = {
    "standardized_365d": {"category": "evaluation", "memory_mb": 14336, "idle_cpu": 15.0, "max_parallel": 1, "disk_mb": 8192},
    "historical_coverage_safe": {"category": "evaluation", "memory_mb": 4096, "idle_cpu": 15.0, "max_parallel": 1, "disk_mb": 2048},
    "historical_research_logit": {"category": "evaluation", "memory_mb": 14336, "idle_cpu": 15.0, "max_parallel": 1, "disk_mb": 4096},
    "genetic_island_search": {"category": "evaluation", "memory_mb": 3072, "idle_cpu": 5.0, "max_parallel": 4, "disk_mb": 2048},
    "market_curvature": {"category": "evaluation", "memory_mb": 2048, "idle_cpu": 5.0, "max_parallel": 4, "disk_mb": 1024},
    "market_residual_walk_forward": {"category": "evaluation", "memory_mb": 2048, "idle_cpu": 5.0, "max_parallel": 2, "disk_mb": 256},
    "four_head_learned_value": {"category": "evaluation", "memory_mb": 4096, "idle_cpu": 5.0, "max_parallel": 2, "disk_mb": 512},
    "listwise_feature_search": {"category": "evaluation", "memory_mb": 14336, "idle_cpu": 15.0, "max_parallel": 1, "disk_mb": 4096},
    "combined_feature_search": {"category": "evaluation", "memory_mb": 14336, "idle_cpu": 15.0, "max_parallel": 1, "disk_mb": 4096},
    "listwise_newton_refine": {"category": "evaluation", "memory_mb": 8192, "idle_cpu": 15.0, "max_parallel": 2, "disk_mb": 4096},
    "listwise_cutoff_refit": {"category": "evaluation", "memory_mb": 8192, "idle_cpu": 15.0, "max_parallel": 1, "disk_mb": 4096},
    "calibrated_mlp_recency_search": {"category": "evaluation", "memory_mb": 16384, "idle_cpu": 15.0, "max_parallel": 1, "disk_mb": 4096},
    "lightgbm_recency_search": {"category": "evaluation", "memory_mb": 14336, "idle_cpu": 15.0, "max_parallel": 1, "disk_mb": 1024},
    "bankroll_policy_search": {"category": "evaluation", "memory_mb": 9216, "idle_cpu": 15.0, "max_parallel": 1, "disk_mb": 1024},
    "bankroll_policy_nested_annual": {"category": "evaluation", "memory_mb": 21504, "idle_cpu": 15.0, "max_parallel": 1, "disk_mb": 4096},
    "conditional_payout_tail": {"category": "evaluation", "memory_mb": 12288, "idle_cpu": 15.0, "max_parallel": 1, "disk_mb": 2048},
    "venue_conditional_order": {"category": "evaluation", "memory_mb": 12288, "idle_cpu": 15.0, "max_parallel": 1, "disk_mb": 2048},
    "evaluation_aggregate": {"category": "aggregation", "memory_mb": 512, "idle_cpu": 3.0, "max_parallel": 1, "disk_mb": 256},
    "gdrive_raw_archive": {"category": "backup", "memory_mb": 512, "idle_cpu": 3.0, "max_parallel": 1, "disk_mb": 256},
    "gdrive_model_cache_archive": {"category": "backup", "memory_mb": 512, "idle_cpu": 3.0, "max_parallel": 1, "disk_mb": 2048},
    "repository_hygiene": {"category": "maintenance", "memory_mb": 256, "idle_cpu": 3.0, "max_parallel": 1, "disk_mb": 256},
    "repository_sync": {"category": "maintenance", "memory_mb": 256, "idle_cpu": 3.0, "max_parallel": 1, "disk_mb": 256},
    "series_feature_cache": {"category": "maintenance", "memory_mb": 512, "idle_cpu": 3.0, "max_parallel": 1, "disk_mb": 256},
    "racer_stats_backfill": {"category": "maintenance", "memory_mb": 512, "idle_cpu": 3.0, "max_parallel": 1, "disk_mb": 256},
    "archive_closing_backfill": {"category": "collection", "memory_mb": 512, "idle_cpu": 3.0, "max_parallel": 1, "disk_mb": 256},
    "archive_market_oracle": {"category": "evaluation", "memory_mb": 4096, "idle_cpu": 15.0, "max_parallel": 1, "disk_mb": 1024},
    "persist_standard_selected_cache": {"category": "maintenance", "memory_mb": 512, "idle_cpu": 3.0, "max_parallel": 1, "disk_mb": 1024},
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
  repository_full_name TEXT NOT NULL DEFAULT '',
  github_issue_number INTEGER,
  github_issue_url TEXT NOT NULL DEFAULT '',
  github_issue_updated_at TIMESTAMPTZ,
  last_synced_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  completed_at TIMESTAMPTZ
);
ALTER TABLE work_tickets ADD COLUMN IF NOT EXISTS repository_full_name TEXT NOT NULL DEFAULT '';
ALTER TABLE work_tickets ADD COLUMN IF NOT EXISTS github_issue_number INTEGER;
ALTER TABLE work_tickets ADD COLUMN IF NOT EXISTS github_issue_url TEXT NOT NULL DEFAULT '';
ALTER TABLE work_tickets ADD COLUMN IF NOT EXISTS github_issue_updated_at TIMESTAMPTZ;
ALTER TABLE work_tickets ADD COLUMN IF NOT EXISTS last_synced_at TIMESTAMPTZ;
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
CREATE UNIQUE INDEX IF NOT EXISTS idx_work_tickets_github_issue
  ON work_tickets(repository_full_name, github_issue_number)
  WHERE github_issue_number IS NOT NULL AND repository_full_name <> '';
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

    conn.execute(
        """
        UPDATE model_evaluation_jobs
        SET min_free_memory_mb = ?
        WHERE status = 'queued'
          AND task_type IN (?, ?)
          AND min_free_memory_mb = ?
        """,
        (
            14336,
            "standardized_365d",
            "listwise_feature_search",
            16384,
        ),
    )
    conn.execute(
        """
        UPDATE model_evaluation_jobs
        SET min_free_memory_mb = ?
        WHERE status = 'queued'
          AND task_type = ?
          AND min_free_memory_mb = ?
        """,
        (14336, "lightgbm_recency_search", 65536),
    )
    conn.execute(
        """
        UPDATE model_evaluation_jobs
        SET min_free_memory_mb = ?
        WHERE status = 'queued'
          AND task_type = ?
          AND min_free_memory_mb = ?
        """,
        (9216, "bankroll_policy_search", 14336),
    )
    conn.execute(
        """
        UPDATE model_evaluation_jobs
        SET min_free_memory_mb = ?
        WHERE status = 'queued'
          AND task_type = ?
          AND min_free_memory_mb = ?
        """,
        (21504, "bankroll_policy_nested_annual", 24576),
    )
    conn.execute(
        """
        UPDATE model_evaluation_jobs
        SET min_free_memory_mb = ?
        WHERE status = 'queued'
          AND task_type = ?
          AND min_free_memory_mb = ?
        """,
        (21504, "bankroll_policy_nested_annual", 23552),
    )
    conn.execute(
        """
        UPDATE model_evaluation_jobs
        SET parameters = jsonb_set(
            parameters,
            '{n_jobs}',
            to_jsonb(CAST(? AS INTEGER)),
            true
        )
        WHERE status = 'queued'
          AND task_type = ?
          AND parameters->>'n_jobs' = ?
        """,
        (4, "lightgbm_recency_search", "16"),
    )


def _available_cpu_ids() -> set[int] | None:
    try:
        return set(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return None


def _read_cpu_times(
    cpu_ids: set[int] | None = None,
    *,
    stat_path: Path = Path("/proc/stat"),
) -> tuple[int, int]:
    lines = stat_path.read_text(encoding="utf-8").splitlines()
    selected: list[list[int]] = []
    for line in lines:
        fields = line.split()
        label = fields[0] if fields else ""
        if cpu_ids is None:
            if label != "cpu":
                continue
        elif not label.startswith("cpu") or not label[3:].isdigit():
            continue
        elif int(label[3:]) not in cpu_ids:
            continue
        selected.append([int(value) for value in fields[1:]])
        if cpu_ids is None:
            break
    if not selected:
        raise RuntimeError("no CPU counters found for the process affinity")
    idle = sum(values[3] + (values[4] if len(values) > 4 else 0) for values in selected)
    total = sum(sum(values) for values in selected)
    return idle, total


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


def _cgroup_reclaimable_file_bytes(
    root: Path = Path("/sys/fs/cgroup"),
) -> int:
    stat_paths = (
        root / "memory.stat",
        root / "memory" / "memory.stat",
    )
    for stat_path in stat_paths:
        try:
            values = {
                key: int(value)
                for key, value, *_ in (
                    line.split()
                    for line in stat_path.read_text(encoding="utf-8").splitlines()
                )
            }
        except (OSError, ValueError):
            continue
        # Use the same reclaimable file-cache component as the standard
        # Linux/Kubernetes cgroup working-set calculation. Prefer v1 totals.
        return max(
            0,
            values.get("total_inactive_file", values.get("inactive_file", 0)),
        )
    return 0


def _cgroup_quota_available_mb(
    memory_limit_mb: int,
    memory_usage_mb: int,
    reclaimable_file_mb: int,
    *,
    safety_reserve_mb: int = 4096,
) -> int:
    reclaimable_mb = min(max(0, reclaimable_file_mb), max(0, memory_usage_mb))
    return max(
        0,
        memory_limit_mb - memory_usage_mb + reclaimable_mb - safety_reserve_mb,
    )


def system_resources(
    sample_seconds: float = 0.15,
    *,
    disk_path: str | Path = "/tmp",
) -> ResourceSnapshot:
    meminfo = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        meminfo[key] = int(value.strip().split()[0])
    host_available_mb = int(meminfo.get("MemAvailable", 0) // 1024)
    memory_limit_mb = None
    memory_usage_mb = None
    reclaimable_file_mb = None
    cgroup = _cgroup_memory()
    if cgroup is not None:
        memory_limit_mb = int(cgroup[0] // 1024**2)
        memory_usage_mb = int(cgroup[1] // 1024**2)
        reclaimable_file_mb = min(
            memory_usage_mb,
            int(_cgroup_reclaimable_file_bytes() // 1024**2),
        )
        quota_available_mb = _cgroup_quota_available_mb(
            memory_limit_mb,
            memory_usage_mb,
            reclaimable_file_mb,
        )
        host_available_mb = min(host_available_mb, quota_available_mb)
    cpu_ids = _available_cpu_ids()
    idle_before, total_before = _read_cpu_times(cpu_ids)
    time.sleep(max(0.0, sample_seconds))
    idle_after, total_after = _read_cpu_times(cpu_ids)
    total_delta = max(1, total_after - total_before)
    idle_percent = max(0.0, min(100.0, (idle_after - idle_before) * 100.0 / total_delta))
    return ResourceSnapshot(
        available_memory_mb=host_available_mb,
        available_disk_mb=int(shutil.disk_usage(disk_path).free // 1024**2),
        idle_cpu_percent=idle_percent,
        cpu_count=len(cpu_ids) if cpu_ids else (os.cpu_count() or 1),
        load_1m=float(os.getloadavg()[0]),
        memory_limit_mb=memory_limit_mb,
        memory_usage_mb=memory_usage_mb,
        reclaimable_file_mb=reclaimable_file_mb,
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
        "reclaimable_file_mb": snapshot.reclaimable_file_mb,
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


def workspace_quota_allows(
    app_root: Path,
    *,
    required_mb: int,
) -> bool:
    """Probe the target filesystem because network-volume statvfs ignores quotas."""
    required_bytes = max(0, int(required_mb)) * 1024**2
    if required_bytes == 0:
        return True
    probe_dir = app_root / "data" / "archive-staging"
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe_path = probe_dir / f".evaluation-quota-{uuid.uuid4().hex}.tmp"
    descriptor: int | None = None
    close_error: OSError | None = None
    try:
        descriptor = os.open(
            probe_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        os.posix_fallocate(descriptor, 0, required_bytes)
        return True
    except OSError as exc:
        if exc.errno in {errno.EDQUOT, errno.ENOSPC}:
            return False
        if exc.errno in {errno.ENOSYS, errno.EOPNOTSUPP}:
            return shutil.disk_usage(probe_dir).free >= required_bytes
        raise
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError as exc:
                if exc.errno not in {errno.EDQUOT, errno.ENOSPC}:
                    close_error = exc
        try:
            probe_path.unlink()
        except FileNotFoundError:
            pass
        if close_error is not None:
            raise close_error


def job_workspace_reservation_mb(job: dict[str, Any], app_root: Path) -> int:
    required_mb = max(0, int(job.get("min_free_disk_mb") or 0))
    if str(job.get("task_type")) != "calibrated_mlp_recency_search":
        return min(required_mb, 256)
    parameters = job.get("parameters") or {}
    if not isinstance(parameters, dict):
        parameters = json.loads(str(parameters))
    drop_feature_groups = _drop_feature_groups(parameters)
    cache_suffix = (
        ""
        if drop_feature_groups == "research_correlates"
        else "__drop_" + drop_feature_groups.replace(",", "_")
    )
    prefix = (
        app_root
        / "data"
        / "models"
        / ("calibrated_shadow_features_16384" + cache_suffix)
    )
    cache_files = (
        Path(f"{prefix}.matrix.npz"),
        Path(f"{prefix}.ranks.npy"),
        Path(f"{prefix}.manifest.json"),
    )
    if all(path.is_file() and path.stat().st_size > 0 for path in cache_files):
        return min(required_mb, 256)
    return min(required_mb, 1024)


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
    semantic_keys = {
        "standardized_365d": ("evaluation_date",),
        "historical_research_logit": ("evaluation_date",),
        "conditional_payout_tail": (
            "training_through",
            "evaluation_from",
            "evaluation_through",
        ),
    }.get(task_type)
    semantic_identity = None
    if semantic_keys and all(name in parameters for name in semantic_keys):
        semantic_identity = {name: parameters[name] for name in semantic_keys}
    elif task_type == "market_residual_walk_forward":
        # Runtime limits control execution, not the evaluated model or period.
        semantic_identity = {
            name: value
            for name, value in parameters.items()
            if name != "timeout_seconds"
        }
    if semantic_identity:
        existing = conn.execute(
            """
            SELECT job_id
            FROM model_evaluation_jobs
            WHERE task_type = ?
              AND model_key = ?
              AND status IN ('queued', 'running', 'completed')
              AND parameters @> CAST(? AS JSONB)
            LIMIT 1
            """,
            (task_type, model_key, _json(semantic_identity)),
        ).fetchone()
        if existing is not None:
            return None
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


def _timeout_retry_parameters(
    parameters: dict[str, Any],
    *,
    task_type: str,
    previous_error: str = "",
) -> dict[str, Any]:
    updated = dict(parameters)
    default = (
        28800
        if task_type in {
            "standardized_365d",
            "historical_coverage_safe",
            "calibrated_mlp_recency_search",
            "lightgbm_recency_search",
        }
        else 21600
    )
    current = updated.get("timeout_seconds", default)
    timeout_match = re.search(
        r"timed out after ([0-9]+(?:\.[0-9]+)?) seconds",
        previous_error,
    )
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        return updated
    observed = (
        float(timeout_match.group(1))
        if timeout_match is not None
        else float(current)
    )
    updated["timeout_seconds"] = min(
        86400,
        max(300, int(current), int(observed) * 2),
    )
    return updated


def _feature_search_checkpoint_path(job: dict[str, Any], *, app_root: Path) -> Path:
    output = (
        app_root / "data" / "models" / "evaluation_queue"
        / f"job-{int(job['job_id']):08d}.json"
    )
    return output.with_name(f".{output.name}.checkpoint.json")


def _valid_feature_search_checkpoint(
    job: dict[str, Any],
    *,
    app_root: Path,
) -> tuple[Path | None, str]:
    """Return a verified persistent checkpoint or a fail-closed reason."""
    if str(job.get("task_type")) not in CHECKPOINT_RECOVERABLE_TASKS:
        return None, "task does not support checkpoint recovery"
    path = _feature_search_checkpoint_path(job, app_root=app_root)
    try:
        persistent_root = (
            app_root / "data" / "models" / "evaluation_queue"
        ).resolve()
        resolved = path.resolve(strict=True)
        if resolved.parent != persistent_root or not resolved.is_file():
            return None, "checkpoint is outside the persistent evaluation directory"
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "checkpoint is missing"
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return None, f"checkpoint is unreadable: {type(exc).__name__}"
    if not isinstance(payload, dict):
        return None, "checkpoint root is not an object"
    signature = payload.get("signature")
    progress = payload.get("progress")
    rows = payload.get("search_results")
    if not isinstance(signature, dict):
        return None, "checkpoint signature is missing"
    if not isinstance(progress, dict) or not isinstance(rows, list) or not rows:
        return None, "checkpoint has no completed candidates"
    from .listwise.feature_search import (
        CACHE_VERSION,
        CHECKPOINT_VERSION,
        _load_checkpoint,
        _validated_source_data_snapshot,
    )

    if signature.get("checkpoint_version") != CHECKPOINT_VERSION:
        return None, "checkpoint version is unsupported"
    if signature.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        return None, "checkpoint feature schema is stale"
    if signature.get("cache_version") != CACHE_VERSION:
        return None, "checkpoint cache version is stale"

    parameters = job.get("parameters") or {}
    if not isinstance(parameters, dict):
        try:
            parameters = json.loads(str(parameters))
        except (TypeError, ValueError, json.JSONDecodeError):
            return None, "job parameters are invalid"
    try:
        expected_signature = {
            "as_of_date": parameters.get("evaluation_date"),
            "n_features": int(parameters.get("n_features", 4096)),
            "batch_races": int(parameters.get("batch_races", 1000)),
            "epochs": int(parameters.get("epochs", 2)),
            "learning_rate": float(parameters.get("learning_rate", 0.02)),
            "targets": [
                value.strip()
                for value in str(
                    parameters.get("targets", "winner,top3_pl")
                ).split(",")
                if value.strip()
            ],
            "alphas": [
                float(value)
                for value in str(
                    parameters.get("alphas", "0.00001,0.0001")
                ).split(",")
                if value.strip()
            ],
        }
        if parameters.get("loss_blend") is not None:
            expected_signature["loss_blend"] = float(parameters["loss_blend"])
    except (TypeError, ValueError):
        return None, "job parameters are invalid"
    for name, expected in expected_signature.items():
        if signature.get(name) != expected:
            return None, f"checkpoint signature mismatch: {name}"

    variants = signature.get("feature_variants")
    if not isinstance(variants, list) or not variants:
        return None, "checkpoint feature variants are missing"
    if str(job["task_type"]) == "combined_feature_search":
        from .listwise.combined_feature_search import (
            COMBINED_FEATURE_VARIANTS,
            RESEARCH_PARTITION_FEATURE_VARIANTS,
        )

        defaults = COMBINED_FEATURE_VARIANTS
        available = dict(
            (*COMBINED_FEATURE_VARIANTS, *RESEARCH_PARTITION_FEATURE_VARIANTS)
        )
    else:
        from .listwise.feature_search import feature_variants

        defaults = tuple(feature_variants())
        available = dict(defaults)
    requested = parameters.get("feature_variants")
    if requested is None:
        expected_variants = defaults
    else:
        requested_names = [
            value.strip()
            for value in str(requested).split(",")
            if value.strip()
        ]
        if any(name not in available for name in requested_names):
            return None, "checkpoint requests unknown feature variants"
        expected_variants = tuple(
            (name, available[name]) for name in requested_names
        )
    expected_variant_payload = [
        [name, list(dropped)] for name, dropped in expected_variants
    ]
    if variants != expected_variant_payload:
        return None, "checkpoint feature variants are invalid"
    variant_names = [str(item[0]) for item in variants]

    race_count = signature.get("race_count")
    train_end = signature.get("train_end")
    selection_end = signature.get("selection_end")
    universe_hash = signature.get("race_universe_sha256")
    if (
        isinstance(race_count, bool)
        or not isinstance(race_count, int)
        or race_count <= 0
        or isinstance(train_end, bool)
        or not isinstance(train_end, int)
        or isinstance(selection_end, bool)
        or not isinstance(selection_end, int)
        or not 0 < train_end < selection_end <= race_count
        or not isinstance(universe_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", universe_hash) is None
    ):
        return None, "checkpoint race-universe signature is invalid"

    try:
        source_snapshot = _validated_source_data_snapshot(
            signature.get("source_data_snapshot")
        )
    except ValueError as exc:
        return None, f"checkpoint source data snapshot is invalid: {exc}"
    if (
        source_snapshot["race_count"] != race_count
        or source_snapshot["race_universe_sha256"] != universe_hash
    ):
        return None, "checkpoint source data snapshot race universe mismatch"

    completed = progress.get("completed_candidates")
    total = progress.get("total_candidates")
    completed_variants = progress.get("completed_variants")
    total_variants = progress.get("total_variants")
    expected_total = (
        len(variant_names)
        * len(expected_signature["targets"])
        * len(expected_signature["alphas"])
    )
    if (
        isinstance(completed, bool)
        or not isinstance(completed, int)
        or completed != len(rows)
        or not 0 < completed <= expected_total
        or total != expected_total
        or total_variants != len(variant_names)
        or isinstance(completed_variants, bool)
        or not isinstance(completed_variants, int)
        or not 0 <= completed_variants <= total_variants
    ):
        return None, "checkpoint progress is inconsistent"


    loaded = _load_checkpoint(resolved, signature)
    if len(loaded) != completed:
        return None, "checkpoint candidates failed integrity validation"
    return resolved, ""


def _evaluation_reservation_mb(resources: ResourceSnapshot) -> int:
    return max(
        8192,
        int(resources.memory_limit_mb or resources.available_memory_mb)
        - EVALUATION_MEMORY_SAFETY_MB,
    )


def claim_job(
    conn: Any,
    *,
    worker_id: str,
    resources: ResourceSnapshot,
) -> dict[str, Any] | None:
    snapshot = _json(resource_snapshot_dict(resources))
    evaluation_reservation_mb = _evaluation_reservation_mb(resources)
    conn.execute("SELECT pg_advisory_xact_lock(?)", (CLAIM_LOCK_ID,))
    candidate = conn.execute(
        """
        SELECT jobs.*
        FROM model_evaluation_jobs AS jobs
        WHERE jobs.status = 'queued'
          AND jobs.available_at <= CURRENT_TIMESTAMP
          AND jobs.attempt < jobs.max_attempts
          AND (
            jobs.parent_job_id IS NULL
            OR EXISTS (
              SELECT 1 FROM model_evaluation_jobs parent
              WHERE parent.job_id = jobs.parent_job_id
                AND parent.status = 'completed'
            )
          )
          AND jobs.min_free_memory_mb <= ?
          AND jobs.min_free_disk_mb <= ?
          AND jobs.min_idle_cpu_percent <= ?
          AND (
            SELECT COUNT(*) FROM model_evaluation_jobs running
            WHERE running.status = 'running'
              AND running.task_type = jobs.task_type
          ) < jobs.max_parallel
          AND (
            jobs.category <> 'evaluation'
            OR jobs.min_free_memory_mb < 8192
            OR (
              jobs.min_free_memory_mb + COALESCE((
                SELECT SUM(running.min_free_memory_mb)
                FROM model_evaluation_jobs running
                WHERE running.status = 'running'
                  AND running.category = 'evaluation'
                  AND running.min_free_memory_mb >= 8192
              ), 0)
            ) <= ?
          )
        ORDER BY jobs.priority DESC, jobs.job_id
        FOR UPDATE SKIP LOCKED
        LIMIT 1
        """,
        (
            resources.available_memory_mb,
            resources.available_disk_mb,
            resources.idle_cpu_percent,
            evaluation_reservation_mb,
        ),
    ).fetchone()
    if candidate is None:
        return None
    candidate_row = {key: candidate[key] for key in candidate.keys()}
    parameters = candidate_row.get("parameters")
    parameters = (
        parameters
        if isinstance(parameters, dict)
        else json.loads(parameters or "{}")
    )
    previous_error = str(candidate_row.get("error") or "")
    if previous_error.startswith("TimeoutExpired:"):
        parameters = _timeout_retry_parameters(
            parameters,
            task_type=str(candidate_row["task_type"]),
            previous_error=previous_error,
        )
    row = conn.execute(
        """
        UPDATE model_evaluation_jobs
        SET status = 'running', worker_id = ?, locked_at = CURRENT_TIMESTAMP,
            started_at = CURRENT_TIMESTAMP,
            attempt = attempt + 1, updated_at = CURRENT_TIMESTAMP,
            parameters = CAST(? AS JSONB), error = NULL,
            last_resource_snapshot = CAST(? AS JSONB)
        WHERE job_id = ? AND status = 'queued'
        RETURNING *
        """,
        (
            worker_id,
            _json(parameters),
            snapshot,
            int(candidate_row["job_id"]),
        ),
    ).fetchone()
    if row is None:
        return None
    result = {key: row[key] for key in row.keys()}
    params = result.get("parameters")
    result["parameters"] = (
        params if isinstance(params, dict) else json.loads(params or "{}")
    )
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


def requeue_stale_jobs(
    conn: Any,
    *,
    stale_minutes: int = 180,
    app_root: Path | None = None,
) -> int:
    if app_root is not None:
        rows = conn.execute(
            """
            SELECT * FROM model_evaluation_jobs
            WHERE status = 'running'
              AND locked_at < CURRENT_TIMESTAMP - (? * INTERVAL '1 minute')
            ORDER BY job_id
            FOR UPDATE SKIP LOCKED
            """,
            (max(1, int(stale_minutes)),),
        ).fetchall()
        for row in rows:
            job = {key: row[key] for key in row.keys()}
            _record_failed_attempt(
                conn,
                job=job,
                error="worker lease expired",
                app_root=app_root,
            )
        return len(rows)
    audit_error = "worker lease expired"
    rows = conn.execute(
        """
        WITH locked_jobs AS MATERIALIZED (
          SELECT job_id
          FROM model_evaluation_jobs
          WHERE status = 'running'
            AND locked_at < CURRENT_TIMESTAMP - (? * INTERVAL '1 minute')
          ORDER BY job_id
          FOR UPDATE SKIP LOCKED
        ), stale_jobs AS (
          UPDATE model_evaluation_jobs AS jobs
          SET status = CASE WHEN attempt < max_attempts THEN 'queued' ELSE 'failed' END,
              available_at = CURRENT_TIMESTAMP,
              worker_id = NULL,
              locked_at = NULL,
              error = COALESCE(error, ?),
              updated_at = CURRENT_TIMESTAMP
          FROM locked_jobs
          WHERE jobs.job_id = locked_jobs.job_id
          RETURNING jobs.job_id, jobs.attempt
        ), closed_runs AS (
          UPDATE model_evaluation_job_runs AS runs
          SET status = 'failed', completed_at = CURRENT_TIMESTAMP,
              error = COALESCE(runs.error, ?)
          FROM stale_jobs
          WHERE runs.job_id = stale_jobs.job_id
            AND runs.attempt = stale_jobs.attempt
            AND runs.status = 'running'
          RETURNING runs.run_id
        )
        SELECT job_id, attempt FROM stale_jobs
        """,
        (max(1, int(stale_minutes)), audit_error, audit_error),
    ).fetchall()
    return len(rows)


def reconcile_queue_state(conn: Any) -> int:
    """Close orphaned runs and cancel exhausted jobs that cannot be claimed."""
    orphaned_run_error = (
        "queue reconciliation closed orphaned running attempt"
    )
    audit_error = (
        "queue reconciliation cancelled exhausted job: "
        "attempt reached max_attempts"
    )
    rows = conn.execute(
        """
        WITH orphaned_runs AS (
          UPDATE model_evaluation_job_runs AS runs
          SET status = 'failed', completed_at = CURRENT_TIMESTAMP,
              error = COALESCE(runs.error, ?)
          WHERE runs.status = 'running'
            AND NOT EXISTS (
              SELECT 1
              FROM model_evaluation_jobs AS jobs
              WHERE jobs.job_id = runs.job_id
                AND jobs.status = 'running'
                AND jobs.attempt = runs.attempt
            )
          RETURNING runs.run_id
        ), locked_jobs AS MATERIALIZED (
          SELECT job_id
          FROM model_evaluation_jobs
          WHERE status = 'queued' AND attempt >= max_attempts
          ORDER BY job_id
          FOR UPDATE SKIP LOCKED
        )
        UPDATE model_evaluation_jobs AS jobs
        SET status = 'cancelled', completed_at = CURRENT_TIMESTAMP,
            worker_id = NULL, locked_at = NULL,
            error = ?, updated_at = CURRENT_TIMESTAMP
        FROM locked_jobs
        WHERE jobs.job_id = locked_jobs.job_id
        RETURNING jobs.job_id
        """,
        (orphaned_run_error, audit_error),
    ).fetchall()
    return len(rows)


def _validated_reconciliation_result_path(
    result_path: str,
    *,
    app_root: Path,
) -> Path:
    root = app_root.resolve(strict=True)
    candidate = Path(result_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("result_path is outside app_root") from exc
    if not resolved.is_file():
        raise ValueError("result_path is not a regular file")
    return resolved


def reconcile_completed_job_runs(
    conn: Any,
    *,
    app_root: Path,
    limit: int = 16,
) -> int:
    """Recover parent jobs whose matching attempt already produced a result."""
    rows = conn.execute(
        """
        SELECT jobs.*, runs.run_id, runs.result_path AS run_result_path
        FROM model_evaluation_jobs AS jobs
        JOIN model_evaluation_job_runs AS runs
          ON runs.job_id = jobs.job_id AND runs.attempt = jobs.attempt
        WHERE jobs.status = 'running'
          AND runs.status = 'completed'
          AND runs.result_path IS NOT NULL
          AND runs.result_path <> ''
        ORDER BY jobs.job_id
        FOR UPDATE OF jobs, runs SKIP LOCKED
        LIMIT ?
        """,
        (max(1, int(limit)),),
    ).fetchall()
    recovered = 0
    for row in rows:
        job = {key: row[key] for key in row.keys()}
        parameters = job.get("parameters")
        job["parameters"] = (
            parameters
            if isinstance(parameters, dict)
            else json.loads(parameters or "{}")
        )
        try:
            result_path = _validated_reconciliation_result_path(
                str(job["run_result_path"]),
                app_root=app_root,
            )
            _payload, summary = _load_result(result_path)
            decision = result_decision(str(job["task_type"]), summary)
        except Exception as exc:
            fail_job(
                conn,
                job=job,
                error=(
                    "completed run reconciliation failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
            continue
        complete_job(
            conn,
            job=job,
            result_path=result_path,
            summary=summary,
            decision=decision,
        )
        recovered += 1
    return recovered


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
        WITH locked_jobs AS MATERIALIZED (
          SELECT job_id
          FROM model_evaluation_jobs
          WHERE status IN {status_sql} {attempt_filter}
          ORDER BY job_id
          FOR UPDATE SKIP LOCKED
        )
        UPDATE model_evaluation_jobs AS jobs
        SET status = 'queued', {reset}
            available_at = CURRENT_TIMESTAMP, completed_at = NULL,
            worker_id = NULL, locked_at = NULL, error = NULL,
            updated_at = CURRENT_TIMESTAMP
        FROM locked_jobs
        WHERE jobs.job_id = locked_jobs.job_id
        RETURNING jobs.job_id
        """
    ).fetchall()
    return len(rows)


def recover_worker_job(
    conn: Any,
    *,
    worker_id: str,
    app_root: Path | None = None,
) -> int:
    if app_root is not None:
        rows = conn.execute(
            """
            SELECT * FROM model_evaluation_jobs
            WHERE status = 'running' AND worker_id = ?
            ORDER BY job_id
            FOR UPDATE SKIP LOCKED
            """,
            (worker_id,),
        ).fetchall()
        for row in rows:
            job = {key: row[key] for key in row.keys()}
            _record_failed_attempt(
                conn,
                job=job,
                error="worker restarted before completion update",
                app_root=app_root,
            )
        return len(rows)
    rows = conn.execute(
        """
        WITH locked_jobs AS MATERIALIZED (
          SELECT job_id
          FROM model_evaluation_jobs
          WHERE status = 'running' AND worker_id = ?
          ORDER BY job_id
          FOR UPDATE SKIP LOCKED
        )
        UPDATE model_evaluation_jobs AS jobs
        SET status = 'queued', available_at = CURRENT_TIMESTAMP,
            worker_id = NULL, locked_at = NULL, updated_at = CURRENT_TIMESTAMP,
            error = COALESCE(error, 'worker restarted before completion update')
        FROM locked_jobs
        WHERE jobs.job_id = locked_jobs.job_id
        RETURNING jobs.job_id, jobs.attempt
        """,
        (worker_id,),
    ).fetchall()
    for row in rows:
        conn.execute(
            """
            UPDATE model_evaluation_job_runs
            SET status = 'failed', completed_at = CURRENT_TIMESTAMP,
                error = COALESCE(error, 'worker restarted before completion update')
            WHERE job_id = ? AND attempt = ? AND status = 'running'
            """,
            (int(row["job_id"]), int(row["attempt"])),
        )
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


def _drop_feature_groups(params: dict[str, Any]) -> str:
    raw = params.get("drop_feature_groups", "research_correlates")
    if not isinstance(raw, str):
        raise ValueError("drop_feature_groups must be a comma-separated string")
    from .feature_tuning import normalize_drop_feature_groups

    try:
        selected = normalize_drop_feature_groups(raw)
    except ValueError as exc:
        raise ValueError(
            "unknown drop_feature_groups: " + str(exc)
        ) from exc
    if not selected:
        raise ValueError("drop_feature_groups must contain at least one group")
    return ",".join(selected)


def _half_lives(params: dict[str, Any], *, minimum_candidates: int = 2) -> str:
    raw = params.get("half_lives", "none,180,365,730")
    if not isinstance(raw, str):
        raise ValueError("half_lives must be a comma-separated string")
    candidates: list[str] = []
    seen: set[float | None] = set()
    for token in raw.split(","):
        token = token.strip()
        if token == "none":
            value = None
            normalized = token
        else:
            try:
                value = float(token)
            except ValueError as exc:
                raise ValueError(
                    "half_lives entries must be none or finite numbers in [30, 3650]"
                ) from exc
            if not math.isfinite(value) or not 30 <= value <= 3650:
                raise ValueError(
                    "half_lives entries must be none or finite numbers in [30, 3650]"
                )
            normalized = str(int(value)) if value.is_integer() else format(value, ".15g")
        if value not in seen:
            seen.add(value)
            candidates.append(normalized)
    if len(candidates) < minimum_candidates:
        raise ValueError(
            "half_lives must contain at least "
            f"{minimum_candidates} distinct candidates"
        )
    return ",".join(candidates)


class JobDependencyUnavailable(RuntimeError):
    """A required upstream artifact has not been generated yet."""


class ObsoleteJob(RuntimeError):
    """A queued job references an artifact superseded by the current schema."""


def _selected_standard_cache_prefix(app_root: Path) -> Path:
    artifact = (
        app_root / "data" / "models" / "standardized_365d_v2"
        / "raw" / "listwise_feature_teacher.json"
    )
    try:
        text = artifact.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise JobDependencyUnavailable(
            f"standardized feature artifact is not available yet: {artifact}"
        ) from exc
    except OSError as exc:
        raise ValueError(
            "standardized feature artifact cannot be read"
        ) from exc
    try:
        payload = json.loads(text)
        selected = payload["selected"]
        variant = selected["feature_variant"]
        selected_cache_dir = payload["selected_cache_dir"]
        n_features = payload["n_features"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(
            "standardized feature artifact is incomplete or invalid"
        ) from exc
    artifact_schema = str(payload.get("feature_schema_version") or "")
    if artifact_schema != FEATURE_SCHEMA_VERSION:
        raise JobDependencyUnavailable(
            "standardized feature artifact uses a stale feature schema: "
            f"{artifact_schema or 'missing'}; current={FEATURE_SCHEMA_VERSION}"
        )
    if not isinstance(variant, str):
        raise ValueError("standardized feature artifact has an invalid feature variant")
    from .listwise.feature_search import feature_variants

    known_variants = {name for name, _dropped in feature_variants()}
    if variant not in known_variants:
        raise ValueError("standardized feature artifact has an unknown feature variant")
    if isinstance(n_features, bool) or not isinstance(n_features, int):
        raise ValueError("standardized feature artifact has invalid n_features")
    if not 1024 <= n_features <= 32768:
        raise ValueError("standardized feature artifact n_features is out of range")
    expected_cache_dir = STANDARDIZED_SELECTED_CACHE_DIR
    if selected_cache_dir != str(expected_cache_dir):
        raise ValueError(
            "standardized feature artifact selected_cache_dir must exactly match "
            f"{expected_cache_dir}"
        )
    cache_prefix = expected_cache_dir / (
        f"listwise_search_{n_features}_{variant}"
    )
    if cache_prefix.parent != expected_cache_dir:
        raise ValueError("standardized feature cache prefix escapes the allowed directory")
    manifest = Path(str(cache_prefix) + ".manifest.json")
    if not manifest.exists():
        raise JobDependencyUnavailable(
            f"selected standardized feature cache is not available yet: {manifest}"
        )
    if not manifest.is_file():
        raise ValueError(
            f"selected standardized feature cache manifest is invalid: {manifest}"
        )
    try:
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            "selected standardized feature cache manifest is invalid"
        ) from exc
    manifest_schema = str(manifest_payload.get("feature_schema_version") or "")
    if manifest_schema != FEATURE_SCHEMA_VERSION:
        raise JobDependencyUnavailable(
            "selected standardized feature cache uses a stale feature schema: "
            f"{manifest_schema or 'missing'}; current={FEATURE_SCHEMA_VERSION}"
        )
    return cache_prefix


def _standardized_holdout_contract(app_root: Path) -> tuple[str, str, str]:
    protocol_path = (
        app_root / "data" / "models" / "standardized_365d_v2"
        / "protocol.json"
    )
    try:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        holdout_start = str(protocol["holdout_start"])
        holdout_end = str(protocol["holdout_end"])
        calendar_days = int(protocol["calendar_days"])
        start_date = datetime.strptime(holdout_start, "%Y-%m-%d").date()
        end_date = datetime.strptime(holdout_end, "%Y-%m-%d").date()
    except FileNotFoundError as exc:
        raise JobDependencyUnavailable(
            f"standardized evaluation protocol is not available yet: {protocol_path}"
        ) from exc
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("standardized evaluation protocol is invalid") from exc
    if calendar_days != 365 or end_date - start_date != timedelta(days=364):
        raise ValueError("standardized evaluation protocol is not an exact 365-day window")
    training_through = (start_date - timedelta(days=1)).isoformat()
    return training_through, holdout_start, holdout_end


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
        return [
            str(python),
            "-m",
            "boatrace_ai.script_snapshot",
            "--app-root",
            str(app_root),
            str(app_root / "scripts" / "run_standardized_365d_evaluations.sh"),
        ], (
            app_root / "data" / "models" / "standardized_365d_v2" / "manifest.json"
        )
    if task_type == "historical_coverage_safe":
        evaluation_date = _date(params, "evaluation_date")
        _integer(params, "timeout_seconds", 28800, 300, 86400)
        unsupported = set(params) - {"evaluation_date", "timeout_seconds"}
        if unsupported:
            raise ValueError(
                "unsupported historical_coverage_safe parameters: "
                + ", ".join(sorted(unsupported))
            )
        model_input = (
            app_root / "data" / "models" / "standardized_365d_v2"
            / "no_odds_v8.joblib"
        )
        return [
            str(python), "-m", "boatrace_ai.historical_candidate_evaluation",
            "--db", db,
            "--output", str(output),
            "--model-input", str(model_input),
            "--evaluation-date", evaluation_date,
        ], output
    if task_type == "historical_research_logit":
        evaluation_date = _date(params, "evaluation_date")
        _integer(params, "timeout_seconds", 28800, 300, 86400)
        unsupported = set(params) - {"evaluation_date", "timeout_seconds"}
        if unsupported:
            raise ValueError(
                "unsupported historical_research_logit parameters: "
                + ", ".join(sorted(unsupported))
            )
        return [
            str(python), "-m", "boatrace_ai.historical_research_evaluation",
            "--db", db,
            "--evaluation-date", evaluation_date,
            "--model-dir", str(app_root / "data" / "models"),
            "--output", str(output),
        ], output
    if task_type == "genetic_island_search":
        allowed = {
            "evaluation_date", "cohort", "generation", "island_id",
            "island_count", "max_generations", "seed", "population_size",
            "local_generations", "elite_count", "train_races",
            "validation_races", "embargo_days", "batch_races", "immigrants", "mutation_rate",
            "random_injections", "migration_interval", "migration_applied",
            "diversity_rescue", "structural_elite_count", "timeout_seconds",
            "genetic_protocol_version",
        }
        unsupported = set(params) - allowed
        if unsupported:
            raise ValueError(
                "unsupported genetic_island_search parameters: "
                + ", ".join(sorted(unsupported))
            )
        evaluation_date = _date(params, "evaluation_date")
        cohort = str(params.get("cohort") or "").strip()
        if not cohort or len(cohort) > 80:
            raise ValueError("genetic cohort is required and must be at most 80 characters")
        generation = _integer(params, "generation", 0, 0, 20)
        _integer(params, "genetic_protocol_version", 4, 1, 99)
        island_id = _integer(params, "island_id", 0, 0, 63)
        island_count = _integer(params, "island_count", 4, 2, 64)
        if island_id >= island_count:
            raise ValueError("genetic island_id must be lower than island_count")
        max_generations = _integer(params, "max_generations", 3, 1, 20)
        seed = _integer(params, "seed", 1, 0, 2_147_483_647)
        population_size = _integer(params, "population_size", 8, 4, 32)
        local_generations = _integer(params, "local_generations", 3, 1, 10)
        elite_count = _integer(params, "elite_count", 2, 1, 8)
        mutation_rate = _number(params, "mutation_rate", 0.35, 0.10, 0.85)
        random_injections = _integer(
            params, "random_injections", 1, 0, population_size // 2
        )
        _integer(params, "migration_interval", 3, 2, 10)
        train_races = _integer(params, "train_races", 12000, 2000, 50000)
        validation_races = _integer(params, "validation_races", 3000, 500, 15000)
        embargo_days = _integer(params, "embargo_days", 1, 1, 14)
        batch_races = _integer(params, "batch_races", 500, 100, 2000)
        _integer(params, "timeout_seconds", 7200, 300, 43200)
        immigrants = params.get("immigrants") or []
        if not isinstance(immigrants, list) or len(immigrants) > 8:
            raise ValueError("genetic immigrants must be a list of at most 8 genomes")
        return [
            str(python), "-m", "boatrace_ai.genetic_islands",
            "--db", db,
            "--output", str(output),
            "--cache-prefix", str(
                app_root / "data/models/standardized_365d_v2/selected_cache"
                / "listwise_search_8192_drop_base_pastlog"
            ),
            "--evaluation-date", evaluation_date,
            "--cohort", cohort,
            "--generation", str(generation),
            "--island-id", str(island_id),
            "--island-count", str(island_count),
            "--max-generations", str(max_generations),
            "--seed", str(seed),
            "--population-size", str(population_size),
            "--local-generations", str(local_generations),
            "--elite-count", str(elite_count),
            "--mutation-rate", str(mutation_rate),
            "--random-injections", str(random_injections),
            "--train-races", str(train_races),
            "--validation-races", str(validation_races),
            "--embargo-days", str(embargo_days),
            "--batch-races", str(batch_races),
            "--immigrants-json", _json(immigrants),
        ], output
    if task_type == "calibrated_mlp_recency_search":
        unsupported = set(params) - {
            "evaluation_date", "timeout_seconds", "half_lives", "calibration_days",
            "drop_feature_groups", "protected_blend",
            "selection_entry_log_loss_tolerance",
        }
        if unsupported:
            raise ValueError(
                "unsupported calibrated_mlp_recency_search parameters: "
                + ", ".join(sorted(unsupported))
            )
        if "evaluation_date" not in params:
            raise ValueError("evaluation_date is required")
        evaluation_date = _date(params, "evaluation_date")
        _integer(params, "timeout_seconds", 28800, 300, 86400)
        half_lives = _half_lives(params, minimum_candidates=1)
        calibration_days = _integer(params, "calibration_days", 180, 30, 730)
        drop_feature_groups = _drop_feature_groups(params)
        selection_tolerance = _number(
            params, "selection_entry_log_loss_tolerance", 0.0005, 0.0, 0.05
        )
        protected_blend = params.get("protected_blend", False)
        if type(protected_blend) is not bool:
            raise ValueError("protected_blend must be a boolean")
        cache_suffix = (
            ""
            if drop_feature_groups == "research_correlates"
            else "__drop_" + drop_feature_groups.replace(",", "_")
        )
        feature_cache = (
            app_root
            / "data"
            / "models"
            / ("calibrated_shadow_features_16384" + cache_suffix)
        )
        command = [
            str(python), "-m", "boatrace_ai.recency_mlp_evaluation",
            "--db", db,
            "--output", str(output),
            "--model-output", str(output.with_suffix(".joblib")),
            "--deployment-model-output", str(
                output.with_name(output.stem + ".deployment.joblib")
            ),
            "--incumbent-prediction", str(
                app_root / "data/models/standardized_365d_v2/raw/no_odds_v8_prediction.json"
            ),
            "--incumbent-bankroll", str(
                app_root / "data/models/standardized_365d_v2/raw/no_odds_v8_bankroll.json"
            ),
            "--evaluation-date", evaluation_date,
            "--feature-cache", str(feature_cache),
            "--drop-feature-groups", drop_feature_groups,
            "--half-lives", half_lives,
            "--calibration-days", str(calibration_days),
            "--selection-entry-log-loss-tolerance", str(selection_tolerance),
        ]
        if protected_blend:
            command.extend([
                "--protected-baseline-model",
                str(app_root / "data/models/standardized_365d_v2/no_odds_v8.joblib"),
            ])
        return command, output
    if task_type == "racer_stats_backfill":
        allowed = {"from_year", "to_year", "sleep_seconds", "timeout_seconds"}
        unsupported = set(params) - allowed
        if unsupported:
            raise ValueError(
                "unsupported racer_stats_backfill parameters: "
                + ", ".join(sorted(unsupported))
            )
        from_year = _integer(params, "from_year", 2016, 2000, 2100)
        to_year = _integer(params, "to_year", 2026, 2000, 2100)
        if from_year > to_year or to_year - from_year > 20:
            raise ValueError("invalid racer statistics year range")
        sleep_seconds = _number(params, "sleep_seconds", 1.5, 0.0, 10.0)
        _integer(params, "timeout_seconds", 3600, 300, 86400)
        return [
            str(python), "-m", "boatrace_ai.racer_stats_backfill",
            "--db", db,
            "--output", str(output),
            "--raw-dir", str(app_root / "data/raw"),
            "--from-year", str(from_year),
            "--to-year", str(to_year),
            "--sleep-seconds", str(sleep_seconds),
        ], output


    if task_type == "lightgbm_recency_search":
        allowed = {
            "evaluation_date", "timeout_seconds", "half_lives",
            "calibration_days", "drop_feature_groups", "n_estimators",
            "num_leaves", "max_depth", "min_child_samples",
            "feature_fraction", "max_bin", "n_jobs", "architecture_presets",
            "incumbent_result",
            "selection_entry_log_loss_tolerance",
        }
        unsupported = set(params) - allowed
        if unsupported:
            raise ValueError(
                "unsupported lightgbm_recency_search parameters: "
                + ", ".join(sorted(unsupported))
            )
        if "evaluation_date" not in params:
            raise ValueError("evaluation_date is required")
        evaluation_date = _date(params, "evaluation_date")
        _integer(params, "timeout_seconds", 86400, 300, 86400)
        half_life_params = dict(params)
        half_life_params.setdefault("half_lives", "none,365")
        half_lives = _half_lives(half_life_params, minimum_candidates=1)
        calibration_days = _integer(params, "calibration_days", 180, 30, 730)
        drop_params = dict(params)
        drop_params.setdefault("drop_feature_groups", "legacy_composites")
        drop_feature_groups = _drop_feature_groups(drop_params)
        n_estimators = _integer(params, "n_estimators", 300, 10, 2000)
        num_leaves = _integer(params, "num_leaves", 31, 2, 512)
        max_depth = _integer(params, "max_depth", -1, -1, 20)
        if max_depth == 0:
            raise ValueError("max_depth must be -1 or in [1, 20]")
        min_child_samples = _integer(
            params, "min_child_samples", 100, 1, 100000
        )
        feature_fraction = _number(
            params, "feature_fraction", 0.6, 0.05, 1.0
        )
        max_bin = _integer(params, "max_bin", 63, 15, 255)
        n_jobs = _integer(params, "n_jobs", 4, 1, 128)
        selection_tolerance = _number(
            params, "selection_entry_log_loss_tolerance", 0.0005, 0.0, 0.05
        )
        architecture_presets = params.get("architecture_presets")
        if architecture_presets is not None:
            if not isinstance(architecture_presets, str):
                raise ValueError("architecture_presets must be a string")
            from .lightgbm_recency_evaluation import parse_architecture_presets

            try:
                architecture_presets = ",".join(
                    parse_architecture_presets(architecture_presets)
                )
            except argparse.ArgumentTypeError as exc:
                raise ValueError(str(exc)) from exc
        incumbent_prediction = (
            app_root
            / "data/models/standardized_365d_v2/raw/no_odds_v8_prediction.json"
        )
        incumbent_bankroll = (
            app_root
            / "data/models/standardized_365d_v2/raw/no_odds_v8_bankroll.json"
        )
        if params.get("incumbent_result") is not None:
            incumbent_result = (
                app_root / str(params["incumbent_result"])
            ).resolve()
            result_root = (
                app_root / "data/models/evaluation_queue"
            ).resolve()
            if (
                result_root not in incumbent_result.parents
                or incumbent_result.suffix != ".json"
            ):
                raise ValueError(
                    "incumbent_result must be a JSON artifact inside "
                    "data/models/evaluation_queue"
                )
            if not incumbent_result.is_file():
                raise JobDependencyUnavailable(
                    f"incumbent result is not available yet: {incumbent_result}"
                )
            incumbent_prediction = incumbent_result
            incumbent_bankroll = incumbent_result
        cache_name = (
            "lightgbm_v6_features_16384_drop_"
            + drop_feature_groups.replace(",", "_")
        )
        feature_cache = app_root / "data" / "models" / cache_name
        command = [
            str(python), "-m", "boatrace_ai.lightgbm_recency_evaluation",
            "--db", db,
            "--output", str(output),
            "--model-output", str(output.with_suffix(".joblib")),
            "--deployment-model-output", str(
                output.with_name(output.stem + ".deployment.joblib")
            ),
            "--incumbent-prediction", str(
                incumbent_prediction
            ),
            "--incumbent-bankroll", str(
                incumbent_bankroll
            ),
            "--evaluation-date", evaluation_date,
            "--feature-cache", str(feature_cache),
            "--no-write-feature-cache",
            "--drop-feature-groups", drop_feature_groups,
            "--half-lives", half_lives,
            "--calibration-days", str(calibration_days),
            "--n-estimators", str(n_estimators),
            "--num-leaves", str(num_leaves),
            "--max-depth", str(max_depth),
            "--min-child-samples", str(min_child_samples),
            "--feature-fraction", str(feature_fraction),
            "--max-bin", str(max_bin),
            "--n-jobs", str(n_jobs),
            "--selection-entry-log-loss-tolerance", str(selection_tolerance),
        ]
        if architecture_presets is not None:
            command.extend(["--architecture-presets", architecture_presets])
        return command, output

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
    if task_type == "four_head_learned_value":
        allowed = {
            "source_model", "training_from", "training_through",
            "outer_from", "outer_through", "projection_dimensions",
            "minimum_inner_training_dates",
            "minimum_purchase_training_dates", "alpha",
            "max_races_per_day", "max_snapshot_age_seconds",
            "timeout_seconds", "purchase_teacher_version",
        }
        unsupported = set(params) - allowed
        if unsupported:
            raise ValueError(
                "unsupported four_head_learned_value parameters: "
                + ", ".join(sorted(unsupported))
            )
        missing = {
            "source_model", "training_from", "training_through",
            "outer_from", "outer_through", "purchase_teacher_version",
        } - set(params)
        if missing:
            raise ValueError(
                "missing four_head_learned_value parameters: "
                + ", ".join(sorted(missing))
            )
        _integer(params, "purchase_teacher_version", 3, 3, 3)
        training_from = _date(params, "training_from")
        training_through = _date(params, "training_through")
        outer_from = _date(params, "outer_from")
        outer_through = _date(params, "outer_through")
        if training_from > training_through or outer_from > outer_through:
            raise ValueError("four-head evaluation periods must be chronological")
        if training_through >= outer_from:
            raise ValueError("four-head outer period must follow training")
        projection_dimensions = _integer(
            params, "projection_dimensions", 8, 1, 128
        )
        minimum_inner = _integer(
            params, "minimum_inner_training_dates", 2, 1, 60
        )
        minimum_purchase = _integer(
            params, "minimum_purchase_training_dates", 2, 1, 60
        )
        alpha = _number(params, "alpha", 1e-3, 1e-9, 1000.0)
        max_snapshot_age = _number(
            params, "max_snapshot_age_seconds", 300.0, 0.0, 300.0
        )
        _integer(params, "timeout_seconds", 7200, 300, 86400)
        model_root = (app_root / "data" / "models").resolve()
        source_model = (app_root / str(params["source_model"])).resolve()
        if model_root not in source_model.parents or source_model.suffix != ".joblib":
            raise ValueError(
                "source_model must be a joblib artifact inside data/models"
            )
        command = [
            str(python), "-m", "boatrace_ai.listwise.four_head_v22_evaluation",
            "--db", db,
            "--source-model", str(source_model),
            "--training-from", training_from,
            "--training-through", training_through,
            "--outer-from", outer_from,
            "--outer-through", outer_through,
            "--projection-dimensions", str(projection_dimensions),
            "--minimum-inner-training-dates", str(minimum_inner),
            "--minimum-purchase-training-dates", str(minimum_purchase),
            "--alpha", str(alpha),
            "--max-snapshot-age-seconds", str(max_snapshot_age),
            "--output", str(output),
        ]
        if params.get("max_races_per_day") is not None:
            command.extend([
                "--max-races-per-day",
                str(_integer(params, "max_races_per_day", 144, 1, 1000)),
            ])
        return command, output
    if task_type == "market_residual_walk_forward":
        allowed = {
            "model_input", "from_date", "through_date", "daily_budget_yen",
            "baseline_model_input", "candidate_weight",
            "min_calibration_days", "calibrator_strategy",
            "minimum_day_coverage", "timeout_seconds",
            "v12_closing_fallback_policy",
            "v25_probability_artifact",
            "closing_odds_min_training_days",
            "closing_odds_min_training_races",
        }
        unsupported = set(params) - allowed
        if unsupported:
            raise ValueError(
                "unsupported market_residual_walk_forward parameters: "
                + ", ".join(sorted(unsupported))
            )
        missing = {"model_input", "from_date"} - set(params)
        if missing:
            raise ValueError(
                "missing market_residual_walk_forward parameters: "
                + ", ".join(sorted(missing))
            )
        from_date = _date(params, "from_date")
        through_date = (
            _date(params, "through_date")
            if params.get("through_date") is not None
            else None
        )
        if through_date is not None and through_date < from_date:
            raise ValueError("market evaluation dates must be chronological")
        _integer(params, "timeout_seconds", 3600, 300, 86400)
        model_root = (app_root / "data" / "models").resolve()
        model_input = (app_root / str(params["model_input"])).resolve()
        has_baseline = "baseline_model_input" in params
        has_candidate_weight = "candidate_weight" in params
        if has_baseline != has_candidate_weight:
            raise ValueError(
                "baseline_model_input and candidate_weight must be provided together"
            )
        cache_period = f"{from_date}_{through_date or 'latest'}"
        scored_cache = (
            app_root
            / "data/models/evaluation_cache/market_scored"
            / f"{model_input.stem}_{cache_period}.races.joblib"
        )
        if model_root not in model_input.parents or model_input.suffix != ".joblib":
            raise ValueError("model_input must be a joblib artifact inside data/models")
        if not model_input.is_file():
            raise JobDependencyUnavailable(
                f"market source model is not available yet: {model_input}"
            )
        baseline_model_input = None
        candidate_weight = None
        if has_baseline:
            raw_baseline = params["baseline_model_input"]
            if not isinstance(raw_baseline, str) or not raw_baseline.strip():
                raise ValueError("baseline_model_input must be a non-empty string")
            raw_weight = params["candidate_weight"]
            if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
                raise ValueError("candidate_weight must be a number in [0, 1]")
            candidate_weight = float(raw_weight)
            if not math.isfinite(candidate_weight) or not 0.0 <= candidate_weight <= 1.0:
                raise ValueError("candidate_weight must be finite and in [0, 1]")
            baseline_model_input = (
                app_root / raw_baseline
            ).resolve()
            if (
                model_root not in baseline_model_input.parents
                or baseline_model_input.suffix != ".joblib"
            ):
                raise ValueError(
                    "baseline_model_input must be a joblib artifact inside data/models"
                )
            if not baseline_model_input.is_file():
                raise JobDependencyUnavailable(
                    "market baseline model is not available yet: "
                    f"{baseline_model_input}"
                )
        v25_probability_artifact = None
        if params.get("v25_probability_artifact") is not None:
            artifact_root = (
                app_root / "data" / "models" / "evaluation_queue"
            ).resolve()
            v25_probability_artifact = (
                app_root / str(params["v25_probability_artifact"])
            ).resolve()
            if (
                artifact_root not in v25_probability_artifact.parents
                or v25_probability_artifact.suffix != ".json"
            ):
                raise ValueError(
                    "v25_probability_artifact must be a JSON artifact inside "
                    "data/models/evaluation_queue"
                )
            if not v25_probability_artifact.is_file():
                raise JobDependencyUnavailable(
                    "V25 probability artifact is not available yet: "
                    f"{v25_probability_artifact}"
                )
        strategy = str(params.get("calibrator_strategy", "newton_residual"))
        if strategy not in {
            "grid",
            "newton_residual",
            "orthogonal_residual",
            "odds_path_return",
            "odds_path_probability",
            "odds_path_closing_return",
            "odds_path_observed_closing_return",
            "odds_path_observed_closing_return_robust_policy_v17",
            "odds_path_observed_closing_return_schedule_quota_v18",
            "odds_path_observed_closing_return_schedule_quota_raw_nonregression_v19",
            "odds_path_observed_closing_return_schedule_quota_dual_head_v20",
            "odds_path_observed_closing_return_schedule_quota_triple_head_v21",
            "odds_path_observed_closing_return_stable_policy_triple_head_v35",
            "odds_path_hit_shrunk_return",
            "odds_path_prequential_shrinkage_return",
            "odds_path_crossfit_conservative_ev",
            "odds_path_market_offset_crossfit_conservative_ev",
            "odds_path_market_offset_discrete_log_ev_v9",
            "odds_path_market_offset_selection_conformal_discrete_ev_v10",
            "odds_path_role_integrated_multihorizon_v11",
            "odds_path_role_integrated_t300_nonlinear_v12",
            "odds_path_role_integrated_edge_conditional_lcb_v13",
            "odds_path_role_integrated_registered_band_lcb_v14",
            "odds_path_role_integrated_selection_free_envelope_v15",
            "odds_path_role_integrated_fixed_band_passthrough_v16",
        }:
            raise ValueError("unsupported market calibrator_strategy")
        command = [
            str(python), "-m", "boatrace_ai.listwise.market_calibration",
            "--db", db,
            "--model", str(model_input),
            "--output", str(output),
            "--scored-cache", str(scored_cache),
            "--from-date", from_date,
            "--daily-budget-yen", str(
                _integer(params, "daily_budget_yen", 10000, 100, 1000000)
            ),
            "--min-calibration-days", str(
                _integer(params, "min_calibration_days", 2, 1, 365)
            ),
            "--closing-odds-min-training-days", str(
                _integer(params, "closing_odds_min_training_days", 7, 2, 30)
            ),
            "--closing-odds-min-training-races", str(
                _integer(params, "closing_odds_min_training_races", 500, 100, 5000)
            ),
            "--calibrator-strategy", strategy,
            "--minimum-day-coverage", str(
                _number(params, "minimum_day_coverage", 1.0, 0.5, 1.0)
            ),
        ]
        if v25_probability_artifact is not None:
            command.extend([
                "--v25-probability-artifact",
                str(v25_probability_artifact),
            ])
        if baseline_model_input is not None and candidate_weight is not None:
            command.extend([
                "--baseline-model", str(baseline_model_input),
                "--candidate-weight", str(candidate_weight),
            ])
        if strategy in {
            "odds_path_role_integrated_t300_nonlinear_v12",
            "odds_path_role_integrated_edge_conditional_lcb_v13",
            "odds_path_role_integrated_registered_band_lcb_v14",
            "odds_path_role_integrated_selection_free_envelope_v15",
            "odds_path_role_integrated_fixed_band_passthrough_v16",
        }:
            fallback_policy = str(
                params.get("v12_closing_fallback_policy", "v11")
            )
            if fallback_policy not in {"v11", "no_bet"}:
                raise ValueError("unsupported v12 closing fallback policy")
            if (
                strategy
                == "odds_path_role_integrated_fixed_band_passthrough_v16"
                and fallback_policy != "no_bet"
            ):
                raise ValueError("V16 requires v12_closing_fallback_policy=no_bet")
            command.extend([
                "--v12-closing-fallback-policy",
                fallback_policy,
            ])
        if through_date is not None:
            command.extend(["--through-date", through_date])
        return command, output
    if task_type in {"listwise_feature_search", "combined_feature_search"}:
        allowed = {
            "evaluation_date",
            "n_features",
            "epochs",
            "batch_races",
            "learning_rate",
            "loss_blend",
            "targets",
            "alphas",
            "feature_variants",
            "reuse_search_job_id",
            "include_decayed_history",
            "ev_threshold",
            "ev_thresholds",
            "timeout_seconds",
        }
        unsupported = set(params) - allowed
        if unsupported:
            raise ValueError(
                f"unsupported {task_type} parameters: "
                + ", ".join(sorted(unsupported))
            )
        _integer(params, "timeout_seconds", 21600, 300, 86400)
        n_features = _integer(params, "n_features", 4096, 1024, 32768)
        epochs = _integer(params, "epochs", 2, 1, 6)
        batch_races = _integer(params, "batch_races", 1000, 250, 5000)
        learning_rate = _number(params, "learning_rate", 0.02, 0.001, 0.2)
        loss_blend = (
            _number(params, "loss_blend", 0.0, 0.0, 1.0)
            if params.get("loss_blend") is not None
            else None
        )
        if task_type == "combined_feature_search" and loss_blend is not None:
            raise ValueError("combined_feature_search does not support loss_blend")
        include_decayed_history = params.get("include_decayed_history", False)
        if not isinstance(include_decayed_history, bool):
            raise ValueError("include_decayed_history must be a boolean")
        if task_type == "combined_feature_search" and include_decayed_history:
            raise ValueError(
                "include_decayed_history is supported only for listwise_feature_search"
            )
        targets = str(params.get("targets", "winner,top3_pl"))
        if targets not in {"winner", "top3_pl", "winner,top3_pl"}:
            raise ValueError("unsupported targets")
        selected_feature_variants = None
        if params.get("feature_variants"):
            if task_type == "combined_feature_search":
                from .listwise.combined_feature_search import (
                    COMBINED_FEATURE_VARIANTS,
                    RESEARCH_PARTITION_FEATURE_VARIANTS,
                )

                available_variants = {
                    name
                    for name, _drops in (
                        *COMBINED_FEATURE_VARIANTS,
                        *RESEARCH_PARTITION_FEATURE_VARIANTS,
                    )
                }
                variant_names = tuple(dict.fromkeys(
                    item.strip()
                    for item in str(params["feature_variants"]).split(",")
                    if item.strip()
                ))
                if not variant_names or any(
                    name not in available_variants for name in variant_names
                ):
                    raise ValueError("unsupported feature_variants")
            else:
                from .listwise.feature_search import parse_feature_variants

                try:
                    parsed_variants = parse_feature_variants(
                        str(params["feature_variants"])
                    )
                except argparse.ArgumentTypeError as exc:
                    raise ValueError("unsupported feature_variants") from exc
                if parsed_variants is None:
                    raise ValueError("unsupported feature_variants")
                variant_names = tuple(name for name, _drops in parsed_variants)
            selected_feature_variants = ",".join(variant_names)
        reuse_search_job_id = None
        if params.get("reuse_search_job_id") is not None:
            if task_type != "listwise_feature_search":
                raise ValueError(
                    "reuse_search_job_id is supported only for listwise_feature_search"
                )
            reuse_search_job_id = _integer(
                params, "reuse_search_job_id", 0, 1, 2_147_483_647
            )
        alpha_values = [
            float(value) for value in str(
                params.get("alphas", "0.00001,0.0001")
            ).split(",") if value.strip()
        ]
        if not 1 <= len(alpha_values) <= 4 or not all(
            math.isfinite(value) and 1e-7 <= value <= 1e-2
            for value in alpha_values
        ):
            raise ValueError("alphas must contain 1-4 values between 1e-7 and 1e-2")
        alphas = ",".join(f"{value:.12g}" for value in alpha_values)
        evaluation_date = _date(params, "evaluation_date")
        cache_root = Path("/tmp/boatrace-evaluation") / f"job-{job_id:08d}"
        combined = task_type == "combined_feature_search"
        search_cache = cache_root / ("combined" if combined else "search")
        selected_suffix = "-combined" if combined else ""
        selected_cache = (
            app_root / "data" / "models" / "evaluation_cache"
            / f"job-{job_id:08d}{selected_suffix}"
        )
        module = (
            "boatrace_ai.listwise.combined_feature_search"
            if combined
            else "boatrace_ai.listwise.feature_search"
        )
        command = [
            str(python), "-m", module,
            "--db", db,
            "--output", str(output),
            "--cache-dir", str(search_cache),
            "--cache-write-mode", "never",
            "--selected-cache-dir", str(selected_cache),
            "--variant-workers", "1",
            "--candidate-workers", "2",
            "--as-of-date", evaluation_date,
            "--n-features", str(n_features),
            "--batch-races", str(batch_races),
            "--epochs", str(epochs),
            "--learning-rate", str(learning_rate),
            "--targets", targets,
            "--alphas", alphas,
            "--daily-budget-yen", "10000",
        ]
        if include_decayed_history:
            command.append("--include-decayed-history")
        if reuse_search_job_id is not None:
            reuse_output = (
                app_root / "data" / "models" / "evaluation_queue"
                / f"job-{reuse_search_job_id:08d}.json"
            )
            command.extend(["--reuse-search-output", str(reuse_output)])
        if loss_blend is not None:
            command.extend(["--loss-blend", str(loss_blend)])
        if selected_feature_variants:
            variant_option = (
                "--combined-feature-variants"
                if combined
                else "--feature-variants"
            )
            command.extend([variant_option, selected_feature_variants])
        if params.get("ev_thresholds"):
            thresholds = [
                float(value) for value in str(params["ev_thresholds"]).split(",")
                if value.strip()
            ]
            if not 1 <= len(thresholds) <= 10 or not all(
                0.8 <= value <= 3.0 for value in thresholds
            ):
                raise ValueError("ev_thresholds must contain 1-10 values between 0.8 and 3.0")
            command.extend([
                "--ev-thresholds", ",".join(f"{value:.6g}" for value in thresholds)
            ])
        else:
            command.extend([
                "--ev-threshold",
                str(_number(params, "ev_threshold", 1.2, 1.0, 3.0)),
            ])
        return command, output
    if task_type == "bankroll_policy_search":
        allowed = {
            "source_job_id", "source_kind", "learning_rate", "epochs", "batch_races",
            "candidate_count", "finalists", "bootstrap_samples",
            "payout_prior_weights", "evaluation_days", "research_only", "seed", "timeout_seconds",
            "coefficient_optimizer", "max_newton_iterations",
            "max_cg_iterations", "gradient_tolerance", "cg_tolerance",
            "ev_calibration_mode", "calibration_fraction",
            "calibration_bootstrap_samples",
        }
        unsupported = set(params) - allowed
        if unsupported:
            raise ValueError(
                "unsupported bankroll_policy_search parameters: "
                + ", ".join(sorted(unsupported))
            )
        source_kind = params.get("source_kind")
        if source_kind is None:
            source_job_id = _integer(
                params, "source_job_id", 0, 1, 9_999_999_999
            )
            source_result = (
                app_root / "data/models/evaluation_queue"
                / f"job-{source_job_id:08d}.json"
            ).resolve()
            if not source_result.is_file():
                raise JobDependencyUnavailable(
                    f"bankroll source result is not available yet: {source_result}"
                )
            source_payload = json.loads(
                source_result.read_text(encoding="utf-8")
            )
            cache_value = source_payload.get("selected_cache_prefix")
            if not cache_value:
                raise ValueError(
                    "bankroll source result lacks selected_cache_prefix"
                )
            cache_prefix = Path(str(cache_value)).resolve()
            cache_root = (
                app_root / "data/models/evaluation_cache"
            ).resolve()
            if cache_root not in cache_prefix.parents:
                raise ValueError(
                    "selected cache must be inside evaluation_cache"
                )
        elif source_kind == "standardized_selected":
            if "source_job_id" in params:
                raise ValueError(
                    "source_job_id cannot be combined with standardized_selected"
                )
            source_result = (
                app_root / "data" / "models" / "standardized_365d_v2"
                / "raw" / "listwise_feature_teacher.json"
            ).resolve()
            cache_prefix = _selected_standard_cache_prefix(app_root)
            source_payload = json.loads(
                source_result.read_text(encoding="utf-8")
            )
        else:
            raise ValueError(
                "source_kind must be standardized_selected when provided"
            )
        source_schema = source_payload.get("feature_schema_version")
        if source_schema not in SUPPORTED_LISTWISE_FEATURE_SCHEMA_VERSIONS:
            raise ObsoleteJob(
                "bankroll source feature schema is obsolete: "
                f"{source_schema} != {FEATURE_SCHEMA_VERSION}"
            )
        learning_rate = _number(
            params, "learning_rate", 0.02, 0.001, 0.2
        )
        epochs = _integer(params, "epochs", 2, 1, 6)
        batch_races = _integer(params, "batch_races", 1000, 250, 5000)
        coefficient_optimizer = str(
            params.get("coefficient_optimizer", "adam")
        )
        if coefficient_optimizer not in {"adam", "newton_cg"}:
            raise ValueError("unsupported coefficient_optimizer")
        max_newton_iterations = _integer(
            params, "max_newton_iterations", 10, 1, 30
        )
        max_cg_iterations = _integer(
            params, "max_cg_iterations", 75, 5, 300
        )
        gradient_tolerance = _number(
            params, "gradient_tolerance", 0.0001, 1e-8, 0.1
        )
        cg_tolerance = _number(
            params, "cg_tolerance", 0.001, 1e-8, 0.1
        )
        ev_calibration_mode = str(params.get("ev_calibration_mode", "none"))
        if ev_calibration_mode not in {
            "none", "contextual_point", "contextual_lcb95"
        }:
            raise ValueError("unsupported ev_calibration_mode")
        calibration_fraction = _number(
            params, "calibration_fraction", 0.50, 0.20, 0.80
        )
        calibration_bootstrap_samples = _integer(
            params, "calibration_bootstrap_samples", 2000, 100, 20000
        )
        candidate_count = _integer(
            params, "candidate_count", 24, 8, 128
        )
        finalists = _integer(params, "finalists", 6, 2, 16)
        if finalists > candidate_count:
            raise ValueError("finalists must not exceed candidate_count")
        bootstrap_samples = _integer(
            params, "bootstrap_samples", 20000, 100, 100000
        )
        seed = _integer(params, "seed", 20260726, 0, 2_147_483_647)
        evaluation_days = _integer(
            params, "evaluation_days", 365, 365, 365
        )
        _integer(params, "timeout_seconds", 43200, 300, 86400)
        research_only = params.get("research_only", False)
        if not isinstance(research_only, bool):
            raise ValueError("research_only must be a boolean")
        prior_values = [
            float(value) for value in str(
                params.get("payout_prior_weights", "10,30,100")
            ).split(",") if value.strip()
        ]
        if not 1 <= len(prior_values) <= 5 or not all(
            math.isfinite(value) and 1.0 <= value <= 1000.0
            for value in prior_values
        ):
            raise ValueError(
                "payout_prior_weights must contain 1-5 values in [1, 1000]"
            )
        return [
            str(python), "-m",
            "boatrace_ai.listwise.bankroll_policy_evaluation",
            "--db", db,
            "--search-result", str(source_result),
            "--cache-prefix", str(cache_prefix),
            "--output", str(output),
            "--learning-rate", str(learning_rate),
            "--epochs", str(epochs),
            "--batch-races", str(batch_races),
            "--coefficient-optimizer", coefficient_optimizer,
            "--max-newton-iterations", str(max_newton_iterations),
            "--max-cg-iterations", str(max_cg_iterations),
            "--gradient-tolerance", str(gradient_tolerance),
            "--cg-tolerance", str(cg_tolerance),
            "--ev-calibration-mode", ev_calibration_mode,
            "--calibration-fraction", str(calibration_fraction),
            "--calibration-bootstrap-samples",
            str(calibration_bootstrap_samples),
            "--candidate-count", str(candidate_count),
            "--finalists", str(finalists),
            "--bootstrap-samples", str(bootstrap_samples),
            "--payout-prior-weights", ",".join(
                f"{value:.12g}" for value in prior_values
            ),
            "--seed", str(seed),
            "--evaluation-days", str(evaluation_days),
            "--research-only", str(research_only).lower(),
        ], output
    if task_type == "bankroll_policy_nested_annual":
        allowed = {
            "source_job_id", "learning_rate", "epochs", "batch_races",
            "targets", "alphas", "candidate_count", "finalists",
            "selection_bootstrap_samples", "aggregate_bootstrap_samples",
            "selection_days", "outer_days", "embargo_days",
            "validation_fraction", "min_validation_races",
            "daily_budget_yen", "ev_threshold", "seed", "timeout_seconds",
        }
        unsupported = set(params) - allowed
        if unsupported:
            raise ValueError(
                "unsupported bankroll_policy_nested_annual parameters: "
                + ", ".join(sorted(unsupported))
            )
        source_job_id = _integer(
            params, "source_job_id", 0, 1, 9_999_999_999
        )
        if source_job_id == 3995:
            raise ObsoleteJob("legacy job 3995 cannot source nested evaluation")
        source_result = (
            app_root / "data/models/evaluation_queue"
            / f"job-{source_job_id:08d}.json"
        ).resolve()
        if not source_result.is_file():
            raise JobDependencyUnavailable(
                f"nested source result is not available yet: {source_result}"
            )
        source_payload = json.loads(source_result.read_text(encoding="utf-8"))
        source_schema = source_payload.get("feature_schema_version")
        if source_schema not in SUPPORTED_LISTWISE_FEATURE_SCHEMA_VERSIONS:
            raise ObsoleteJob(
                "nested source feature schema is obsolete: "
                f"{source_schema} != {FEATURE_SCHEMA_VERSION}"
            )
        cache_value = source_payload.get("selected_cache_prefix")
        if not cache_value:
            raise ValueError("nested source result lacks selected_cache_prefix")
        cache_prefix = Path(str(cache_value)).resolve()
        cache_root = (app_root / "data/models/evaluation_cache").resolve()
        if cache_root not in cache_prefix.parents:
            raise ValueError("nested selected cache must be inside evaluation_cache")
        if not cache_prefix.with_suffix(".manifest.json").is_file():
            raise JobDependencyUnavailable(
                f"nested selected cache is not available yet: {cache_prefix}"
            )

        targets = str(params.get("targets", "winner,top3_pl"))
        if targets not in {"winner", "top3_pl", "winner,top3_pl"}:
            raise ValueError("unsupported nested prediction targets")
        alpha_values = [
            float(value) for value in str(
                params.get("alphas", "0.00001,0.0001,0.001")
            ).split(",") if value.strip()
        ]
        if not 1 <= len(alpha_values) <= 4 or not all(
            math.isfinite(value) and 1e-7 <= value <= 1e-2
            for value in alpha_values
        ):
            raise ValueError("nested alphas must contain 1-4 values in [1e-7, 1e-2]")
        candidate_count = _integer(params, "candidate_count", 64, 8, 128)
        finalists = _integer(params, "finalists", 8, 2, 16)
        if finalists > candidate_count:
            raise ValueError("nested finalists must not exceed candidate_count")
        selection_days = _integer(params, "selection_days", 365, 365, 365)
        outer_days = _integer(params, "outer_days", 365, 365, 365)
        checkpoint_dir = (
            app_root / "data/models/evaluation_cache/nested_annual"
            / f"job-{job_id:08d}"
        )
        command = [
            str(python), "-m",
            "boatrace_ai.listwise.bankroll_policy_nested_evaluation",
            "--db", db,
            "--search-result", str(source_result),
            "--cache-prefix", str(cache_prefix),
            "--output", str(output),
            "--checkpoint-dir", str(checkpoint_dir),
            "--source-job-id", str(source_job_id),
            "--folds", "5",
            "--selection-days", str(selection_days),
            "--outer-days", str(outer_days),
            "--embargo-days", str(
                _integer(params, "embargo_days", 0, 0, 30)
            ),
            "--targets", targets,
            "--alphas", ",".join(f"{value:.12g}" for value in alpha_values),
            "--learning-rate", str(
                _number(params, "learning_rate", 0.02, 0.001, 0.2)
            ),
            "--epochs", str(_integer(params, "epochs", 2, 1, 6)),
            "--batch-races", str(
                _integer(params, "batch_races", 1000, 250, 5000)
            ),
            "--validation-fraction", str(
                _number(params, "validation_fraction", 0.2, 0.05, 0.4)
            ),
            "--min-validation-races", str(
                _integer(params, "min_validation_races", 1000, 500, 15000)
            ),
            "--daily-budget-yen", str(
                _integer(params, "daily_budget_yen", 10000, 100, 1000000)
            ),
            "--ev-threshold", str(
                _number(params, "ev_threshold", 1.2, 1.0, 3.0)
            ),
            "--candidate-count", str(candidate_count),
            "--finalists", str(finalists),
            "--selection-bootstrap-samples", str(
                _integer(
                    params, "selection_bootstrap_samples", 20000, 100, 100000
                )
            ),
            "--aggregate-bootstrap-samples", str(
                _integer(
                    params, "aggregate_bootstrap_samples", 20000, 100, 100000
                )
            ),
            "--seed", str(
                _integer(params, "seed", 20260728, 0, 2_147_483_647)
            ),
        ]
        _integer(params, "timeout_seconds", 86400, 300, 86400)
        return command, output
    if task_type == "conditional_payout_tail":
        allowed = {
            "training_through", "evaluation_from", "evaluation_through",
            "timeout_seconds",
        }
        unsupported = set(params) - allowed
        if unsupported:
            raise ValueError(
                "unsupported conditional_payout_tail parameters: "
                + ", ".join(sorted(unsupported))
            )
        required = allowed - {"timeout_seconds"}
        missing = required - set(params)
        if missing:
            raise ValueError(
                "missing conditional_payout_tail parameters: "
                + ", ".join(sorted(missing))
            )
        training_through = _date(params, "training_through")
        evaluation_from = _date(params, "evaluation_from")
        evaluation_through = _date(params, "evaluation_through")
        training_date = datetime.strptime(training_through, "%Y-%m-%d").date()
        evaluation_start = datetime.strptime(evaluation_from, "%Y-%m-%d").date()
        evaluation_end = datetime.strptime(evaluation_through, "%Y-%m-%d").date()
        if training_date + timedelta(days=1) != evaluation_start:
            raise ValueError(
                "conditional payout training and evaluation ranges must be adjacent"
            )
        if evaluation_start > evaluation_end:
            raise ValueError(
                "conditional payout evaluation dates must be chronological"
            )
        if evaluation_end - evaluation_start != timedelta(days=364):
            raise ValueError(
                "conditional payout evaluation range must be exactly 365 days"
            )
        _integer(params, "timeout_seconds", 21600, 300, 86400)
        cache_prefix = _selected_standard_cache_prefix(app_root)
        expected_contract = _standardized_holdout_contract(app_root)
        requested_contract = (
            training_through,
            evaluation_from,
            evaluation_through,
        )
        if requested_contract != expected_contract:
            raise ObsoleteJob(
                "conditional payout window does not match the current standardized "
                f"protocol: requested={requested_contract}; expected={expected_contract}"
            )
        baseline_model = (
            app_root / "data" / "models" / "standardized_365d_v2"
            / "listwise_newton.joblib"
        )
        return [
            str(python), "-m", "boatrace_ai.listwise.conditional_order",
            "--db", db,
            "--cache-prefix", str(cache_prefix),
            "--baseline-model", str(baseline_model),
            "--training-through", training_through,
            "--evaluation-from", evaluation_from,
            "--evaluation-through", evaluation_through,
            "--model-output", str(output.with_suffix(".joblib")),
            "--output", str(output),
            "--validation-days", "365",
            "--batch-races", "4000",
            "--payout-mean-corrections", "0.0", "0.5", "1.0",
            "--payout-threshold-candidates",
            "1.05", "1.10", "1.20", "1.30", "1.50", "2.00",
            "--promote-legacy-cache",
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
    if task_type == "persist_standard_selected_cache":
        allowed = {"artifact_mtime_after", "timeout_seconds"}
        unsupported = set(params) - allowed
        if unsupported:
            raise ValueError(
                "unsupported persist cache parameters: "
                + ", ".join(sorted(unsupported))
            )
        if "artifact_mtime_after" not in params:
            raise ValueError("artifact_mtime_after is required")
        artifact_mtime_after = _number(
            params, "artifact_mtime_after", 0.0, 1.0, 4_102_444_800.0
        )
        timeout_seconds = _integer(
            params, "timeout_seconds", 21600, 600, 86400
        )
        artifact = (
            app_root / "data/models/standardized_365d_v2/raw"
            / "listwise_feature_teacher.json"
        )
        destination = (
            app_root / "data/models/standardized_365d_v2/selected_cache"
        )
        return [
            str(python),
            str(app_root / "scripts/persist_selected_feature_cache.py"),
            "--artifact", str(artifact),
            "--destination-dir", str(destination),
            "--wait-for-mtime-after", str(artifact_mtime_after),
            "--wait-timeout-seconds", str(max(300, timeout_seconds - 300)),
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
    if task_type == "gdrive_model_cache_archive":
        allowed = {"paths", "timeout_seconds"}
        unsupported = set(params) - allowed
        if unsupported:
            raise ValueError(
                "unsupported model cache archive parameters: "
                + ", ".join(sorted(unsupported))
            )
        paths = params.get("paths")
        if not isinstance(paths, list) or not paths:
            raise ValueError("model cache archive paths must be a non-empty list")
        model_root = (app_root / "data" / "models").resolve()
        resolved_paths: list[Path] = []
        for value in paths:
            if not isinstance(value, str) or not value:
                raise ValueError("model cache archive paths must contain strings")
            candidate = (app_root / value).resolve()
            if model_root not in candidate.parents:
                raise ValueError("model cache archive path must be inside data/models")
            resolved_paths.append(candidate)
        return [
            str(python), "-m", "boatrace_ai.maintenance_tasks",
            "backup-model-cache", "--app-root", str(app_root),
            "--output", str(output),
            *[item for path in resolved_paths for item in ("--path", str(path))],
        ], output
    if task_type == "repository_hygiene":
        return [
            str(python), "-m", "boatrace_ai.maintenance_tasks",
            "repository-hygiene", "--app-root", str(app_root),
            "--output", str(output),
        ], output
    if task_type == "repository_sync":
        return [
            str(python), "-m", "boatrace_ai.maintenance_tasks",
            "repository-sync", "--db", db,
            "--app-root", str(app_root), "--output", str(output),
        ], output
    if task_type == "series_feature_cache":
        return [
            str(python), "-m", "boatrace_ai.cache_entry_series_features",
            "--db", db,
            "--batch-size", "1000",
            "--from-date", _date(params, "from_date"),
            "--output", str(output),
        ], output
    if task_type == "archive_closing_backfill":
        allowed = {
            "from_date", "through_date", "sleep_seconds", "max_pages",
            "timeout_seconds", "source",
        }
        unsupported = set(params) - allowed
        if unsupported:
            raise ValueError(
                "unsupported archive closing backfill parameters: "
                + ", ".join(sorted(unsupported))
            )
        from_date = _date(params, "from_date")
        through_date = _date(params, "through_date")
        if from_date > through_date:
            raise ValueError("archive closing from_date must not exceed through_date")
        sleep_seconds = _number(params, "sleep_seconds", 1.0, 0.5, 60.0)
        _integer(params, "timeout_seconds", 86400, 600, 86400)
        source = str(params.get("source") or "mirror")
        if source not in {"mirror", "official"}:
            raise ValueError("archive closing source must be mirror or official")
        module = (
            "boatrace_ai.official_closing_odds"
            if source == "official"
            else "boatrace_ai.archive_closing_odds"
        )
        command = [
            str(python), "-m", module,
            "--db", db,
            "--from-date", from_date,
            "--through-date", through_date,
            "--sleep-seconds", str(sleep_seconds),
            "--output", str(output),
        ]
        if params.get("max_pages") is not None:
            command.extend([
                "--max-pages",
                str(_integer(params, "max_pages", 1, 1, 100000)),
            ])
        return command, output
    if task_type == "archive_market_oracle":
        allowed = {
            "from_date", "through_date", "model_input", "daily_budget_yen",
            "timeout_seconds", "temporal_calibration_through",
        }
        unsupported = set(params) - allowed
        if unsupported:
            raise ValueError(
                "unsupported archive market oracle parameters: "
                + ", ".join(sorted(unsupported))
            )
        from_date = _date(params, "from_date")
        through_date = _date(params, "through_date")
        if from_date > through_date:
            raise ValueError("oracle from_date must not be after through_date")
        temporal_calibration_through = params.get(
            "temporal_calibration_through"
        )
        if temporal_calibration_through is not None:
            temporal_calibration_through = _date(
                params, "temporal_calibration_through"
            )
            if not from_date <= temporal_calibration_through < through_date:
                raise ValueError(
                    "oracle temporal cutoff must be inside evaluation period"
                )
        model_input = (app_root / str(params["model_input"])).resolve()
        model_root = (app_root / "data" / "models").resolve()
        if model_root not in model_input.parents:
            raise ValueError("oracle model_input must be inside data/models")
        daily_budget = _integer(params, "daily_budget_yen", 10000, 100, 10000000)
        _integer(params, "timeout_seconds", 43200, 600, 172800)
        command = [
            str(python), "-m", "boatrace_ai.listwise.archive_market_oracle",
            "--db", db,
            "--model", str(model_input),
            "--from-date", from_date,
            "--through-date", through_date,
            "--daily-budget-yen", str(daily_budget),
            "--output", str(output),
        ]
        if temporal_calibration_through is not None:
            command.extend([
                "--temporal-calibration-through", temporal_calibration_through,
            ])
        return command, output
    if task_type == "listwise_cutoff_refit":
        allowed = {
            "source_model", "training_cutoff", "evaluation_from",
            "evaluation_through", "batch_races", "max_newton_iterations",
            "max_cg_iterations", "gradient_tolerance", "cg_tolerance",
            "timeout_seconds",
        }
        unsupported = set(params) - allowed
        if unsupported:
            raise ValueError(
                "unsupported listwise cutoff refit parameters: "
                + ", ".join(sorted(unsupported))
            )
        source_model = (app_root / str(params["source_model"])).resolve()
        model_root = (app_root / "data" / "models").resolve()
        if model_root not in source_model.parents:
            raise ValueError("cutoff refit source_model must be inside data/models")
        training_cutoff = _date(params, "training_cutoff")
        evaluation_from = _date(params, "evaluation_from")
        evaluation_through = _date(params, "evaluation_through")
        if not training_cutoff < evaluation_from <= evaluation_through:
            raise ValueError(
                "cutoff refit dates must satisfy training_cutoff < "
                "evaluation_from <= evaluation_through"
            )
        batch_races = _integer(params, "batch_races", 2000, 100, 10000)
        max_newton_iterations = _integer(
            params, "max_newton_iterations", 5, 1, 20
        )
        max_cg_iterations = _integer(params, "max_cg_iterations", 30, 1, 200)
        gradient_tolerance = _number(
            params, "gradient_tolerance", 1e-4, 1e-8, 1.0
        )
        cg_tolerance = _number(params, "cg_tolerance", 1e-3, 1e-8, 1.0)
        _integer(params, "timeout_seconds", 43200, 600, 172800)
        model_output = output.with_suffix(".joblib")
        cache_dir = app_root / "data" / "models" / "evaluation_cache" / (
            f"job-{int(job['job_id']):08d}-cutoff"
        )
        return [
            str(python), "-m", "boatrace_ai.listwise.cutoff_refit",
            "--db", db,
            "--source-model", str(source_model),
            "--model-output", str(model_output),
            "--output", str(output),
            "--cache-dir", str(cache_dir),
            "--training-cutoff", training_cutoff,
            "--evaluation-from", evaluation_from,
            "--evaluation-through", evaluation_through,
            "--batch-races", str(batch_races),
            "--max-newton-iterations", str(max_newton_iterations),
            "--max-cg-iterations", str(max_cg_iterations),
            "--gradient-tolerance", str(gradient_tolerance),
            "--cg-tolerance", str(cg_tolerance),
        ], output
    if task_type == "listwise_newton_refine":
        search_result = app_root / str(params["search_result"])
        if app_root not in search_result.resolve().parents:
            raise ValueError("search_result must be inside app root")
        try:
            search_payload = json.loads(search_result.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise JobDependencyUnavailable(
                f"feature search result is not available yet: {search_result}"
            ) from exc
        search_schema = str(search_payload.get("feature_schema_version") or "")
        if search_schema not in SUPPORTED_LISTWISE_FEATURE_SCHEMA_VERSIONS:
            raise ObsoleteJob(
                f"feature schema {search_schema or 'missing'} is obsolete; "
                f"current={FEATURE_SCHEMA_VERSION}"
            )
        model_output = output.with_suffix(".joblib")
        cache = Path(str(params.get("cache_dir") or "/tmp/boatrace-evaluation/newton"))
        if "ev_threshold" in params:
            ev_threshold = _number(params, "ev_threshold", 1.2, 1.0, 3.0)
        else:
            ev_threshold = float(
                (search_payload.get("policy") or {}).get("ev_threshold") or 1.2
            )
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
            "--ev-threshold", str(ev_threshold),
        ], output
    raise ValueError(f"unsupported task_type: {task_type}")


METRIC_KEYS = (
    "cached", "evaluated_races", "evaluation_races", "evaluation_days", "entry_log_loss",
    "comparison_role", "coefficient_optimizer", "ev_calibration_mode",
    "ev_calibration_usage", "evaluation_from", "evaluation_through",
    "selection_races", "holdout_races",
    "comparison_role", "coefficient_optimizer", "ev_calibration_mode",
    "ev_calibration_usage", "evaluation_from", "evaluation_through",
    "selection_races", "holdout_races",
    "entry_brier", "winner_log_loss", "trifecta_log_loss",
    "calibrated_trifecta_log_loss", "model_trifecta_log_loss",
    "market_trifecta_log_loss", "calibrated_trifecta_top5_hit_rate",
    "winner_top1_accuracy", "trifecta_top1_hit_rate",
    "trifecta_top5_hit_rate",
    "roi", "profit_yen", "stake_yen", "return_yen", "max_drawdown_yen",
    "roi_ci95_lower", "roi_ci95_upper", "probability_roi_above_one",
    "profit_ci95_lower_yen", "profit_ci95_upper_yen",
    "selected_races", "hit_races", "tickets", "hit_tickets",
    "race_selection_rate", "avg_tickets_per_selected_race",
    "ticket_hit_rate", "ticket_hit_rate_ci95_lower", "ticket_hit_rate_ci95_upper",
    "benchmark_target_days", "benchmark_days", "benchmark_status",
    "benchmark_population_races", "benchmark_payout_races",
    "benchmark_odds_eligible_races", "benchmark_missing_odds_races",
    "benchmark_odds_coverage", "benchmark_evaluated_races",
    "benchmark_evaluation_coverage", "population_race_selection_rate",
    "race_hit_rate", "race_hit_rate_ci95_lower", "race_hit_rate_ci95_upper",
    "largest_hit_return_share", "effective_hit_count", "roi_without_largest_hit",
    "profit_without_largest_hit_yen", "largest_hit_excluded_roi",
    "closing_odds_log_mae", "baseline_closing_odds_log_mae",
    "closing_odds_rank_correlation", "closing_odds_interval_coverage",
    "closing_snapshot_age_seconds", "closing_snapshot_age_seconds_p90",
    "closing_q20_pinball_loss", "closing_q20_lower_coverage",
    "closing_q20_target_coverage", "closing_q20_evaluation_races",
    "daily_cluster_bootstrap_roi_lower_95",
    "promotion_eligible", "prediction_deployment_eligible",
    "deployment_model_artifact_saved", "incremental_confidence_pass", "converged",
    "gradient_norm", "elapsed_seconds", "source_files_before", "source_files_after",
    "source_bytes_before", "source_bytes_after", "archived_files_removed",
    "archived_bytes_removed", "staging_files", "action", "ahead", "behind",
    "active_evaluations",
)


def summarize_result(payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    def visit(value: Any, depth: int = 0) -> None:
        if depth > 5 or not isinstance(value, dict):
            return
        tail_diagnostics = value.get("tail_portfolio_diagnostics")
        if (
            "tail_portfolio_diagnostics" not in summary
            and isinstance(tail_diagnostics, dict)
        ):
            summary["tail_portfolio_diagnostics"] = tail_diagnostics
        for key in METRIC_KEYS:
            if key in value and key not in summary and not isinstance(value[key], (dict, list)):
                summary[key] = value[key]
        for key in (
            "metrics", "holdout", "holdout_after_newton", "bankroll",
            "holdout_prediction_metrics", "selection_prediction_metrics",
            "bankroll_confidence",
            "closing_odds_forecast",
            "conditional_order", "venue_conditional_order",
            "momentum_newton_residual",
        ):
            if key in value:
                visit(value[key], depth + 1)

    visit(payload)
    apply_archive_residual_summary(payload, summary)
    chronological = payload.get("chronological_bankroll")
    if isinstance(chronological, dict):
        summary["primary_bankroll"] = (
            "chronological"
            if chronological.get("primary_promotion_bankroll")
            else summary.get("primary_bankroll", "legacy")
        )
        for key in (
            "race_days",
            "evaluated_races",
            "tickets",
            "hit_tickets",
            "stake_yen",
            "return_yen",
            "profit_yen",
            "roi",
            "max_drawdown_yen",
            "winning_days",
            "profitable_day_fraction",
            "roi_without_largest_hit",
            "largest_hit_return_share",
            "effective_hit_count",
            "daily_cluster_bootstrap_roi_lower_95",
            "bootstrap_probability_roi_above_one",
            "normalized_drawdown",
            "daily_stake_limit_fraction",
        ):
            value = chronological.get(key)
            if not isinstance(value, (dict, list)):
                summary[f"chronological_{key}"] = value
    purchase_value = payload.get("purchase_value_diagnostics")
    if isinstance(purchase_value, dict):
        summary["purchase_value_diagnostics"] = dict(purchase_value)
        for key in (
            "tickets", "predicted_mean", "observed_mean",
            "pearson_correlation", "calibration_mae",
            "positive_predicted_tickets", "positive_predicted_fraction",
            "positive_predicted_mean", "positive_observed_capped_roi",
        ):
            if key in purchase_value:
                summary[f"purchase_value_{key}"] = purchase_value[key]
    purchase_diagnostics = payload.get("purchase_decision_diagnostics")
    if isinstance(purchase_diagnostics, dict):
        preserved = dict(purchase_diagnostics)
        summary["purchase_decision_diagnostics"] = preserved
        for key in (
            "threshold_pass_candidates",
            "candidates_after_race_cap",
            "candidates_before_allocation",
            "allocation_candidate_tickets",
            "purchases_after_allocation",
            "zero_purchase_days",
            "zero_reason_counts",
            "raw_selected_candidates",
            "guarded_threshold_candidates",
            "safe_ev_max",
            "safe_ev_p95",
            "safe_ev_p99",
        ):
            if key in preserved:
                summary[key] = preserved[key]
    selection_conformal = payload.get("selection_conformal")
    if isinstance(selection_conformal, dict):
        preserved = dict(selection_conformal)
        summary["selection_conformal"] = preserved
        for key in (
            "selection_evaluation_candidates",
            "selection_raw_covered_candidates",
            "selection_guarded_covered_candidates",
            "selection_raw_closing_coverage",
            "selection_guarded_closing_coverage",
            "selection_closing_ratio_mean",
            "selection_closing_ratio_p10",
            "selection_closing_ratio_median",
            "haircut_latest",
            "haircut_min",
            "haircut_max",
            "training_days_latest",
            "training_candidates_latest",
            "trained_through_date_latest",
        ):
            if key in preserved:
                summary[key] = preserved[key]
    closing_identity = payload.get("closing_model_identity")
    if isinstance(closing_identity, dict):
        preserved_identity = dict(closing_identity)
        summary["closing_model_identity"] = preserved_identity
        summary["closing_model_requested"] = preserved_identity.get(
            "requested_model"
        )
        summary["closing_model_selected"] = preserved_identity.get(
            "selected_model_latest",
            preserved_identity.get("selected_model"),
        )
        summary["closing_fallback_policy"] = preserved_identity.get(
            "fallback_policy"
        )
        for key in (
            "evaluation_folds",
            "v12_ready_folds",
            "v12_adopted_folds",
            "v11_fallback_folds",
            "no_bet_folds",
        ):
            if key in preserved_identity:
                summary[f"closing_{key}"] = preserved_identity[key]
    if payload.get("source_role") in {
        "secondary_archive_candidate_unverified",
        "secondary_archive_research_only",
        "primary_official_historical_closing",
    }:
        for key in (
            "targets", "stored", "not_found", "invalid", "fetch_failed",
            "remaining", "from_date", "through_date", "source_key",
            "source_role",
        ):
            if key in payload:
                summary[f"archive_{key}"] = payload[key]
    top5_flat = payload.get("holdout_top5_flat_diagnostic")
    if isinstance(top5_flat, dict):
        for key in (
            "evaluated_races", "tickets", "hit_races", "hit_rate",
            "stake_yen", "return_yen", "profit_yen", "roi",
            "average_hit_payout_yen", "breakeven_average_hit_payout_yen",
        ):
            if key in top5_flat:
                summary[f"top5_flat_{key}"] = top5_flat[key]
    ev_calibration = payload.get("holdout_candidate_ev_calibration")
    if isinstance(ev_calibration, list):
        clean_bins = [
            {key: row.get(key) for key in (
                "lower_inclusive", "upper_exclusive", "tickets", "hits",
                "realized_roi", "mean_estimated_ev",
                "realized_to_estimated_ratio",
            )}
            for row in ev_calibration if isinstance(row, dict)
        ]
        summary["holdout_candidate_ev_calibration"] = clean_bins
        high_bins = [
            row for row in ev_calibration
            if isinstance(row, dict)
            and float(row.get("lower_inclusive") or 0.0) >= 2.0
        ]
        high_stake = sum(int(row.get("flat_stake_yen") or 0) for row in high_bins)
        high_return = sum(int(row.get("flat_return_yen") or 0) for row in high_bins)
        summary["high_ev_tickets"] = sum(
            int(row.get("tickets") or 0) for row in high_bins
        )
        summary["high_ev_realized_roi"] = (
            high_return / high_stake if high_stake else None
        )
    promotion_gate = payload.get("promotion_gate")
    if isinstance(promotion_gate, dict):
        checks = {
            str(key): bool(value)
            for key, value in promotion_gate.items()
            if isinstance(value, bool)
        }
        summary["promotion_gate_passed"] = sum(checks.values())
        summary["promotion_gate_total"] = len(checks)
        summary["promotion_gate_failed"] = [
            key for key, passed in checks.items() if not passed
        ]
    prediction_deployment_gate = payload.get("prediction_deployment_gate")
    if isinstance(prediction_deployment_gate, dict):
        prediction_checks = {
            str(key): bool(value)
            for key, value in prediction_deployment_gate.items()
            if key != "pass" and isinstance(value, bool)
        }
        summary["prediction_deployment_gate_passed"] = sum(
            prediction_checks.values()
        )
        summary["prediction_deployment_gate_total"] = len(prediction_checks)
        summary["prediction_deployment_gate_failed"] = [
            key for key, passed in prediction_checks.items() if not passed
        ]
    holdout_stability = payload.get("holdout_temporal_stability")
    if isinstance(holdout_stability, dict):
        summary["holdout_temporal_minimum_roi"] = (
            holdout_stability.get("minimum_roi")
        )
        summary["holdout_temporal_fold_rois"] = [
            row.get("roi")
            for row in (holdout_stability.get("folds") or [])
            if isinstance(row, dict) and row.get("roi") is not None
        ]
    registered_policy = payload.get("registered_ev_band_walk_forward")
    if isinstance(registered_policy, dict):
        for key in (
            "status",
            "registered_after",
            "evaluation_days",
            "evaluated_races",
            "tickets",
            "hit_tickets",
            "stake_yen",
            "return_yen",
            "roi",
            "profit_yen",
            "winning_days",
            "profitable_day_fraction",
            "largest_hit_return_share",
            "effective_hit_count",
            "roi_without_largest_hit",
            "daily_cluster_bootstrap_roi_lower_95",
            "probability_roi_above_one",
        ):
            if key in registered_policy:
                summary[f"registered_ev_band_{key}"] = registered_policy[key]
    prospective_policy = payload.get("prospective_normalized_ev_walk_forward")
    if isinstance(prospective_policy, dict):
        for key in (
            "status",
            "registered_after",
            "evaluation_days",
            "evaluated_races",
            "tickets",
            "hit_tickets",
            "stake_yen",
            "return_yen",
            "roi",
            "profit_yen",
            "winning_days",
            "profitable_day_fraction",
            "largest_hit_return_share",
            "effective_hit_count",
            "roi_without_largest_hit",
            "daily_cluster_bootstrap_roi_lower_95",
            "probability_roi_above_one",
        ):
            if key in prospective_policy:
                summary[f"prospective_normalized_ev_{key}"] = prospective_policy[key]
    prospective_top5 = payload.get("prospective_top5_narrow_ev_walk_forward")
    if isinstance(prospective_top5, dict):
        for key in (
            "status",
            "registered_after",
            "evaluation_days",
            "evaluated_races",
            "tickets",
            "hit_tickets",
            "stake_yen",
            "return_yen",
            "roi",
            "profit_yen",
            "winning_days",
            "profitable_day_fraction",
            "largest_hit_return_share",
            "effective_hit_count",
            "roi_without_largest_hit",
            "daily_cluster_bootstrap_roi_lower_95",
            "probability_roi_above_one",
        ):
            if key in prospective_top5:
                summary[f"prospective_top5_narrow_ev_{key}"] = (
                    prospective_top5[key]
                )
    retrospective_top5 = payload.get("top5_narrow_retrospective_diagnostic")
    if isinstance(retrospective_top5, dict):
        for key in (
            "status",
            "evaluation_days",
            "evaluated_races",
            "tickets",
            "hit_tickets",
            "roi",
            "profit_yen",
            "roi_without_largest_hit",
            "effective_hit_count",
            "promotion_evidence",
        ):
            if key in retrospective_top5:
                summary[f"top5_narrow_retrospective_{key}"] = (
                    retrospective_top5[key]
                )
    v33_forecast_diagnostic = payload.get(
        "v33_v25_top1_narrow_forecast_only_diagnostic"
    )
    if not isinstance(v33_forecast_diagnostic, dict):
        forecast_rows = []
        for fold in payload.get("folds") or []:
            if (
                not isinstance(fold, dict)
                or fold.get("closing_odds_policy_input")
                != "oof_forecast_final_from_real_t5"
            ):
                continue
            bankroll = fold.get(
                "v33_v25_top1_narrow_retrospective_bankroll"
            )
            if isinstance(bankroll, dict):
                forecast_rows.append(bankroll)
        if forecast_rows:
            stake_yen = sum(int(row.get("stake_yen") or 0) for row in forecast_rows)
            return_yen = sum(
                int(row.get("return_yen") or 0) for row in forecast_rows
            )
            largest_hit = max(
                int(row.get("largest_hit_return_yen") or 0)
                for row in forecast_rows
            )
            hit_square_sum = sum(
                int(row.get("hit_return_square_sum_yen2") or 0)
                for row in forecast_rows
            )
            hit_hhi = (
                hit_square_sum / (return_yen * return_yen)
                if return_yen else None
            )
            v33_forecast_diagnostic = {
                "status": "diagnostic_only_not_promotion_evidence",
                "evaluation_days": len(forecast_rows),
                "evaluated_races": sum(
                    int(row.get("evaluated_races") or 0) for row in forecast_rows
                ),
                "tickets": sum(int(row.get("tickets") or 0) for row in forecast_rows),
                "hit_tickets": sum(
                    int(row.get("hit_tickets") or 0) for row in forecast_rows
                ),
                "stake_yen": stake_yen,
                "return_yen": return_yen,
                "profit_yen": return_yen - stake_yen,
                "roi": return_yen / stake_yen if stake_yen else 0.0,
                "roi_without_largest_hit": (
                    (return_yen - largest_hit) / stake_yen
                    if stake_yen else None
                ),
                "effective_hit_count": 1.0 / hit_hhi if hit_hhi else None,
                "promotion_evidence": False,
            }
    for payload_key, prefix in (
        (
            "v33_v25_top1_narrow_retrospective_diagnostic",
            "v33_v25_top1_narrow_retrospective",
        ),
        (
            "v33_v25_top1_narrow_forecast_only_diagnostic",
            "v33_v25_top1_narrow_forecast_only",
        ),
        (
            "v33_v25_top1_narrow_prospective_walk_forward",
            "v33_v25_top1_narrow_prospective",
        ),
    ):
        diagnostic = (
            v33_forecast_diagnostic
            if payload_key == "v33_v25_top1_narrow_forecast_only_diagnostic"
            else payload.get(payload_key)
        )
        if not isinstance(diagnostic, dict):
            continue
        for key in (
            "status", "evaluation_days", "evaluated_races", "tickets",
            "hit_tickets", "stake_yen", "return_yen", "roi", "profit_yen",
            "roi_without_largest_hit", "effective_hit_count",
            "promotion_evidence",
        ):
            if key in diagnostic:
                summary[f"{prefix}_{key}"] = diagnostic[key]
    prospective_v4 = payload.get(
        "prospective_observed_closing_return_v4_walk_forward"
    )
    if isinstance(prospective_v4, dict):
        for key in (
            "status",
            "registered_after",
            "evaluation_days",
            "evaluated_races",
            "tickets",
            "hit_tickets",
            "roi",
            "profit_yen",
            "roi_without_largest_hit",
        ):
            if key in prospective_v4:
                summary[f"prospective_observed_closing_v4_{key}"] = (
                    prospective_v4[key]
                )
    prospective_v7 = payload.get(
        "prospective_crossfit_conservative_ev_v7_walk_forward"
    )
    if isinstance(prospective_v7, dict):
        for key in (
            "status",
            "registered_after",
            "evaluation_days",
            "evaluated_races",
            "tickets",
            "hit_tickets",
            "stake_yen",
            "return_yen",
            "profit_yen",
            "roi",
            "roi_without_largest_hit",
            "daily_cluster_bootstrap_roi_lower_95",
            "effective_hit_count",
            "largest_hit_return_share",
            "calibrated_trifecta_log_loss",
            "model_trifecta_log_loss",
            "market_trifecta_log_loss",
            "closing_q20_pinball_loss",
            "closing_q20_lower_coverage",
            "promotion_eligible",
        ):
            if key in prospective_v7:
                summary[f"prospective_v7_{key}"] = prospective_v7[key]
    prospective_v8 = payload.get(
        "prospective_market_offset_crossfit_conservative_ev_v8_walk_forward"
    )
    if isinstance(prospective_v8, dict):
        for key in (
            "status",
            "registered_after",
            "evaluation_days",
            "evaluated_races",
            "tickets",
            "hit_tickets",
            "stake_yen",
            "return_yen",
            "profit_yen",
            "roi",
            "roi_without_largest_hit",
            "daily_cluster_bootstrap_roi_lower_95",
            "effective_hit_count",
            "largest_hit_return_share",
            "calibrated_trifecta_log_loss",
            "model_trifecta_log_loss",
            "market_trifecta_log_loss",
            "closing_q20_pinball_loss",
            "closing_q20_lower_coverage",
            "promotion_eligible",
        ):
            if key in prospective_v8:
                summary[f"prospective_v8_{key}"] = prospective_v8[key]
    prospective_v9 = payload.get(
        "prospective_market_offset_discrete_log_ev_v9_walk_forward"
    )
    if isinstance(prospective_v9, dict):
        for key in (
            "status",
            "registered_after",
            "evaluation_days",
            "evaluated_races",
            "tickets",
            "hit_tickets",
            "stake_yen",
            "return_yen",
            "profit_yen",
            "roi",
            "roi_without_largest_hit",
            "daily_cluster_bootstrap_roi_lower_95",
            "effective_hit_count",
            "largest_hit_return_share",
            "calibrated_trifecta_log_loss",
            "model_trifecta_log_loss",
            "market_trifecta_log_loss",
            "closing_q20_pinball_loss",
            "closing_q20_lower_coverage",
            "promotion_eligible",
        ):
            if key in prospective_v9:
                summary[f"prospective_v9_{key}"] = prospective_v9[key]
    prospective_v10 = payload.get(
        "prospective_market_offset_selection_conformal_discrete_ev_v10_walk_forward"
    )
    if isinstance(prospective_v10, dict):
        for key in (
            "status",
            "registered_after",
            "evaluation_days",
            "evaluated_races",
            "tickets",
            "hit_tickets",
            "stake_yen",
            "return_yen",
            "profit_yen",
            "roi",
            "roi_without_largest_hit",
            "daily_cluster_bootstrap_roi_lower_95",
            "effective_hit_count",
            "largest_hit_return_share",
            "calibrated_trifecta_log_loss",
            "model_trifecta_log_loss",
            "market_trifecta_log_loss",
            "closing_q20_pinball_loss",
            "closing_q20_lower_coverage",
            "promotion_eligible",
        ):
            if key in prospective_v10:
                summary[f"prospective_v10_{key}"] = prospective_v10[key]
        conformal = prospective_v10.get("selection_conformal")
        if isinstance(conformal, dict):
            summary["prospective_v10_selection_conformal"] = dict(conformal)
    prospective_v12 = payload.get(
        "prospective_role_integrated_v12_walk_forward"
    )
    if isinstance(prospective_v12, dict):
        for key in (
            "status",
            "registered_after",
            "evaluation_days",
            "evaluated_races",
            "tickets",
            "hit_tickets",
            "stake_yen",
            "return_yen",
            "profit_yen",
            "roi",
            "roi_without_largest_hit",
            "daily_cluster_bootstrap_roi_lower_95",
            "effective_hit_count",
            "largest_hit_return_share",
            "calibrated_trifecta_log_loss",
            "model_trifecta_log_loss",
            "market_trifecta_log_loss",
            "closing_q20_pinball_loss",
            "closing_q20_lower_coverage",
            "promotion_eligible",
        ):
            if key in prospective_v12:
                summary[f"prospective_v12_{key}"] = prospective_v12[key]
        prospective_identity = prospective_v12.get("closing_model_identity")
        if isinstance(prospective_identity, dict):
            summary["prospective_v12_closing_model_identity"] = dict(
                prospective_identity
            )
    edge_calibration = payload.get("edge_conditional_probability_calibration")
    if isinstance(edge_calibration, dict):
        summary["edge_conditional_probability_calibration"] = dict(edge_calibration)
        for key in (
            "evaluation_days", "coverage_days", "daily_lower_bound_coverage",
            "candidate_count", "raw_expected_hits", "adjusted_expected_hits",
            "observed_hits", "raw_overprediction_hits",
            "adjusted_overprediction_hits", "overprediction_reduction_hits",
            "relative_overprediction_reduction",
            "observed_hits_to_adjusted_predicted_hits_ratio", "missing_t300_races",
        ):
            if key in edge_calibration:
                summary[f"edge_conditional_{key}"] = edge_calibration[key]
    divergence = payload.get("strict_prior_divergence_bands")
    if isinstance(divergence, dict):
        summary["strict_prior_divergence_bands"] = dict(divergence)
    prospective_v13 = payload.get("prospective_role_integrated_v13_walk_forward")
    if isinstance(prospective_v13, dict):
        for key in (
            "status", "registered_after", "evaluation_days", "evaluated_races",
            "tickets", "hit_tickets", "stake_yen", "return_yen", "profit_yen",
            "roi", "roi_without_largest_hit",
            "daily_cluster_bootstrap_roi_lower_95", "effective_hit_count",
            "largest_hit_return_share", "calibrated_trifecta_log_loss",
            "model_trifecta_log_loss", "market_trifecta_log_loss",
            "closing_q20_lower_coverage", "promotion_eligible",
        ):
            if key in prospective_v13:
                summary[f"prospective_v13_{key}"] = prospective_v13[key]
        for key in (
            "conditional_calibration", "strict_prior_divergence_bands",
            "closing_model_identity",
        ):
            if isinstance(prospective_v13.get(key), dict):
                summary[f"prospective_v13_{key}"] = dict(prospective_v13[key])
    selected_v14 = payload.get("selected_candidate_calibration")
    if isinstance(selected_v14, dict):
        summary["selected_candidate_calibration"] = dict(selected_v14)
        for key in (
            "evaluation_days", "candidate_count", "raw_predicted_hits",
            "adjusted_predicted_hits", "observed_hits",
            "observed_hits_to_raw_predicted_hits_ratio",
            "observed_hits_to_adjusted_predicted_hits_ratio",
            "adjusted_predicted_hits_to_observed_hits_ratio",
            "day_bootstrap_observed_to_adjusted_predicted_ratio_lower_95",
            "candidate_binary_brier_score", "candidate_binary_log_loss",
            "inconsistent_t300_snapshot_races",
        ):
            if key in selected_v14:
                summary[f"v14_calibration_{key}"] = selected_v14[key]
    prospective_v14 = payload.get("prospective_role_integrated_v14_walk_forward")
    if isinstance(prospective_v14, dict):
        for key in (
            "status", "registered_after", "evaluation_days", "evaluated_races",
            "tickets", "hit_tickets", "stake_yen", "return_yen", "profit_yen",
            "roi", "roi_without_largest_hit", "profit_without_largest_hit_yen",
            "daily_cluster_bootstrap_roi_lower_95", "promotion_eligible",
        ):
            if key in prospective_v14:
                summary[f"prospective_v14_{key}"] = prospective_v14[key]
        for key in (
            "selected_candidate_calibration", "strict_prior_divergence_bands",
            "closing_model_identity", "promotion_gate",
        ):
            if isinstance(prospective_v14.get(key), dict):
                summary[f"prospective_v14_{key}"] = dict(prospective_v14[key])
    closing_envelope_v15 = payload.get("closing_envelope_conformal")
    if isinstance(closing_envelope_v15, dict):
        summary["closing_envelope_conformal"] = dict(closing_envelope_v15)
    fixed_band_ranking_diagnostics = payload.get(
        "fixed_band_ranking_diagnostics"
    )
    if isinstance(fixed_band_ranking_diagnostics, dict):
        summary["fixed_band_ranking_diagnostics"] = dict(
            fixed_band_ranking_diagnostics
        )
    prospective_v15 = payload.get("prospective_role_integrated_v15_walk_forward")
    if isinstance(prospective_v15, dict):
        for key in (
            "status", "registered_after", "evaluation_days", "evaluated_races",
            "tickets", "hit_tickets", "stake_yen", "return_yen", "profit_yen",
            "roi", "roi_without_largest_hit", "profit_without_largest_hit_yen",
            "daily_cluster_bootstrap_roi_lower_95", "effective_hit_count",
            "largest_hit_return_share", "max_drawdown_yen",
            "selected_races", "hit_races", "profitable_days",
            "profitable_day_fraction", "race_selection_rate",
            "promotion_eligible",
        ):
            if key in prospective_v15:
                summary[f"prospective_v15_{key}"] = prospective_v15[key]
        for key in ("closing_envelope_conformal", "promotion_gate"):
            if isinstance(prospective_v15.get(key), dict):
                summary[f"prospective_v15_{key}"] = dict(prospective_v15[key])
    prospective_v16 = payload.get("prospective_role_integrated_v16_walk_forward")
    if isinstance(prospective_v16, dict):
        for key in (
            "status", "registered_after", "evaluation_days", "evaluated_races",
            "tickets", "hit_tickets", "stake_yen", "return_yen", "profit_yen",
            "roi", "roi_without_largest_hit", "profit_without_largest_hit_yen",
            "daily_cluster_bootstrap_roi_lower_95", "effective_hit_count",
            "largest_hit_return_share", "max_drawdown_yen",
            "selected_races", "hit_races", "profitable_days",
            "profitable_day_fraction", "race_selection_rate",
            "promotion_eligible",
        ):
            if key in prospective_v16:
                summary[f"prospective_v16_{key}"] = prospective_v16[key]
        for key in ("closing_envelope_conformal", "promotion_gate"):
            if isinstance(prospective_v16.get(key), dict):
                summary[f"prospective_v16_{key}"] = dict(prospective_v16[key])
    empirical_policy = payload.get("empirical_lcb_walk_forward")
    if isinstance(empirical_policy, dict):
        for key in (
            "status",
            "evaluation_days",
            "evaluated_races",
            "calibration_ready_folds",
            "minimum_ready_evaluation_days",
            "minimum_tickets",
            "sample_size_pass",
            "eligible_days",
            "no_bet_days",
            "profitable_days",
            "tickets",
            "hit_tickets",
            "stake_yen",
            "return_yen",
            "profit_yen",
            "roi",
            "roi_without_largest_hit",
            "largest_hit_return_share",
            "max_drawdown_yen",
        ):
            if key in empirical_policy:
                summary[f"empirical_lcb_{key}"] = empirical_policy[key]
        empirical_tail = empirical_policy.get("tail_portfolio_diagnostics")
        if isinstance(empirical_tail, dict):
            summary["empirical_lcb_tail_portfolio_diagnostics"] = empirical_tail
            ordinary = empirical_tail.get("normal")
            if isinstance(ordinary, dict):
                summary["empirical_lcb_roi_lower95"] = ordinary.get(
                    "daily_cluster_bootstrap_roi_lower_95"
                )
    if payload.get("model") and str(payload.get("model")).startswith(
        "genetic_listwise_island_v"
    ):
        champion = payload.get("champion")
        if isinstance(champion, dict):
            champion_metrics = champion.get("metrics")
            if isinstance(champion_metrics, dict):
                visit(champion_metrics, 1)
            if champion.get("fitness") is not None:
                summary["genetic_fitness"] = champion["fitness"]
        summary["genetic_cohort"] = payload.get("cohort")
        summary["genetic_generation"] = payload.get("generation")
        summary["genetic_island_id"] = payload.get("island_id")
        summary["genetic_evaluated_individuals"] = (
            int(len(payload.get("history") or []))
            * int((payload.get("population_size") or 0))
        )
        summary["genetic_history"] = [
            {
                key: row.get(key)
                for key in (
                    "local_generation", "best_fitness", "min_fitness",
                    "q1_fitness", "median_fitness", "q3_fitness",
                    "max_fitness", "std_fitness", "mutation_rate",
                    "random_injections", "unique_genomes",
                )
                if row.get(key) is not None
            }
            for row in (payload.get("history") or [])
            if isinstance(row, dict)
        ]
    if payload.get("protocol_id") == "standard_365d_v2":
        models = payload.get("models")
        promotion = payload.get("promotion_decision")
        if isinstance(models, list) and isinstance(promotion, dict):
            model_rows = {
                str(row.get("model_id")): row
                for row in models
                if isinstance(row, dict) and row.get("model_id")
            }
            selected_id = str(promotion.get("selected_model_id") or "")
            selected = model_rows.get(selected_id)
            if selected is not None:
                visit(selected, 1)
                summary["model"] = selected_id
            incumbent_id = str(promotion.get("incumbent_model_id") or "")
            candidates = [
                row for model_id, row in model_rows.items()
                if model_id != incumbent_id and row.get("roi") is not None
            ]
            if candidates:
                best = max(candidates, key=lambda row: float(row["roi"]))
                summary["best_candidate_model"] = best.get("model_id")
                summary["best_candidate_roi"] = best.get("roi")
                summary["best_candidate_profit_yen"] = best.get("profit_yen")
            summary["comparison_ready"] = payload.get("comparison_ready")
            summary["valid_model_count"] = payload.get("valid_model_count")
            summary["promotion_eligible"] = bool(
                promotion.get("eligible_candidate_ids")
            )
            summary["status"] = promotion.get("status")
    payout_walk_forward = payload.get("conditional_payout_walk_forward")
    if isinstance(payout_walk_forward, dict):
        bankroll = payout_walk_forward.get("bankroll")
        if isinstance(bankroll, dict):
            for key in (
                "roi", "profit_yen", "stake_yen", "return_yen",
                "max_drawdown_yen",
            ):
                summary[f"payout_feature_candidate_{key}"] = bankroll.get(key)
            policy = bankroll.get("policy")
            if isinstance(policy, dict):
                summary["payout_feature_candidate_schema"] = policy.get(
                    "payout_tail_schema"
                ) or policy.get(
                    "payout_feature_schema"
                )
        confidence = payout_walk_forward.get("bankroll_confidence")
        if isinstance(confidence, dict):
            for key, value in confidence.items():
                if not isinstance(value, (dict, list)):
                    summary[f"payout_feature_{key}"] = value
        gate = payout_walk_forward.get("diagnostic_gate")
        if isinstance(gate, dict):
            for key, value in gate.items():
                if not isinstance(value, (dict, list)):
                    summary[f"payout_feature_gate_{key}"] = value
        summary["payout_feature_promotion_eligible"] = (
            payout_walk_forward.get("promotion_eligible")
        )
    payout_comparison = payload.get("payout_feature_comparison")
    if isinstance(payout_comparison, dict):
        candidate = payout_comparison.get("candidate_bankroll")
        legacy = payout_comparison.get("legacy_bankroll")
        confidence = payout_comparison.get("confidence")
        if isinstance(candidate, dict):
            summary["payout_feature_candidate_roi"] = candidate.get("roi")
        if isinstance(legacy, dict):
            summary["payout_feature_legacy_roi"] = legacy.get("roi")
        if isinstance(confidence, dict):
            for key in (
                "roi_delta",
                "roi_delta_ci95_lower",
                "roi_delta_ci95_upper",
                "probability_roi_delta_above_zero",
            ):
                summary[f"payout_feature_{key}"] = confidence.get(key)
        gate = payout_comparison.get("gate")
        if isinstance(gate, dict):
            summary["payout_feature_promotion_eligible"] = gate.get("pass")
            for key in (
                "roi_ci95_lower",
                "roi_delta_ci95_lower",
                "roi_pass",
                "profit_pass",
                "baseline_improved",
            ):
                summary[f"payout_feature_gate_{key}"] = gate.get(key)
        summary["payout_feature_candidate_schema"] = payout_comparison.get(
            "candidate_schema"
        )
        summary["payout_feature_legacy_schema"] = payout_comparison.get(
            "legacy_schema"
        )
    nested_evaluation = payload.get("evaluation")
    if (
        payload.get("model") == "bankroll_policy_nested_annual_v1"
        and isinstance(nested_evaluation, dict)
    ):
        aggregate = nested_evaluation.get("aggregate")
        if isinstance(aggregate, dict):
            visit(aggregate, 1)
            summary["fold_count"] = aggregate.get("fold_count")
            summary["minimum_fold_roi"] = aggregate.get("minimum_fold_roi")
            summary["profitable_folds"] = aggregate.get("profitable_folds")
            summary["largest_hit_excluded_roi"] = aggregate.get(
                "largest_hit_excluded_roi"
            )
            summary["fold_rois"] = [
                row.get("roi") for row in (aggregate.get("folds") or [])
                if isinstance(row, dict)
            ]
            confidence = aggregate.get("bootstrap")
            if isinstance(confidence, dict):
                summary["roi_ci95_lower"] = confidence.get("roi_ci95_lower")
                summary["roi_ci95_upper"] = confidence.get("roi_ci95_upper")
                summary["probability_roi_above_one"] = confidence.get(
                    "probability_roi_above_one"
                )
        gate = nested_evaluation.get("promotion_gate")
        if isinstance(gate, dict):
            checks = {key: bool(value) for key, value in gate.items()}
            summary["promotion_gate_passed"] = sum(checks.values())
            summary["promotion_gate_total"] = len(checks)
            summary["promotion_gate_failed"] = [
                key for key, passed in checks.items() if not passed
            ]
        summary["promotion_eligible"] = payload.get("promotion_eligible")
    summary.setdefault("model", payload.get("model"))
    summary.setdefault("status", payload.get("status"))
    if payload.get("model") == "odds_path_role_integrated_edge_conditional_lcb_v13":
        summary.update({
            "status": "research_invalid_deprecated",
            "research_invalid": True,
            "deprecated": True,
            "promotion_eligible": False,
        })
    gate = payload.get("promotion_gate")
    gate_primary = (
        gate.get("primary_bankroll") if isinstance(gate, dict) else None
    )
    canonical = canonicalize_primary_bankroll(
        summary,
        chronological_bankroll=(
            payload.get("chronological_bankroll")
            if isinstance(payload.get("chronological_bankroll"), dict)
            else None
        ),
        primary_bankroll=payload.get("primary_bankroll") or gate_primary,
    )
    canonical = canonicalize_probability_metrics(
        canonical,
        probability_metrics=(
            payload.get("probability_metrics")
            if isinstance(payload.get("probability_metrics"), dict)
            else None
        ),
        market_comparison=(
            payload.get("market_comparison")
            if isinstance(payload.get("market_comparison"), dict)
            else None
        ),
    )
    return {key: value for key, value in canonical.items() if value is not None}


def result_decision(task_type: str, summary: dict[str, Any]) -> str:
    if task_type == "genetic_island_search":
        return "speculative_generation_complete"
    if task_type == "market_residual_walk_forward":
        if summary.get("promotion_eligible") is True:
            return "promotion_candidate"
        return "accumulate_formal_evidence"
    if task_type == "bankroll_policy_nested_annual":
        if summary.get("promotion_eligible") is True:
            return "promotion_candidate"
        return "nested_gate_failed"
    if task_type == "conditional_payout_tail":
        if summary.get("payout_feature_promotion_eligible") is True:
            return "payout_feature_promotion_candidate"
        return "reject_or_research_only"
    if task_type in {"listwise_feature_search", "combined_feature_search"}:
        return "refine_selected_candidate"
    if summary.get("promotion_eligible") is True:
        return "promotion_candidate"
    if summary.get("payout_feature_promotion_eligible") is True:
        return "payout_feature_promotion_candidate"
    if summary.get("incremental_confidence_pass") is True:
        return "confirm_on_new_holdout"
    roi = summary.get("roi")
    profit = summary.get("profit_yen")
    if roi is not None and float(roi) >= 1.0 and float(profit or 0) > 0:
        return "bankroll_gate_pass"
    if task_type == "evaluation_aggregate":
        return "aggregation_complete"
    if task_type in {"gdrive_raw_archive", "gdrive_model_cache_archive"}:
        return "backup_complete"
    if task_type == "archive_closing_backfill":
        return "collection_complete"
    if task_type == "repository_sync":
        if str(summary.get("action") or "").startswith("deferred_"):
            return "repository_sync_deferred"
        return "maintenance_complete"
    if task_type in {
        "repository_hygiene",
        "series_feature_cache",
        "racer_stats_backfill",
        "persist_standard_selected_cache",
    }:
        return "maintenance_complete"
    return "reject_or_research_only"


def _attach_tail_portfolio_diagnostics(value: Any) -> bool:
    changed = False
    if isinstance(value, list):
        for item in value:
            changed = _attach_tail_portfolio_diagnostics(item) or changed
        return changed
    if not isinstance(value, dict):
        return False

    daily = value.get("daily")
    if isinstance(daily, list):
        rows: list[dict[str, Any]] = []
        found_rows = False
        for day in daily:
            if not isinstance(day, dict):
                continue
            raw_rows = day.pop("_tail_portfolio_rows", None)
            if isinstance(raw_rows, list):
                found_rows = True
                rows.extend(row for row in raw_rows if isinstance(row, dict))
        if found_rows:
            diagnostics = diagnose_tail_portfolio(rows)
            diagnostics["odds_field"] = "estimated_odds_at_purchase"
            value["tail_portfolio_diagnostics"] = diagnostics
            changed = True

    diagnostics = value.get("tail_portfolio_diagnostics")
    for child in value.values():
        if child is not diagnostics:
            changed = _attach_tail_portfolio_diagnostics(child) or changed
    return changed


def _load_result(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("evaluation result must be a JSON object")
    if _attach_tail_portfolio_diagnostics(payload):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
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


def _record_failed_attempt(
    conn: Any,
    *,
    job: dict[str, Any],
    error: str,
    app_root: Path | None = None,
) -> bool:
    remaining = int(job["attempt"]) < int(job["max_attempts"])
    recoverable_task = (
        str(job.get("task_type")) in CHECKPOINT_RECOVERABLE_TASKS
    )
    checkpoint: Path | None = None
    checkpoint_error = ""
    if recoverable_task:
        if app_root is None:
            checkpoint_error = "application root is unavailable"
        else:
            checkpoint, checkpoint_error = _valid_feature_search_checkpoint(
                job,
                app_root=app_root,
            )
    requeue = remaining and (not recoverable_task or checkpoint is not None)
    if checkpoint is not None:
        audit_error = (
            f"{error}; checkpoint recovery queued: {checkpoint}"
            if requeue
            else f"{error}; checkpoint recovery exhausted: {checkpoint}"
        )
    elif recoverable_task:
        audit_error = (
            f"{error}; checkpoint recovery unavailable: {checkpoint_error}"
        )
    else:
        audit_error = error
    audit_error = audit_error[-8000:]
    status = "queued" if requeue else "failed"
    conn.execute(
        """
        UPDATE model_evaluation_jobs
        SET status = ?, available_at = CURRENT_TIMESTAMP + INTERVAL '15 minutes',
            worker_id = NULL, locked_at = NULL, updated_at = CURRENT_TIMESTAMP,
            completed_at = CASE WHEN ? = 'failed' THEN CURRENT_TIMESTAMP ELSE NULL END,
            error = ?
        WHERE job_id = ?
        """,
        (status, status, audit_error, int(job["job_id"])),
    )
    conn.execute(
        """
        UPDATE model_evaluation_job_runs
        SET status = 'failed', completed_at = CURRENT_TIMESTAMP, error = ?
        WHERE job_id = ? AND attempt = ?
        """,
        (audit_error, int(job["job_id"]), int(job["attempt"])),
    )
    return requeue


def fail_job(
    conn: Any,
    *,
    job: dict[str, Any],
    error: str,
    app_root: Path | None = None,
) -> None:
    _record_failed_attempt(
        conn,
        job=job,
        error=error,
        app_root=app_root,
    )


def cancel_obsolete_job(
    conn: Any,
    *,
    job: dict[str, Any],
    reason: str,
) -> None:
    audit_error = f"obsolete job cancelled: {reason}"[-8000:]
    conn.execute(
        """
        UPDATE model_evaluation_jobs
        SET status = 'cancelled', worker_id = NULL, locked_at = NULL,
            updated_at = CURRENT_TIMESTAMP, completed_at = CURRENT_TIMESTAMP,
            error = ?
        WHERE job_id = ?
        """,
        (audit_error, int(job["job_id"])),
    )
    conn.execute(
        """
        UPDATE model_evaluation_job_runs
        SET status = 'failed', completed_at = CURRENT_TIMESTAMP, error = ?
        WHERE job_id = ? AND attempt = ?
        """,
        (audit_error, int(job["job_id"]), int(job["attempt"])),
    )


def defer_job(
    conn: Any,
    *,
    job: dict[str, Any],
    reason: str,
) -> None:
    audit_error = f"dependency deferred: {reason}"[-8000:]
    conn.execute(
        """
        UPDATE model_evaluation_jobs
        SET status = 'queued',
            available_at = CURRENT_TIMESTAMP + INTERVAL '15 minutes',
            max_attempts = max_attempts + 1,
            worker_id = NULL, locked_at = NULL,
            updated_at = CURRENT_TIMESTAMP, completed_at = NULL,
            error = ?
        WHERE job_id = ?
        """,
        (audit_error, int(job["job_id"])),
    )
    conn.execute(
        """
        UPDATE model_evaluation_job_runs
        SET status = 'failed', completed_at = CURRENT_TIMESTAMP, error = ?
        WHERE job_id = ? AND attempt = ?
        """,
        (
            audit_error,
            int(job["job_id"]),
            int(job["attempt"]),
        ),
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
    timeout_default = (
        28800
        if job["task_type"] in {
            "standardized_365d",
            "historical_coverage_safe",
            "calibrated_mlp_recency_search",
            "lightgbm_recency_search",
        }
        else 21600
    )
    timeout = _integer(job.get("parameters") or {}, "timeout_seconds", timeout_default, 300, 86400)
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
    if (
        job["task_type"] not in {"listwise_feature_search", "combined_feature_search"}
        or decision != "refine_selected_candidate"
    ):
        return None
    relative = f"data/models/evaluation_queue/job-{int(job['job_id']):08d}.json"
    selected_suffix = (
        "-combined" if job["task_type"] == "combined_feature_search" else ""
    )
    parent_cache = str(
        app_root / "data" / "models" / "evaluation_cache"
        / f"job-{int(job['job_id']):08d}{selected_suffix}"
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


def enqueue_refined_market_evaluation(
    conn: Any,
    job: dict[str, Any],
    *,
    app_root: Path,
) -> int | None:
    """Evaluate a refined historical model only after its artifact exists."""
    if job.get("task_type") != "listwise_newton_refine":
        return None
    parameters = job.get("parameters") or {}
    if not isinstance(parameters, dict):
        try:
            parameters = json.loads(str(parameters))
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    search_result = parameters.get("search_result")
    if not isinstance(search_result, str):
        return None
    search_path = (app_root / search_result).resolve()
    result_root = (app_root / "data/models/evaluation_queue").resolve()
    if result_root not in search_path.parents or search_path.suffix != ".json":
        return None
    try:
        search_payload = json.loads(search_path.read_text(encoding="utf-8"))
        through = datetime.strptime(
            str(search_payload["as_of_date"]), "%Y-%m-%d"
        ).date()
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    source_job_id = int(job["job_id"])
    model_input = (
        f"data/models/evaluation_queue/job-{source_job_id:08d}.joblib"
    )
    from_date = through - timedelta(days=13)
    range_key = f"{from_date:%Y%m%d}-{through.day:02d}"
    return enqueue_job(
        conn,
        task_type="market_residual_walk_forward",
        model_key=f"{job['model_key']}:v21_market:{range_key}",
        parameters={
            "model_input": model_input,
            "from_date": from_date.isoformat(),
            "through_date": through.isoformat(),
            "daily_budget_yen": 10000,
            "calibrator_strategy": (
                "odds_path_observed_closing_return_schedule_quota_triple_head_v21"
            ),
            "min_calibration_days": 2,
            "minimum_day_coverage": 1.0,
            "timeout_seconds": 7200,
        },
        priority=int(job["priority"]) + 2,
        max_attempts=2,
        parent_job_id=source_job_id,
    )


def reconcile_refined_market_evaluations(
    conn: Any,
    *,
    app_root: Path,
) -> list[int]:
    """Recover market evaluations missed across worker code reloads."""
    rows = conn.execute(
        """
        SELECT refined.*
        FROM model_evaluation_jobs AS refined
        WHERE refined.task_type = 'listwise_newton_refine'
          AND refined.status = 'completed'
          AND refined.completed_at >= CURRENT_TIMESTAMP - INTERVAL '48 hours'
          AND NOT EXISTS (
            SELECT 1
            FROM model_evaluation_jobs AS child
            WHERE child.parent_job_id = refined.job_id
              AND child.task_type = 'market_residual_walk_forward'
              AND child.status IN ('queued', 'running', 'completed')
          )
        ORDER BY refined.completed_at, refined.job_id
        """
    ).fetchall()
    inserted: list[int] = []
    for row in rows:
        job_id = enqueue_refined_market_evaluation(
            conn, dict(row), app_root=app_root
        )
        if job_id is not None:
            inserted.append(job_id)
    return inserted


def advance_genetic_islands(
    conn: Any,
    job: dict[str, Any],
    *,
    app_root: Path,
) -> list[int]:
    if job.get("task_type") != "genetic_island_search":
        return []
    params = dict(job.get("parameters") or {})
    cohort = str(params["cohort"])
    generation = int(params["generation"])
    island_count = int(params["island_count"])
    max_generations = int(params["max_generations"])
    conn.execute("SELECT pg_advisory_xact_lock(?)", (CLAIM_LOCK_ID + 100,))
    rows = conn.execute(
        """
        SELECT job_id, parameters, result_path
        FROM model_evaluation_jobs
        WHERE task_type = 'genetic_island_search'
          AND status = 'completed'
          AND parameters->>'cohort' = ?
          AND CAST(parameters->>'generation' AS INTEGER) = ?
        ORDER BY CAST(parameters->>'island_id' AS INTEGER)
        """,
        (cohort, generation),
    ).fetchall()
    by_island = {int(row["parameters"]["island_id"]): row for row in rows}
    if set(by_island) != set(range(island_count)):
        return []
    results: dict[int, dict[str, Any]] = {}
    for island_id, row in by_island.items():
        result_path = Path(str(row["result_path"]))
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise JobDependencyUnavailable(
                f"genetic island result is unavailable: {result_path}"
            ) from exc
        if not isinstance(result, dict) or not isinstance(result.get("elites"), list):
            raise ValueError(f"invalid genetic island result: {result_path}")
        results[island_id] = result

    inserted: list[int] = []
    if generation + 1 < max_generations:
        current_best = max(
            float(result["champion"]["fitness"]) for result in results.values()
        )
        prior = conn.execute(
            """
            SELECT MAX(CAST(result_summary->>'genetic_fitness' AS DOUBLE PRECISION))
                   AS best_fitness
            FROM model_evaluation_jobs
            WHERE task_type = 'genetic_island_search'
              AND status = 'completed'
              AND parameters->>'cohort' = ?
              AND CAST(parameters->>'generation' AS INTEGER) < ?
            """,
            (cohort, generation),
        ).fetchone()
        prior_best = float(prior["best_fitness"]) if prior and prior["best_fitness"] is not None else None
        base_mutation = float(params.get("mutation_rate") or 0.35)
        mutation_rate = (
            min(0.80, base_mutation + 0.15)
            if prior_best is not None and current_best <= prior_best + 1e-5
            else max(0.35, base_mutation - 0.10)
        )
        structural_elites = {
            (
                result["champion"]["genome"].get("target"),
                round(
                    float(
                        result["champion"]["genome"].get("loss_blend")
                        if result["champion"]["genome"].get("loss_blend") is not None
                        else (
                            0.0
                            if result["champion"]["genome"].get("target") == "winner"
                            else 1.0
                        )
                    ),
                    4,
                ),
                round(float(result["champion"]["genome"].get("learning_rate") or 0), 4),
                int(result["champion"]["genome"].get("epochs") or 0),
            )
            for result in results.values()
        }
        diversity_rescue = len(structural_elites) <= max(2, island_count // 2)
        migration_interval = int(params.get("migration_interval") or 3)
        migration_applied = (
            not diversity_rescue
            and (generation + 1) % migration_interval == 0
        )
        random_injections = int(params.get("random_injections") or 1)
        if diversity_rescue:
            mutation_rate = max(mutation_rate, 0.80)
            random_injections = max(
                random_injections,
                int(params.get("population_size") or 8) // 3,
            )
        for island_id in range(island_count):
            donor = results[(island_id - 1) % island_count]
            source_elites = list(results[island_id]["elites"][:2])
            if migration_applied:
                source_elites = [
                    results[island_id]["elites"][0], donor["elites"][0]
                ]
            immigrants = []
            seen_immigrants: set[tuple[Any, ...]] = set()
            for row in source_elites:
                if not isinstance(row, dict) or not isinstance(row.get("genome"), dict):
                    continue
                genome = row["genome"]
                key = (
                    genome.get("target"),
                    float(
                        genome.get("loss_blend")
                        if genome.get("loss_blend") is not None
                        else (0.0 if genome.get("target") == "winner" else 1.0)
                    ),
                    float(genome.get("alpha") or 0),
                    float(genome.get("learning_rate") or 0),
                    int(genome.get("epochs") or 0),
                )
                if key in seen_immigrants:
                    continue
                seen_immigrants.add(key)
                immigrants.append(genome)
            next_params = dict(params)
            next_params.update({
                "generation": generation + 1,
                "island_id": island_id,
                "seed": int(params["seed"]) + 1_000_003 + island_id,
                "immigrants": immigrants,
                "mutation_rate": mutation_rate,
                "random_injections": random_injections,
                "migration_interval": migration_interval,
                "migration_applied": migration_applied,
                "diversity_rescue": diversity_rescue,
                "structural_elite_count": len(structural_elites),
            })
            job_id = enqueue_job(
                conn,
                task_type="genetic_island_search",
                model_key=(
                    f"genetic-listwise-{cohort}-g{generation + 1:02d}"
                    f"-i{island_id:02d}"
                ),
                parameters=next_params,
                priority=int(job.get("priority") or 50),
                max_attempts=3,
                parent_job_id=int(by_island[island_id]["job_id"]),
            )
            if job_id is not None:
                inserted.append(job_id)
        return inserted

    ranked_champions = sorted(
        (result["champion"] for result in results.values()),
        key=lambda row: float(row["fitness"]),
        reverse=True,
    )
    champions: list[dict[str, Any]] = []
    seen_genomes: set[tuple[Any, ...]] = set()
    for champion in ranked_champions:
        genome = champion["genome"]
        predictive_key = (
            genome.get("target"),
            float(
                genome.get("loss_blend")
                if genome.get("loss_blend") is not None
                else (0.0 if genome.get("target") == "winner" else 1.0)
            ),
            float(genome.get("alpha") or 0),
            float(genome.get("learning_rate") or 0),
            int(genome.get("epochs") or 0),
        )
        if predictive_key in seen_genomes:
            continue
        seen_genomes.add(predictive_key)
        champions.append(champion)
    for rank, champion in enumerate(champions[:island_count], start=1):
        genome = dict(champion["genome"])
        genome_version = int(genome.get("genome_version") or 1)
        loss_blend = float(
            genome.get("loss_blend")
            if genome.get("loss_blend") is not None
            else (0.0 if genome.get("target") == "winner" else 1.0)
        )
        validation_target = str(genome.get("target") or "")
        if validation_target not in {"winner", "top3_pl"}:
            validation_target = "winner" if loss_blend < 0.5 else "top3_pl"
        model_key = (
            f"genetic-champion-v{genome_version}-{cohort}"
            f"-g{generation:02d}-r{rank:02d}"
        )
        validation_id = enqueue_job(
            conn,
            task_type="listwise_feature_search",
            model_key=model_key,
            parameters={
                "evaluation_date": params["evaluation_date"],
                "n_features": 8192,
                "epochs": int(genome["epochs"]),
                "batch_races": 1000,
                "learning_rate": float(genome["learning_rate"]),
                "targets": validation_target,
                "loss_blend": loss_blend,
                "alphas": f"{float(genome['alpha']):.12g}",
                "ev_thresholds": "1.0,1.1,1.2,1.35,1.5",
                "timeout_seconds": 43200,
            },
            priority=max(1, int(job.get("priority") or 50) + 10 - rank),
            max_attempts=3,
            parent_job_id=int(job["job_id"]),
        )
        if validation_id is not None:
            inserted.append(validation_id)
    return inserted


def seed_periodic_jobs(conn: Any, *, now: datetime | None = None) -> list[int]:
    now = now or datetime.now(timezone.utc)
    inserted: list[int] = []

    def already_scheduled(task_type: str, key: str) -> bool:
        row = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM model_evaluation_jobs
            WHERE dedupe_key = ?
               OR (task_type = ? AND status IN ('queued','running'))
            """,
            (key, task_type),
        ).fetchone()
        return bool(row and int(row["count"]))

    schedules = (
        ("gdrive_raw_archive", "raw-data", 3600, 10, 1800),
        ("evaluation_aggregate", "all-models", 900, 30, 900),
        ("series_feature_cache", "official-series", 1800, 45, 600),
        ("repository_sync", "repository", 1800, 25, 300),
        ("repository_hygiene", "repository", 21600, 20, 300),
    )
    epoch = int(now.timestamp())
    for task_type, model_key, interval, priority, timeout in schedules:
        bucket = epoch - epoch % interval
        parameters = {
            "schedule_bucket": datetime.fromtimestamp(
                bucket, timezone.utc
            ).isoformat(),
            "timeout_seconds": timeout,
        }
        if task_type == "series_feature_cache":
            parameters["from_date"] = (
                now.astimezone(JST).date() - timedelta(days=14)
            ).isoformat()
        key = dedupe_key(task_type, model_key, parameters)
        if already_scheduled(task_type, key):
            continue
        job_id = enqueue_job(
            conn,
            task_type=task_type,
            model_key=model_key,
            parameters=parameters,
            priority=priority,
            max_attempts=3,
        )
        if job_id is not None:
            inserted.append(job_id)
    return inserted


DEFAULT_WORK_TICKETS = (
    ("OPS-QUEUE-001", "DBジョブ基盤と資源監視", "運用基盤", "評価・集計・バックアップをDBキューから実行する", "4ランナーが資源条件付きで取得し完了履歴をDBへ残す", 100, "in_progress", 70),
    ("OPS-BACKUP-001", "GDriveバックアップのキュー移行", "バックアップ", "生データ転送を定期DBジョブとして管理する", "排他付き転送が完了し元データ削除と結果記録を確認する", 90, "in_progress", 65),
    ("OPS-REPO-SYNC-001", "Gitリポジトリの定期確認と安全な更新", "運用基盤", "DB定期ジョブでoriginを確認し安全条件を満たす時だけfast-forwardする", "dirty・履歴分岐・評価実行中は更新せず監査結果を残し、cleanかつidle時だけff-onlyで更新する", 92, "in_progress", 20),
    ("MODEL-OPT-001", "モデル再設計と収益ゲート収束", "モデル", "特徴量・教師・構造を同一評価軸で反復検証する", "未使用holdoutでROI・損益・確率指標の昇格基準を満たす", 100, "in_progress", 55),
    (
        "MODEL-GENETIC-001",
        "日次遺伝的アイランド探索と監査付き昇格",
        "モデル",
        "最新確定データから複数の小型モデル島を並列評価し、移住と世代更新で候補を成長させる",
        "投機評価と365日昇格評価を分離し、全ゲート合格候補だけを原子的に本番反映して監視・ロールバックできる",
        100,
        "in_progress",
        30,
    ),
    ("UI-MODEL-001", "モデル性能ページの評価表現統一", "WebUI", "評価母集団と指標表現を統一する", "全モデルが同じ列定義と評価群で比較できる", 70, "queued", 20),
    ("UI-PRED-001", "タイムラインとGantt的中判定の統一", "WebUI", "主系予測と購入的中を別指標として表示する", "同一レースで各表示の意味と判定が一致する", 80, "queued", 25),
    (
        "OPS-EVAL-PERF-001",
        "特徴探索の並列化と再現性保証",
        "モデル基盤",
        "特徴バリアント生成を資源制約付きで並列化し、評価待ち時間を短縮する。GitHub Issue: https://github.com/ryo100794/boat/issues/1",
        "workers=1/2で候補順・selected・holdout hash・資金評価が一致し、checkpoint再開可能。Git commit SHAとDBイベントを記録し、リモートが同SHAで稼働する",
        90,
        "in_progress",
        35,
    ),
    (
        "OPS-EVAL-MEM-001",
        "Memory-safe feature-search parallelism for 32GB runtime",
        "Model infrastructure",
        "Process feature variants sequentially and parallelize only candidates sharing one dataset",
        "variant_workers=1 and candidate_workers=1/2 produce identical results while sharing the dataset and scaler",
        98,
        "in_progress",
        15,
    ),
    (
        "OPS-GITHUB-SYNC-001",
        "GitHub IssueとDB懸案事項の同期",
        "運用基盤",
        "GitHub Issueとwork_ticketsを安定した対応キーで同期する",
        "冪等な双方向同期、競合方針、dry-run、監査イベントをテストで確認する",
        90,
        "in_progress",
        40,
    ),
    (
        "DOCS-HIERARCHY-001",
        "再現可能な文書階層と定期監査",
        "文書",
        "READMEを入口に既存文書を粗から詳細へ整理し、話題別文書の乱立を防ぐ",
        "canonical文書階層を明記し、6時間ごとのhygiene監査結果をDBへ残す",
        70,
        "in_progress",
        20,
    ),
    (
        "MODEL-FEATURE-COMBINE-001",
        "Combined feature ablation and retraining",
        "Model",
        "Run selection-only search combining base_pastlog+research_correlates with inert series_cached/series_relative ablations",
        "Compare against single ablations on the same fixed 365-day holdout and evaluation axes without holdout leakage",
        88,
        "queued",
        10,
    ),
    (
        "MODEL-SERIES-CACHE-001",
        "公式series特徴キャッシュの継続更新",
        "モデル",
        "公式raw JSONのseries成績をPostgreSQL特徴キャッシュへ増分反映する",
        "raw保有艇を欠落なく反映し、負のtrendを保持し方向を実績較正したv4で365日評価を記録する",
        95,
        "in_progress",
        75,
    ),
    (
        "MODEL-PAYOUT-001",
        "条件付き払戻分布の評価",
        "モデル",
        "条件付き払戻tail補正を365日holdoutで評価する",
        "従来モデルと同一母集団でROI・損益・確率指標を比較する",
        70,
        "queued",
        55,
    ),
    (
        "MODEL-RECENCY-001",
        "時間減衰モデルの評価",
        "モデル",
        "過去ログ中心モデルに時間減衰とcalibrationを導入して評価する",
        "365日holdoutで基準モデルとのpaired比較を記録する",
        80,
        "in_progress",
        50,
    ),
    (
        "MODEL-VENUE-001",
        "場条件付き着順モデルの評価",
        "モデル",
        "場ごとの傾向を過学習させず着順モデルへ反映する",
        "場別と全体の365日holdout指標および資金評価を記録する",
        70,
        "in_progress",
        30,
    ),
    (
        "MODEL-SEGMENT-001",
        "弱点セグメントの改善",
        "モデル",
        "場・時期・オッズ有無などの弱点区分を抽出して改善候補を評価する",
        "未使用holdoutで全体性能を損なわず対象区分の改善を確認する",
        60,
        "queued",
        10,
    ),
    (
        "MODEL-MARKET-RESIDUAL-001",
        "履歴モデルとT-5公式オッズの残差評価",
        "モデル",
        "365日最良履歴モデルの確率を事前分布としT-5公式オッズ残差を日付単位walk-forwardで較正する",
        "全適格レースを日単位foldで市場単独・履歴単独と比較し、LogLoss・1着・3T5・ROI・損益・最大DD・ROI片側95%下限を記録する。30日未満はshadow限定とする",
        96,
        "in_progress",
        25,
    ),
    (
        "MODEL-HISTORICAL-RESIDUAL-001",
        "Protected historical residual model",
        "Model",
        "Keep no_odds_v8 baseline probabilities fixed and learn historical-feature residual corrections on selection data only",
        "On the untouched 365-day holdout, do not degrade log loss, winner top1, or trifecta top5 versus no_odds_v8; compare ROI, profit, and drawdown under the identical bankroll protocol without selecting blend weights on holdout",
        99,
        "queued",
        5,
    ),
    (
        "MODEL-MARKET-POLICY-CAL-001",
        "T-5期待値・払戻選択層の較正",
        "モデル",
        "市場残差の予測改善と購入損失を分離し、T-5直値・終値予測・保守的分位点の期待値方針を日次前進評価する",
        "方針選択は過去日だけで行い、30日・1,000RでROI片側95%下限1超・損益プラス・日別安定性を満たす。未達時はno-betを維持する",
        97,
        "in_progress",
        20,
    ),
    (
        "MODEL-V21-PROSPECTIVE-EVIDENCE-001",
        "Frozen V21 prospective confirmation",
        "Model",
        "Freeze the V21 triple-head architecture on 2026-07-30 and collect only unseen, append-only T300 shadow evidence from 2026-07-31. Job 8666 remains exploratory because its six days informed the head selection.",
        "After at least 30 fully covered unseen days, 1000 races, and 20 effective hits, require positive profit, ROI and largest-hit-excluded ROI above one, daily bootstrap lower95 above one, profitable-day fraction at least 0.6, probability calibration non-regression, and race/day market confidence. Any model or policy change resets evidence to D1; real betting stays disabled.",
        116,
        "in_progress",
        5,
    ),
    (
        "TEST-BASELINE-FAILURES-001",
        "Resolve five repository baseline test failures",
        "Quality",
        "Fix the existing empirical market walk-forward, packed bankroll, and tail portfolio expectation mismatches found by the full repository test run.",
        "The five failing tests reproduce independently, their behavioral contract is reconciled without changing frozen V21 artifacts or prospective evidence, and the full suite passes with zero failures.",
        82,
        "queued",
        5,
    ),
    (
        "MODEL-V23-PROSPECTIVE-EVIDENCE-001",
        "V23上位5点・終値予測EV帯の完全未見検証",
        "モデル運用",
        "V21確率・順位ヘッドを固定し、事前登録済みtop5かつ終値予測EV 1.00以上1.05以下を100円固定で選ぶV23を2026-08-01からappend-only影運用する",
        "30完全日、1000レース、200券、20有効的中以上を蓄積し、ROIと最大払戻除外ROIと日クラスタbootstrap片側95%下限がすべて1超、黒字日率0.6以上、LogLoss市場非劣化と3T5改善信頼度0.95以上、欠損・判断遅延・モデル例外0を満たす。実投票は別のgo/no-go承認まで無効",
        117,
        "in_progress",
        35,
    ),
    (
        "MODEL-EDGE-CONDITIONAL-LCB-V13-001",
        "高EV候補向け階層的確率LCB V13",
        "モデル",
        "V12確定オッズ下限とは独立に、strict-prior whole-day crossfitの確率をrank・確率帯・model/normalized-T300-market乖離帯で日クラスタ下方較正する",
        "V12と同一評価窓・同一資金条件でROI・最大払戻除外ROI・日クラスタbootstrap片側下限を比較し、高EV候補のexpected-vs-hit過大評価と条件付きcoverageが改善する。result/payoutは購入後settlement限定とし、再現可能な評価jobとテストを残す",
        98,
        "in_progress",
        55,
    ),
    (
        "MODEL-REGISTERED-BAND-LCB-V14-001",
        "固定T300乖離帯の保守的確率補正V14",
        "モデル",
        "2026-07-29に固定したlog(model/T300市場確率)=[0.5,1.0)だけを対象に、疑似カウントなしの日単位bootstrap下限とV12確定オッズ下限を組み合わせる",
        "履歴結果は探索扱いとし昇格証拠にしない。2026-07-30以降のstrict-prior評価で5日・300候補・20的中以上、最大払戻除外ROI>1、日次ROI bootstrap片側95%下限>1、確率観測/予測比の片側95%下限>1、T300 snapshot不一致0を満たす",
        99,
        "in_progress",
        35,
    ),
    (
        "MODEL-WEEKEND-PILOT-20260801",
        "週末V14上限制御パイロット",
        "モデル運用",
        "2026-07-30 shadow、2026-07-31 go/no-go、2026-08-01 capped pilot、2026-08-02 continuation auditを日付固定で実施する",
        "2026-07-30は実投票無効のshadow、2026-07-31は最大払戻除外ROI>1かつ日次bootstrap下限>1かつデータ欠損0の場合だけgo。未達なら実投票を無効のままにする。2026-08-01の初期実投票上限は元資金10,000円に対し2,000円/日、2026-08-02に継続可否を監査する",
        100,
        "queued",
        0,
    ),
    (
        "UI-MODEL-DAILY-001",
        "モデル個別の日次集計と表タップ選択",
        "WebUI",
        "モデル別の日次資金集計を正規化し評価表の行から個別集計を選択する",
        "全モデルに日次系列または欠損理由がありプルダウンなしで行タップとキーボード選択ができる",
        85,
        "in_progress",
        10,
    ),
)


def seed_work_tickets(conn: Any) -> int:
    changed = 0
    for key, title, area, description, acceptance, priority, status, progress in DEFAULT_WORK_TICKETS:
        if conn.execute(
            "SELECT 1 FROM work_tickets WHERE ticket_key = ?",
            (key,),
        ).fetchone() is not None:
            continue
        conn.execute(
            """
            INSERT INTO work_tickets(
              ticket_key, title, area, description, acceptance_criteria,
              priority, status, progress, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'codex')
            ON CONFLICT(ticket_key) DO NOTHING
            """,
            (key, title, area, description, acceptance, priority, status, progress),
        )
        changed += 1
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


def standardized_evaluation_due(
    conn: Any,
    *,
    evaluation_date: str,
    cadence_days: int = 7,
) -> bool:
    target = datetime.strptime(evaluation_date, "%Y-%m-%d").date()
    row = conn.execute(
        """
        SELECT parameters->>'evaluation_date' AS evaluation_date, status
        FROM model_evaluation_jobs
        WHERE task_type = ?
          AND status IN ('queued', 'running', 'completed')
          AND parameters->>'evaluation_date' IS NOT NULL
        ORDER BY parameters->>'evaluation_date' DESC NULLS LAST, job_id DESC
        LIMIT 1
        """,
        ("standardized_365d",),
    ).fetchone()
    if row is None:
        return True
    if str(row["status"]) in {"queued", "running"}:
        return False
    latest = datetime.strptime(str(row["evaluation_date"]), "%Y-%m-%d").date()
    return (target - latest).days >= max(1, int(cadence_days))


def cancel_superseded_daily_jobs(
    conn: Any,
    *,
    evaluation_date: str,
) -> list[int]:
    """Cancel older queued daily jobs only when an exact newer track exists."""
    rows = conn.execute(
        """
        UPDATE model_evaluation_jobs AS old
        SET status = 'cancelled', completed_at = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP,
            decision = 'superseded_by_newer_daily_evaluation',
            worker_id = NULL, locked_at = NULL
        WHERE old.status = 'queued'
          AND COALESCE(
                old.parameters->>'evaluation_date',
                old.parameters->>'evaluation_through'
              ) < ?
          AND EXISTS (
            SELECT 1
            FROM model_evaluation_jobs AS newer
            WHERE newer.job_id <> old.job_id
              AND newer.task_type = old.task_type
              AND newer.model_key = old.model_key
              AND newer.status IN ('queued', 'running', 'completed')
              AND COALESCE(
                    newer.parameters->>'evaluation_date',
                    newer.parameters->>'evaluation_through'
                  ) = ?
          )
        RETURNING old.job_id
        """,
        (evaluation_date, evaluation_date),
    ).fetchall()
    return [int(row["job_id"]) for row in rows]


def seed_default_jobs(
    conn: Any,
    *,
    evaluation_date: str,
    include_standardized: bool = True,
) -> list[int]:
    inserted: list[int] = []

    def add(**kwargs: Any) -> None:
        job_id = enqueue_job(conn, **kwargs)
        if job_id is not None:
            inserted.append(job_id)

    if include_standardized:
        add(
            task_type="standardized_365d",
            model_key="all_registered_models",
            parameters={
                "evaluation_date": evaluation_date,
                "timeout_seconds": 86400,
            },
            priority=70,
            max_attempts=3,
        )
    evaluation_end = datetime.strptime(evaluation_date, "%Y-%m-%d").date()
    evaluation_start = evaluation_end - timedelta(days=364)
    add(
        task_type="calibrated_mlp_recency_search",
        model_key="calibrated_mlp_recency_drop_base_pastlog",
        parameters={
            "evaluation_date": evaluation_date,
            "half_lives": "none,180,365,730",
            "calibration_days": 180,
            "drop_feature_groups": "base_pastlog",
            "timeout_seconds": 86400,
        },
        priority=90,
        max_attempts=3,
    )
    add(
        task_type="conditional_payout_tail",
        model_key="conditional_payout_tail_365d_v1",
        parameters={
            "training_through": (evaluation_start - timedelta(days=1)).isoformat(),
            "evaluation_from": evaluation_start.isoformat(),
            "evaluation_through": evaluation_end.isoformat(),
            "timeout_seconds": 86400,
        },
        priority=90,
        max_attempts=3,
    )
    add(
        task_type="historical_research_logit",
        model_key="no_odds_v9_research_logit",
        parameters={
            "evaluation_date": evaluation_date,
            "timeout_seconds": 86400,
        },
        priority=89,
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
    add(
        task_type="combined_feature_search",
        model_key="listwise_combined_8192",
        parameters={
            "evaluation_date": evaluation_date,
            "n_features": 8192,
            "targets": "winner,top3_pl",
            "alphas": "0.00001,0.0001",
            "learning_rate": 0.02,
            "epochs": 2,
            "batch_races": 1000,
            "ev_threshold": 1.2,
            "timeout_seconds": 21600,
        },
        priority=85,
        max_attempts=2,
    )
    return inserted


GENETIC_PROTOCOL_VERSION = 4


def seed_daily_genetic_jobs(
    conn: Any,
    *,
    evaluation_date: str,
    now: datetime | None = None,
    island_count: int = 4,
    max_generations: int = 12,
) -> list[int]:
    existing = conn.execute(
        """
        SELECT 1 FROM model_evaluation_jobs
        WHERE task_type = 'genetic_island_search'
          AND parameters->>'evaluation_date' = ?
          AND COALESCE(parameters->>'genetic_protocol_version', '0') = ?
        LIMIT 1
        """,
        (evaluation_date, str(GENETIC_PROTOCOL_VERSION)),
    ).fetchone()
    if existing is not None:
        return []
    generated = (now or datetime.now(timezone.utc)).replace(microsecond=0)
    cohort = generated.strftime("%Y%m%dT%H%M%SZ")
    base_seed = int(generated.timestamp()) % 2_000_000_000
    inserted: list[int] = []
    for island_id in range(island_count):
        job_id = enqueue_job(
            conn,
            task_type="genetic_island_search",
            model_key=f"genetic-listwise-{cohort}-g00-i{island_id:02d}",
            parameters={
                "evaluation_date": evaluation_date,
                "genetic_protocol_version": GENETIC_PROTOCOL_VERSION,
                "cohort": cohort,
                "generation": 0,
                "island_id": island_id,
                "island_count": island_count,
                "max_generations": max_generations,
                "seed": base_seed + island_id,
                "population_size": 8,
                "local_generations": 3,
                "elite_count": 2,
                "mutation_rate": 0.35,
                "random_injections": 1,
                "migration_interval": 3,
                "train_races": 12000,
                "validation_races": 3000,
                "batch_races": 500,
                "immigrants": [],
                "timeout_seconds": 7200,
            },
            priority=55,
            max_attempts=3,
        )
        if job_id is not None:
            inserted.append(job_id)
    return inserted


def genetic_cache_evaluation_date(app_root: Path) -> str | None:
    manifest_path = (
        app_root / "data/models/standardized_365d_v2/selected_cache"
        / "listwise_search_8192_drop_base_pastlog.manifest.json"
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    last_race_id = str(manifest.get("last_race_id") or "")
    try:
        return datetime.strptime(last_race_id[:10], "%Y-%m-%d").date().isoformat()
    except ValueError:
        return None


MARKET_FORMAL_FROM_DATE = "2026-07-18"
DEPLOYMENT_DAILY_MARKET_PRIORITIES = {
    "odds_path_role_integrated_t300_nonlinear_v12": 104,
    "odds_path_role_integrated_registered_band_lcb_v14": 103,
    "odds_path_role_integrated_fixed_band_passthrough_v16": 102,
    "odds_path_observed_closing_return_schedule_quota_v18": 101,
    "odds_path_observed_closing_return_schedule_quota_dual_head_v20": 100,
}
MARKET_EVALUATION_SOURCES = (
    (
        "protected_mlp_prediction",
        "calibrated_mlp_recency_search",
        "calibrated_mlp_prediction_deployment",
        "newton_residual",
        "deployment",
    ),
    (
        "calibrated_mlp_recency_selected",
        "calibrated_mlp_recency_search",
        "calibrated_mlp_recency_card_features",
        "newton_residual",
    ),
    (
        "calibrated_lightgbm_recency_selected",
        "lightgbm_recency_search",
        "calibrated_lightgbm_recency_period_v6_4cpu",
        "newton_residual",
    ),
    (
        "odds_path_operational_daily",
        "lightgbm_recency_search",
        "calibrated_lightgbm_recency_period_v6_4cpu",
        "odds_path_return",
    ),
    (
        "odds_path_probability_only_daily",
        "lightgbm_recency_search",
        "calibrated_lightgbm_recency_period_v6_4cpu",
        "odds_path_probability",
    ),
    (
        "odds_path_observed_closing_return_v4_daily",
        "lightgbm_recency_search",
        "calibrated_lightgbm_recency_period_v6_4cpu",
        "odds_path_observed_closing_return",
    ),
    (
        "odds_path_observed_closing_return_robust_policy_v17_daily",
        "lightgbm_recency_search",
        "calibrated_lightgbm_recency_period_v6_4cpu",
        "odds_path_observed_closing_return_robust_policy_v17",
    ),
    (
        "odds_path_observed_closing_return_schedule_quota_v18_daily",
        "lightgbm_recency_search",
        "calibrated_lightgbm_recency_period_v6_4cpu",
        "odds_path_observed_closing_return_schedule_quota_v18",
    ),
    (
        "odds_path_observed_closing_return_schedule_quota_raw_nonregression_v19_daily",
        "lightgbm_recency_search",
        "calibrated_lightgbm_recency_period_v6_4cpu",
        "odds_path_observed_closing_return_schedule_quota_raw_nonregression_v19",
    ),
    (
        "odds_path_observed_closing_return_schedule_quota_dual_head_v20_daily",
        "lightgbm_recency_search",
        "calibrated_lightgbm_recency_period_v6_4cpu",
        "odds_path_observed_closing_return_schedule_quota_dual_head_v20",
    ),
    (
        "odds_path_observed_closing_return_schedule_quota_triple_head_v21_daily",
        "lightgbm_recency_search",
        "calibrated_lightgbm_recency_period_v6_4cpu",
        "odds_path_observed_closing_return_schedule_quota_triple_head_v21",
    ),
    (
        "odds_path_prequential_shrinkage_return_v6_daily",
        "lightgbm_recency_search",
        "calibrated_lightgbm_recency_period_v6_4cpu",
        "odds_path_prequential_shrinkage_return",
    ),
    (
        "odds_path_crossfit_conservative_ev_v7_daily",
        "lightgbm_recency_search",
        "calibrated_lightgbm_recency_period_v6_4cpu",
        "odds_path_crossfit_conservative_ev",
    ),
    (
        "odds_path_market_offset_crossfit_conservative_ev_v8_daily",
        "lightgbm_recency_search",
        "calibrated_lightgbm_recency_period_v6_4cpu",
        "odds_path_market_offset_crossfit_conservative_ev",
    ),
    (
        "odds_path_market_offset_discrete_log_ev_v9_daily",
        "lightgbm_recency_search",
        "calibrated_lightgbm_recency_period_v6_4cpu",
        "odds_path_market_offset_discrete_log_ev_v9",
    ),
    (
        "odds_path_market_offset_selection_conformal_discrete_ev_v10_daily",
        "lightgbm_recency_search",
        "calibrated_lightgbm_recency_period_v6_4cpu",
        "odds_path_market_offset_selection_conformal_discrete_ev_v10",
    ),
    (
        "odds_path_role_integrated_multihorizon_v11_daily",
        "lightgbm_recency_search",
        "calibrated_lightgbm_recency_period_v6_4cpu",
        "odds_path_role_integrated_multihorizon_v11",
    ),
    (
        "odds_path_role_integrated_t300_nonlinear_v12_daily",
        "lightgbm_recency_search",
        "calibrated_lightgbm_recency_period_v6_4cpu",
        "odds_path_role_integrated_t300_nonlinear_v12",
    ),
    (
        "odds_path_role_integrated_edge_conditional_lcb_v13_daily",
        "lightgbm_recency_search",
        "calibrated_lightgbm_recency_period_v6_4cpu",
        "odds_path_role_integrated_edge_conditional_lcb_v13",
    ),
    (
        "odds_path_role_integrated_fixed_band_passthrough_v16_daily",
        "lightgbm_recency_search",
        "calibrated_lightgbm_recency_period_v6_4cpu",
        "odds_path_role_integrated_fixed_band_passthrough_v16",
    ),
    (
        "odds_path_role_integrated_selection_free_envelope_v15_daily",
        "lightgbm_recency_search",
        "calibrated_lightgbm_recency_period_v6_4cpu",
        "odds_path_role_integrated_selection_free_envelope_v15",
    ),
    (
        "odds_path_role_integrated_registered_band_lcb_v14_daily",
        "lightgbm_recency_search",
        "calibrated_lightgbm_recency_period_v6_4cpu",
        "odds_path_role_integrated_registered_band_lcb_v14",
    ),
)


def seed_daily_market_jobs(
    conn: Any,
    *,
    app_root: Path,
    evaluation_date: str,
) -> list[int]:
    through = datetime.strptime(evaluation_date, "%Y-%m-%d").date()
    formal_from = datetime.strptime(MARKET_FORMAL_FROM_DATE, "%Y-%m-%d").date()
    if through < formal_from:
        return []
    model_root = (app_root / "data" / "models").resolve()
    inserted: list[int] = []
    for source_spec in MARKET_EVALUATION_SOURCES:
        label, task_type, source_key, calibrator_strategy, *artifact_kinds = source_spec
        if (
            calibrator_strategy in {
                "odds_path_role_integrated_selection_free_envelope_v15",
                "odds_path_role_integrated_fixed_band_passthrough_v16",
            }
            and through.isoformat() <= "2026-07-29"
        ):
            continue
        artifact_kind = artifact_kinds[0] if artifact_kinds else "evaluation"
        source = conn.execute(
            """
            SELECT result_path
            FROM model_evaluation_jobs
            WHERE task_type = ? AND model_key = ?
              AND status = ? AND result_path IS NOT NULL
            ORDER BY completed_at DESC, job_id DESC
            LIMIT 1
            """,
            (task_type, source_key, "completed"),
        ).fetchone()
        if source is None:
            continue
        result_path = Path(str(source["result_path"]))
        if not result_path.is_absolute():
            result_path = app_root / result_path
        model_input = (
            result_path.with_name(result_path.stem + ".deployment.joblib")
            if artifact_kind == "deployment"
            else result_path.with_suffix(".joblib")
        ).resolve()
        if model_root not in model_input.parents or not model_input.is_file():
            continue
        parameters = {
            "model_input": model_input.relative_to(app_root).as_posix(),
            "from_date": MARKET_FORMAL_FROM_DATE,
            "through_date": through.isoformat(),
            "daily_budget_yen": 10000,
            "min_calibration_days": (
                5
                if calibrator_strategy in {
                    "odds_path_role_integrated_selection_free_envelope_v15",
                    "odds_path_role_integrated_fixed_band_passthrough_v16",
                }
                else 2
            ),
            "calibrator_strategy": calibrator_strategy,
            "minimum_day_coverage": 1.0,
            **(
                {"v12_closing_fallback_policy": (
                    "no_bet"
                    if calibrator_strategy in {
                        "odds_path_role_integrated_selection_free_envelope_v15",
                        "odds_path_role_integrated_fixed_band_passthrough_v16",
                    }
                    else "v11"
                )}
                if calibrator_strategy
                in {
                    "odds_path_role_integrated_t300_nonlinear_v12",
                    "odds_path_role_integrated_edge_conditional_lcb_v13",
                    "odds_path_role_integrated_registered_band_lcb_v14",
                    "odds_path_role_integrated_selection_free_envelope_v15",
                    "odds_path_role_integrated_fixed_band_passthrough_v16",
                }
                else {}
            ),
            "timeout_seconds": (
                14_400
                if calibrator_strategy
                in {
                    "odds_path_market_offset_selection_conformal_discrete_ev_v10",
                    "odds_path_role_integrated_multihorizon_v11",
                    "odds_path_role_integrated_t300_nonlinear_v12",
                    "odds_path_role_integrated_edge_conditional_lcb_v13",
                    "odds_path_role_integrated_registered_band_lcb_v14",
                    "odds_path_role_integrated_selection_free_envelope_v15",
                    "odds_path_role_integrated_fixed_band_passthrough_v16",
                }
                else 7200
                if calibrator_strategy in {
                    "odds_path_return",
                    "odds_path_probability",
                    "odds_path_prequential_shrinkage_return",
                    "odds_path_crossfit_conservative_ev",
                    "odds_path_market_offset_crossfit_conservative_ev",
                    "odds_path_market_offset_discrete_log_ev_v9",
                }
                else 3600
            ),
        }
        range_key = formal_from.strftime("%Y%m%d") + f"-{through.day:02d}"
        job_id = enqueue_job(
            conn,
            task_type="market_residual_walk_forward",
            model_key=f"{label}:market_residual:{range_key}",
            parameters=parameters,
            priority=(
                DEPLOYMENT_DAILY_MARKET_PRIORITIES[calibrator_strategy]
                if calibrator_strategy in DEPLOYMENT_DAILY_MARKET_PRIORITIES
                else 99
                if calibrator_strategy
                == "odds_path_role_integrated_selection_free_envelope_v15"
                else 97
                if calibrator_strategy
                == "odds_path_observed_closing_return_schedule_quota_triple_head_v21"
                else 98
                if calibrator_strategy in {
                    "odds_path_return",
                    "odds_path_probability",
                }
                else 95
                if calibrator_strategy == "odds_path_prequential_shrinkage_return"
                else 94
                if calibrator_strategy == "odds_path_crossfit_conservative_ev"
                else 93
                if calibrator_strategy
                == "odds_path_market_offset_crossfit_conservative_ev"
                else 92
                if calibrator_strategy
                == "odds_path_market_offset_discrete_log_ev_v9"
                else 91
                if calibrator_strategy
                == "odds_path_market_offset_selection_conformal_discrete_ev_v10"
                else 90
                if calibrator_strategy
                == "odds_path_role_integrated_multihorizon_v11"
                else 89
                if calibrator_strategy
                == "odds_path_role_integrated_t300_nonlinear_v12"
                else 88
                if calibrator_strategy
                == "odds_path_role_integrated_edge_conditional_lcb_v13"
                else 87
                if calibrator_strategy
                == "odds_path_role_integrated_registered_band_lcb_v14"
                else 96
            ),
            max_attempts=2,
        )
        if job_id is not None:
            inserted.append(job_id)
    return inserted


def _configure_worker_database_memory() -> None:
    os.environ.setdefault("BOATRACE_PG_APPLICATION_NAME", "boatrace_evaluator")
    os.environ.setdefault("BOATRACE_PG_WORK_MEM", "128MB")


def run_worker(args: argparse.Namespace) -> int:
    _configure_worker_database_memory()
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
        recover_worker_job(conn, worker_id=worker_id, app_root=app_root)
        if is_leader:
            seed_work_tickets(conn)
    while True:
        try:
            now = time.monotonic()
            if is_leader:
                seeded_defaults = False
                scheduled_periodic = False
                with connection(args.db) as conn:
                    reconcile_completed_job_runs(
                        conn,
                        app_root=app_root,
                    )
                    requeue_stale_jobs(
                        conn,
                        stale_minutes=args.stale_minutes,
                        app_root=app_root,
                    )
                    reconcile_queue_state(conn)
                    reconcile_refined_market_evaluations(
                        conn,
                        app_root=app_root,
                    )
                    if (
                        args.seed_defaults
                        and now - last_seed >= args.seed_interval
                    ):
                        evaluation_date = (
                            datetime.now(JST).date() - timedelta(days=1)
                        ).isoformat()
                        seed_default_jobs(
                            conn,
                            evaluation_date=evaluation_date,
                            include_standardized=standardized_evaluation_due(
                                conn, evaluation_date=evaluation_date
                            ),
                        )
                        cancel_superseded_daily_jobs(
                            conn,
                            evaluation_date=evaluation_date,
                        )
                        seed_daily_market_jobs(
                            conn,
                            app_root=app_root,
                            evaluation_date=evaluation_date,
                        )
                        if genetic_cache_evaluation_date(app_root) == evaluation_date:
                            seed_daily_genetic_jobs(
                                conn,
                                evaluation_date=evaluation_date,
                            )
                        seeded_defaults = True
                    if (
                        args.schedule_periodic
                        and now - last_schedule >= 60.0
                    ):
                        seed_periodic_jobs(conn)
                        scheduled_periodic = True
                if seeded_defaults:
                    last_seed = now
                if scheduled_periodic:
                    last_schedule = now
            resources = system_resources(disk_path=app_root)
            with connection(args.db) as conn:
                job = claim_job(conn, worker_id=worker_id, resources=resources)
            if job is None:
                if args.once:
                    return 0
                time.sleep(args.poll_seconds)
                continue
            try:
                if not workspace_quota_allows(
                    app_root,
                    required_mb=job_workspace_reservation_mb(
                        job, app_root
                    ),
                ):
                    raise JobDependencyUnavailable(
                        "workspace quota cannot reserve the job disk requirement"
                    )
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
                    enqueue_refined_market_evaluation(
                        conn,
                        job,
                        app_root=app_root,
                    )
                    advance_genetic_islands(
                        conn,
                        job,
                        app_root=app_root,
                    )
            except JobDependencyUnavailable as exc:
                with connection(args.db) as conn:
                    defer_job(conn, job=job, reason=str(exc))
            except ObsoleteJob as exc:
                with connection(args.db) as conn:
                    cancel_obsolete_job(conn, job=job, reason=str(exc))
            except Exception as exc:
                with connection(args.db) as conn:
                    fail_job(
                        conn,
                        job=job,
                        error=f"{type(exc).__name__}: {exc}",
                        app_root=app_root,
                    )
            if args.once:
                return 0
        except KeyboardInterrupt:
            return 130
        except Exception as exc:
            print(f"evaluation worker error: {type(exc).__name__}: {exc}", flush=True)
            if args.once:
                raise
            time.sleep(args.poll_seconds)


def run_scheduler(args: argparse.Namespace) -> int:
    """Seed maintenance work independently from long-running evaluations."""
    _configure_worker_database_memory()
    app_root = Path(args.app_root).resolve()
    last_seed = 0.0
    last_schedule = 0.0
    with connection(args.db) as conn:
        ensure_schema(conn)
        seed_work_tickets(conn)
    while True:
        try:
            now = time.monotonic()
            seeded_defaults = False
            scheduled_periodic = False
            with connection(args.db) as conn:
                reconcile_completed_job_runs(
                    conn,
                    app_root=app_root,
                )
                requeue_stale_jobs(
                    conn,
                    stale_minutes=args.stale_minutes,
                    app_root=app_root,
                )
                reconcile_queue_state(conn)
                if args.seed_defaults and now - last_seed >= args.seed_interval:
                    evaluation_date = (
                        datetime.now(JST).date() - timedelta(days=1)
                    ).isoformat()
                    seed_default_jobs(
                        conn,
                        evaluation_date=evaluation_date,
                        include_standardized=standardized_evaluation_due(
                            conn, evaluation_date=evaluation_date
                        ),
                    )
                    cancel_superseded_daily_jobs(
                        conn,
                        evaluation_date=evaluation_date,
                    )
                    seed_daily_market_jobs(
                        conn,
                        app_root=app_root,
                        evaluation_date=evaluation_date,
                    )
                    if genetic_cache_evaluation_date(app_root) == evaluation_date:
                        seed_daily_genetic_jobs(
                            conn,
                            evaluation_date=evaluation_date,
                        )
                    seeded_defaults = True
                if now - last_schedule >= args.schedule_interval:
                    seed_periodic_jobs(conn)
                    scheduled_periodic = True
            if seeded_defaults:
                last_seed = now
            if scheduled_periodic:
                last_schedule = now
            if args.once:
                return 0
            time.sleep(args.poll_seconds)
        except KeyboardInterrupt:
            return 130
        except Exception as exc:
            print(f"evaluation scheduler error: {type(exc).__name__}: {exc}", flush=True)
            if args.once:
                raise
            time.sleep(args.poll_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PostgreSQL-backed model evaluation queue")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in (
        "init", "seed", "enqueue", "retry", "reprioritize", "status",
        "resummarize", "run", "schedule",
    ):
        command = sub.add_parser(name)
        command.add_argument("--db", default=DEFAULT_DSN)
        if name == "seed":
            command.add_argument("--evaluation-date", required=True)
        if name == "enqueue":
            command.add_argument("--task-type", choices=sorted(TASK_PROFILES), required=True)
            command.add_argument("--model-key", required=True)
            command.add_argument("--parameters-file", type=Path, required=True)
            command.add_argument("--priority", type=int, default=0)
            command.add_argument("--max-attempts", type=int, default=2)
            command.add_argument("--parent-job-id", type=int)
        if name == "retry":
            command.add_argument("--include-failed", action="store_true")
            command.add_argument("--include-running", action="store_true")
        if name == "reprioritize":
            command.add_argument("--job-id", type=int, required=True)
            command.add_argument("--priority", type=int, required=True)
            command.add_argument("--reason", required=True)
            command.add_argument("--ticket-key")
        if name == "resummarize":
            command.add_argument("--job-id", type=int, required=True)
            command.add_argument("--app-root", type=Path, default=Path("/workspace/boat"))
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
        if name == "schedule":
            command.add_argument("--app-root", default="/workspace/boat")
            command.add_argument("--poll-seconds", type=float, default=5.0)
            command.add_argument("--schedule-interval", type=float, default=60.0)
            command.add_argument("--seed-interval", type=float, default=3600.0)
            command.add_argument("--stale-minutes", type=int, default=180)
            command.add_argument("--seed-defaults", action="store_true")
            command.add_argument("--once", action="store_true")
    return parser


def load_job_parameters(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid parameters file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("parameters file must contain one JSON object")
    return value


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


def reprioritize_job(
    conn: Any,
    *,
    job_id: int,
    priority: int,
    reason: str,
    ticket_key: str | None = None,
) -> dict[str, Any]:
    if job_id < 1:
        raise ValueError("job_id must be positive")
    if not 0 <= priority <= 1000:
        raise ValueError("priority must be between 0 and 1000")
    note = str(reason).strip()
    if not note or len(note) > 500:
        raise ValueError("reason must contain between 1 and 500 characters")
    row = conn.execute(
        """
        UPDATE model_evaluation_jobs
        SET priority = ?, updated_at = CURRENT_TIMESTAMP
        WHERE job_id = ? AND status = 'queued'
        RETURNING job_id, priority, status
        """,
        (int(priority), int(job_id)),
    ).fetchone()
    if row is None:
        raise ValueError("job must exist and be queued")
    if ticket_key:
        ticket = conn.execute(
            """
            UPDATE work_tickets
            SET status = 'in_progress',
                progress = GREATEST(progress, 70),
                updated_at = CURRENT_TIMESTAMP
            WHERE ticket_key = ?
            RETURNING ticket_key, progress
            """,
            (ticket_key,),
        ).fetchone()
        if ticket is None:
            raise ValueError(f"unknown ticket: {ticket_key}")
        conn.execute(
            """
            INSERT INTO work_ticket_events(ticket_key, status, progress, note)
            VALUES (?, 'in_progress', ?, ?)
            """,
            (ticket_key, int(ticket["progress"]), note),
        )
    return {key: row[key] for key in row.keys()}


def resummarize_completed_job(
    conn: Any, *, job_id: int, app_root: Path
) -> dict[str, Any]:
    if job_id < 1:
        raise ValueError("job_id must be positive")
    row = conn.execute(
        """
        SELECT job_id, status, result_path
        FROM model_evaluation_jobs
        WHERE job_id = ?
        """,
        (job_id,),
    ).fetchone()
    if row is None or str(row["status"]) != "completed" or not row["result_path"]:
        raise ValueError("job must be completed and have a result_path")
    root = app_root.resolve()
    result_path = Path(str(row["result_path"]))
    if not result_path.is_absolute():
        result_path = root / result_path
    result_path = result_path.resolve()
    if root != result_path and root not in result_path.parents:
        raise ValueError("result_path must be inside app_root")
    _payload, summary = _load_result(result_path)
    conn.execute(
        """
        UPDATE model_evaluation_jobs
        SET result_summary = CAST(? AS JSONB), updated_at = CURRENT_TIMESTAMP
        WHERE job_id = ? AND status = 'completed'
        """,
        (_json(summary), job_id),
    )
    conn.execute(
        """
        UPDATE model_improvement_candidates
        SET metrics = CAST(? AS JSONB)
        WHERE job_id = ?
        """,
        (_json(summary), job_id),
    )
    return {
        "job_id": job_id,
        "result_path": str(result_path),
        "summary_keys": sorted(summary),
        "tail_diagnostics": "tail_portfolio_diagnostics" in summary,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return run_worker(args)
    if args.command == "schedule":
        return run_scheduler(args)
    with connection(args.db) as conn:
        ensure_schema(conn)
        if args.command == "seed":
            print(_json({"inserted": seed_default_jobs(conn, evaluation_date=args.evaluation_date)}))
        elif args.command == "enqueue":
            job_id = enqueue_job(
                conn,
                task_type=args.task_type,
                model_key=args.model_key,
                parameters=load_job_parameters(args.parameters_file),
                priority=args.priority,
                max_attempts=args.max_attempts,
                parent_job_id=args.parent_job_id,
            )
            print(_json({"job_id": job_id, "inserted": job_id is not None}))
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
        elif args.command == "reprioritize":
            print(
                _json(
                    reprioritize_job(
                        conn,
                        job_id=args.job_id,
                        priority=args.priority,
                        reason=args.reason,
                        ticket_key=args.ticket_key,
                    )
                )
            )
        elif args.command == "resummarize":
            print(
                _json(
                    resummarize_completed_job(
                        conn,
                        job_id=args.job_id,
                        app_root=args.app_root,
                    )
                )
            )
        elif args.command == "status":
            print(json.dumps(status_rows(conn), ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
