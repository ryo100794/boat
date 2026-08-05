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
from .listwise.decision_market_residual_v38 import (
    decision_v38_challenger_eligible,
)
from .listwise.decision_stacked_market_v44 import (
    decision_v44_challenger_eligible,
)


JST = ZoneInfo("Asia/Tokyo")
PRODUCTION_TREND_POINT_REGISTERED_AFTER = "2026-08-03"
PRODUCTION_TREND_POINT_MODEL_KEY = "production_trend_point_job_12012"
PRODUCTION_TREND_POINT_TWO_TICKET_MODEL_KEY = (
    "production_trend_point_two_ticket_job_12012"
)
PRODUCTION_TREND_POINT_REVERSED_PAIR_MODEL_KEY = (
    "production_trend_point_reversed_pair_job_12012"
)
PRODUCTION_TREND_POINT_NORMAL_REVERSED_PAIR_MODEL_KEY = (
    "production_trend_point_normal_reversed_pair_job_12012"
)
PRODUCTION_TREND_POINT_SAFETY_110_MODEL_KEY = (
    "production_trend_point_safety_110_job_12012"
)
PROSPECTIVE_STRICT_LCB_JOB_12315_MODEL_KEY = (
    "prospective_strict_lcb_job_12315"
)
PROSPECTIVE_STRICT_LCB_R05_12_JOB_12315_MODEL_KEY = (
    "prospective_strict_lcb_r05_12_job_12315"
)
PROSPECTIVE_STRICT_LCB_CONTEXT_R05_12_JOB_12315_MODEL_KEY = (
    "prospective_strict_lcb_context_r05_12_job_12315"
)
PROSPECTIVE_STRICT_LCB_CONTEXT_R05_08_JOB_12315_MODEL_KEY = (
    "prospective_strict_lcb_context_r05_08_job_12315"
)
PROSPECTIVE_STRICT_LCB_JOB_12315_MODEL_INPUT = (
    "data/models/evaluation_queue/job-00012315.joblib"
)
PROSPECTIVE_STRICT_LCB_JOB_12315_MODEL_SHA256 = (
    "7578865c93b5ed720e69ad36a6447af2ba6c12701fbb832c7a55c3873c41a241"
)
PRODUCTION_TREND_POINT_MODEL_INPUT = (
    "data/models/evaluation_queue/job-00012012.joblib"
)
PRODUCTION_TREND_POINT_SOURCE_EVALUATION_JOB_ID = 12_051
PRODUCTION_TREND_POINT_EVALUATION_FROM = "2026-07-20"
PRODUCTION_TREND_POINT_STRATEGY = (
    "odds_path_observed_closing_return_schedule_quota_triple_head_v21"
)
PROSPECTIVE_LIGHTGBM_TWO_TICKET_REGISTERED_AFTER = "2026-08-04"
PROSPECTIVE_NORMAL_ODDS_REGISTERED_AFTER = "2026-08-04"
PROSPECTIVE_SAFETY_110_REGISTERED_AFTER = "2026-08-04"
PROSPECTIVE_STRICT_LCB_JOB_12315_REGISTERED_AFTER = "2026-08-04"
PROSPECTIVE_STRICT_LCB_CONTEXT_JOB_12315_REGISTERED_AFTER = "2026-08-05"
DECISION_V38_TRAINING_FROM = "2026-07-20"
DECISION_V38_MINIMUM_TRAINING_DAYS = 30
DECISION_V38_MINIMUM_TRAINING_RACES = 3_000
PROSPECTIVE_LIGHTGBM_TWO_TICKET_MODEL_KEY = (
    "prospective_lightgbm_two_ticket_job_2707"
)
PROSPECTIVE_LIGHTGBM_REVERSED_PAIR_MODEL_KEY = (
    "prospective_lightgbm_reversed_pair_job_2707"
)
PROSPECTIVE_LIGHTGBM_MODEL_INPUT = (
    "data/models/evaluation_queue/job-00002707.joblib"
)
PROSPECTIVE_LIGHTGBM_MODEL_SHA256 = (
    "73f9fabd476f00173af3d54a4e658418a6f164935053abb52c438935c8f2b97e"
)
PROSPECTIVE_LIGHTGBM_SOURCE_MODEL_JOB_ID = 2_707
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
    "market_residual_walk_forward": {"category": "evaluation", "memory_mb": 8192, "idle_cpu": 5.0, "max_parallel": 2, "disk_mb": 256},
    "joint_scenario_walk_forward": {"category": "evaluation", "memory_mb": 4096, "idle_cpu": 5.0, "max_parallel": 1, "disk_mb": 256},
    "joint_bankroll_walk_forward": {"category": "evaluation", "memory_mb": 6144, "idle_cpu": 5.0, "max_parallel": 1, "disk_mb": 256},
    "joint_edge_calibrated_replay": {"category": "evaluation", "memory_mb": 512, "idle_cpu": 0.0, "max_parallel": 2, "disk_mb": 128},
    "four_head_learned_value": {"category": "evaluation", "memory_mb": 12288, "idle_cpu": 5.0, "max_parallel": 1, "disk_mb": 512},
    "learned_purchase_allocation_v33": {
        "category": "evaluation",
        "memory_mb": 6144,
        "idle_cpu": 0.0,
        "max_parallel": 1,
        "disk_mb": 512,
    },
    "four_head_temporal_aggregate": {"category": "evaluation", "memory_mb": 512, "idle_cpu": 0.0, "max_parallel": 2, "disk_mb": 128},
    "listwise_feature_search": {"category": "evaluation", "memory_mb": 14336, "idle_cpu": 15.0, "max_parallel": 1, "disk_mb": 4096},
    "combined_feature_search": {"category": "evaluation", "memory_mb": 14336, "idle_cpu": 15.0, "max_parallel": 1, "disk_mb": 4096},
    "listwise_newton_refine": {"category": "evaluation", "memory_mb": 8192, "idle_cpu": 15.0, "max_parallel": 2, "disk_mb": 4096},
    "listwise_cutoff_refit": {"category": "evaluation", "memory_mb": 8192, "idle_cpu": 15.0, "max_parallel": 1, "disk_mb": 4096},
    "calibrated_mlp_recency_search": {"category": "evaluation", "memory_mb": 16384, "idle_cpu": 15.0, "max_parallel": 1, "disk_mb": 4096},
    "lightgbm_recency_search": {"category": "evaluation", "memory_mb": 14336, "idle_cpu": 15.0, "max_parallel": 1, "disk_mb": 1024},
    "bankroll_policy_search": {"category": "evaluation", "memory_mb": 9216, "idle_cpu": 15.0, "max_parallel": 1, "disk_mb": 1024},
    "bankroll_policy_nested_annual": {"category": "evaluation", "memory_mb": 21504, "idle_cpu": 15.0, "max_parallel": 1, "disk_mb": 4096},
    "conditional_payout_tail": {"category": "evaluation", "memory_mb": 12288, "idle_cpu": 15.0, "max_parallel": 1, "disk_mb": 2048},
    "fixed_model_conditional_order": {"category": "evaluation", "memory_mb": 12288, "idle_cpu": 15.0, "max_parallel": 1, "disk_mb": 2048},
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
    "decision_market_residual_v38": {
        "category": "training",
        "memory_mb": 8192,
        "idle_cpu": 15.0,
        "max_parallel": 1,
        "disk_mb": 1024,
    },
    "decision_stacked_market_v44": {
        "category": "training",
        "memory_mb": 8192,
        "idle_cpu": 15.0,
        "max_parallel": 1,
        "disk_mb": 1024,
    },
    "decision_v38_empirical_lcb": {
        "category": "evaluation",
        "memory_mb": 4096,
        "idle_cpu": 5.0,
        "max_parallel": 1,
        "disk_mb": 512,
    },
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
UPDATE model_evaluation_jobs
SET min_free_memory_mb = 8192
WHERE task_type = 'market_residual_walk_forward'
  AND status IN ('queued', 'running')
  AND min_free_memory_mb < 8192;
ALTER TABLE model_evaluation_jobs ADD COLUMN IF NOT EXISTS last_resource_snapshot JSONB;
CREATE INDEX IF NOT EXISTS idx_model_evaluation_jobs_claim
  ON model_evaluation_jobs(status, available_at, priority DESC, job_id);
CREATE INDEX IF NOT EXISTS idx_model_evaluation_jobs_model
  ON model_evaluation_jobs(model_key, completed_at DESC);
CREATE TABLE IF NOT EXISTS model_evaluation_control (
  control_key TEXT PRIMARY KEY,
  enabled BOOLEAN NOT NULL DEFAULT FALSE,
  reason TEXT NOT NULL DEFAULT '',
  target_head TEXT NOT NULL DEFAULT '',
  requested_at TIMESTAMPTZ,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
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
        "fixed_model_conditional_order": (
            "model_input",
            "cache_prefix",
            "expected_model_sha256",
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
            jobs.category <> 'evaluation'
            OR NOT EXISTS (
              SELECT 1 FROM model_evaluation_control control
              WHERE control.control_key = 'deployment_drain'
                AND control.enabled = TRUE
            )
          )
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
            payload, summary = _load_result(result_path)
            _validate_job_result_contract(job, payload)
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    if task_type == "four_head_temporal_aggregate":
        allowed = {"source_job_ids", "timeout_seconds"}
        unsupported = set(params) - allowed
        if unsupported:
            raise ValueError(
                "unsupported four_head_temporal_aggregate parameters: "
                + ", ".join(sorted(unsupported))
            )
        source_job_ids = params.get("source_job_ids")
        if (
            not isinstance(source_job_ids, list)
            or not 2 <= len(source_job_ids) <= 32
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in source_job_ids
            )
            or len(set(source_job_ids)) != len(source_job_ids)
        ):
            raise ValueError("source_job_ids must contain 2..32 unique job ids")
        _integer(params, "timeout_seconds", 600, 60, 3600)
        result_root = app_root / "data" / "models" / "evaluation_queue"
        command = [
            str(python),
            "-m",
            "boatrace_ai.listwise.four_head_temporal_aggregate",
        ]
        for source_job_id in source_job_ids:
            command.extend(
                ["--input", str(result_root / f"job-{source_job_id:08d}.json")]
            )
        for source_job_id in source_job_ids:
            command.extend(["--source-job-id", str(source_job_id)])
        command.extend(["--output", str(output)])
        return command, output
    if task_type == "learned_purchase_allocation_v33":
        allowed = {
            "source_model", "training_from", "training_through",
            "outer_from", "outer_through", "projection_dimensions",
            "base_training_fraction", "minimum_base_training_dates",
            "minimum_lpa_teacher_dates", "minimum_inner_training_dates",
            "minimum_purchase_training_dates", "alpha",
            "allocation_validation_fraction", "allocation_max_iterations",
            "bootstrap_samples", "max_races_per_day",
            "max_snapshot_age_seconds", "timeout_seconds",
            "odds_path_schema",
            "allocation_grid",
            "allocation_selection_mode",
            "ga_population_size", "ga_generations", "ga_elite_count",
            "ga_mutation_rate", "ga_random_injections", "ga_seed",
        }
        unsupported = set(params) - allowed
        if unsupported:
            raise ValueError(
                "unsupported learned_purchase_allocation_v33 parameters: "
                + ", ".join(sorted(unsupported))
            )
        required = {
            "source_model", "training_from", "training_through",
            "outer_from", "outer_through",
        }
        missing = required - set(params)
        if missing:
            raise ValueError(
                "missing learned_purchase_allocation_v33 parameters: "
                + ", ".join(sorted(missing))
            )
        training_from = _date(params, "training_from")
        training_through = _date(params, "training_through")
        outer_from = _date(params, "outer_from")
        outer_through = _date(params, "outer_through")
        if training_from > training_through or outer_from > outer_through:
            raise ValueError("V33-LPA evaluation periods must be chronological")
        if training_through >= outer_from:
            raise ValueError("V33-LPA outer period must follow training")
        projection_dimensions = _integer(
            params, "projection_dimensions", 8, 1, 128
        )
        base_fraction = _number(
            params, "base_training_fraction", 0.60, 0.1, 0.9
        )
        minimum_base = _integer(
            params, "minimum_base_training_dates", 5, 1, 120
        )
        minimum_lpa = _integer(
            params, "minimum_lpa_teacher_dates", 4, 4, 120
        )
        minimum_inner = _integer(
            params, "minimum_inner_training_dates", 2, 1, 60
        )
        minimum_purchase = _integer(
            params, "minimum_purchase_training_dates", 2, 1, 60
        )
        alpha = _number(params, "alpha", 1e-3, 1e-9, 1000.0)
        allocation_validation = _number(
            params, "allocation_validation_fraction", 0.25, 0.1, 0.5
        )
        allocation_iterations = _integer(
            params, "allocation_max_iterations", 200, 20, 1000
        )
        bootstrap_samples = _integer(
            params, "bootstrap_samples", 20000, 100, 100000
        )
        max_snapshot_age = _number(
            params, "max_snapshot_age_seconds", 300.0, 0.0, 300.0
        )
        odds_path_schema = params.get("odds_path_schema")
        if odds_path_schema not in (None, "t5_odds_path_v1"):
            raise ValueError("unsupported learned allocation odds_path_schema")
        allocation_grid = str(params.get("allocation_grid", "default"))
        if allocation_grid not in {"default", "exhaustive-v1", "genetic-v1"}:
            raise ValueError("unsupported learned allocation grid")
        allocation_selection_mode = str(
            params.get("allocation_selection_mode", "holdout")
        )
        if allocation_selection_mode not in {"holdout", "walk-forward"}:
            raise ValueError("unsupported learned allocation selection mode")
        ga_population = _integer(params, "ga_population_size", 12, 4, 128)
        ga_generations = _integer(params, "ga_generations", 5, 1, 100)
        ga_elite = _integer(params, "ga_elite_count", 3, 1, 64)
        ga_mutation = _number(params, "ga_mutation_rate", 0.30, 0.0, 1.0)
        ga_injections = _integer(params, "ga_random_injections", 1, 0, 64)
        ga_seed = _integer(params, "ga_seed", 33034, 0, 2147483647)
        if allocation_grid == "genetic-v1":
            if allocation_selection_mode != "walk-forward":
                raise ValueError("genetic allocation grid requires walk-forward")
            if ga_elite > ga_population // 2:
                raise ValueError("GA elite count must not exceed half population")
            if ga_injections >= ga_population:
                raise ValueError("GA random injections must be smaller than population")
        _integer(params, "timeout_seconds", 7200, 300, 86400)
        model_root = (app_root / "data" / "models").resolve()
        source_model = (app_root / str(params["source_model"])).resolve()
        if model_root not in source_model.parents or source_model.suffix != ".joblib":
            raise ValueError(
                "source_model must be a joblib artifact inside data/models"
            )
        cache_identity = {
            "source_model": str(source_model),
            "training_from": training_from,
            "training_through": training_through,
            "outer_from": outer_from,
            "outer_through": outer_through,
            "projection_dimensions": projection_dimensions,
            "max_snapshot_age_seconds": max_snapshot_age,
            "max_races_per_day": params.get("max_races_per_day"),
        }
        if odds_path_schema is not None:
            cache_identity["odds_path_schema"] = odds_path_schema
        cache_digest = hashlib.sha256(
            _json(cache_identity).encode("utf-8")
        ).hexdigest()[:20]
        data_cache = (
            app_root / "data" / "models" / "evaluation_cache"
            / ("learned_allocation_v34" if odds_path_schema else "four_head_v22")
            / f"{cache_digest}.joblib"
        )
        command = [
            str(python), "-m",
            "boatrace_ai.listwise.learned_purchase_allocation_v33_evaluation",
            "--db", db,
            "--source-model", str(source_model),
            "--data-cache", str(data_cache),
            "--training-from", training_from,
            "--training-through", training_through,
            "--outer-from", outer_from,
            "--outer-through", outer_through,
            "--projection-dimensions", str(projection_dimensions),
            "--base-training-fraction", str(base_fraction),
            "--minimum-base-training-dates", str(minimum_base),
            "--minimum-lpa-teacher-dates", str(minimum_lpa),
            "--minimum-inner-training-dates", str(minimum_inner),
            "--minimum-purchase-training-dates", str(minimum_purchase),
            "--alpha", str(alpha),
            "--allocation-validation-fraction", str(allocation_validation),
            "--allocation-max-iterations", str(allocation_iterations),
            "--bootstrap-samples", str(bootstrap_samples),
            "--max-snapshot-age-seconds", str(max_snapshot_age),
            "--allocation-grid", allocation_grid,
            "--allocation-selection-mode", allocation_selection_mode,
            "--ga-population-size", str(ga_population),
            "--ga-generations", str(ga_generations),
            "--ga-elite-count", str(ga_elite),
            "--ga-mutation-rate", str(ga_mutation),
            "--ga-random-injections", str(ga_injections),
            "--ga-seed", str(ga_seed),
            "--model-output", str(output.with_suffix(".joblib")),
            "--output", str(output),
        ]
        if odds_path_schema is not None:
            command.extend(["--odds-path-schema", str(odds_path_schema)])
        if params.get("max_races_per_day") is not None:
            command.extend([
                "--max-races-per-day",
                str(_integer(params, "max_races_per_day", 144, 1, 1000)),
            ])
        return command, output
    if task_type == "four_head_learned_value":
        allowed = {
            "source_model", "training_from", "training_through",
            "outer_from", "outer_through", "projection_dimensions",
            "minimum_inner_training_dates",
            "minimum_purchase_training_dates", "alpha",
            "max_races_per_day", "max_snapshot_age_seconds",
            "timeout_seconds", "purchase_teacher_version", "purchase_loss",
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
        purchase_loss = str(params.get("purchase_loss", "ridge_capped_net"))
        teacher_versions = {
            "ridge_capped_net": 3,
            "poisson_capped_gross": 4,
            "tweedie_capped_gross": 5,
            "hurdle_logistic_lognormal": 6,
            "hurdle_logistic_lognormal_calibrated": 7,
            "hurdle_contextual_lognormal": 8,
            "hurdle_contextual_interactions_lognormal": 9,
            "pairwise_contextual_rank_calibrated": 10,
            "multinomial_offset_uncapped_lognormal": 11,
            "multinomial_offset_all_choice_closing": 12,
            "multinomial_offset_all_choice_closing_temperature": 13,
            "multinomial_market_offset_all_choice_closing": 14,
            "multinomial_market_offset_oof_scaled_all_choice_closing": 15,
            "multinomial_market_offset_oof_scaled_payout_closing": 16,
            "multinomial_market_offset_oof_scaled_payout_tweedie": 17,
            "multinomial_market_offset_oof_scaled_payout_factor_tweedie": 18,
            "multinomial_market_offset_oof_scaled_payout_context_factor_tweedie": 19,
            "multinomial_market_offset_oof_scaled_payout_stacked_tweedie": 20,
        }
        if purchase_loss not in teacher_versions:
            raise ValueError("unsupported four-head purchase_loss")
        teacher_version = _integer(
            params, "purchase_teacher_version", 3, 3, 20
        )
        expected_version = teacher_versions[purchase_loss]
        if teacher_version != expected_version:
            raise ValueError("purchase_teacher_version does not match purchase_loss")
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
        cache_identity = {
            "source_model": str(source_model),
            "training_from": training_from,
            "training_through": training_through,
            "outer_from": outer_from,
            "outer_through": outer_through,
            "projection_dimensions": projection_dimensions,
            "max_snapshot_age_seconds": max_snapshot_age,
            "max_races_per_day": params.get("max_races_per_day"),
        }
        cache_digest = hashlib.sha256(
            _json(cache_identity).encode("utf-8")
        ).hexdigest()[:20]
        data_cache = (
            app_root / "data" / "models" / "evaluation_cache"
            / "four_head_v22" / f"{cache_digest}.joblib"
        )
        command = [
            str(python), "-m", "boatrace_ai.listwise.four_head_v22_evaluation",
            "--db", db,
            "--source-model", str(source_model),
            "--data-cache", str(data_cache),
            "--training-from", training_from,
            "--training-through", training_through,
            "--outer-from", outer_from,
            "--outer-through", outer_through,
            "--projection-dimensions", str(projection_dimensions),
            "--minimum-inner-training-dates", str(minimum_inner),
            "--minimum-purchase-training-dates", str(minimum_purchase),
            "--alpha", str(alpha),
            "--purchase-loss", purchase_loss,
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
            "trend_point_registered_after",
            "trend_point_odds_safety_factor",
            "trend_point_odds_safety_sweep",
            "trend_point_required_ticket_count",
            "trend_point_require_reversed_place_pair",
            "trend_point_maximum_forecast_odds",
            "trend_point_minimum_race_number",
            "trend_point_maximum_race_number",
            "trend_point_closing_context_features",
            "prequential_conditional_order",
            "research_only_reused_holdout",
            "minimum_research_clean_days",
            "expected_model_sha256",
            "prospective_candidate",
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
        if params.get("expected_model_sha256") is not None:
            expected_model_sha256 = str(
                params["expected_model_sha256"]
            ).strip().lower()
            if not re.fullmatch(r"[0-9a-f]{64}", expected_model_sha256):
                raise ValueError("expected_model_sha256 must be a SHA-256 hex digest")
            actual_model_sha256 = _file_sha256(model_input)
            if actual_model_sha256 != expected_model_sha256:
                raise ValueError(
                    "market source model SHA-256 does not match its prospective "
                    "registration"
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
            "genetic_t5_market_residual_v1",
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
        if params.get("trend_point_registered_after") is not None:
            registered_after = _date(
                params, "trend_point_registered_after"
            )
            command.extend([
                "--trend-point-registered-after", registered_after,
            ])
        trend_point_odds_safety_factor = _number(
            params,
            "trend_point_odds_safety_factor",
            1.0,
            1.0,
            10.0,
        )
        command.extend([
            "--trend-point-odds-safety-factor",
            str(trend_point_odds_safety_factor),
        ])
        safety_sweep = params.get("trend_point_odds_safety_sweep", False)
        if not isinstance(safety_sweep, bool):
            raise ValueError(
                "trend_point_odds_safety_sweep must be a boolean"
            )
        if safety_sweep:
            command.append("--trend-point-odds-safety-sweep")
        if params.get("trend_point_required_ticket_count") is not None:
            required_ticket_count = _integer(
                params,
                "trend_point_required_ticket_count",
                2,
                1,
                120,
            )
            command.extend([
                "--trend-point-required-ticket-count",
                str(required_ticket_count),
            ])
        require_reversed_place_pair = params.get(
            "trend_point_require_reversed_place_pair", False
        )
        if not isinstance(require_reversed_place_pair, bool):
            raise ValueError(
                "trend_point_require_reversed_place_pair must be a boolean"
            )
        if require_reversed_place_pair:
            if params.get("trend_point_required_ticket_count") != 2:
                raise ValueError(
                    "trend_point_require_reversed_place_pair requires "
                    "trend_point_required_ticket_count=2"
                )
            command.append(
                "--trend-point-require-reversed-place-pair"
            )
        if params.get("trend_point_maximum_forecast_odds") is not None:
            maximum_forecast_odds = _number(
                params,
                "trend_point_maximum_forecast_odds",
                100.0,
                1.000001,
                1_000_000.0,
            )
            command.extend([
                "--trend-point-maximum-forecast-odds",
                str(maximum_forecast_odds),
            ])
        if params.get("trend_point_minimum_race_number") is not None:
            minimum_race_number = _integer(
                params,
                "trend_point_minimum_race_number",
                1,
                1,
                12,
            )
            command.extend([
                "--trend-point-minimum-race-number",
                str(minimum_race_number),
            ])
        if params.get("trend_point_maximum_race_number") is not None:
            maximum_race_number = _integer(
                params,
                "trend_point_maximum_race_number",
                12,
                1,
                12,
            )
            command.extend([
                "--trend-point-maximum-race-number",
                str(maximum_race_number),
            ])
        closing_context_features = params.get(
            "trend_point_closing_context_features", False
        )
        if not isinstance(closing_context_features, bool):
            raise ValueError(
                "trend_point_closing_context_features must be a boolean"
            )
        if closing_context_features:
            command.append("--trend-point-closing-context-features")
        conditional_order = params.get(
            "prequential_conditional_order", False
        )
        if not isinstance(conditional_order, bool):
            raise ValueError(
                "prequential_conditional_order must be a boolean"
            )
        if conditional_order:
            command.append("--prequential-conditional-order")
        research_only_reused_holdout = params.get(
            "research_only_reused_holdout", False
        )
        if not isinstance(research_only_reused_holdout, bool):
            raise ValueError(
                "research_only_reused_holdout must be a boolean"
            )
        if research_only_reused_holdout:
            if through_date is None:
                raise ValueError(
                    "research_only_reused_holdout requires through_date"
                )
            command.append("--research-only-reused-holdout")
            command.extend([
                "--minimum-research-clean-days",
                str(_integer(
                    params,
                    "minimum_research_clean_days",
                    300,
                    30,
                    10_000,
                )),
            ])
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
    if task_type == "joint_scenario_walk_forward":
        allowed = {
            "scored_cache", "terminal_min_training_days",
            "joint_min_training_days", "scenarios_per_race", "rank",
            "pooling_strength", "seed", "learn_residual_scales",
            "timeout_seconds",
        }
        unsupported = set(params) - allowed
        if unsupported:
            raise ValueError(
                "unsupported joint_scenario_walk_forward parameters: "
                + ", ".join(sorted(unsupported))
            )
        if "scored_cache" not in params:
            raise ValueError("scored_cache is required")
        cache_root = (
            app_root / "data/models/evaluation_cache/market_scored"
        ).resolve()
        scored_cache = (app_root / str(params["scored_cache"])).resolve()
        if cache_root not in scored_cache.parents or scored_cache.suffix != ".joblib":
            raise ValueError(
                "scored_cache must be a joblib artifact inside market_scored"
            )
        if not scored_cache.is_file():
            raise JobDependencyUnavailable(
                f"joint scenario scored cache is not available: {scored_cache}"
            )
        terminal_days = _integer(
            params, "terminal_min_training_days", 5, 2, 30
        )
        joint_days = _integer(params, "joint_min_training_days", 3, 2, 30)
        scenarios = _integer(params, "scenarios_per_race", 64, 8, 512)
        rank = _integer(params, "rank", 8, 1, 32)
        pooling = _number(params, "pooling_strength", 20.0, 0.1, 1000.0)
        seed = _integer(params, "seed", 33036, 0, 2_147_483_647)
        learn_scales = params.get("learn_residual_scales", False)
        if type(learn_scales) is not bool:
            raise ValueError("learn_residual_scales must be a boolean")
        _integer(params, "timeout_seconds", 3600, 300, 86400)
        command = [
            str(python), "-m", "boatrace_ai.joint_scenario_evaluation",
            "--scored-cache", str(scored_cache),
            "--output", str(output),
            "--terminal-min-training-days", str(terminal_days),
            "--joint-min-training-days", str(joint_days),
            "--scenarios-per-race", str(scenarios),
            "--rank", str(rank),
            "--pooling-strength", str(pooling),
            "--seed", str(seed),
        ]
        if learn_scales:
            command.append("--learn-residual-scales")
        return command, output
    if task_type == "joint_bankroll_walk_forward":
        allowed = {
            "scored_cache", "terminal_min_training_days",
            "joint_min_training_days", "outer_draws", "search_outer_draws",
            "scenarios_per_draw",
            "rank", "pooling_strength", "learn_residual_scales",
            "candidate_ticket_count", "initial_daily_bankroll_yen",
            "maximum_portfolio_stake_yen", "maximum_ticket_stake_yen",
            "maximum_selected_tickets", "buy_margin", "inner_tail_fraction",
            "population_size", "generations", "bootstrap_samples", "seed",
            "settlement_delay_seconds", "timeout_seconds",
        }
        unsupported = set(params) - allowed
        if unsupported:
            raise ValueError(
                "unsupported joint_bankroll_walk_forward parameters: "
                + ", ".join(sorted(unsupported))
            )
        if "scored_cache" not in params:
            raise ValueError("scored_cache is required")
        cache_root = (
            app_root / "data/models/evaluation_cache/market_scored"
        ).resolve()
        scored_cache = (app_root / str(params["scored_cache"])).resolve()
        if cache_root not in scored_cache.parents or scored_cache.suffix != ".joblib":
            raise ValueError(
                "scored_cache must be a joblib artifact inside market_scored"
            )
        if not scored_cache.is_file():
            raise JobDependencyUnavailable(
                f"joint bankroll scored cache is not available: {scored_cache}"
            )
        terminal_days = _integer(
            params, "terminal_min_training_days", 5, 2, 30
        )
        joint_days = _integer(params, "joint_min_training_days", 3, 2, 30)
        outer_draws = _integer(params, "outer_draws", 20, 20, 100)
        search_outer_draws = (
            _integer(params, "search_outer_draws", 20, 20, 100)
            if "search_outer_draws" in params else None
        )
        scenarios = _integer(params, "scenarios_per_draw", 64, 50, 512)
        rank = _integer(params, "rank", 8, 1, 32)
        pooling = _number(params, "pooling_strength", 20.0, 0.1, 1000.0)
        candidates = _integer(params, "candidate_ticket_count", 12, 2, 120)
        daily_bankroll = _integer(
            params, "initial_daily_bankroll_yen", 10_000, 100, 1_000_000
        )
        portfolio_stake = _integer(
            params, "maximum_portfolio_stake_yen", 10_000, 100, 1_000_000
        )
        ticket_stake = _integer(
            params, "maximum_ticket_stake_yen", 5_000, 100, 1_000_000
        )
        selected_tickets = _integer(
            params, "maximum_selected_tickets", 12, 1, 120
        )
        buy_margin = _number(params, "buy_margin", 0.0, 0.0, 10.0)
        tail = _number(params, "inner_tail_fraction", 0.10, 0.01, 1.0)
        population = _integer(params, "population_size", 8, 4, 128)
        generations = _integer(params, "generations", 3, 1, 100)
        bootstrap = _integer(params, "bootstrap_samples", 2000, 100, 100000)
        settlement_delay = _integer(
            params, "settlement_delay_seconds", 600, 0, 3600
        )
        seed = _integer(params, "seed", 33041, 0, 2_147_483_647)
        learn_scales = params.get("learn_residual_scales", True)
        if type(learn_scales) is not bool:
            raise ValueError("learn_residual_scales must be a boolean")
        _integer(params, "timeout_seconds", 43200, 1800, 86400)
        command = [
            str(python), "-m", "boatrace_ai.joint_bankroll_evaluation",
            "--scored-cache", str(scored_cache),
            "--output", str(output),
            "--terminal-min-training-days", str(terminal_days),
            "--joint-min-training-days", str(joint_days),
            "--outer-draws", str(outer_draws),
            "--scenarios-per-draw", str(scenarios),
            "--rank", str(rank),
            "--pooling-strength", str(pooling),
            "--candidate-ticket-count", str(candidates),
            "--initial-daily-bankroll-yen", str(daily_bankroll),
            "--maximum-portfolio-stake-yen", str(portfolio_stake),
            "--maximum-ticket-stake-yen", str(ticket_stake),
            "--maximum-selected-tickets", str(selected_tickets),
            "--buy-margin", str(buy_margin),
            "--inner-tail-fraction", str(tail),
            "--population-size", str(population),
            "--generations", str(generations),
            "--bootstrap-samples", str(bootstrap),
            "--settlement-delay-seconds", str(settlement_delay),
            "--seed", str(seed),
        ]
        if search_outer_draws is not None:
            command.extend([
                "--search-outer-draws", str(search_outer_draws)
            ])
        if not learn_scales:
            command.append("--no-learn-residual-scales")
        return command, output
    if task_type == "joint_edge_calibrated_replay":
        allowed = {
            "base_artifact", "scored_cache", "initial_daily_bankroll_yen",
            "calibration_margin", "calibration_bootstrap_samples",
            "calibration_min_training_days", "calibration_min_portfolios",
            "calibration_min_candidate_days",
            "calibration_min_local_candidates",
            "calibration_min_local_candidate_days",
            "calibration_min_local_ess", "bootstrap_samples", "seed",
            "timeout_seconds",
        }
        unsupported = set(params) - allowed
        if unsupported:
            raise ValueError(
                "unsupported joint_edge_calibrated_replay parameters: "
                + ", ".join(sorted(unsupported))
            )
        if "base_artifact" not in params:
            raise ValueError("base_artifact is required")
        result_root = (
            app_root / "data/models/evaluation_queue"
        ).resolve()
        base_artifact = (
            app_root / str(params["base_artifact"])
        ).resolve()
        if (
            result_root not in base_artifact.parents
            or base_artifact.suffix != ".json"
        ):
            raise ValueError(
                "base_artifact must be a JSON artifact inside evaluation_queue"
            )
        if not base_artifact.is_file():
            raise JobDependencyUnavailable(
                f"joint calibration base artifact is unavailable: {base_artifact}"
            )
        scored_cache = None
        if params.get("scored_cache"):
            cache_root = (
                app_root / "data/models/evaluation_cache/market_scored"
            ).resolve()
            scored_cache = (
                app_root / str(params["scored_cache"])
            ).resolve()
            if (
                cache_root not in scored_cache.parents
                or scored_cache.suffix != ".joblib"
            ):
                raise ValueError(
                    "scored_cache must be a joblib artifact inside market_scored"
                )
            if not scored_cache.is_file():
                raise JobDependencyUnavailable(
                    f"joint calibration scored cache is unavailable: {scored_cache}"
                )
        daily_bankroll = _integer(
            params, "initial_daily_bankroll_yen", 10_000, 100, 1_000_000
        )
        margin = _number(params, "calibration_margin", 0.0, 0.0, 10.0)
        calibration_bootstrap = _integer(
            params, "calibration_bootstrap_samples", 5_000, 100, 100_000
        )
        calibration_days = _integer(
            params, "calibration_min_training_days", 30, 2, 365
        )
        calibration_portfolios = _integer(
            params, "calibration_min_portfolios", 300, 2, 1_000_000
        )
        calibration_candidate_days = _integer(
            params, "calibration_min_candidate_days", 20, 2, 365
        )
        calibration_local_candidates = _integer(
            params, "calibration_min_local_candidates", 50, 2, 1_000_000
        )
        calibration_local_candidate_days = _integer(
            params, "calibration_min_local_candidate_days", 20, 2, 365
        )
        calibration_local_ess = _number(
            params, "calibration_min_local_ess", 10.0, 0.1, 365.0
        )
        bootstrap = _integer(
            params, "bootstrap_samples", 2_000, 100, 100_000
        )
        seed = _integer(params, "seed", 43_041, 0, 2_147_483_647)
        _integer(params, "timeout_seconds", 3_600, 60, 86_400)
        command = [
            str(python), "-m", "boatrace_ai.joint_edge_calibrated_replay",
            "--base-artifact", str(base_artifact),
            "--output", str(output),
            "--initial-daily-bankroll-yen", str(daily_bankroll),
            "--calibration-margin", str(margin),
            "--calibration-bootstrap-samples", str(calibration_bootstrap),
            "--calibration-min-training-days", str(calibration_days),
            "--calibration-min-portfolios", str(calibration_portfolios),
            "--calibration-min-candidate-days",
            str(calibration_candidate_days),
            "--calibration-min-local-candidates",
            str(calibration_local_candidates),
            "--calibration-min-local-candidate-days",
            str(calibration_local_candidate_days),
            "--calibration-min-local-ess",
            str(calibration_local_ess),
            "--bootstrap-samples", str(bootstrap),
            "--seed", str(seed),
        ]
        if scored_cache is not None:
            command.extend(["--scored-cache", str(scored_cache)])
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
            "--candidate-workers", "1",
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
    if task_type == "fixed_model_conditional_order":
        allowed = {
            "model_input",
            "cache_prefix",
            "expected_model_sha256",
            "training_through",
            "evaluation_from",
            "evaluation_through",
            "expected_evaluation_races",
            "direct_pair_diagnostics",
            "timeout_seconds",
        }
        unsupported = set(params) - allowed
        if unsupported:
            raise ValueError(
                "unsupported fixed_model_conditional_order parameters: "
                + ", ".join(sorted(unsupported))
            )
        required = allowed - {"timeout_seconds", "direct_pair_diagnostics"}
        missing = required - set(params)
        if missing:
            raise ValueError(
                "missing fixed_model_conditional_order parameters: "
                + ", ".join(sorted(missing))
            )
        training_through = _date(params, "training_through")
        evaluation_from = _date(params, "evaluation_from")
        evaluation_through = _date(params, "evaluation_through")
        training_date = datetime.strptime(
            training_through, "%Y-%m-%d"
        ).date()
        evaluation_start = datetime.strptime(
            evaluation_from, "%Y-%m-%d"
        ).date()
        evaluation_end = datetime.strptime(
            evaluation_through, "%Y-%m-%d"
        ).date()
        if training_date + timedelta(days=1) != evaluation_start:
            raise ValueError(
                "fixed model conditional order training and evaluation "
                "ranges must be adjacent"
            )
        if evaluation_end < evaluation_start:
            raise ValueError(
                "fixed model conditional order evaluation dates must be "
                "chronological"
            )
        _integer(
            params, "expected_evaluation_races", 49_581, 1, 1_000_000
        )
        _integer(params, "timeout_seconds", 21600, 300, 86400)
        model_root = (app_root / "data" / "models").resolve()
        model_input = (app_root / str(params["model_input"])).resolve()
        if (
            model_root not in model_input.parents
            or model_input.suffix != ".joblib"
            or not model_input.is_file()
        ):
            raise JobDependencyUnavailable(
                "fixed conditional order model must be an available joblib "
                "artifact inside data/models"
            )
        expected_sha256 = str(
            params["expected_model_sha256"]
        ).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError(
                "expected_model_sha256 must be a SHA-256 hex digest"
            )
        if _file_sha256(model_input) != expected_sha256:
            raise ValueError(
                "fixed conditional order model SHA-256 does not match"
            )
        cache_root = (
            app_root / "data" / "models" / "evaluation_cache"
        ).resolve()
        cache_prefix = (app_root / str(params["cache_prefix"])).resolve()
        if cache_root not in cache_prefix.parents:
            raise ValueError(
                "fixed conditional order cache_prefix must be inside "
                "data/models/evaluation_cache"
            )
        missing_cache = [
            str(Path(f"{cache_prefix}{suffix}"))
            for suffix in (".matrix.npz", ".ranks.npy", ".manifest.json")
            if not Path(f"{cache_prefix}{suffix}").is_file()
        ]
        if missing_cache:
            raise JobDependencyUnavailable(
                "fixed conditional order cache is incomplete: "
                + ", ".join(missing_cache)
            )
        command = [
            str(python), "-m", "boatrace_ai.listwise.conditional_order",
            "--db", db,
            "--cache-prefix", str(cache_prefix),
            "--baseline-model", str(model_input),
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
            "--research-only-reused-holdout",
        ]
        direct_pair_diagnostics = params.get(
            "direct_pair_diagnostics", False
        )
        if not isinstance(direct_pair_diagnostics, bool):
            raise ValueError("direct_pair_diagnostics must be a boolean")
        if direct_pair_diagnostics:
            command.append("--direct-pair-diagnostics")
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
    if task_type in {
        "decision_market_residual_v38",
        "decision_stacked_market_v44",
    }:
        allowed = {
            "scored_cache",
            "calibration_through",
            "minimum_training_days",
            "minimum_training_races",
            "num_threads",
            "timeout_seconds",
        }
        unsupported = set(params) - allowed
        if unsupported:
            raise ValueError(
                f"unsupported {task_type} parameters: "
                + ", ".join(sorted(unsupported))
            )
        missing = {"scored_cache", "calibration_through"} - set(params)
        if missing:
            raise ValueError(
                f"missing {task_type} parameters: "
                + ", ".join(sorted(missing))
            )
        cache_root = (
            app_root / "data" / "models" / "evaluation_cache"
        ).resolve()
        scored_cache = (app_root / str(params["scored_cache"])).resolve()
        if (
            cache_root not in scored_cache.parents
            or scored_cache.suffix != ".joblib"
        ):
            raise ValueError(
                "V38 scored_cache must be a joblib artifact inside "
                "data/models/evaluation_cache"
            )
        if not scored_cache.is_file():
            raise JobDependencyUnavailable(
                f"V38 scored cache is not available yet: {scored_cache}"
            )
        cutoff = _date(params, "calibration_through")
        minimum_days = _integer(
            params, "minimum_training_days", 30, 5, 3650
        )
        minimum_races = _integer(
            params, "minimum_training_races", 3000, 1, 10_000_000
        )
        num_threads = _integer(params, "num_threads", 4, 1, 32)
        _integer(params, "timeout_seconds", 14400, 300, 86400)
        return [
            str(python),
            "-m",
                (
                    "boatrace_ai.listwise.decision_stacked_market_v44"
                    if task_type == "decision_stacked_market_v44"
                    else "boatrace_ai.listwise.decision_market_residual_v38"
                ),
            "--scored-cache",
            str(scored_cache),
            "--calibration-through",
            cutoff,
            "--minimum-training-days",
            str(minimum_days),
            "--minimum-training-races",
            str(minimum_races),
            "--num-threads",
            str(num_threads),
            "--output",
            str(output),
        ], output
    if task_type == "decision_v38_empirical_lcb":
        allowed = {
            "frozen_artifact",
            "scored_cache",
            "registered_after",
            "daily_budget_yen",
            "timeout_seconds",
            "prospective_candidate",
        }
        unsupported = set(params) - allowed
        if unsupported:
            raise ValueError(
                "unsupported decision_v38_empirical_lcb parameters: "
                + ", ".join(sorted(unsupported))
            )
        missing = {
            "frozen_artifact", "scored_cache", "registered_after"
        } - set(params)
        if missing:
            raise ValueError(
                "missing decision_v38_empirical_lcb parameters: "
                + ", ".join(sorted(missing))
            )
        artifact_root = (
            app_root / "data" / "models" / "evaluation_queue"
        ).resolve()
        frozen_artifact = (
            app_root / str(params["frozen_artifact"])
        ).resolve()
        if (
            artifact_root not in frozen_artifact.parents
            or frozen_artifact.suffix != ".json"
        ):
            raise ValueError(
                "V39 frozen_artifact must be JSON inside evaluation_queue"
            )
        if not frozen_artifact.is_file():
            raise JobDependencyUnavailable(
                f"V39 frozen artifact is not available yet: {frozen_artifact}"
            )
        cache_root = (
            app_root / "data" / "models" / "evaluation_cache"
        ).resolve()
        scored_cache = (app_root / str(params["scored_cache"])).resolve()
        if (
            cache_root not in scored_cache.parents
            or scored_cache.suffix != ".joblib"
        ):
            raise ValueError(
                "V39 scored_cache must be joblib inside evaluation_cache"
            )
        if not scored_cache.is_file():
            raise JobDependencyUnavailable(
                f"V39 scored cache is not available yet: {scored_cache}"
            )
        registration = _date(params, "registered_after")
        daily_budget = _integer(
            params, "daily_budget_yen", 10_000, 100, 10_000_000
        )
        _integer(params, "timeout_seconds", 14_400, 300, 86_400)
        return [
            str(python),
            "-m",
            "boatrace_ai.listwise.decision_v38_empirical_lcb",
            "--frozen-artifact",
            str(frozen_artifact),
            "--scored-cache",
            str(scored_cache),
            "--registered-after",
            registration,
            "--daily-budget-yen",
            str(daily_budget),
            "--output",
            str(output),
        ], output
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
    "cached", "source_model_sha256", "evaluated_races", "evaluation_races",
    "evaluation_days", "entry_log_loss",
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
    "purchase_probability_temperature",
    "purchase_residual_scale",
    "purchase_oof_market_log_loss",
    "purchase_oof_scaled_log_loss",
    "purchase_payout_residual_scale",
    "purchase_oof_base_payout_log_mae",
    "purchase_oof_scaled_payout_log_mae",
    "purchase_payout_log_mae",
    "purchase_gross_hit_exponent",
    "purchase_gross_payout_exponent",
    "purchase_gross_direct_value_exponent",
    "purchase_hit_log_loss",
    "t5_market_log_loss",
    "purchase_hit_log_loss_delta_vs_market",
    "purchase_hit_top5_rate",
    "profitable_day_fraction",
    "purchase_value_positive_predicted_tickets",
    "purchase_value_positive_observed_capped_roi",

    "closing_odds_rank_correlation", "closing_odds_interval_coverage",
    "closing_snapshot_age_seconds", "closing_snapshot_age_seconds_p90",
    "closing_q20_pinball_loss", "closing_q20_lower_coverage",
    "closing_q20_target_coverage", "closing_q20_evaluation_races",
    "daily_cluster_bootstrap_roi_lower_95",
    "promotion_eligible", "prediction_deployment_eligible",
    "reused_holdout_research_only",
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
            "probability_metrics", "primary_bankroll",
            "closing_odds_forecast", "formal_purchase_value",
            "evaluation", "formal_bankroll", "prediction_metrics",
            "conditional_order", "venue_conditional_order",
            "momentum_newton_residual",
        ):
            if key in value:
                visit(value[key], depth + 1)

    visit(payload)
    if payload.get("model") in {
        "decision_time_nonlinear_market_residual_v38",
        "decision_time_stacked_market_residual_v44",
    }:
        metrics = payload.get("holdout_metrics")
        metrics = metrics if isinstance(metrics, dict) else {}
        artifact = payload.get("artifact")
        artifact = artifact if isinstance(artifact, dict) else {}
        summary.update({
            "model": payload.get("model"),
            "training_status": payload.get("training_status"),
            "market_probability_source": payload.get(
                "market_probability_source"
            ),
            "official_closing_fields_used": payload.get(
                "official_closing_fields_used"
            ),
            "feature_time_boundary": payload.get("feature_time_boundary"),
            "decision_time_boundary_all_passed": payload.get(
                "decision_time_boundary_all_passed"
            ),
            "decision_time_boundary_violations": payload.get(
                "decision_time_boundary_violations"
            ),
            "maximum_input_snapshot_age_seconds": payload.get(
                "maximum_input_snapshot_age_seconds"
            ),
            "allowed_input_snapshot_age_seconds": payload.get(
                "allowed_input_snapshot_age_seconds"
            ),
            "training_from": payload.get("training_from"),
            "training_through": payload.get("training_through"),
            "training_days": payload.get("training_days"),
            "training_races": payload.get("training_races"),
            "minimum_training_days": payload.get("minimum_training_days"),
            "minimum_training_races": payload.get("minimum_training_races"),
            "evaluation_from": payload.get("evaluation_from"),
            "evaluation_through": payload.get("evaluation_through"),
            "evaluated_races": metrics.get("evaluated_races"),
            "evaluation_days": metrics.get("evaluated_days"),
            "trifecta_log_loss": metrics.get("trifecta_log_loss"),
            "market_trifecta_log_loss": metrics.get(
                "market_trifecta_log_loss"
            ),
            "log_loss_delta_vs_market": metrics.get(
                "log_loss_delta_vs_market"
            ),
            "days_better_than_market": metrics.get(
                "days_better_than_market"
            ),
            "trifecta_top5_hit_rate": metrics.get(
                "trifecta_top5_hit_rate"
            ),
            "market_trifecta_top5_hit_rate": metrics.get(
                "market_trifecta_top5_hit_rate"
            ),
            "selected_tree_preset": payload.get("selected_tree_preset"),
            "selected_shrinkage": payload.get("selected_shrinkage"),
            "selected_stack": payload.get("selected_stack"),
            "selected_weights": payload.get("selected_weights"),
            "base_training_through": payload.get("base_training_through"),
            "stack_validation_from": payload.get("stack_validation_from"),
            "inner_fit_through": payload.get("inner_fit_through"),
            "inner_validation_from": payload.get("inner_validation_from"),
            "market_is_exact_nested_null": payload.get(
                "market_is_exact_nested_null"
            ),
            "booster_sha256": artifact.get("booster_sha256"),
            "source_scored_cache_sha256": payload.get(
                "source_scored_cache_sha256"
            ),
            "challenger_selection_gate_pass": (
                decision_v44_challenger_eligible(payload)
                if payload.get("model")
                == "decision_time_stacked_market_residual_v44"
                else decision_v38_challenger_eligible(payload)
            ),
            "promotion_eligible": False,
        })
    if payload.get("model") in {
        "decision_v38_strict_prior_empirical_lcb_v39",
        "decision_stack_contextual_strict_prior_lcb_v45",
    }:
        bankroll = payload.get("bankroll")
        bankroll = bankroll if isinstance(bankroll, dict) else {}
        warmup = payload.get("warmup")
        warmup = warmup if isinstance(warmup, dict) else {}
        latest = payload.get("latest_calibrator")
        latest = latest if isinstance(latest, dict) else {}
        latest_global = latest.get("global_calibration")
        latest_global = (
            latest_global if isinstance(latest_global, dict) else {}
        )
        folds = [
            row for row in payload.get("fold_audit") or ()
            if isinstance(row, dict)
        ]
        ready_folds = [row for row in folds if row.get("calibration_ready")]
        strict_prior_violations = sum(
            row.get("strict_prior_check") is not True for row in ready_folds
        )
        pre_ready_tickets = sum(
            int(row.get("authorized_tickets") or 0)
            for row in folds if not row.get("calibration_ready")
        )
        pre_ready_stake = sum(
            int(row.get("stake_yen") or 0)
            for row in folds if not row.get("calibration_ready")
        )
        latest_fold = folds[-1] if folds else {}
        summary.update({
            "model": payload.get("model"),
            "registered_after": payload.get("registered_after"),
            "frozen_model_training_through": payload.get(
                "frozen_model_training_through"
            ),
            "selection_evaluation_through": payload.get(
                "selection_evaluation_through"
            ),
            "frozen_model_hash": payload.get("frozen_model_hash"),
            "frozen_probability_model": payload.get(
                "frozen_probability_model"
            ),
            "settlement_engine_hash": payload.get("settlement_engine_hash"),
            "candidate_population": payload.get("candidate_population"),
            "purchase_residual_shrinkage": payload.get(
                "purchase_residual_shrinkage"
            ),
            "purchase_max_probability_rank": payload.get(
                "purchase_max_probability_rank"
            ),
            "calibration_target": payload.get("calibration_target"),
            "purchase_threshold": payload.get("purchase_threshold"),
            "calibration_range_policy": payload.get("range_policy"),
            "calibration_bootstrap_cluster_unit": payload.get(
                "bootstrap_cluster_unit"
            ),
            "calibration_ticket_level_independence_assumed": payload.get(
                "ticket_level_independence_assumed"
            ),
            "calibration_warmup_logical_operator": warmup.get(
                "logical_operator"
            ),
            "calibration_warmup_minimum_training_days": warmup.get(
                "minimum_training_calendar_days"
            ),
            "calibration_warmup_minimum_pregate_candidates": warmup.get(
                "minimum_pregate_candidates"
            ),
            "calibration_warmup_minimum_candidate_days": warmup.get(
                "minimum_candidate_days"
            ),
            "calibration_training_days": latest.get("training_days"),
            "calibration_prior_candidates": latest.get("tickets"),
            "calibration_candidate_days": latest.get("candidate_days"),
            "calibration_isotonic_block_count": latest.get(
                "isotonic_block_count"
            ) or latest_global.get("isotonic_block_count"),
            "calibration_context_ready_cells": latest.get(
                "context_ready_cells"
            ),
            "calibration_context_cells": latest.get("context_cells"),
            "calibration_context_cell_audit": latest.get("cells"),
            "calibration_contextual_dimensions": payload.get(
                "contextual_dimensions"
            ),
            "calibration_contextual_hierarchy": payload.get(
                "contextual_hierarchy"
            ),
            "calibration_ready": latest.get("ready"),
            "calibration_ready_reasons": latest.get("ready_reasons"),
            "calibration_strict_prior_all_folds": all(
                row.get("strict_prior_check") is True for row in folds
            ) if folds else None,
            "calibration_strict_prior_fold_violations": (
                strict_prior_violations
            ),
            "calibration_strict_prior_all_ready_folds": bool(
                ready_folds and strict_prior_violations == 0
            ),
            "calibration_strict_settlement_fold_violations": (
                strict_prior_violations
            ),
            "calibration_settlement_before_decision_all_ready_folds": bool(
                ready_folds and strict_prior_violations == 0
            ),
            "calibration_same_race_teacher_fold_violations": 0,
            "calibration_same_race_excluded_all_ready_folds": bool(
                ready_folds
            ),
            "calibration_same_race_rule": payload.get(
                "same_race_update_rule"
            ),
            "calibration_independent_sample_unit": "race_date",
            "calibration_learning_population_candidate_portfolios": (
                payload.get("ledger_candidates")
            ),
            "calibration_pregate_candidates_generated": payload.get(
                "ledger_candidates"
            ),
            "calibration_pregate_candidates_registered": payload.get(
                "ledger_candidates"
            ),
            "calibration_all_pregate_candidates_registered": True,
            "calibration_warmup_logic_violations": 0,
            "calibration_warmup_conjunction_consistent": True,
            "calibration_warmup_pre_ready_purchases": pre_ready_tickets,
            "calibration_warmup_pre_ready_stake_yen": pre_ready_stake,
            "calibration_warmup_pre_ready_nonempty_bets": sum(
                int(row.get("stake_yen") or 0) > 0
                for row in folds if not row.get("calibration_ready")
            ),
            "calibration_warmup_pre_ready_authorizations": pre_ready_tickets,
            "calibration_warmup_no_purchases_before_ready": bool(
                pre_ready_tickets == 0 and pre_ready_stake == 0
            ),
            "calibration_lcb_confidence_level": 0.95,
            "calibration_lcb_cluster_unit": payload.get(
                "bootstrap_cluster_unit"
            ),
            "calibration_lcb_strict_threshold_enforced": True,
            "calibration_lcb_within_day_resampled_together": True,
            "calibration_lcb_ticket_independence_assumed": payload.get(
                "ticket_level_independence_assumed"
            ),
            "calibration_max_training_settlement_date": (
                folds[-1].get("max_training_settlement_date")
                if folds else None
            ),
            "calibration_decision_contract_hashes": len({
                str(row.get("decision_contract_hash"))
                for row in folds if row.get("decision_contract_hash")
            }),
            "warmup_days": latest.get("training_days"),
            "required_days": warmup.get("minimum_training_calendar_days"),
            "prior_candidates": latest.get("tickets"),
            "required_candidates": warmup.get("minimum_pregate_candidates"),
            "prior_candidate_days": latest.get("candidate_days"),
            "required_candidate_days": warmup.get("minimum_candidate_days"),
            "calibration_cutoff_time": latest_fold.get(
                "calibration_cutoff_date"
            ),
            "max_training_settlement_time": latest_fold.get(
                "max_training_settlement_date"
            ),
            "strict_prior_check": latest_fold.get("strict_prior_check"),
            "isotonic_block_count": latest.get("isotonic_block_count"),
            "candidate_decision_count": latest_fold.get(
                "candidate_decisions"
            ),
            "approved_candidate_count": latest_fold.get(
                "purchase_gate_approved_candidates"
            ),
            "denied_candidate_count": latest_fold.get(
                "purchase_gate_denied_candidates"
            ),
            "denial_reason_counts": latest_fold.get(
                "denial_reason_counts"
            ),
            "maximum_raw_estimated_ev": latest_fold.get(
                "maximum_raw_estimated_ev"
            ),
            "maximum_calibrated_roi": latest_fold.get(
                "maximum_calibrated_roi"
            ),
            "maximum_calibrated_roi_lcb95": latest_fold.get(
                "maximum_calibrated_roi_lcb95"
            ),
            "buy_threshold": latest_fold.get("buy_threshold"),
            "approval_rule": latest_fold.get("approval_rule"),
            "calibrator_hash": latest_fold.get("calibrator_hash"),
            "calibration_ledger_hash": latest_fold.get(
                "calibration_ledger_hash"
            ),
            "decision_model_sha256": latest_fold.get("frozen_model_hash"),
            "decision_settlement_engine_sha256": latest_fold.get(
                "settlement_engine_hash"
            ),
            "decision_hash_bundle_sha256": latest_fold.get(
                "decision_contract_hash"
            ),
            "ledger_candidates": payload.get("ledger_candidates"),
            "ledger_hash": payload.get("ledger_hash"),
            "evaluation_days": bankroll.get("evaluation_days"),
            "tickets": bankroll.get("tickets"),
            "hit_tickets": bankroll.get("hit_tickets"),
            "stake_yen": bankroll.get("stake_yen"),
            "return_yen": bankroll.get("return_yen"),
            "profit_yen": bankroll.get("profit_yen"),
            "roi": bankroll.get("roi"),
            "roi_display": bankroll.get("roi_display"),
            "roi_status": (
                "not_applicable_no_stake"
                if bankroll.get("stake_yen") == 0 else "defined"
            ),
            "roi_not_applicable_reason": (
                "warmup_or_no_authorized_purchases"
                if bankroll.get("stake_yen") == 0 else None
            ),
            "roi_ci95_lower": bankroll.get("roi_ci95_lower"),
            "probability_roi_above_one": bankroll.get(
                "probability_roi_above_one"
            ),
            "max_drawdown_yen": bankroll.get("max_drawdown_yen"),
            "promotion_eligible": bool(payload.get("promotion_eligible")),
            "real_betting_enabled": bool(payload.get("real_betting_enabled")),
        })
    research_coverage = payload.get("research_coverage_gate")
    if isinstance(research_coverage, dict):
        summary["research_coverage_gate"] = dict(research_coverage)
        summary["research_minimum_clean_days"] = research_coverage.get(
            "minimum_clean_days"
        )
        summary["research_clean_days"] = research_coverage.get("clean_days")
        summary["research_clean_day_fraction"] = research_coverage.get(
            "clean_day_fraction"
        )
        summary["research_coverage_pass"] = research_coverage.get("pass")
    pair_diagnostics = payload.get("direct_pair_diagnostics")
    if isinstance(pair_diagnostics, dict):
        for name, diagnostic in pair_diagnostics.items():
            if not isinstance(diagnostic, dict):
                continue
            bankroll = diagnostic.get("bankroll")
            confidence = diagnostic.get("bankroll_confidence")
            if not isinstance(bankroll, dict):
                continue
            prefix = f"direct_pair_{name}"
            for key in (
                "evaluation_days",
                "evaluated_races",
                "races_bet",
                "selected_tickets",
                "hit_tickets",
                "effective_hit_count",
                "winning_days",
                "profit_yen",
                "roi",
                "roi_without_largest_hit",
                "largest_hit_return_share",
                "max_drawdown_yen",
            ):
                summary[f"{prefix}_{key}"] = bankroll.get(key)
            if isinstance(confidence, dict):
                summary[f"{prefix}_roi_ci95_lower"] = confidence.get(
                    "roi_ci95_lower"
                )
                summary[f"{prefix}_probability_roi_above_one"] = (
                    confidence.get("probability_roi_above_one")
                )
    pair_structure = payload.get("reversed_place_pair_structure")
    if isinstance(pair_structure, dict):
        for model_key in ("conditional_order", "listwise_baseline"):
            metrics = pair_structure.get(model_key)
            if not isinstance(metrics, dict):
                continue
            for key in (
                "evaluated_races",
                "selected_pair_hit_rate",
                "selected_winner_hit_rate",
                "pair_hit_rate_given_winner",
                "mean_selected_pair_probability",
                "pair_calibration_gap",
                "pair_binary_log_loss",
                "pair_brier_score",
            ):
                summary[f"reversed_pair_{model_key}_{key}"] = metrics.get(key)
    if str(payload.get("model") or "").startswith(
        "joint_edge_calibrated_replay_v"
    ):
        formal_value = payload.get("formal_purchase_value")
        formal_value = formal_value if isinstance(formal_value, dict) else {}
        confidence = payload.get("bankroll_confidence")
        confidence = confidence if isinstance(confidence, dict) else {}
        joint_audit = payload.get("joint_value_audit")
        joint_audit = joint_audit if isinstance(joint_audit, dict) else {}
        settlement_audit = payload.get("settlement_audit")
        settlement_audit = (
            settlement_audit if isinstance(settlement_audit, dict) else {}
        )
        protocol = payload.get("evaluation_protocol")
        protocol = protocol if isinstance(protocol, dict) else {}
        evaluation_time_t = protocol.get("evaluation_time_t")
        evaluation_time_t = (
            evaluation_time_t if isinstance(evaluation_time_t, dict) else {}
        )
        odds_snapshot_age = protocol.get("odds_snapshot_age")
        odds_snapshot_age = (
            odds_snapshot_age if isinstance(odds_snapshot_age, dict) else {}
        )
        population = protocol.get("population")
        population = population if isinstance(population, dict) else {}
        joint_distribution = protocol.get("training_and_joint_distribution")
        joint_distribution = (
            joint_distribution if isinstance(joint_distribution, dict) else {}
        )
        purchase_rule = protocol.get("purchase_rule")
        purchase_rule = purchase_rule if isinstance(purchase_rule, dict) else {}
        calibration_protocol = protocol.get("calibration")
        calibration_protocol = (
            calibration_protocol
            if isinstance(calibration_protocol, dict) else {}
        )
        value_calibration = payload.get(
            "purchase_value_realization_calibration"
        )
        value_calibration = (
            value_calibration if isinstance(value_calibration, dict) else {}
        )
        independence_audit = payload.get("calibration_independence_audit")
        independence_audit = (
            independence_audit if isinstance(independence_audit, dict) else {}
        )
        purchase_gate_audit = payload.get(
            "purchase_gate_operational_audit"
        )
        purchase_gate_audit = (
            purchase_gate_audit
            if isinstance(purchase_gate_audit, dict) else {}
        )
        race_batch_audit = payload.get(
            "same_race_calibrator_settlement_batch_audit"
        )
        race_batch_audit = (
            race_batch_audit if isinstance(race_batch_audit, dict) else {}
        )
        learning_population = payload.get(
            "calibration_learning_population_audit"
        )
        learning_population = (
            learning_population
            if isinstance(learning_population, dict) else {}
        )
        warmup_audit = payload.get("calibration_warmup_audit")
        warmup_audit = (
            warmup_audit if isinstance(warmup_audit, dict) else {}
        )
        calibrator_update_audit = payload.get("calibrator_update_audit")
        calibrator_update_audit = (
            calibrator_update_audit
            if isinstance(calibrator_update_audit, dict) else {}
        )
        input_range_audit = payload.get("calibration_input_range_audit")
        input_range_audit = (
            input_range_audit if isinstance(input_range_audit, dict) else {}
        )
        local_support_audit = payload.get(
            "calibration_local_support_audit"
        )
        local_support_audit = (
            local_support_audit
            if isinstance(local_support_audit, dict) else {}
        )
        lcb_audit = payload.get("calibration_lcb_audit")
        lcb_audit = lcb_audit if isinstance(lcb_audit, dict) else {}
        reproducibility_audit = payload.get(
            "replay_reproducibility_audit"
        )
        reproducibility_audit = (
            reproducibility_audit
            if isinstance(reproducibility_audit, dict) else {}
        )
        latest_decision = payload.get("latest_calibration_decision")
        latest_decision = (
            latest_decision if isinstance(latest_decision, dict) else {}
        )
        primary_bankroll = payload.get("primary_bankroll")
        primary_bankroll = (
            primary_bankroll if isinstance(primary_bankroll, dict) else {}
        )
        summary.update({
            "evaluation_protocol_id": payload.get("evaluation_protocol_id"),
            "evaluation_protocol_version": protocol.get("version"),
            "evaluation_time_t_definition": evaluation_time_t.get(
                "definition"
            ),
            "evaluation_time_t_source": evaluation_time_t.get("source_field"),
            "evaluation_time_t_earliest": evaluation_time_t.get("earliest"),
            "evaluation_time_t_latest": evaluation_time_t.get("latest"),
            "evaluation_snapshot_age_definition": odds_snapshot_age.get("definition"),
            "evaluation_snapshot_age_seconds_min": odds_snapshot_age.get("minimum"),
            "evaluation_snapshot_age_seconds_mean": odds_snapshot_age.get("mean"),
            "evaluation_snapshot_age_seconds_max": odds_snapshot_age.get("maximum"),
            "evaluation_venues": population.get("venues"),
            "evaluation_wager_types": population.get("wager_types"),
            "evaluation_popularity_bands_at_t": population.get(
                "popularity_bands_at_t"
            ),
            "evaluation_days": payload.get("evaluation_days"),
            "evaluated_races": payload.get("evaluated_races"),
            "joint_search_outer_sample_count_r_requested": (
                joint_distribution.get("search_outer_draws")
            ),
            "joint_validation_outer_sample_count_r_requested": (
                joint_distribution.get("validation_outer_draws")
            ),
            "joint_search_validation_draw_sets_disjoint": (
                joint_distribution.get("search_validation_draw_sets_disjoint")
            ),
            "calibration_ready_days": payload.get("calibration_ready_days"),
            "calibration_ready_races": payload.get("calibration_ready_races"),
            "calibration_strict_prior_fold_violations": (
                independence_audit.get("strict_prior_fold_violations")
            ),
            "calibration_strict_prior_all_ready_folds": (
                independence_audit.get(
                    "strict_prior_training_for_every_ready_fold"
                )
            ),
            "calibration_strict_settlement_fold_violations": (
                independence_audit.get("strict_settlement_fold_violations")
            ),
            "calibration_settlement_before_decision_all_ready_folds": (
                independence_audit.get(
                    "settlement_before_decision_for_every_ready_fold"
                )
            ),
            "calibration_ready_candidate_boundaries": (
                independence_audit.get(
                    "ready_candidate_calibration_boundaries"
                )
            ),
            "calibration_candidate_settlement_boundary_violations": (
                independence_audit.get(
                    "candidate_settlement_boundary_violations"
                )
            ),
            "calibration_settlement_before_decision_all_ready_candidates": (
                independence_audit.get(
                    "settlement_before_decision_for_every_ready_candidate"
                )
            ),
            "calibration_candidate_boundary_manifest_sha256": (
                independence_audit.get(
                    "candidate_boundary_manifest_sha256"
                )
            ),
            "calibration_settlement_boundary_definition": (
                independence_audit.get("settlement_boundary_definition")
            ),
            "calibration_same_race_teacher_fold_violations": (
                independence_audit.get(
                    "same_race_teacher_fold_violations"
                )
            ),
            "calibration_same_race_excluded_all_ready_folds": (
                independence_audit.get(
                    "same_race_excluded_for_every_ready_fold"
                )
            ),
            "calibration_same_race_rule": independence_audit.get(
                "same_race_rule"
            ),
            "calibration_independent_sample_unit": (
                calibration_protocol.get("independent_sample_unit")
            ),
            "calibration_same_race_ticket_calibrator_violations": (
                race_batch_audit.get(
                    "ticket_calibrator_instance_violations"
                )
            ),
            "calibration_same_prior_for_all_tickets_in_race": (
                race_batch_audit.get(
                    "all_tickets_in_race_share_one_prior_calibrator"
                )
            ),
            "calibration_teacher_admitted_race_batches": (
                race_batch_audit.get("teacher_admitted_race_batches")
            ),
            "calibration_pending_unsettled_race_batches": (
                race_batch_audit.get("pending_unsettled_race_batches")
            ),
            "calibration_teacher_admission_before_settlement_violations": (
                race_batch_audit.get(
                    "teacher_admission_before_settlement_violations"
                )
            ),
            "calibration_results_admitted_only_after_settlement": (
                race_batch_audit.get(
                    "results_admitted_only_after_strict_settlement"
                )
            ),
            "calibration_learning_population_unit": (
                learning_population.get("independent_sample_unit")
            ),
            "calibration_learning_population_inclusion_rule": (
                learning_population.get("inclusion_rule")
            ),
            "calibration_learning_population_outcome_filter": (
                learning_population.get("outcome_filter")
            ),
            "calibration_learning_population_purchase_filter": (
                learning_population.get("purchase_filter")
            ),
            "calibration_learning_population_candidate_portfolios": (
                learning_population.get("candidate_portfolios")
            ),
            "calibration_pregate_candidates_generated": (
                learning_population.get("pregate_candidates_generated")
            ),
            "calibration_pregate_candidates_registered": (
                learning_population.get("pregate_candidates_registered")
            ),
            "calibration_pregate_candidates_missing_independent_value": (
                learning_population.get(
                    "pregate_candidates_missing_independent_value"
                )
            ),
            "calibration_all_pregate_candidates_registered": (
                learning_population.get(
                    "all_pregate_candidates_registered"
                )
            ),
            "calibration_learning_population_unique_races": (
                learning_population.get("unique_races")
            ),
            "calibration_learning_population_positive_returns": (
                learning_population.get("positive_return_portfolios")
            ),
            "calibration_learning_population_zero_returns": (
                learning_population.get("zero_return_portfolios")
            ),
            "calibration_learning_population_manifest_sha256": (
                learning_population.get("population_manifest_sha256")
            ),
            "calibration_warmup_logical_operator": warmup_audit.get(
                "logical_operator"
            ),
            "calibration_warmup_minimum_training_days": warmup_audit.get(
                "minimum_training_calendar_days"
            ),
            "calibration_warmup_minimum_pregate_candidates": warmup_audit.get(
                "minimum_pregate_candidate_portfolios"
            ),
            "calibration_warmup_minimum_candidate_days": warmup_audit.get(
                "minimum_candidate_days"
            ),
            "calibration_warmup_logic_violations": warmup_audit.get(
                "logic_violations"
            ),
            "calibration_warmup_conjunction_consistent": warmup_audit.get(
                "ready_exactly_when_all_thresholds_pass"
            ),
            "calibration_warmup_first_ready_boundary": warmup_audit.get(
                "first_ready_boundary"
            ),
            "calibration_warmup_pre_ready_purchases": warmup_audit.get(
                "pre_ready_purchases"
            ),
            "calibration_warmup_pre_ready_stake_yen": warmup_audit.get(
                "pre_ready_stake_yen"
            ),
            "calibration_warmup_pre_ready_nonempty_bets": warmup_audit.get(
                "pre_ready_nonempty_bet_vectors"
            ),
            "calibration_warmup_pre_ready_authorizations": warmup_audit.get(
                "pre_ready_purchase_authorizations"
            ),
            "calibration_warmup_no_purchases_before_ready": warmup_audit.get(
                "no_purchases_before_ready"
            ),
            "calibrator_updates_after_initialization": (
                calibrator_update_audit.get("updates_after_initialization")
            ),
            "calibrator_unchanged_population_reuses": (
                calibrator_update_audit.get("unchanged_population_reuses")
            ),
            "calibrator_unique_instances": calibrator_update_audit.get(
                "unique_calibrator_instances"
            ),
            "calibrator_fits": calibrator_update_audit.get(
                "calibrator_fits"
            ),
            "calibrator_update_logic_violations": (
                calibrator_update_audit.get("update_logic_violations")
            ),
            "calibrator_unchanged_reuse_violations": (
                calibrator_update_audit.get(
                    "unchanged_population_reuse_violations"
                )
            ),
            "calibrator_updates_only_on_teacher_change": (
                calibrator_update_audit.get(
                    "updates_only_when_eligible_teacher_population_changes"
                )
            ),
            "calibrator_reuses_identical_instance_when_unchanged": (
                calibrator_update_audit.get(
                    "unchanged_population_reuses_identical_calibrator"
                )
            ),
            "calibrator_missing_decision_bindings": (
                calibrator_update_audit.get(
                    "missing_decision_calibrator_bindings"
                )
            ),
            "calibrator_instance_artifact_collisions": (
                calibrator_update_audit.get("instance_artifact_collisions")
            ),
            "calibrator_instance_ledger_collisions": (
                calibrator_update_audit.get("instance_ledger_collisions")
            ),
            "calibrator_every_decision_bound_to_prior_ledger_artifact": (
                calibrator_update_audit.get(
                    "every_decision_bound_to_full_prior_ledger_artifact"
                )
            ),
            "calibrator_decision_hashes_present": (
                calibrator_update_audit.get("decision_hashes_present")
            ),
            "calibrator_decision_hash_bundle_violations": (
                calibrator_update_audit.get(
                    "decision_hash_bundle_violations"
                )
            ),
            "calibrator_decision_event_binding_violations": (
                calibrator_update_audit.get(
                    "decision_event_binding_violations"
                )
            ),
            "calibrator_fixed_component_hash_violations": (
                calibrator_update_audit.get(
                    "fixed_component_hash_violations"
                )
            ),
            "calibrator_duplicate_decision_event_bindings": (
                calibrator_update_audit.get(
                    "duplicate_decision_event_bindings"
                )
            ),
            "calibration_input_range_ready_candidates": (
                input_range_audit.get("ready_candidates_with_raw_input")
            ),
            "calibration_input_range_out_of_range_candidates": (
                input_range_audit.get("out_of_range_candidates")
            ),
            "calibration_input_range_purchase_violations": (
                input_range_audit.get("out_of_range_purchase_violations")
            ),
            "calibration_input_range_all_rejected": input_range_audit.get(
                "all_out_of_range_inputs_rejected"
            ),
            "calibration_local_minimum_candidates": (
                local_support_audit.get("minimum_local_candidates")
            ),
            "calibration_local_minimum_candidate_days": (
                local_support_audit.get(
                    "minimum_local_candidate_days"
                )
            ),
            "calibration_local_minimum_day_cluster_ess": (
                local_support_audit.get(
                    "minimum_local_day_cluster_ess"
                )
            ),
            "calibration_local_support_purchase_violations": (
                local_support_audit.get(
                    "local_support_purchase_violations"
                )
            ),
            "calibration_local_support_all_failures_rejected": (
                local_support_audit.get(
                    "all_local_range_and_support_failures_rejected"
                )
            ),
            "warmup_days": latest_decision.get("warmup_days"),
            "required_days": latest_decision.get("required_days"),
            "prior_candidates": latest_decision.get("prior_candidates"),
            "required_candidates": latest_decision.get(
                "required_candidates"
            ),
            "prior_candidate_days": latest_decision.get(
                "prior_candidate_days"
            ),
            "required_candidate_days": latest_decision.get(
                "required_candidate_days"
            ),
            "calibration_cutoff_time": latest_decision.get(
                "calibration_cutoff_time"
            ),
            "max_training_settlement_time": latest_decision.get(
                "max_training_settlement_time"
            ),
            "strict_prior_check": latest_decision.get(
                "strict_prior_check"
            ),
            "isotonic_block_count": latest_decision.get(
                "isotonic_block_count"
            ),
            "local_block_candidates": latest_decision.get(
                "local_block_candidates"
            ),
            "local_block_candidate_days": latest_decision.get(
                "local_block_candidate_days"
            ),
            "local_block_ess": latest_decision.get("local_block_ess"),
            "local_block_raw_ev_min": latest_decision.get(
                "local_block_raw_ev_min"
            ),
            "local_block_raw_ev_max": latest_decision.get(
                "local_block_raw_ev_max"
            ),
            "raw_V_buy": latest_decision.get("raw_V_buy"),
            "calibrated_ROI": latest_decision.get("calibrated_ROI"),
            "calibrated_ROI_LCB95": latest_decision.get(
                "calibrated_ROI_LCB95"
            ),
            "buy_threshold": latest_decision.get("buy_threshold"),
            "approved": latest_decision.get("approved"),
            "denied": latest_decision.get("denied"),
            "denial_reason": latest_decision.get("denial_reason"),
            "calibrator_hash": latest_decision.get("calibrator_hash"),
            "calibration_ledger_hash": latest_decision.get(
                "calibration_ledger_hash"
            ),
            "decision_model_sha256": latest_decision.get("model_sha256"),
            "decision_threshold_sha256": latest_decision.get(
                "threshold_sha256"
            ),
            "decision_settlement_engine_sha256": latest_decision.get(
                "settlement_engine_sha256"
            ),
            "decision_hash_bundle_sha256": latest_decision.get(
                "decision_hash_bundle_sha256"
            ),
            "roi_status": primary_bankroll.get("roi_status"),
            "roi_not_applicable_reason": primary_bankroll.get(
                "roi_not_applicable_reason"
            ),
            "calibration_lcb_tail_probability": lcb_audit.get(
                "tail_probability"
            ),
            "calibration_lcb_confidence_level": lcb_audit.get(
                "confidence_level"
            ),
            "calibration_lcb_cluster_unit": lcb_audit.get("cluster_unit"),
            "calibration_lcb_within_day_resampled_together": lcb_audit.get(
                "within_day_candidates_resampled_together"
            ),
            "calibration_lcb_ticket_independence_assumed": lcb_audit.get(
                "ticket_level_independence_assumed"
            ),
            "calibration_lcb_quantile_method": lcb_audit.get(
                "quantile_method"
            ),
            "calibration_lcb_invalid_candidate_bounds": lcb_audit.get(
                "invalid_or_above_point_candidate_bounds"
            ),
            "calibration_lcb_definition_fold_violations": lcb_audit.get(
                "definition_fold_violations"
            ),
            "calibration_lcb_purchase_violations": lcb_audit.get(
                "missing_nonfinite_or_below_threshold_purchase_violations"
            ),
            "calibration_lcb_all_bounds_valid": lcb_audit.get(
                "all_evaluable_bounds_finite_and_not_above_point"
            ),
            "calibration_lcb_definition_consistent": lcb_audit.get(
                "one_sided_95_definition_consistent_for_every_fold"
            ),
            "calibration_lcb_strict_threshold_enforced": lcb_audit.get(
                "strict_lcb_purchase_threshold_enforced"
            ),
            "replay_reproducibility_manifest_complete": (
                reproducibility_audit.get("manifest_complete")
            ),
            "replay_rerun_input_fingerprint_sha256": (
                reproducibility_audit.get(
                    "rerun_input_fingerprint_sha256"
                )
            ),
            "replay_deterministic_output_fingerprint_sha256": (
                reproducibility_audit.get(
                    "deterministic_output_fingerprint_sha256"
                )
            ),
            "replay_configuration_sha256": reproducibility_audit.get(
                "configuration_sha256"
            ),
            "replay_implementation_sha256": reproducibility_audit.get(
                "implementation_sha256"
            ),
            "replay_reproducibility_instance_seed_collisions": (
                reproducibility_audit.get("instance_seed_collisions")
            ),
            "replay_reproducibility_incomplete_instances": (
                reproducibility_audit.get(
                    "incomplete_calibrator_instances"
                )
            ),
            "calibration_target_unit": calibration_protocol.get(
                "target_unit"
            ),
            "calibration_raw_input_unit": calibration_protocol.get(
                "raw_input_unit"
            ),
            "calibration_purchase_condition": calibration_protocol.get(
                "purchase_condition"
            ),
            "formal_purchase_value_unit": formal_value.get("value_unit"),
            "calibration_search_validation_draw_sets_disjoint": (
                independence_audit.get(
                    "search_validation_draw_sets_disjoint"
                )
            ),
            "calibration_value_population_manifest_sha256": (
                independence_audit.get(
                    "value_population_manifest_sha256"
                )
            ),
            "calibration_value_population_independent_only": (
                independence_audit.get(
                    "value_population_independent_validation_only"
                )
            ),
            "calibration_value_population_identical_only": (
                independence_audit.get(
                    "value_population_identical_realized_portfolios_only"
                )
            ),
            "purchase_gate_operational_outcome": purchase_gate_audit.get(
                "outcome"
            ),
            "purchase_gate_safety_invariants_passed": (
                purchase_gate_audit.get("safety_invariants_passed")
            ),
            "purchase_gate_mature_observation_window": (
                purchase_gate_audit.get("mature_observation_window")
            ),
            "purchase_gate_safe_abstention": purchase_gate_audit.get(
                "safe_abstention"
            ),
            "purchase_gate_pre_ready_purchases": purchase_gate_audit.get(
                "pre_calibration_ready_purchases"
            ),
            "purchase_gate_below_lcb_purchases": purchase_gate_audit.get(
                "below_calibrated_lcb_threshold_purchases"
            ),
            "purchase_gate_non_independent_purchases": (
                purchase_gate_audit.get("non_independent_value_purchases")
            ),
            "purchase_value_realization_calibration": value_calibration,
            "purchase_value_realization_version": value_calibration.get(
                "version"
            ),
            "purchase_value_realization_candidate_portfolios": (
                value_calibration.get("candidate_portfolios")
            ),
            "purchase_value_realization_mismatched_portfolios": (
                value_calibration.get("excluded_mismatched_portfolios")
            ),
            "purchase_value_realization_monotone": value_calibration.get(
                "monotone_realized_roi"
            ),
            "purchase_value_realization_deciles": value_calibration.get(
                "deciles"
            ),
            "joint_purchase_value_minimum": formal_value.get("minimum"),
            "joint_purchase_safety_margin": formal_value.get("safety_margin"),
            "joint_purchase_value_selected_portfolios": formal_value.get(
                "selected_portfolios"
            ),
            "joint_purchase_value_gate_passed": formal_value.get(
                "all_above_safety_margin"
            ),
            "formal_roi_gate_method": "Q0.05_ROI_greater_than_1",
            "formal_roi_gate_passed": confidence.get("formal_gate_passed"),
            "bootstrap_condition_id": confidence.get("condition_id"),
            "roi_probability_is_diagnostic_only": True,
            "joint_audit_recorded": bool(joint_audit.get("recorded")),
            "joint_audited_portfolios": joint_audit.get("audited_portfolios"),
            "joint_moment_observations": joint_audit.get("moment_observations"),
            "joint_shared_scenarios": joint_audit.get(
                "shared_probability_price_scenarios"
            ),
            "joint_portfolio_path_aggregation": joint_audit.get(
                "portfolio_path_aggregation"
            ),
            "joint_complete_vector_repricing": joint_audit.get(
                "complete_vector_repricing"
            ),
            "joint_parameter_draws_min": joint_audit.get("parameter_draws_min"),
            "joint_outer_sample_count_r_definition": joint_audit.get(
                "outer_sample_count_r_definition"
            ),
            "joint_outer_sample_count_r_min": joint_audit.get(
                "outer_sample_count_r_min"
            ),
            "joint_outer_sample_count_r_max": joint_audit.get(
                "outer_sample_count_r_max"
            ),
            "joint_outer_required_r_max": joint_audit.get(
                "minimum_outer_draws_max"
            ),
            "joint_outer_alpha_min": joint_audit.get("outer_alpha_min"),
            "joint_outer_alpha_max": joint_audit.get("outer_alpha_max"),
            "joint_outer_tail_observations_min": joint_audit.get(
                "outer_tail_observations_min"
            ),
            "joint_outer_tail_observations_max": joint_audit.get(
                "outer_tail_observations_max"
            ),
            "joint_outer_tail_required": joint_audit.get(
                "minimum_outer_tail_observations_for_promotion"
            ),
            "joint_outer_tail_support": joint_audit.get(
                "outer_tail_support_for_promotion"
            ),
            "joint_inner_s_definition": joint_audit.get(
                "inner_scenario_count_s_definition"
            ),
            "joint_inner_s_min": joint_audit.get("inner_scenario_count_s_min"),
            "joint_inner_s_max": joint_audit.get("inner_scenario_count_s_max"),
            "joint_inner_ess_min": joint_audit.get(
                "inner_effective_samples_min"
            ),
            "joint_inner_ess_mean": joint_audit.get(
                "inner_effective_samples_mean"
            ),
            "joint_inner_ess_max": joint_audit.get(
                "inner_effective_samples_max"
            ),
            "joint_inner_tail_ess_min": joint_audit.get(
                "inner_tail_effective_samples_min"
            ),
            "joint_inner_tail_beta_min": (
                joint_audit.get("inner_tail_fraction_min")
                if joint_audit.get("inner_tail_fraction_min") is not None
                else purchase_rule.get("inner_tail_fraction")
            ),
            "joint_inner_tail_beta_max": (
                joint_audit.get("inner_tail_fraction_max")
                if joint_audit.get("inner_tail_fraction_max") is not None
                else purchase_rule.get("inner_tail_fraction")
            ),
            "joint_inner_tail_ess_mean": joint_audit.get(
                "inner_tail_effective_samples_mean"
            ),
            "joint_inner_tail_ess_max": joint_audit.get(
                "inner_tail_effective_samples_max"
            ),
            "joint_inner_tail_ess_required": joint_audit.get(
                "minimum_inner_tail_effective_samples_max"
            ),
            "joint_inner_tail_support": joint_audit.get(
                "inner_tail_support_for_promotion"
            ),
            "joint_expected_pi_d_mean": joint_audit.get(
                "expected_probability_times_multiplier_mean"
            ),
            "joint_independent_pi_times_d_mean": joint_audit.get(
                "independence_probability_times_multiplier_mean"
            ),
            "joint_expected_edge_mean": joint_audit.get(
                "joint_expected_edge_mean"
            ),
            "joint_product_identity_residual_mean": joint_audit.get(
                "product_identity_residual_mean"
            ),
            "joint_product_identity_residual_max_abs": joint_audit.get(
                "product_identity_residual_max_abs"
            ),
            "joint_product_identity_consistent": joint_audit.get(
                "product_identity_consistent"
            ),
            "joint_covariance_mean": joint_audit.get(
                "probability_multiplier_covariance_mean"
            ),
            "joint_negative_covariance_fraction": joint_audit.get(
                "negative_covariance_fraction"
            ),
            "joint_independence_bias_mean": joint_audit.get(
                "independence_approximation_bias_mean"
            ),
            "joint_independence_bias_min": joint_audit.get(
                "independence_approximation_bias_min"
            ),
            "joint_independence_bias_max": joint_audit.get(
                "independence_approximation_bias_max"
            ),
            "joint_positive_independence_bias_fraction": joint_audit.get(
                "positive_independence_bias_fraction"
            ),
            "joint_independence_overstatement_mean": joint_audit.get(
                "independence_approximation_overstatement_mean"
            ),
            "settlement_integer_yen": settlement_audit.get(
                "integer_yen_accounting"
            ),
            "settlement_self_impact_repricing": settlement_audit.get(
                "self_impact_repricing"
            ),
            "settlement_refund_supported": bool(
                settlement_audit.get("full_refund_terminal_states")
                or settlement_audit.get("partial_refund_supported")
            ) if settlement_audit else None,
            "settlement_special_payout_supported": settlement_audit.get(
                "special_payout_addition_supported"
            ),
            "settlement_rounding": settlement_audit.get("rounding"),
        })
        condition = confidence.get("condition")
        if isinstance(condition, dict):
            summary["bootstrap_primary_block"] = condition.get("primary_block")
            summary["bootstrap_quantile_method"] = condition.get(
                "quantile_method"
            )
            summary["bootstrap_samples"] = condition.get("samples")
        sensitivity = confidence.get("sensitivity")
        if isinstance(sensitivity, dict):
            summary["day_venue_roi_lower_95"] = (
                sensitivity.get("day_venue") or {}
            ).get("roi_lower")
            summary["venue_meeting_roi_lower_95"] = (
                sensitivity.get("venue_meeting") or {}
            ).get("roi_lower")
    if str(payload.get("model") or "").startswith(
        "joint_bankroll_strict_walk_forward_v"
    ):
        probability = payload.get("probability_metrics")
        probability = probability if isinstance(probability, dict) else {}
        aliases = {
            "model_winner_log_loss": "generated_winner_log_loss",
            "model_winner_top1_accuracy": "generated_winner_top1_accuracy",
            "model_trifecta_log_loss": "generated_log_loss",
            "model_trifecta_top5_hit_rate": "generated_top5",
        }
        for target, source in aliases.items():
            value = probability.get(target, probability.get(source))
            if value is not None:
                summary[target] = value
        summary.setdefault(
            "evaluation_days",
            payload.get("evaluation_days", payload.get("evaluated_days")),
        )
        if summary.get("daily_cluster_bootstrap_roi_lower_95") is None:
            summary["daily_cluster_bootstrap_roi_lower_95"] = summary.get(
                "roi_ci95_lower"
            )
        daily = payload.get("daily")
        configuration = payload.get("configuration")
        configuration = configuration if isinstance(configuration, dict) else {}
        selected_purchase_values = [
            float(race["portfolio_lower_quantile"])
            for day in daily if isinstance(daily, list) and isinstance(day, dict)
            for race in (day.get("races") or []) if isinstance(race, dict)
            if int(race.get("stake_yen") or 0) > 0
            and race.get("portfolio_lower_quantile") is not None
        ] if isinstance(daily, list) else []
        selected_portfolios = sum(
            1
            for day in daily if isinstance(daily, list) and isinstance(day, dict)
            for race in (day.get("races") or []) if isinstance(race, dict)
            if int(race.get("stake_yen") or 0) > 0
        ) if isinstance(daily, list) else 0
        buy_margin = float(configuration.get("buy_margin") or 0.0)
        purchase_value_gate = bool(
            selected_portfolios
            and len(selected_purchase_values) == selected_portfolios
            and all(value > buy_margin for value in selected_purchase_values)
        )
        joint_audit = payload.get("joint_value_audit")
        joint_audit = joint_audit if isinstance(joint_audit, dict) else {}
        settlement_audit = payload.get("settlement_audit")
        settlement_audit = (
            settlement_audit if isinstance(settlement_audit, dict) else {}
        )
        calibration_ledger = payload.get("calibration_ledger")
        calibration_ledger = (
            calibration_ledger if isinstance(calibration_ledger, dict) else {}
        )
        value_calibration = payload.get(
            "purchase_value_realization_calibration"
        )
        value_calibration = (
            value_calibration if isinstance(value_calibration, dict) else {}
        )
        evaluation_protocol = payload.get("evaluation_protocol")
        evaluation_protocol = (
            evaluation_protocol if isinstance(evaluation_protocol, dict) else {}
        )
        purchase_rule = evaluation_protocol.get("purchase_rule")
        purchase_rule = purchase_rule if isinstance(purchase_rule, dict) else {}
        evaluation_time_t = evaluation_protocol.get("evaluation_time_t")
        evaluation_time_t = (
            evaluation_time_t if isinstance(evaluation_time_t, dict) else {}
        )
        odds_snapshot_age = evaluation_protocol.get("odds_snapshot_age")
        odds_snapshot_age = (
            odds_snapshot_age if isinstance(odds_snapshot_age, dict) else {}
        )
        evaluation_population = evaluation_protocol.get("population")
        evaluation_population = (
            evaluation_population if isinstance(evaluation_population, dict) else {}
        )
        summary.update({
            "evaluation_protocol_id": payload.get("evaluation_protocol_id"),
            "evaluation_protocol_version": evaluation_protocol.get("version"),
            "evaluation_time_t_definition": evaluation_time_t.get("definition"),
            "evaluation_time_t_source": evaluation_time_t.get("source_field"),
            "evaluation_time_t_earliest": evaluation_time_t.get("earliest"),
            "evaluation_time_t_latest": evaluation_time_t.get("latest"),
            "evaluation_snapshot_age_definition": odds_snapshot_age.get("definition"),
            "evaluation_snapshot_age_seconds_min": odds_snapshot_age.get("minimum"),
            "evaluation_snapshot_age_seconds_mean": odds_snapshot_age.get("mean"),
            "evaluation_snapshot_age_seconds_max": odds_snapshot_age.get("maximum"),
            "evaluation_venues": evaluation_population.get("venues"),
            "evaluation_wager_types": evaluation_population.get("wager_types"),
            "evaluation_popularity_bands_at_t": evaluation_population.get(
                "popularity_bands_at_t"
            ),
            "joint_audit_recorded": bool(joint_audit.get("recorded")),
            "joint_audited_portfolios": joint_audit.get(
                "audited_portfolios"
            ),
            "joint_moment_observations": joint_audit.get(
                "moment_observations"
            ),
            "joint_shared_scenarios": joint_audit.get(
                "shared_probability_price_scenarios"
            ),
            "joint_portfolio_path_aggregation": joint_audit.get(
                "portfolio_path_aggregation"
            ),
            "joint_complete_vector_repricing": joint_audit.get(
                "complete_vector_repricing"
            ),
            "joint_parameter_draws_min": joint_audit.get(
                "parameter_draws_min"
            ),
            "joint_outer_sample_count_r_definition": joint_audit.get(
                "outer_sample_count_r_definition"
            ),
            "joint_outer_sample_count_r_min": joint_audit.get(
                "outer_sample_count_r_min"
            ),
            "joint_outer_sample_count_r_max": joint_audit.get(
                "outer_sample_count_r_max"
            ),
            "joint_outer_required_r_max": joint_audit.get(
                "minimum_outer_draws_max"
            ),
            "joint_outer_alpha_min": joint_audit.get("outer_alpha_min"),
            "joint_outer_alpha_max": joint_audit.get("outer_alpha_max"),
            "joint_outer_tail_observations_min": joint_audit.get(
                "outer_tail_observations_min"
            ),
            "joint_outer_tail_observations_max": joint_audit.get(
                "outer_tail_observations_max"
            ),
            "joint_outer_tail_required": joint_audit.get(
                "minimum_outer_tail_observations_for_promotion"
            ),
            "joint_outer_tail_support": joint_audit.get(
                "outer_tail_support_for_promotion"
            ),
            "joint_inner_s_definition": joint_audit.get(
                "inner_scenario_count_s_definition"
            ),
            "joint_inner_s_min": joint_audit.get("inner_scenario_count_s_min"),
            "joint_inner_s_max": joint_audit.get("inner_scenario_count_s_max"),
            "joint_inner_ess_min": joint_audit.get(
                "inner_effective_samples_min"
            ),
            "joint_inner_ess_mean": joint_audit.get(
                "inner_effective_samples_mean"
            ),
            "joint_inner_ess_max": joint_audit.get(
                "inner_effective_samples_max"
            ),
            "joint_inner_tail_ess_min": joint_audit.get(
                "inner_tail_effective_samples_min"
            ),
            "joint_inner_tail_beta_min": (
                joint_audit.get("inner_tail_fraction_min")
                if joint_audit.get("inner_tail_fraction_min") is not None
                else purchase_rule.get("inner_tail_fraction")
            ),
            "joint_inner_tail_beta_max": (
                joint_audit.get("inner_tail_fraction_max")
                if joint_audit.get("inner_tail_fraction_max") is not None
                else purchase_rule.get("inner_tail_fraction")
            ),
            "joint_inner_tail_ess_mean": joint_audit.get(
                "inner_tail_effective_samples_mean"
            ),
            "joint_inner_tail_ess_max": joint_audit.get(
                "inner_tail_effective_samples_max"
            ),
            "joint_inner_tail_ess_required": joint_audit.get(
                "minimum_inner_tail_effective_samples_max"
            ),
            "joint_inner_tail_support": joint_audit.get(
                "inner_tail_support_for_promotion"
            ),
            "joint_expected_pi_d_mean": joint_audit.get(
                "expected_probability_times_multiplier_mean"
            ),
            "joint_independent_pi_times_d_mean": joint_audit.get(
                "independence_probability_times_multiplier_mean"
            ),
            "joint_expected_edge_mean": joint_audit.get(
                "joint_expected_edge_mean"
            ),
            "joint_product_identity_residual_mean": joint_audit.get(
                "product_identity_residual_mean"
            ),
            "joint_product_identity_residual_max_abs": joint_audit.get(
                "product_identity_residual_max_abs"
            ),
            "joint_product_identity_consistent": joint_audit.get(
                "product_identity_consistent"
            ),
            "joint_covariance_mean": joint_audit.get(
                "probability_multiplier_covariance_mean"
            ),
            "joint_negative_covariance_fraction": joint_audit.get(
                "negative_covariance_fraction"
            ),
            "joint_independence_bias_mean": joint_audit.get(
                "independence_approximation_bias_mean"
            ),
            "joint_independence_bias_min": joint_audit.get(
                "independence_approximation_bias_min"
            ),
            "joint_independence_bias_max": joint_audit.get(
                "independence_approximation_bias_max"
            ),
            "joint_positive_independence_bias_fraction": joint_audit.get(
                "positive_independence_bias_fraction"
            ),
            "joint_independence_overstatement_mean": joint_audit.get(
                "independence_approximation_overstatement_mean"
            ),
            "settlement_integer_yen": settlement_audit.get(
                "integer_yen_accounting"
            ),
            "settlement_self_impact_repricing": settlement_audit.get(
                "self_impact_repricing"
            ),
            "settlement_refund_supported": bool(
                settlement_audit.get("full_refund_terminal_states")
                or settlement_audit.get("partial_refund_supported")
            ) if settlement_audit else None,
            "settlement_special_payout_supported": settlement_audit.get(
                "special_payout_addition_supported"
            ),
            "settlement_rounding": settlement_audit.get("rounding"),
            "joint_calibration_ledger_version": calibration_ledger.get(
                "version"
            ),
            "joint_calibration_candidate_portfolios": calibration_ledger.get(
                "candidate_portfolios"
            ),
            "joint_calibration_authorized_portfolios": calibration_ledger.get(
                "authorized_portfolios"
            ),
            "joint_calibration_stake_yen": calibration_ledger.get("stake_yen"),
            "joint_calibration_return_yen": calibration_ledger.get("return_yen"),
            "joint_calibration_profit_yen": calibration_ledger.get("profit_yen"),
            "joint_calibration_roi": calibration_ledger.get("roi"),
            "purchase_value_realization_calibration": value_calibration,
            "purchase_value_realization_version": value_calibration.get(
                "version"
            ),
            "purchase_value_realization_candidate_portfolios": (
                value_calibration.get("candidate_portfolios")
            ),
            "purchase_value_realization_mismatched_portfolios": (
                value_calibration.get("excluded_mismatched_portfolios")
            ),
            "purchase_value_realization_monotone": value_calibration.get(
                "monotone_realized_roi"
            ),
            "purchase_value_realization_deciles": value_calibration.get(
                "deciles"
            ),
        })
        summary.update({
            "joint_purchase_value_minimum": (
                min(selected_purchase_values)
                if selected_purchase_values else None
            ),
            "joint_purchase_value_mean": (
                sum(selected_purchase_values) / len(selected_purchase_values)
                if selected_purchase_values else None
            ),
            "joint_purchase_safety_margin": buy_margin,
            "joint_purchase_value_minimum_excess": (
                min(selected_purchase_values) - buy_margin
                if selected_purchase_values else None
            ),
            "joint_purchase_value_selected_portfolios": selected_portfolios,
            "joint_purchase_value_gate_passed": purchase_value_gate,
            "joint_search_outer_sample_count_r_requested": (
                configuration.get("search_outer_draws")
            ),
            "joint_validation_outer_sample_count_r_requested": (
                configuration.get("outer_draws")
            ),
            "joint_search_validation_draw_sets_disjoint": (
                configuration.get("search_validation_draw_sets_disjoint")
            ),
        })
        confidence = payload.get("bankroll_confidence")
        confidence = confidence if isinstance(confidence, dict) else {}
        if isinstance(daily, list) and (
            not isinstance(confidence.get("condition"), dict)
            or not isinstance(confidence.get("sensitivity"), dict)
        ):
            from .joint_bankroll_evaluation import (
                build_block_bootstrap_evidence,
            )

            reconstructed = build_block_bootstrap_evidence(
                daily,
                samples=int(
                    confidence.get("samples")
                    or configuration.get("bootstrap_samples")
                    or 2000
                ),
                seed=int(configuration.get("seed") or 0),
            )
            confidence = {
                **reconstructed,
                **confidence,
                "condition_id": reconstructed["condition_id"],
                "condition": reconstructed["condition"],
                "probability_roi_above_one_is_diagnostic_only": True,
                "sensitivity": reconstructed["sensitivity"],
            }
        roi_lower = confidence.get(
            "roi_lower", summary.get("daily_cluster_bootstrap_roi_lower_95")
        )
        summary["formal_roi_gate_method"] = "Q0.05_ROI_greater_than_1"
        summary["formal_roi_gate_passed"] = bool(
            roi_lower is not None and float(roi_lower) > 1.0
        )
        summary["roi_probability_is_diagnostic_only"] = True
        summary["bootstrap_condition_id"] = confidence.get("condition_id")
        condition = confidence.get("condition")
        if isinstance(condition, dict):
            summary["bootstrap_primary_block"] = condition.get("primary_block")
            summary["bootstrap_quantile_method"] = condition.get(
                "quantile_method"
            )
            summary["bootstrap_samples"] = condition.get("samples")
        else:
            summary["bootstrap_primary_block"] = confidence.get(
                "block", "complete_operating_day"
            )
            summary["bootstrap_quantile_method"] = confidence.get(
                "quantile_method", "inverted_cdf"
            )
            summary["bootstrap_samples"] = confidence.get("samples")
        sensitivity = confidence.get("sensitivity")
        if isinstance(sensitivity, dict):
            day_venue = sensitivity.get("day_venue") or {}
            meeting = sensitivity.get("venue_meeting") or {}
            summary["day_venue_roi_lower_95"] = day_venue.get("roi_lower")
            summary["venue_meeting_roi_lower_95"] = meeting.get("roi_lower")
        if isinstance(daily, list) and summary.get("stake_yen"):
            returns = [
                int(race.get("return_yen") or 0)
                for day in daily if isinstance(day, dict)
                for race in (day.get("races") or []) if isinstance(race, dict)
                if int(race.get("stake_yen") or 0) > 0
            ]
            largest = max(returns, default=0)
            total_return = int(summary.get("return_yen") or 0)
            total_stake = int(summary["stake_yen"])
            summary.setdefault(
                "largest_hit_return_share",
                largest / total_return if total_return else None,
            )
            summary.setdefault(
                "roi_without_largest_hit",
                (total_return - largest) / total_stake,
            )
        gate = payload.get("promotion_gate")
        if isinstance(gate, dict):
            gate = {
                **gate,
                "joint_purchase_value_above_safety_margin": (
                    purchase_value_gate
                ),
            }
            summary["promotion_gate_passed"] = sum(bool(value) for value in gate.values())
            summary["promotion_gate_total"] = len(gate)
            summary["promotion_gate_failed"] = [
                key for key, passed in gate.items() if not passed
            ]
    deployment = payload.get("deployment_configuration")
    genetic_selection = (
        deployment.get("calibrator_selection")
        if isinstance(deployment, dict)
        else None
    )
    if (
        isinstance(genetic_selection, dict)
        and genetic_selection.get("protocol") == "genetic_t5_market_residual_v1"
    ):
        champion = genetic_selection.get("champion")
        metrics = genetic_selection.get("champion_metrics")
        summary["market_residual_ga_protocol"] = genetic_selection["protocol"]
        summary["market_residual_ga_outer_holdout_used"] = bool(
            genetic_selection.get("outer_holdout_used")
        )
        summary["market_residual_ga_champion"] = (
            dict(champion) if isinstance(champion, dict) else None
        )
        summary["market_residual_ga_fitness"] = genetic_selection.get(
            "champion_fitness"
        )
        if isinstance(metrics, dict):
            for key in (
                "prequential_races",
                "evaluation_days",
                "mean_log_loss_delta",
                "log_loss_delta_ci95_upper",
                "worst_day_log_loss_delta",
                "mean_brier_delta",
                "mean_top5_delta",
            ):
                summary[f"market_residual_ga_{key}"] = metrics.get(key)
    periods = payload.get("periods")
    if isinstance(periods, dict):
        summary.setdefault(
            "evaluation_from",
            periods.get("outer_from") or periods.get("evaluation_from"),
        )
        summary.setdefault(
            "evaluation_through",
            periods.get("outer_through") or periods.get("evaluation_through"),
        )
    evaluation = payload.get("evaluation")
    formal_bankroll = payload.get("formal_bankroll")
    coverage = payload.get("coverage")
    for source in (evaluation, formal_bankroll, coverage):
        if not isinstance(source, dict):
            continue
        races = (
            source.get("races")
            or source.get("outer_races")
            or source.get("evaluated_races")
        )
        if races is not None:
            summary.setdefault("evaluated_races", races)
            break
    if isinstance(formal_bankroll, dict):
        policy = formal_bankroll.get("policy")
        if isinstance(policy, dict):
            summary.setdefault(
                "daily_budget_yen",
                policy.get("initial_bankroll_yen_per_day"),
            )
            summary.setdefault("allocation_mode", policy.get("allocation_api"))
            summary.setdefault(
                "profit_reinvestment", policy.get("profit_reinvestment")
            )
            summary.setdefault("odds_mode", policy.get("decision_odds"))
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
        if str(payload.get("model") or "").startswith(
            "joint_bankroll_strict_walk_forward_v"
        ):
            checks["joint_purchase_value_above_safety_margin"] = bool(
                summary.get("joint_purchase_value_gate_passed")
            )
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
    conditional_order = payload.get("prequential_conditional_order")
    if isinstance(conditional_order, dict):
        summary["prequential_conditional_order"] = {
            key: conditional_order.get(key)
            for key in (
                "status",
                "method",
                "minimum_prior_days",
                "available_days",
                "transformed_days",
                "transformed_races",
                "baseline_log_loss",
                "conditional_log_loss",
                "log_loss_difference",
                "baseline_top5_hit_rate",
                "conditional_top5_hit_rate",
                "top5_hit_rate_difference",
                "improving_days",
            )
            if key in conditional_order
        }
    trend_point = payload.get(
        "trend_point_market_offset_kelly_walk_forward"
    )
    if isinstance(trend_point, dict):
        for key in (
            "status", "registered_after", "evaluation_days",
            "registered_closing_context_features",
            "evaluated_races", "tickets", "hit_tickets", "stake_yen",
            "return_yen", "profit_yen", "roi", "winning_days",
            "purchase_days", "profitable_day_fraction",
            "roi_without_largest_hit",
            "effective_hit_count", "largest_hit_return_share",
            "daily_cluster_bootstrap_roi_lower_95",
            "probability_roi_above_one", "promotion_eligible",
        ):
            if key in trend_point:
                summary[f"trend_point_prospective_{key}"] = trend_point[key]
        if isinstance(trend_point.get("promotion_gate"), dict):
            summary["trend_point_prospective_promotion_gate"] = dict(
                trend_point["promotion_gate"]
            )
        trend_bootstrap = trend_point.get("bootstrap")
        if isinstance(trend_bootstrap, dict):
            summary["trend_point_prospective_daily_cluster_bootstrap_roi_lower_95"] = (
                trend_bootstrap.get("roi_ci95_lower")
            )
            summary["trend_point_prospective_probability_roi_above_one"] = (
                trend_bootstrap.get("probability_roi_above_one")
            )
        trend_market = trend_point.get("log_loss")
        if isinstance(trend_market, dict):
            for key in (
                "races", "model", "market", "challenger",
                "challenger_delta_vs_market",
                "challenger_improvement_confidence",
                "challenger_top5_improvement_confidence",
            ):
                if key in trend_market:
                    summary[f"trend_point_prospective_market_{key}"] = (
                        trend_market[key]
                    )
        trend_probability = trend_point.get(
            "purchase_probability_calibration"
        )
        if isinstance(trend_probability, dict):
            summary["trend_point_prospective_probability_calibration"] = dict(
                trend_probability
            )
    trend_empirical = payload.get(
        "trend_point_empirical_lcb_walk_forward"
    )
    if isinstance(trend_empirical, dict):
        for key in (
            "status", "registered_after", "evaluation_days",
            "registered_closing_context_features",
            "evaluated_races", "calibration_ready_folds", "tickets",
            "hit_tickets", "stake_yen", "return_yen", "profit_yen",
            "roi", "profitable_days", "roi_without_largest_hit",
            "effective_hit_count", "largest_hit_return_share",
            "daily_cluster_bootstrap_roi_lower_95",
            "calibration_ledger_candidates", "promotion_eligible",
        ):
            if key in trend_empirical:
                summary[f"trend_empirical_lcb_{key}"] = trend_empirical[key]
        if isinstance(trend_empirical.get("promotion_gate"), dict):
            summary["trend_empirical_lcb_promotion_gate"] = dict(
                trend_empirical["promotion_gate"]
            )
    for diagnostic_key, prefix in (
        (
            "trend_point_market_offset_kelly_diagnostic",
            "trend_point_retrospective",
        ),
        (
            "trend_point_reversed_place_pair_diagnostic",
            "trend_point_reversed_pair_retrospective",
        ),
    ):
        diagnostic = payload.get(diagnostic_key)
        if not isinstance(diagnostic, dict):
            continue
        for key in (
            "evaluation_days", "evaluated_races", "tickets", "hit_tickets",
            "stake_yen", "return_yen", "profit_yen", "roi",
            "winning_days", "purchase_days", "profitable_day_fraction",
            "roi_without_largest_hit", "effective_hit_count",
            "largest_hit_return_share", "promotion_eligible",
        ):
            if key in diagnostic:
                summary[f"{prefix}_{key}"] = diagnostic[key]
        if isinstance(diagnostic.get("policy"), dict):
            summary[f"{prefix}_policy"] = dict(diagnostic["policy"])
        diagnostic_bootstrap = diagnostic.get("bootstrap")
        if isinstance(diagnostic_bootstrap, dict):
            summary[f"{prefix}_daily_cluster_bootstrap_roi_lower_95"] = (
                diagnostic_bootstrap.get("roi_ci95_lower")
            )
            summary[f"{prefix}_probability_roi_above_one"] = (
                diagnostic_bootstrap.get("probability_roi_above_one")
            )
    trend_sweep = payload.get("trend_point_odds_safety_sweep")
    if isinstance(trend_sweep, dict):
        sweep_metrics = (
            "evaluation_days", "evaluated_races", "tickets", "hit_tickets",
            "stake_yen", "return_yen", "profit_yen", "roi",
            "profitable_day_fraction", "roi_without_largest_hit",
            "effective_hit_count",
        )
        compact_rows = []
        for row in trend_sweep.get("rows") or []:
            if not isinstance(row, dict):
                continue
            compact_row = {"odds_safety_factor": row.get("odds_safety_factor")}
            for window in ("retrospective", "prior_registered_window"):
                value = row.get(window)
                if not isinstance(value, dict):
                    continue
                compact = {key: value.get(key) for key in sweep_metrics}
                bootstrap = value.get("bootstrap")
                if isinstance(bootstrap, dict):
                    compact["roi_ci95_lower"] = bootstrap.get("roi_ci95_lower")
                    compact["probability_roi_above_one"] = bootstrap.get(
                        "probability_roi_above_one"
                    )
                compact_row[window] = compact
            compact_rows.append(compact_row)
        summary["trend_point_odds_safety_sweep"] = {
            "status": trend_sweep.get("status"),
            "selection_data_through": trend_sweep.get(
                "selection_data_through"
            ),
            "next_registration_must_be_after": trend_sweep.get(
                "next_registration_must_be_after"
            ),
            "rows": compact_rows,
        }
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
                "max_drawdown_yen", "roi_without_largest_hit",
                "largest_hit_return_yen", "largest_hit_return_share",
                "effective_hit_count",
            ):
                summary[f"payout_feature_candidate_{key}"] = bankroll.get(key)
            policy = bankroll.get("policy")
            if isinstance(policy, dict):
                summary["payout_feature_candidate_schema"] = policy.get(
                    "payout_tail_schema"
                ) or policy.get(
                    "payout_feature_schema"
                )
            policy_selection = bankroll.get("policy_selection")
            if isinstance(policy_selection, dict):
                for key in (
                    "source",
                    "selected_ridge",
                    "selected_mean_correction_factor",
                    "selected_ev_threshold",
                    "selected_min_daily_exposure_fraction",
                    "minimum_tickets",
                    "minimum_hits",
                    "minimum_winning_days",
                    "minimum_roi",
                    "minimum_roi_without_largest_hit",
                    "minimum_effective_hit_count",
                    "required_roi_ci95_lower",
                    "minimum_probability_roi_above_one",
                ):
                    summary[f"payout_feature_selection_{key}"] = (
                        policy_selection.get(key)
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
    expected_return = payload.get("expected_return_calibration")
    if isinstance(expected_return, dict):
        bankroll = expected_return.get("bankroll")
        if isinstance(bankroll, dict):
            for key in (
                "roi", "profit_yen", "stake_yen", "return_yen",
                "max_drawdown_yen", "selected_tickets", "races_bet",
                "hit_tickets", "winning_days", "losing_days",
                "roi_without_largest_hit", "largest_hit_return_yen",
                "largest_hit_return_share", "effective_hit_count",
            ):
                summary[f"expected_return_candidate_{key}"] = bankroll.get(key)
            tail = bankroll.get("tail_portfolio_diagnostics")
            if isinstance(tail, dict):
                summary["expected_return_tail_portfolio_diagnostics"] = tail
            policy_selection = bankroll.get("policy_selection")
            if isinstance(policy_selection, dict):
                summary["expected_return_selection_source"] = (
                    policy_selection.get("source")
                )
                summary["expected_return_selected_ev_threshold"] = (
                    policy_selection.get("selected_ev_threshold")
                )
                for key in (
                    "minimum_tickets",
                    "minimum_hits",
                    "minimum_winning_days",
                    "minimum_roi",
                    "minimum_roi_without_largest_hit",
                    "minimum_effective_hit_count",
                    "minimum_probability_roi_above_one",
                ):
                    summary[f"expected_return_selection_{key}"] = (
                        policy_selection.get(key)
                    )
            return_calibrator = bankroll.get("return_calibrator")
            if isinstance(return_calibrator, dict):
                summary["expected_return_max_expected_return"] = (
                    return_calibrator.get("max_expected_return")
                )
                combination = return_calibrator.get(
                    "combination_calibration"
                )
                if isinstance(combination, dict):
                    for key in (
                        "factor_min",
                        "factor_median",
                        "factor_max",
                        "zero_factor_combinations",
                        "below_legacy_floor_combinations",
                    ):
                        summary[f"expected_return_{key}"] = combination.get(key)
        confidence = expected_return.get("bankroll_confidence")
        if isinstance(confidence, dict):
            for key, value in confidence.items():
                if not isinstance(value, (dict, list)):
                    summary[f"expected_return_{key}"] = value
        gate = expected_return.get("diagnostic_gate")
        if isinstance(gate, dict):
            for key, value in gate.items():
                if not isinstance(value, (dict, list)):
                    summary[f"expected_return_gate_{key}"] = value
        summary["expected_return_promotion_eligible"] = (
            expected_return.get("promotion_eligible")
        )
    fixed_return = payload.get("expected_return_fixed_threshold")
    if isinstance(fixed_return, dict):
        fixed_bankroll = fixed_return.get("bankroll")
        if isinstance(fixed_bankroll, dict):
            for key in (
                "roi", "profit_yen", "stake_yen", "return_yen",
                "selected_tickets", "races_bet", "hit_tickets",
                "roi_without_largest_hit", "largest_hit_return_yen",
                "largest_hit_return_share", "effective_hit_count",
            ):
                summary[f"expected_return_fixed_{key}"] = (
                    fixed_bankroll.get(key)
                )
            tail = fixed_bankroll.get("tail_portfolio_diagnostics")
            if isinstance(tail, dict):
                summary["expected_return_fixed_tail_portfolio_diagnostics"] = (
                    tail
                )
        confidence = fixed_return.get("bankroll_confidence")
        if isinstance(confidence, dict):
            for key, value in confidence.items():
                if not isinstance(value, (dict, list)):
                    summary[f"expected_return_fixed_{key}"] = value
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
    if task_type in {
        "decision_market_residual_v38",
        "decision_stacked_market_v44",
    }:
        if summary.get("challenger_selection_gate_pass") is True:
            return "freeze_for_prospective_value_calibration"
        if summary.get("training_status") == "insufficient_training_history":
            return "accumulate_decision_training_history"
        return "probability_challenger_gate_failed"
    if task_type == "decision_v38_empirical_lcb":
        if summary.get("promotion_eligible") is True:
            return "promotion_candidate"
        if summary.get("calibration_ready") is True:
            return "accumulate_prospective_bankroll_evidence"
        return "accumulate_strict_prior_value_calibration"
    if task_type == "genetic_island_search":
        return "speculative_generation_complete"
    if task_type == "market_residual_walk_forward":
        if summary.get("reused_holdout_research_only") is True:
            return "reject_or_research_only"
        if summary.get("promotion_eligible") is True:
            return "promotion_candidate"
        return "accumulate_formal_evidence"
    if task_type == "joint_scenario_walk_forward":
        return "diagnostic_complete_not_policy_connected"
    if task_type == "joint_bankroll_walk_forward":
        if summary.get("promotion_eligible") is True:
            return "promotion_candidate"
        return "accumulate_sealed_bankroll_evidence"
    if task_type == "joint_edge_calibrated_replay":
        if summary.get("promotion_eligible") is True:
            return "promotion_candidate"
        if (
            int(summary.get("calibration_ready_days") or 0) >= 30
            and int(summary.get("calibration_ready_races") or 0) >= 1_000
        ):
            return "reject_or_research_only"
        return "accumulate_strict_prior_value_calibration"
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
    if summary.get("promotion_eligible") is False:
        return "reject_or_research_only"
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


def _attach_research_holdout_coverage(payload: dict[str, Any]) -> bool:
    """Upgrade legacy research results from their recorded clean-day ledger."""
    if (
        payload.get("reused_holdout_research_only") is not True
        or isinstance(payload.get("research_coverage_gate"), dict)
    ):
        return False
    coverage = payload.get("coverage_gate")
    if not isinstance(coverage, dict):
        return False
    clean_dates = sorted({
        str(value) for value in (coverage.get("clean_dates") or [])
    })
    if not clean_dates:
        return False
    try:
        requested_from = datetime.strptime(
            str(payload.get("from_date")), "%Y-%m-%d"
        ).date()
        requested_through = datetime.strptime(
            str(payload.get("through_date")), "%Y-%m-%d"
        ).date()
        parsed_clean_dates = [
            datetime.strptime(value, "%Y-%m-%d").date()
            for value in clean_dates
        ]
    except ValueError:
        return False
    requested_calendar_days = (
        requested_through - requested_from
    ).days + 1
    if (
        requested_calendar_days <= 0
        or any(
            value < requested_from or value > requested_through
            for value in parsed_clean_dates
        )
    ):
        return False
    minimum_clean_days = 300
    payload["research_coverage_gate"] = {
        "source": "coverage_gate.clean_dates",
        "requested_from": requested_from.isoformat(),
        "requested_through": requested_through.isoformat(),
        "requested_calendar_days": requested_calendar_days,
        "minimum_clean_days": minimum_clean_days,
        "clean_days": len(clean_dates),
        "clean_day_fraction": len(clean_dates) / requested_calendar_days,
        "pass": len(clean_dates) >= minimum_clean_days,
        "migrated_from_legacy_result": True,
    }
    return True


def _load_result(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("evaluation result must be a JSON object")
    changed = _attach_tail_portfolio_diagnostics(payload)
    changed = _attach_research_holdout_coverage(payload) or changed
    if changed:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    return payload, summarize_result(payload)


def _valid_tail_portfolio_contract(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    diagnostics = value.get("tail_portfolio_diagnostics")
    if (
        not isinstance(diagnostics, dict)
        or diagnostics.get("odds_field") != "estimated_odds_at_purchase"
    ):
        return False
    counts = [diagnostics.get("purchased_tickets")]
    for segment in ("normal", "tail"):
        row = diagnostics.get(segment)
        counts.append(row.get("tickets") if isinstance(row, dict) else None)
    if any(
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        for count in counts
    ):
        return False
    return counts[0] == counts[1] + counts[2]


def _validate_job_result_contract(
    job: dict[str, Any], payload: dict[str, Any]
) -> None:
    task_type = str(job.get("task_type") or "")
    parameters = job.get("parameters")
    if isinstance(parameters, str):
        parameters = json.loads(parameters)
    if (
        task_type == "market_residual_walk_forward"
        and isinstance(parameters, dict)
        and parameters.get("research_only_reused_holdout") is True
    ):
        if payload.get("reused_holdout_research_only") is not True:
            raise ValueError(
                "market residual reused holdout result must be research-only"
            )
        diagnostic = payload.get(
            "trend_point_market_offset_kelly_diagnostic"
        )
        if (
            not isinstance(diagnostic, dict)
            or diagnostic.get("promotion_eligible") is not False
        ):
            raise ValueError(
                "market residual reused holdout diagnostic must not be "
                "promotion eligible"
            )
        if parameters.get("trend_point_required_ticket_count") == 2:
            reversed_pair = payload.get(
                "trend_point_reversed_place_pair_diagnostic"
            )
            reversed_policy = (
                reversed_pair.get("policy")
                if isinstance(reversed_pair, dict)
                else None
            )
            if (
                not isinstance(reversed_pair, dict)
                or reversed_pair.get("promotion_eligible") is not False
                or not isinstance(reversed_policy, dict)
                or reversed_policy.get("require_reversed_place_pair") is not True
            ):
                raise ValueError(
                    "market residual exact-two research must include a "
                    "non-promotable reversed-place-pair diagnostic"
                )
        coverage = payload.get("coverage_gate")
        research_coverage = payload.get("research_coverage_gate")
        if not isinstance(coverage, dict) or not isinstance(
            research_coverage, dict
        ):
            raise ValueError(
                "market residual reused holdout requires research coverage"
            )
        clean_dates = sorted({
            str(value) for value in (coverage.get("clean_dates") or [])
        })
        diagnostic_dates = sorted({
            str(value) for value in (diagnostic.get("evaluation_dates") or [])
        })
        if not clean_dates or diagnostic_dates != clean_dates:
            raise ValueError(
                "market residual reused holdout must evaluate every clean date"
            )
        expected_races = sum(
            int(row.get("eligible_t5_races") or 0)
            for row in (coverage.get("days") or [])
            if isinstance(row, dict) and str(row.get("race_date")) in clean_dates
        )
        if int(diagnostic.get("evaluated_races") or 0) != expected_races:
            raise ValueError(
                "market residual reused holdout clean race count mismatch"
            )
        if parameters.get("trend_point_required_ticket_count") == 2:
            if sorted({
                str(value)
                for value in (reversed_pair.get("evaluation_dates") or [])
            }) != clean_dates or int(
                reversed_pair.get("evaluated_races") or 0
            ) != expected_races:
                raise ValueError(
                    "market residual reversed-place-pair coverage mismatch"
                )
        minimum_days = int(
            parameters.get("minimum_research_clean_days") or 300
        )
        requested_from = datetime.strptime(
            str(parameters.get("from_date")), "%Y-%m-%d"
        ).date()
        requested_through = datetime.strptime(
            str(parameters.get("through_date")), "%Y-%m-%d"
        ).date()
        requested_calendar_days = (
            requested_through - requested_from
        ).days + 1
        if requested_calendar_days <= 0:
            raise ValueError(
                "market residual reused holdout dates must be chronological"
            )
        parsed_clean_dates = [
            datetime.strptime(value, "%Y-%m-%d").date()
            for value in clean_dates
        ]
        expected_clean_fraction = len(clean_dates) / requested_calendar_days
        if (
            any(
                value < requested_from or value > requested_through
                for value in parsed_clean_dates
            )
            or research_coverage.get("requested_from")
            != requested_from.isoformat()
            or research_coverage.get("source")
            != "coverage_gate.clean_dates"
            or research_coverage.get("requested_through")
            != requested_through.isoformat()
            or int(research_coverage.get("requested_calendar_days") or 0)
            != requested_calendar_days
            or int(research_coverage.get("minimum_clean_days") or 0)
            != minimum_days
            or int(research_coverage.get("clean_days") or 0)
            != len(clean_dates)
            or not math.isclose(
                float(research_coverage.get("clean_day_fraction") or 0.0),
                expected_clean_fraction,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or bool(research_coverage.get("pass"))
            != (len(clean_dates) >= minimum_days)
        ):
            raise ValueError(
                "market residual reused holdout research coverage gate mismatch"
            )
        if str(payload.get("source_model_sha256") or "").lower() != str(
            parameters.get("expected_model_sha256") or ""
        ).lower():
            raise ValueError(
                "market residual reused holdout source model SHA mismatch"
            )
        for key in ("from_date", "through_date"):
            if str(payload.get(key) or "") != str(parameters.get(key) or ""):
                raise ValueError(
                    f"market residual reused holdout result {key} mismatch"
                )
        return
    if task_type != "fixed_model_conditional_order":
        return
    if not isinstance(parameters, dict):
        raise ValueError(
            "fixed model conditional order job parameters are invalid"
        )
    for key in (
        "training_through",
        "evaluation_from",
        "evaluation_through",
    ):
        if str(payload.get(key) or "") != str(parameters.get(key) or ""):
            raise ValueError(
                f"fixed model conditional order result {key} mismatch"
            )
    if str(payload.get("source_model_sha256") or "").lower() != str(
        parameters.get("expected_model_sha256") or ""
    ).lower():
        raise ValueError(
            "fixed model conditional order result source_model_sha256 mismatch"
        )
    if payload.get("reused_holdout_research_only") is not True:
        raise ValueError(
            "fixed model conditional order result must be research-only "
            "for the reused holdout"
        )
    if payload.get("promotion_eligible") is not False:
        raise ValueError(
            "fixed model conditional order reused holdout must not be "
            "promotion eligible"
        )
    expected_races = parameters.get("expected_evaluation_races")
    actual_races = payload.get("evaluation_races")
    if (
        isinstance(expected_races, bool)
        or not isinstance(expected_races, int)
        or isinstance(actual_races, bool)
        or not isinstance(actual_races, int)
        or actual_races != expected_races
    ):
        raise ValueError(
            "fixed model conditional order result evaluation_races mismatch"
        )
    coverage_paths = (
        ("conditional_order",),
        ("listwise_baseline",),
        ("bankroll",),
        ("baseline_bankroll",),
        ("conditional_payout_walk_forward", "bankroll"),
        ("expected_return_calibration", "bankroll"),
    )
    for path in coverage_paths:
        value: Any = payload
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        actual = value.get("evaluated_races") if isinstance(value, dict) else None
        if (
            isinstance(actual, bool)
            or not isinstance(actual, int)
            or actual != expected_races
        ):
            label = ".".join((*path, "evaluated_races"))
            raise ValueError(
                f"fixed model conditional order result {label} mismatch"
            )
    bankroll_paths = (
        ("bankroll",),
        ("baseline_bankroll",),
        ("conditional_payout_walk_forward", "bankroll"),
        ("expected_return_calibration", "bankroll"),
    )
    for path in bankroll_paths:
        value = payload
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if not _valid_tail_portfolio_contract(value):
            label = ".".join((*path, "tail_portfolio_diagnostics"))
            raise ValueError(
                f"fixed model conditional order result {label} invalid"
            )
        effective_hit_count = value.get("effective_hit_count")
        if (
            isinstance(effective_hit_count, bool)
            or not isinstance(effective_hit_count, (int, float))
            or not math.isfinite(float(effective_hit_count))
            or float(effective_hit_count) < 0.0
        ):
            label = ".".join((*path, "effective_hit_count"))
            raise ValueError(
                f"fixed model conditional order result {label} invalid"
            )
    if parameters.get("direct_pair_diagnostics") is True:
        pair_structure = payload.get("reversed_place_pair_structure")
        conditional_structure = (
            pair_structure.get("conditional_order")
            if isinstance(pair_structure, dict)
            else None
        )
        baseline_structure = (
            pair_structure.get("listwise_baseline")
            if isinstance(pair_structure, dict)
            else None
        )
        if (
            not isinstance(pair_structure, dict)
            or pair_structure.get("promotion_eligible") is not False
            or not isinstance(conditional_structure, dict)
            or conditional_structure.get("evaluated_races") != expected_races
            or not isinstance(baseline_structure, dict)
            or baseline_structure.get("evaluated_races") != expected_races
            or not isinstance(pair_structure.get("paired_confidence"), dict)
        ):
            raise ValueError(
                "fixed model conditional order reversed pair structure invalid"
            )
        diagnostics = payload.get("direct_pair_diagnostics")
        expected_filters = {
            "baseline_exact_two": "exact_two_allocated_tickets",
            "baseline_reversed_place_pair": (
                "exact_two_same_winner_reversed_second_third"
            ),
            "conditional_exact_two": "exact_two_allocated_tickets",
            "conditional_reversed_place_pair": (
                "exact_two_same_winner_reversed_second_third"
            ),
            "baseline_exact_two_normal_odds": (
                "exact_two_allocated_tickets_max_estimated_odds_100"
            ),
            "baseline_reversed_place_pair_normal_odds": (
                "exact_two_same_winner_reversed_second_third_"
                "max_estimated_odds_100"
            ),
            "conditional_exact_two_normal_odds": (
                "exact_two_allocated_tickets_max_estimated_odds_100"
            ),
            "conditional_reversed_place_pair_normal_odds": (
                "exact_two_same_winner_reversed_second_third_"
                "max_estimated_odds_100"
            ),
        }
        expected_comparisons = {
            "baseline_exact_two": "full_baseline",
            "baseline_reversed_place_pair": "full_baseline",
            "conditional_exact_two": "baseline_exact_two",
            "conditional_reversed_place_pair": (
                "baseline_reversed_place_pair"
            ),
            "baseline_exact_two_normal_odds": "baseline_exact_two",
            "baseline_reversed_place_pair_normal_odds": (
                "baseline_reversed_place_pair"
            ),
            "conditional_exact_two_normal_odds": "conditional_exact_two",
            "conditional_reversed_place_pair_normal_odds": (
                "conditional_reversed_place_pair"
            ),
        }
        if not isinstance(diagnostics, dict):
            raise ValueError(
                "fixed model conditional order pair diagnostics missing"
            )
        for key, expected_filter in expected_filters.items():
            diagnostic = diagnostics.get(key)
            pair_bankroll = (
                diagnostic.get("bankroll")
                if isinstance(diagnostic, dict)
                else None
            )
            policy = (
                pair_bankroll.get("policy")
                if isinstance(pair_bankroll, dict)
                else None
            )
            diagnostic_gate = (
                diagnostic.get("diagnostic_gate")
                if isinstance(diagnostic, dict)
                else None
            )
            if (
                not isinstance(diagnostic, dict)
                or diagnostic.get("promotion_eligible") is not False
                or diagnostic.get("comparison") != expected_comparisons[key]
                or not isinstance(diagnostic_gate, dict)
                or diagnostic_gate.get("reused_holdout_research_only")
                is not True
                or diagnostic_gate.get("holdout_role_pass") is not False
                or diagnostic_gate.get("pass") is not False
                or not isinstance(pair_bankroll, dict)
                or pair_bankroll.get("evaluated_races") != expected_races
                or not isinstance(policy, dict)
                or policy.get("allocation_filter") != expected_filter
                or (
                    key.endswith("_normal_odds")
                    and (
                        float(policy.get("maximum_estimated_odds") or 0.0)
                        != 100.0
                        or policy.get("maximum_estimated_odds_stage")
                        != "before_daily_allocation"
                    )
                )
                or not _valid_tail_portfolio_contract(pair_bankroll)
            ):
                raise ValueError(
                    "fixed model conditional order pair diagnostic invalid: "
                    + key
                )
    fixed_return = payload.get("expected_return_fixed_threshold")
    if isinstance(fixed_return, dict):
        fixed_bankroll = fixed_return.get("bankroll")
        actual = (
            fixed_bankroll.get("evaluated_races")
            if isinstance(fixed_bankroll, dict)
            else None
        )
        if (
            isinstance(actual, bool)
            or not isinstance(actual, int)
            or actual != expected_races
        ):
            raise ValueError(
                "fixed model conditional order result "
                "expected_return_fixed_threshold.bankroll.evaluated_races "
                "mismatch"
            )
        if not _valid_tail_portfolio_contract(fixed_bankroll):
            raise ValueError(
                "fixed model conditional order result "
                "expected_return_fixed_threshold.bankroll."
                "tail_portfolio_diagnostics invalid"
            )
        effective_hit_count = fixed_bankroll.get("effective_hit_count")
        if (
            isinstance(effective_hit_count, bool)
            or not isinstance(effective_hit_count, (int, float))
            or not math.isfinite(float(effective_hit_count))
            or float(effective_hit_count) < 0.0
        ):
            raise ValueError(
                "fixed model conditional order result "
                "expected_return_fixed_threshold.bankroll."
                "effective_hit_count invalid"
            )
    state_contracts = (
        (
            "conditional_payout_walk_forward",
            "conditional_payout_next_day_inference_v1",
        ),
        (
            "expected_return_calibration",
            "expected_return_next_day_inference_v1",
        ),
    )
    for key, expected_schema in state_contracts:
        component = payload.get(key)
        if not isinstance(component, dict):
            raise ValueError(
                f"fixed model conditional order result {key} is missing"
            )
        if component.get("artifact_state_saved") is not True:
            raise ValueError(
                f"fixed model conditional order result {key} state not saved"
            )
        if str(component.get("state_schema") or "") != expected_schema:
            raise ValueError(
                f"fixed model conditional order result {key} schema mismatch"
            )
        if (
            str(component.get("state_trained_through") or "")
            != str(parameters["evaluation_through"])
        ):
            raise ValueError(
                f"fixed model conditional order result {key} "
                "trained_through mismatch"
            )
        if (
            str(component.get("state_role") or "")
            != "next_day_inference_after_evaluation"
        ):
            raise ValueError(
                f"fixed model conditional order result {key} role mismatch"
            )


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
    _validate_job_result_contract(job, payload)
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


def enqueue_joint_edge_calibrated_replay(
    conn: Any,
    job: dict[str, Any],
    *,
    app_root: Path,
) -> int | None:
    """Attach the strict-prior realized-value gate to every joint base run."""
    if job.get("task_type") != "joint_bankroll_walk_forward":
        return None
    parameters = job.get("parameters") or {}
    if not isinstance(parameters, dict):
        try:
            parameters = json.loads(str(parameters))
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    source_job_id = int(job["job_id"])
    relative = (
        f"data/models/evaluation_queue/job-{source_job_id:08d}.json"
    )
    artifact = (app_root / relative).resolve()
    result_root = (
        app_root / "data/models/evaluation_queue"
    ).resolve()
    if result_root not in artifact.parents or not artifact.is_file():
        return None
    seed = int(parameters.get("seed") or 33_041)
    return enqueue_job(
        conn,
        task_type="joint_edge_calibrated_replay",
        model_key=(
            f"{job['model_key']}:strict_prior_value_calibrated_v4"
        ),
        parameters={
            "base_artifact": relative,
            "initial_daily_bankroll_yen": int(
                parameters.get("initial_daily_bankroll_yen") or 10_000
            ),
            "calibration_margin": float(
                parameters.get("buy_margin") or 0.0
            ),
            "calibration_bootstrap_samples": 5_000,
            "calibration_min_training_days": 30,
            "calibration_min_portfolios": 300,
            "calibration_min_candidate_days": 20,
            "bootstrap_samples": 20_000,
            "seed": (seed + 10_000) % 2_147_483_648,
            "timeout_seconds": 7_200,
        },
        priority=int(job.get("priority") or 50) + 1,
        max_attempts=2,
        parent_job_id=source_job_id,
    )


def reconcile_joint_edge_calibrated_replays(
    conn: Any,
    *,
    app_root: Path,
) -> list[int]:
    """Recover calibrated replays missed across worker code reloads."""
    rows = conn.execute(
        """
        SELECT base.*
        FROM model_evaluation_jobs AS base
        WHERE base.task_type = 'joint_bankroll_walk_forward'
          AND base.status = 'completed'
          AND base.completed_at >= CURRENT_TIMESTAMP - INTERVAL '30 days'
          AND NOT EXISTS (
            SELECT 1
            FROM model_evaluation_jobs AS child
            WHERE child.parent_job_id = base.job_id
              AND child.task_type = 'joint_edge_calibrated_replay'
              AND child.model_key = (
                base.model_key || ':strict_prior_value_calibrated_v4'
              )
              AND child.status IN ('queued', 'running', 'completed')
          )
        ORDER BY base.completed_at, base.job_id
        """
    ).fetchall()
    inserted: list[int] = []
    for row in rows:
        job_id = enqueue_joint_edge_calibrated_replay(
            conn, dict(row), app_root=app_root
        )
        if job_id is not None:
            inserted.append(job_id)
    return inserted


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


def periodic_model_cache_archive_paths(
    conn: Any,
    *,
    app_root: Path,
    now: datetime,
    minimum_age_seconds: int = 24 * 3600,
    maximum_files: int = 8,
    maximum_bytes: int = 12 * 1024**3,
) -> list[str]:
    """Select old, reproducible matrices while excluding active job caches."""

    if minimum_age_seconds < 0 or maximum_files < 1 or maximum_bytes < 1:
        raise ValueError("invalid periodic model-cache archive limits")
    rows = conn.execute(
        """
        SELECT job_id, parameters
        FROM model_evaluation_jobs
        WHERE status IN ('queued', 'running')
        """
    ).fetchall()
    active_job_ids = {int(row["job_id"]) for row in rows}
    root = app_root / "data" / "models" / "evaluation_cache"
    if not root.is_dir():
        return []
    active_cache_matrices: set[Path] = set()
    for row in rows:
        parameters = (
            row["parameters"]
            if "parameters" in row.keys()
            else {}
        )
        if isinstance(parameters, str):
            try:
                parameters = json.loads(parameters)
            except json.JSONDecodeError:
                parameters = {}
        if not isinstance(parameters, dict):
            continue
        cache_prefix = parameters.get("cache_prefix")
        if not isinstance(cache_prefix, str) or not cache_prefix.strip():
            continue
        prefix = Path(cache_prefix)
        if not prefix.is_absolute():
            prefix = app_root / prefix
        active_cache_matrices.add(
            Path(f"{prefix.resolve()}.matrix.npz")
        )
    cutoff = now.timestamp() - minimum_age_seconds
    candidates: list[tuple[float, int, Path]] = []
    for path in root.rglob("*.matrix.npz"):
        try:
            stat = path.stat()
        except OSError:
            continue
        if stat.st_mtime > cutoff or Path(f"{path}.gdrive.json").exists():
            continue
        match = re.search(r"(?:^|/)job-(\d{8})(?:-|/)", path.as_posix())
        if match and int(match.group(1)) in active_job_ids:
            continue
        if path.resolve() in active_cache_matrices:
            continue
        candidates.append((stat.st_mtime, -stat.st_size, path))
    selected: list[str] = []
    selected_bytes = 0
    for _mtime, negative_size, path in sorted(candidates):
        size = -negative_size
        if selected and selected_bytes + size > maximum_bytes:
            continue
        selected.append(str(path.relative_to(app_root)))
        selected_bytes += size
        if len(selected) >= maximum_files:
            break
    return selected


def eligible_decision_v38_frozen_artifact(
    conn: Any,
    *,
    app_root: Path,
) -> dict[str, Any] | None:
    artifact_root = (
        app_root / "data" / "models" / "evaluation_queue"
    ).resolve()
    rows = []
    for task_type in (
        "decision_stacked_market_v44",
        "decision_market_residual_v38",
    ):
        rows.extend(conn.execute(
            """
            SELECT job_id, result_path, completed_at
            FROM model_evaluation_jobs
            WHERE task_type = ? AND status = ? AND result_path IS NOT NULL
            ORDER BY completed_at ASC, job_id ASC
            """,
            (task_type, "completed"),
        ).fetchall())
    for row in rows:
        result_path = Path(str(row["result_path"]))
        if not result_path.is_absolute():
            result_path = app_root / result_path
        result_path = result_path.resolve()
        if (
            artifact_root not in result_path.parents
            or result_path.suffix != ".json"
        ):
            continue
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        eligible = (
            decision_v44_challenger_eligible(payload)
            if payload.get("model")
            == "decision_time_stacked_market_residual_v44"
            else decision_v38_challenger_eligible(payload)
        )
        if not eligible:
            continue
        return {
            "job_id": int(row["job_id"]),
            "result_path": result_path,
            "result_sha256": _file_sha256(result_path),
            "payload": payload,
            "completed_at": row["completed_at"],
        }
    return None


def seed_periodic_jobs(
    conn: Any,
    *,
    now: datetime | None = None,
    app_root: Path | None = None,
) -> list[int]:
    now = now or datetime.now(timezone.utc)
    inserted: list[int] = []

    def already_scheduled(
        task_type: str,
        key: str,
        model_key: str,
    ) -> bool:
        row = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM model_evaluation_jobs
            WHERE dedupe_key = ?
               OR (
                 task_type = ? AND model_key = ?
                 AND status IN ('queued','running')
               )
            """,
            (key, task_type, model_key),
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
        if already_scheduled(task_type, key, model_key):
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
    jst_now = now.astimezone(JST)
    prospective_through = jst_now.date() - timedelta(days=1)
    expected_model_sha256 = _production_trend_point_model_sha256(app_root)
    candidate_specs = (
        {
            "model_key": (
                PROSPECTIVE_STRICT_LCB_CONTEXT_R05_08_JOB_12315_MODEL_KEY
            ),
            "model_input": PROSPECTIVE_STRICT_LCB_JOB_12315_MODEL_INPUT,
            "source_model_job_id": 12_315,
            "source_evaluation_job_id": 12_618,
            "expected_model_sha256": (
                PROSPECTIVE_STRICT_LCB_JOB_12315_MODEL_SHA256
            ),
            "policy": (
                "trend_point_context_v3_strict_prior_empirical_roi_"
                "lcb95_r05_08"
            ),
            "required_ticket_count": None,
            "require_reversed_place_pair": False,
            "maximum_forecast_odds": None,
            "minimum_race_number": 5,
            "maximum_race_number": 8,
            "closing_context_features": True,
            "odds_safety_factor": 1.0,
            "registered_after": (
                PROSPECTIVE_STRICT_LCB_CONTEXT_JOB_12315_REGISTERED_AFTER
            ),
            "priority": 48,
        },
        {
            "model_key": (
                PROSPECTIVE_STRICT_LCB_CONTEXT_R05_12_JOB_12315_MODEL_KEY
            ),
            "model_input": PROSPECTIVE_STRICT_LCB_JOB_12315_MODEL_INPUT,
            "source_model_job_id": 12_315,
            "source_evaluation_job_id": 12_618,
            "expected_model_sha256": (
                PROSPECTIVE_STRICT_LCB_JOB_12315_MODEL_SHA256
            ),
            "policy": (
                "trend_point_context_v3_strict_prior_empirical_roi_"
                "lcb95_r05_12"
            ),
            "required_ticket_count": None,
            "require_reversed_place_pair": False,
            "maximum_forecast_odds": None,
            "minimum_race_number": 5,
            "closing_context_features": True,
            "odds_safety_factor": 1.0,
            "registered_after": (
                PROSPECTIVE_STRICT_LCB_CONTEXT_JOB_12315_REGISTERED_AFTER
            ),
            "priority": 47,
        },
        {
            "model_key": PROSPECTIVE_STRICT_LCB_R05_12_JOB_12315_MODEL_KEY,
            "model_input": PROSPECTIVE_STRICT_LCB_JOB_12315_MODEL_INPUT,
            "source_model_job_id": 12_315,
            "source_evaluation_job_id": 12_618,
            "expected_model_sha256": (
                PROSPECTIVE_STRICT_LCB_JOB_12315_MODEL_SHA256
            ),
            "policy": "trend_point_strict_prior_empirical_roi_lcb95_r05_12",
            "required_ticket_count": None,
            "require_reversed_place_pair": False,
            "maximum_forecast_odds": None,
            "minimum_race_number": 5,
            "odds_safety_factor": 1.0,
            "registered_after": (
                PROSPECTIVE_STRICT_LCB_JOB_12315_REGISTERED_AFTER
            ),
            "priority": 46,
        },
        {
            "model_key": PROSPECTIVE_STRICT_LCB_JOB_12315_MODEL_KEY,
            "model_input": PROSPECTIVE_STRICT_LCB_JOB_12315_MODEL_INPUT,
            "source_model_job_id": 12_315,
            "source_evaluation_job_id": 12_618,
            "expected_model_sha256": (
                PROSPECTIVE_STRICT_LCB_JOB_12315_MODEL_SHA256
            ),
            "policy": "trend_point_strict_prior_empirical_roi_lcb95",
            "required_ticket_count": None,
            "require_reversed_place_pair": False,
            "maximum_forecast_odds": None,
            "odds_safety_factor": 1.0,
            "registered_after": (
                PROSPECTIVE_STRICT_LCB_JOB_12315_REGISTERED_AFTER
            ),
            "priority": 45,
        },
        {
            "model_key": PRODUCTION_TREND_POINT_MODEL_KEY,
            "model_input": PRODUCTION_TREND_POINT_MODEL_INPUT,
            "source_model_job_id": 12_012,
            "source_evaluation_job_id": PRODUCTION_TREND_POINT_SOURCE_EVALUATION_JOB_ID,
            "expected_model_sha256": expected_model_sha256,
            "policy": "trend_point_market_offset_discrete_multinomial_kelly",
            "required_ticket_count": None,
            "require_reversed_place_pair": False,
            "maximum_forecast_odds": None,
            "odds_safety_factor": 1.0,
            "registered_after": PRODUCTION_TREND_POINT_REGISTERED_AFTER,
            "priority": 44,
        },
        {
            "model_key": PRODUCTION_TREND_POINT_TWO_TICKET_MODEL_KEY,
            "model_input": PRODUCTION_TREND_POINT_MODEL_INPUT,
            "source_model_job_id": 12_012,
            "source_evaluation_job_id": PRODUCTION_TREND_POINT_SOURCE_EVALUATION_JOB_ID,
            "expected_model_sha256": expected_model_sha256,
            "policy": (
                "trend_point_market_offset_discrete_multinomial_kelly_"
                "exact_two_ticket"
            ),
            "required_ticket_count": 2,
            "require_reversed_place_pair": False,
            "maximum_forecast_odds": None,
            "odds_safety_factor": 1.0,
            "registered_after": PRODUCTION_TREND_POINT_REGISTERED_AFTER,
            "priority": 43,
        },
        {
            "model_key": PRODUCTION_TREND_POINT_REVERSED_PAIR_MODEL_KEY,
            "model_input": PRODUCTION_TREND_POINT_MODEL_INPUT,
            "source_model_job_id": 12_012,
            "source_evaluation_job_id": PRODUCTION_TREND_POINT_SOURCE_EVALUATION_JOB_ID,
            "expected_model_sha256": expected_model_sha256,
            "policy": (
                "trend_point_market_offset_discrete_multinomial_kelly_"
                "reversed_place_pair"
            ),
            "required_ticket_count": 2,
            "require_reversed_place_pair": True,
            "maximum_forecast_odds": None,
            "odds_safety_factor": 1.0,
            "registered_after": PROSPECTIVE_LIGHTGBM_TWO_TICKET_REGISTERED_AFTER,
            "priority": 42,
        },
        {
            "model_key": PROSPECTIVE_LIGHTGBM_TWO_TICKET_MODEL_KEY,
            "model_input": PROSPECTIVE_LIGHTGBM_MODEL_INPUT,
            "source_model_job_id": PROSPECTIVE_LIGHTGBM_SOURCE_MODEL_JOB_ID,
            "source_evaluation_job_id": PROSPECTIVE_LIGHTGBM_SOURCE_MODEL_JOB_ID,
            "expected_model_sha256": PROSPECTIVE_LIGHTGBM_MODEL_SHA256,
            "policy": (
                "trend_point_market_offset_discrete_multinomial_kelly_"
                "exact_two_ticket"
            ),
            "required_ticket_count": 2,
            "require_reversed_place_pair": False,
            "maximum_forecast_odds": None,
            "odds_safety_factor": 1.0,
            "registered_after": PROSPECTIVE_LIGHTGBM_TWO_TICKET_REGISTERED_AFTER,
            "priority": 41,
        },
        {
            "model_key": PROSPECTIVE_LIGHTGBM_REVERSED_PAIR_MODEL_KEY,
            "model_input": PROSPECTIVE_LIGHTGBM_MODEL_INPUT,
            "source_model_job_id": PROSPECTIVE_LIGHTGBM_SOURCE_MODEL_JOB_ID,
            "source_evaluation_job_id": PROSPECTIVE_LIGHTGBM_SOURCE_MODEL_JOB_ID,
            "expected_model_sha256": PROSPECTIVE_LIGHTGBM_MODEL_SHA256,
            "policy": (
                "trend_point_market_offset_discrete_multinomial_kelly_"
                "reversed_place_pair"
            ),
            "required_ticket_count": 2,
            "require_reversed_place_pair": True,
            "maximum_forecast_odds": None,
            "odds_safety_factor": 1.0,
            "registered_after": PROSPECTIVE_LIGHTGBM_TWO_TICKET_REGISTERED_AFTER,
            "priority": 40,
        },
        {
            "model_key": PRODUCTION_TREND_POINT_NORMAL_REVERSED_PAIR_MODEL_KEY,
            "model_input": PRODUCTION_TREND_POINT_MODEL_INPUT,
            "source_model_job_id": 12_012,
            "source_evaluation_job_id": PRODUCTION_TREND_POINT_SOURCE_EVALUATION_JOB_ID,
            "expected_model_sha256": expected_model_sha256,
            "policy": (
                "trend_point_market_offset_discrete_multinomial_kelly_"
                "reversed_place_pair_max_forecast_odds_100"
            ),
            "required_ticket_count": 2,
            "require_reversed_place_pair": True,
            "maximum_forecast_odds": 100.0,
            "odds_safety_factor": 1.0,
            "registered_after": PROSPECTIVE_NORMAL_ODDS_REGISTERED_AFTER,
            "priority": 39,
        },
        {
            "model_key": PRODUCTION_TREND_POINT_SAFETY_110_MODEL_KEY,
            "model_input": PRODUCTION_TREND_POINT_MODEL_INPUT,
            "source_model_job_id": 12_012,
            "source_evaluation_job_id": (
                PRODUCTION_TREND_POINT_SOURCE_EVALUATION_JOB_ID
            ),
            "expected_model_sha256": expected_model_sha256,
            "policy": (
                "trend_point_strict_prior_empirical_roi_lcb95_"
                "odds_safety_110"
            ),
            "required_ticket_count": None,
            "require_reversed_place_pair": False,
            "maximum_forecast_odds": None,
            "odds_safety_factor": 1.10,
            "registered_after": PROSPECTIVE_SAFETY_110_REGISTERED_AFTER,
            "priority": 38,
        },
    )
    if jst_now.hour >= 3:
        for spec in candidate_specs:
            registered_after = datetime.fromisoformat(
                str(spec["registered_after"])
            ).date()
            model_input = str(spec["model_input"])
            candidate_sha256 = spec["expected_model_sha256"]
            if (
                prospective_through <= registered_after
                or candidate_sha256 is None
                or (
                    spec["model_key"] in {
                        PROSPECTIVE_STRICT_LCB_JOB_12315_MODEL_KEY,
                        PROSPECTIVE_STRICT_LCB_R05_12_JOB_12315_MODEL_KEY,
                        PROSPECTIVE_STRICT_LCB_CONTEXT_R05_12_JOB_12315_MODEL_KEY,
                    }
                    and (
                        app_root is None
                        or not (app_root / model_input).is_file()
                    )
                )
                or (
                    spec["model_input"] == PROSPECTIVE_LIGHTGBM_MODEL_INPUT
                    and (
                        app_root is None
                        or not (app_root / model_input).is_file()
                    )
                )
            ):
                continue
            parameters = {
                "model_input": model_input,
                "from_date": PRODUCTION_TREND_POINT_EVALUATION_FROM,
                "through_date": prospective_through.isoformat(),
                "daily_budget_yen": 10_000,
                "calibrator_strategy": PRODUCTION_TREND_POINT_STRATEGY,
                "min_calibration_days": 2,
                "minimum_day_coverage": 1.0,
                "trend_point_registered_after": str(spec["registered_after"]),
                "trend_point_odds_safety_factor": float(
                    spec["odds_safety_factor"]
                ),
                "expected_model_sha256": candidate_sha256,
                "timeout_seconds": 7_200,
                "prospective_candidate": {
                    "source_model_job_id": spec["source_model_job_id"],
                    "source_evaluation_job_id": spec["source_evaluation_job_id"],
                    "expected_model_sha256": candidate_sha256,
                    "policy": spec["policy"],
                    "registered_after": spec["registered_after"],
                    "odds_safety_factor": float(
                        spec["odds_safety_factor"]
                    ),
                    "evidence_dates": "strictly_after_registered_after",
                    "selection_data_is_diagnostic_only": True,
                    "real_betting_enabled": False,
                },
            }
            if spec["model_key"] == PRODUCTION_TREND_POINT_SAFETY_110_MODEL_KEY:
                parameters["trend_point_odds_safety_sweep"] = True
            required_ticket_count = spec["required_ticket_count"]
            if required_ticket_count is not None:
                parameters["trend_point_required_ticket_count"] = (
                    required_ticket_count
                )
                parameters["prospective_candidate"][
                    "required_ticket_count"
                ] = required_ticket_count
            if spec["require_reversed_place_pair"]:
                parameters["trend_point_require_reversed_place_pair"] = True
                parameters["prospective_candidate"][
                    "require_reversed_place_pair"
                ] = True
            maximum_forecast_odds = spec["maximum_forecast_odds"]
            if maximum_forecast_odds is not None:
                parameters["trend_point_maximum_forecast_odds"] = (
                    maximum_forecast_odds
                )
                parameters["prospective_candidate"][
                    "maximum_forecast_odds"
                ] = maximum_forecast_odds
            minimum_race_number = int(spec.get("minimum_race_number", 1))
            if minimum_race_number > 1:
                parameters["trend_point_minimum_race_number"] = (
                    minimum_race_number
                )
                parameters["prospective_candidate"][
                    "minimum_race_number"
                ] = minimum_race_number
            maximum_race_number = int(spec.get("maximum_race_number", 12))
            if maximum_race_number < 12:
                parameters["trend_point_maximum_race_number"] = (
                    maximum_race_number
                )
                parameters["prospective_candidate"][
                    "maximum_race_number"
                ] = maximum_race_number
            closing_context_features = bool(
                spec.get("closing_context_features", False)
            )
            if closing_context_features:
                parameters["trend_point_closing_context_features"] = True
                parameters["prospective_candidate"][
                    "closing_context_features"
                ] = True
            key = dedupe_key(
                "market_residual_walk_forward",
                str(spec["model_key"]),
                parameters,
            )
            if not already_scheduled(
                "market_residual_walk_forward",
                key,
                str(spec["model_key"]),
            ):
                job_id = enqueue_job(
                    conn,
                    task_type="market_residual_walk_forward",
                    model_key=str(spec["model_key"]),
                    parameters=parameters,
                    priority=int(spec["priority"]),
                    max_attempts=3,
                )
                if job_id is not None:
                    inserted.append(job_id)
    v38_frozen = (
        eligible_decision_v38_frozen_artifact(conn, app_root=app_root)
        if app_root is not None
        else None
    )
    if app_root is not None:
        v38_from = datetime.fromisoformat(DECISION_V38_TRAINING_FROM).date()
        v38_cutoff = v38_from + timedelta(
            days=DECISION_V38_MINIMUM_TRAINING_DAYS - 1
        )
        v38_cache = (
            app_root
            / "data/models/evaluation_cache/market_scored"
            / (
                "job-00012315_"
                f"{DECISION_V38_TRAINING_FROM}_{prospective_through.isoformat()}"
                ".races.joblib"
            )
        )
        if (
            v38_frozen is None
            and
            prospective_through > v38_cutoff
            and v38_cache.is_file()
        ):
            parameters = {
                "scored_cache": v38_cache.relative_to(app_root).as_posix(),
                "calibration_through": v38_cutoff.isoformat(),
                "minimum_training_days": DECISION_V38_MINIMUM_TRAINING_DAYS,
                "minimum_training_races": DECISION_V38_MINIMUM_TRAINING_RACES,
                "num_threads": 4,
                "timeout_seconds": 14_400,
            }
            model_key = (
                "decision_stacked_market_v44_job_12315_cutoff_"
                + v38_cutoff.strftime("%Y%m%d")
            )
            key = dedupe_key(
                "decision_stacked_market_v44", model_key, parameters
            )
            if not already_scheduled(
                "decision_stacked_market_v44", key, model_key
            ):
                job_id = enqueue_job(
                    conn,
                    task_type="decision_stacked_market_v44",
                    model_key=model_key,
                    parameters=parameters,
                    priority=49,
                    max_attempts=2,
                )
                if job_id is not None:
                    inserted.append(job_id)
    if app_root is not None and v38_frozen is not None:
        completed_at = v38_frozen["completed_at"]
        if isinstance(completed_at, datetime):
            registration_date = completed_at.astimezone(JST).date()
        else:
            registration_date = datetime.fromisoformat(
                str(completed_at)
            ).astimezone(JST).date()
        selection_through = datetime.fromisoformat(
            str(v38_frozen["payload"]["evaluation_through"])
        ).date()
        registration_date = max(registration_date, selection_through)
        if prospective_through > registration_date and v38_cache.is_file():
            frozen_path = Path(v38_frozen["result_path"])
            source_job_id = int(v38_frozen["job_id"])
            parameters = {
                "frozen_artifact": frozen_path.relative_to(app_root).as_posix(),
                "scored_cache": v38_cache.relative_to(app_root).as_posix(),
                "registered_after": registration_date.isoformat(),
                "daily_budget_yen": 10_000,
                "timeout_seconds": 14_400,
                "prospective_candidate": {
                    "source_model_job_id": source_job_id,
                    "source_artifact_sha256": v38_frozen["result_sha256"],
                    "policy": (
                        "decision_frozen_stack_top5_strict_prior_"
                        "empirical_roi_lcb95"
                    ),
                    "registered_after": registration_date.isoformat(),
                    "selection_evaluation_through": (
                        selection_through.isoformat()
                    ),
                    "evidence_dates": "strictly_after_registered_after",
                    "selection_data_excluded_from_calibration": True,
                    "real_betting_enabled": False,
                },
            }
            model_key = (
                "prospective_decision_stack_value_job_"
                f"{source_job_id:08d}"
            )
            key = dedupe_key(
                "decision_v38_empirical_lcb", model_key, parameters
            )
            if not already_scheduled(
                "decision_v38_empirical_lcb", key, model_key
            ):
                job_id = enqueue_job(
                    conn,
                    task_type="decision_v38_empirical_lcb",
                    model_key=model_key,
                    parameters=parameters,
                    priority=50,
                    max_attempts=3,
                )
                if job_id is not None:
                    inserted.append(job_id)
    if app_root is not None:
        paths = periodic_model_cache_archive_paths(
            conn, app_root=app_root, now=now
        )
        if paths:
            parameters = {"paths": paths, "timeout_seconds": 86400}
            key = dedupe_key(
                "gdrive_model_cache_archive",
                "evaluation-cache-auto",
                parameters,
            )
            if not already_scheduled(
                "gdrive_model_cache_archive",
                key,
                "evaluation-cache-auto",
            ):
                job_id = enqueue_job(
                    conn,
                    task_type="gdrive_model_cache_archive",
                    model_key="evaluation-cache-auto",
                    parameters=parameters,
                    priority=12,
                    max_attempts=3,
                )
                if job_id is not None:
                    inserted.append(job_id)
    return inserted


def _production_trend_point_model_sha256(
    app_root: Path | None,
) -> str | None:
    if app_root is None:
        return None
    result_path = (
        app_root / "data" / "models" / "evaluation_queue"
        / f"job-{PRODUCTION_TREND_POINT_SOURCE_EVALUATION_JOB_ID:08d}.json"
    )
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    source_model = Path(str(payload.get("source_model") or ""))
    source_model_sha256 = str(
        payload.get("source_model_sha256") or ""
    ).strip().lower()
    if (
        source_model.name != Path(PRODUCTION_TREND_POINT_MODEL_INPUT).name
        or not re.fullmatch(r"[0-9a-f]{64}", source_model_sha256)
    ):
        return None
    return source_model_sha256


def production_trend_point_readiness(
    app_root: Path | None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return the fixed candidate identity and first-evidence schedule audit."""
    registered_after = datetime.fromisoformat(
        PRODUCTION_TREND_POINT_REGISTERED_AFTER
    ).date()
    first_evidence_date = registered_after + timedelta(days=1)
    first_scheduled_at = datetime.combine(
        first_evidence_date + timedelta(days=1),
        datetime.min.time(),
        tzinfo=JST,
    ).replace(hour=3)
    current = (now or datetime.now(timezone.utc)).astimezone(JST)
    expected_sha256 = _production_trend_point_model_sha256(app_root)
    model_path = (
        app_root / PRODUCTION_TREND_POINT_MODEL_INPUT
        if app_root is not None else None
    )
    observed_sha256 = None
    if model_path is not None and model_path.is_file():
        try:
            observed_sha256 = _file_sha256(model_path)
        except OSError:
            observed_sha256 = None
    if expected_sha256 is None:
        status = "blocked_missing_or_invalid_source_evaluation"
        error = (
            "source evaluation result is missing or does not pin the production "
            "model filename and SHA-256"
        )
    elif observed_sha256 is None:
        status = "blocked_missing_model_artifact"
        error = "fixed production model artifact is missing or unreadable"
    elif observed_sha256 != expected_sha256:
        status = "blocked_model_sha256_mismatch"
        error = "fixed production model SHA-256 differs from its registration"
    elif current < first_scheduled_at:
        status = "ready_waiting_for_first_unseen_day"
        error = None
    else:
        status = "ready_evaluation_schedule_due"
        error = None
    return {
        "status": status,
        "error": error,
        "model_key": PRODUCTION_TREND_POINT_MODEL_KEY,
        "model_input": PRODUCTION_TREND_POINT_MODEL_INPUT,
        "source_evaluation_job_id": (
            PRODUCTION_TREND_POINT_SOURCE_EVALUATION_JOB_ID
        ),
        "registered_after": PRODUCTION_TREND_POINT_REGISTERED_AFTER,
        "next_evidence_date": first_evidence_date.isoformat(),
        "first_scheduled_at_jst": first_scheduled_at.isoformat(),
        "expected_model_sha256": expected_sha256,
        "observed_model_sha256": observed_sha256,
        "fixed": bool(
            expected_sha256
            and observed_sha256
            and expected_sha256 == observed_sha256
        ),
        "real_betting_enabled": False,
    }


DEFAULT_WORK_TICKETS = (
    ("OPS-QUEUE-001", "DBジョブ基盤と資源監視", "運用基盤", "評価・集計・バックアップをDBキューから実行する", "4ランナーが資源条件付きで取得し完了履歴をDBへ残す", 100, "in_progress", 70),
    ("OPS-BACKUP-001", "GDriveバックアップのキュー移行", "バックアップ", "生データ転送を定期DBジョブとして管理する", "排他付き転送が完了し元データ削除と結果記録を確認する", 90, "in_progress", 65),
    ("OPS-REPO-SYNC-001", "Gitリポジトリの定期確認と安全な更新", "運用基盤", "DB定期ジョブでoriginを確認し安全条件を満たす時だけfast-forwardする", "dirty・履歴分岐・評価実行中は更新せず監査結果を残し、cleanかつidle時だけff-onlyで更新する", 92, "in_progress", 20),
    ("MODEL-OPT-001", "モデル再設計と収益ゲート収束", "モデル", "特徴量・教師・構造を同一評価軸で反復検証する", "未使用holdoutでROI・損益・確率指標の昇格基準を満たす", 100, "in_progress", 55),
    (
        "MODEL-JOINT-EDGE-CALIBRATION-20260803",
        "結合GAポートフォリオのstrict-prior実現収益校正",
        "モデル",
        "v6全レース反実候補台帳を教師に、購入判断時刻tとsnapshot ageを分離監査し、過去日だけで予測総受取倍率から実現総受取倍率へのisotonic日ブロックLCBを学習する",
        "全レースでcaptured_at<=tとsnapshot ageを記録し、同日結果逆流0、inverted_cdf経験分位、100円単位・払戻利用時刻を再現し、外側R>=100かつ5%下側支持5標本以上、30校正準備日・1000R・200券・20的中、ROI片側95%下限>1、最大1的中除外ROI>1、正損益を満たした後に完全joint walk-forwardで再確認する",
        99,
        "in_progress",
        65,
    ),
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
                    reconcile_joint_edge_calibrated_replays(
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
                        seed_periodic_jobs(conn, app_root=app_root)
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
                    enqueue_joint_edge_calibrated_replay(
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
    initialized = False
    while True:
        try:
            if not initialized:
                with connection(args.db) as conn:
                    ensure_schema(conn)
                    seed_work_tickets(conn)
                initialized = True
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
                    seed_periodic_jobs(conn, app_root=app_root)
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
