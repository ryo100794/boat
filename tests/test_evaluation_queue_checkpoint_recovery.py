from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from boatrace_ai import evaluation_queue
from boatrace_ai.feature_schema import FEATURE_SCHEMA_VERSION


class _Rows:
    def __init__(self, rows=()):
        self.rows = list(rows)

    def fetchall(self):
        return self.rows


class LifecycleConnection:
    def __init__(self, running_jobs=()):
        self.running_jobs = [dict(job) for job in running_jobs]
        self.parent_updates = []
        self.run_updates = []
        self.selects = []

    def execute(self, statement, parameters=()):
        sql = " ".join(statement.split())
        if sql.startswith("SELECT * FROM model_evaluation_jobs"):
            self.selects.append((sql, parameters))
            return _Rows(self.running_jobs)
        if sql.startswith("UPDATE model_evaluation_jobs"):
            self.parent_updates.append((sql, parameters))
            return _Rows()
        if sql.startswith("UPDATE model_evaluation_job_runs"):
            self.run_updates.append((sql, parameters))
            return _Rows()
        raise AssertionError(f"unexpected SQL: {sql}")


def _job(*, attempt: int = 1, max_attempts: int = 3) -> dict:
    return {
        "job_id": 7475,
        "task_type": "listwise_feature_search",
        "category": "evaluation",
        "model_key": "feature-search",
        "parameters": {
            "evaluation_date": "2026-07-29",
            "n_features": 4096,
            "batch_races": 1000,
            "epochs": 2,
            "learning_rate": 0.02,
            "targets": "winner,top3_pl",
            "alphas": "0.00001,0.0001",
            "feature_variants": "full",
        },
        "status": "running",
        "attempt": attempt,
        "max_attempts": max_attempts,
        "worker_id": "evaluator-01",
    }


def _write_checkpoint(app_root: Path, job: dict, *, mutation=None) -> Path:
    source_snapshot = {
        "snapshot_version": 1,
        "race_count": 1000,
        "race_universe_sha256": "a" * 64,
        "source_watermark_sha256": "b" * 64,
        "trifecta_payouts_sha256": "c" * 64,
        "selected_cache_manifest_sha256": "d" * 64,
    }
    source_snapshot["snapshot_sha256"] = hashlib.sha256(
        json.dumps(
            source_snapshot,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    signature = {
        "checkpoint_version": 2,
        "cache_version": 2,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "as_of_date": "2026-07-29",
        "race_count": 1000,
        "race_universe_sha256": "a" * 64,
        "source_data_snapshot": source_snapshot,
        "train_end": 600,
        "selection_end": 800,
        "n_features": 4096,
        "batch_races": 1000,
        "epochs": 2,
        "learning_rate": 0.02,
        "targets": ["winner", "top3_pl"],
        "alphas": [0.00001, 0.0001],
        "feature_variants": [["full", []]],
    }
    row = {
        "feature_variant": "full",
        "drop_feature_groups": [],
        "target": "winner",
        "alpha": 0.00001,
        "entry_log_loss": 0.4,
        "ranking_log_loss": 1.1,
        "winner_top1_accuracy": 0.5,
        "trifecta_top5_hit_rate": 0.3,
        "training_history": [],
    }
    payload = {
        "signature": signature,
        "progress": {
            "completed_candidates": 1,
            "total_candidates": 4,
            "completed_variants": 0,
            "total_variants": 1,
            "last_completed": row,
        },
        "search_results": [row],
    }
    if mutation is not None:
        mutation(payload)
    path = evaluation_queue._feature_search_checkpoint_path(
        job,
        app_root=app_root,
    )
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "incident",
    [
        "TimeoutExpired: command timed out after 21600 seconds",
        "RuntimeError: worker process exited with status 137",
    ],
)
def test_feature_search_failure_requeues_from_valid_checkpoint(
    tmp_path: Path,
    incident: str,
) -> None:
    job = _job()
    checkpoint = _write_checkpoint(tmp_path, job)
    conn = LifecycleConnection()

    evaluation_queue.fail_job(
        conn,
        job=job,
        error=incident,
        app_root=tmp_path,
    )

    _sql, parent = conn.parent_updates[0]
    assert parent[0:2] == ("queued", "queued")
    assert parent[2].startswith(incident)
    assert str(checkpoint.resolve()) in parent[2]
    assert "checkpoint recovery queued" in parent[2]
    assert incident in parent[2]
    _run_sql, run = conn.run_updates[0]
    assert run == (parent[2], 7475, 1)
    if incident.startswith("TimeoutExpired:"):
        retry_parameters = evaluation_queue._timeout_retry_parameters(
            {"timeout_seconds": 21600},
            task_type="listwise_feature_search",
            previous_error=parent[2],
        )
        assert retry_parameters["timeout_seconds"] == 43200


@pytest.mark.parametrize(
    "checkpoint_state",
    [
        "missing",
        "invalid-json",
        "signature-mismatch",
        "legacy-v1",
        "source-identity-missing",
    ],
)
def test_feature_search_does_not_requeue_invalid_or_missing_checkpoint(
    tmp_path: Path,
    checkpoint_state: str,
) -> None:
    job = _job()
    if checkpoint_state != "missing":
        path = _write_checkpoint(tmp_path, job)
        if checkpoint_state == "invalid-json":
            path.write_text("{", encoding="utf-8")
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if checkpoint_state == "signature-mismatch":
                payload["signature"]["as_of_date"] = "2026-07-28"
            elif checkpoint_state == "legacy-v1":
                payload["signature"]["checkpoint_version"] = 1
                payload["signature"].pop("source_data_snapshot")
            else:
                payload["signature"].pop("source_data_snapshot")
            path.write_text(json.dumps(payload), encoding="utf-8")
    conn = LifecycleConnection()

    evaluation_queue.fail_job(
        conn,
        job=job,
        error="TimeoutExpired: timeout",
        app_root=tmp_path,
    )

    _sql, parent = conn.parent_updates[0]
    assert parent[0:2] == ("failed", "failed")
    assert "checkpoint recovery unavailable" in parent[2]
    assert conn.run_updates[0][1][0] == parent[2]


def test_feature_search_does_not_requeue_after_max_attempts(
    tmp_path: Path,
) -> None:
    job = _job(attempt=3, max_attempts=3)
    checkpoint = _write_checkpoint(tmp_path, job)
    conn = LifecycleConnection()

    evaluation_queue.fail_job(
        conn,
        job=job,
        error="TimeoutExpired: timeout",
        app_root=tmp_path,
    )

    _sql, parent = conn.parent_updates[0]
    assert parent[0:2] == ("failed", "failed")
    assert "checkpoint recovery exhausted" in parent[2]
    assert str(checkpoint.resolve()) in parent[2]


def test_stale_worker_loss_requeues_from_checkpoint_and_closes_run(
    tmp_path: Path,
) -> None:
    job = _job()
    _write_checkpoint(tmp_path, job)
    conn = LifecycleConnection([job])

    recovered = evaluation_queue.requeue_stale_jobs(
        conn,
        stale_minutes=90,
        app_root=tmp_path,
    )

    assert recovered == 1
    assert conn.selects[0][1] == (90,)
    assert conn.parent_updates[0][1][0:2] == ("queued", "queued")
    assert "worker lease expired" in conn.parent_updates[0][1][2]
    assert conn.run_updates[0][1][1:] == (7475, 1)


def test_worker_restart_requeues_from_checkpoint_and_closes_run(
    tmp_path: Path,
) -> None:
    job = _job()
    _write_checkpoint(tmp_path, job)
    conn = LifecycleConnection([job])

    recovered = evaluation_queue.recover_worker_job(
        conn,
        worker_id="evaluator-01",
        app_root=tmp_path,
    )

    assert recovered == 1
    assert conn.selects[0][1] == ("evaluator-01",)
    assert conn.parent_updates[0][1][0:2] == ("queued", "queued")
    assert "worker restarted before completion update" in (
        conn.parent_updates[0][1][2]
    )


def test_non_feature_job_retains_bounded_normal_retry() -> None:
    job = _job()
    job["task_type"] = "market_residual_walk_forward"
    conn = LifecycleConnection()

    evaluation_queue.fail_job(
        conn,
        job=job,
        error="RuntimeError: transient",
        app_root=Path("/unused"),
    )

    assert conn.parent_updates[0][1][0:3] == (
        "queued",
        "queued",
        "RuntimeError: transient",
    )
