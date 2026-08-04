import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from boatrace_ai.evaluation_queue import (
    periodic_model_cache_archive_paths,
    seed_periodic_jobs,
)


class _Rows:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0] if self.rows else None


class _Queue:
    def __init__(self, active=()):
        self.active = list(active)

    def execute(self, query, params=()):
        if "SELECT job_id" in query:
            return _Rows([
                (
                    value
                    if isinstance(value, dict)
                    else {"job_id": value, "parameters": {}}
                )
                for value in self.active
            ])
        if "SELECT COUNT(*) AS count" in query:
            return _Rows([{"count": 0}])
        raise AssertionError(query)


def _matrix(root: Path, job_id: int, *, age_hours: int, size: int = 8) -> Path:
    path = (
        root
        / "data/models/evaluation_cache"
        / f"job-{job_id:08d}"
        / "candidate.matrix.npz"
    )
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x" * size)
    stamp = (datetime.now(timezone.utc) - timedelta(hours=age_hours)).timestamp()
    os.utime(path, (stamp, stamp))
    return path


def test_periodic_cache_selection_excludes_recent_active_and_archived(tmp_path):
    old = _matrix(tmp_path, 100, age_hours=48)
    _matrix(tmp_path, 101, age_hours=48)
    _matrix(tmp_path, 102, age_hours=1)
    archived = _matrix(tmp_path, 103, age_hours=48)
    Path(f"{archived}.gdrive.json").write_text("{}")

    selected = periodic_model_cache_archive_paths(
        _Queue(active=(101,)),
        app_root=tmp_path,
        now=datetime.now(timezone.utc),
    )

    assert selected == [str(old.relative_to(tmp_path))]


def test_periodic_cache_selection_protects_active_dependency_prefix(tmp_path):
    dependent = _matrix(tmp_path, 11732, age_hours=48)
    other = _matrix(tmp_path, 11733, age_hours=48)
    prefix = str(dependent)[: -len(".matrix.npz")]
    queue = _Queue(active=({
        "job_id": 12499,
        "parameters": {
            "cache_prefix": str(Path(prefix).relative_to(tmp_path)),
        },
    },))

    selected = periodic_model_cache_archive_paths(
        queue,
        app_root=tmp_path,
        now=datetime.now(timezone.utc),
    )

    assert selected == [str(other.relative_to(tmp_path))]


def test_periodic_seed_registers_explicit_model_cache_paths(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        "boatrace_ai.evaluation_queue.periodic_model_cache_archive_paths",
        lambda conn, **kwargs: ["data/models/evaluation_cache/job-1/a.matrix.npz"],
    )
    monkeypatch.setattr(
        "boatrace_ai.evaluation_queue.enqueue_job",
        lambda conn, **kwargs: calls.append(kwargs) or len(calls),
    )

    inserted = seed_periodic_jobs(
        _Queue(), now=datetime(2026, 8, 2, tzinfo=timezone.utc), app_root=tmp_path
    )

    assert inserted == [1, 2, 3, 4, 5, 6]
    cache = calls[-1]
    assert cache["task_type"] == "gdrive_model_cache_archive"
    assert cache["parameters"]["paths"] == [
        "data/models/evaluation_cache/job-1/a.matrix.npz"
    ]
    assert cache["parameters"]["timeout_seconds"] == 86400
