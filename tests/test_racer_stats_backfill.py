from __future__ import annotations

from pathlib import Path

import pytest

from boatrace_ai import racer_stats_backfill
from boatrace_ai.evaluation_queue import TASK_PROFILES, build_command


class FakeConnection:
    def execute(self, statement: str):
        assert "FROM racer_period_stats" in statement
        return self

    def fetchone(self) -> dict[str, int]:
        return {
            "row_count": 42,
            "period_count": 2,
            "first_year": 2025,
            "last_year": 2026,
        }


def test_backfill_reports_persisted_official_period_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_fetch(_conn: object, **kwargs: object) -> int:
        captured.update(kwargs)
        return 42

    monkeypatch.setattr(racer_stats_backfill, "fetch_racer_stats", fake_fetch)
    result = racer_stats_backfill.run_backfill(
        FakeConnection(),
        from_year=2025,
        to_year=2026,
        raw_dir=tmp_path,
        sleep_seconds=0.0,
    )

    assert result["status"] == "completed"
    assert result["stored_rows"] == 42
    assert result["database_rows"] == 42
    assert result["period_count"] == 2
    assert captured == {
        "from_year": 2025,
        "to_year": 2026,
        "raw_dir": tmp_path,
        "sleep_seconds": 0.0,
    }


def test_backfill_rejects_unbounded_year_range(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        racer_stats_backfill.run_backfill(
            FakeConnection(),
            from_year=2000,
            to_year=2026,
            raw_dir=tmp_path,
            sleep_seconds=0.0,
        )


def test_queue_builds_fixed_racer_stats_command(tmp_path: Path) -> None:
    job = {
        "job_id": 7,
        "task_type": "racer_stats_backfill",
        "parameters": {
            "from_year": 2016,
            "to_year": 2026,
            "sleep_seconds": 0.5,
            "timeout_seconds": 3600,
        },
    }
    command, output = build_command(
        job,
        app_root=tmp_path,
        python=tmp_path / ".venv/bin/python",
        db="postgresql://test",
    )

    assert TASK_PROFILES["racer_stats_backfill"] == {
        "category": "maintenance",
        "memory_mb": 512,
        "disk_mb": 256,
        "idle_cpu": 3.0,
        "max_parallel": 1,
    }
    assert command[:3] == [
        str(tmp_path / ".venv/bin/python"),
        "-m",
        "boatrace_ai.racer_stats_backfill",
    ]
    assert command[command.index("--from-year") + 1] == "2016"
    assert command[command.index("--to-year") + 1] == "2026"
    assert command[command.index("--raw-dir") + 1] == str(tmp_path / "data/raw")
    assert output == tmp_path / "data/models/evaluation_queue/job-00000007.json"


def test_queue_rejects_arbitrary_racer_stats_parameters(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported"):
        build_command(
            {
                "job_id": 8,
                "task_type": "racer_stats_backfill",
                "parameters": {"from_year": 2016, "command": "arbitrary"},
            },
            app_root=tmp_path,
            python=tmp_path / "python",
            db="postgresql://test",
        )
