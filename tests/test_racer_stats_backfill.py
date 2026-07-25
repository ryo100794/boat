from __future__ import annotations

from pathlib import Path

import pytest

from boatrace_ai import racer_stats_backfill
from boatrace_ai.evaluation_queue import TASK_PROFILES, build_command
from boatrace_ai.ingestion.parsers import parse_racer_stats_bytes
from boatrace_ai.official import racer_stats_url


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


def test_official_racer_stats_urls_match_download_page() -> None:
    assert racer_stats_url(2026, 1).endswith("/kibetsu/fan2510.lzh")
    assert racer_stats_url(2026, 2).endswith("/kibetsu/fan2604.lzh")
    assert racer_stats_url(2016, 1).endswith("/kibetsu/fan1510.lzh")
    assert racer_stats_url(2016, 2).endswith("/kibetsu/fan1604.lzh")


def test_racer_stats_parser_reads_avg_st_and_trailing_origin() -> None:
    line = bytearray(b" " * 416)
    line[0:4] = b"3415"
    line[4:20] = b"RACER           "
    line[35:39] = "大阪".encode("cp932")
    line[39:41] = b"A1"
    line[49:51] = b"44"
    line[54:56] = b"50"
    line[58:62] = b"0756"
    line[62:66] = b"0459"
    line[72:75] = b"122"
    line[79:82] = b"016"
    line[410:416] = "大阪  ".encode("cp932")

    rows = parse_racer_stats_bytes(bytes(line), year=2026, half=1)
    assert rows[0]["avg_st"] == 0.16
    assert rows[0]["origin"] == "大阪"
