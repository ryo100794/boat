from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from boatrace_ai.evaluation_queue import (
    JobDependencyUnavailable,
    ObsoleteJob,
    TASK_PROFILES,
    build_command,
    result_decision,
)


def _job(root: Path, **overrides: object) -> tuple[dict, Path]:
    source_job_id = 2707
    artifact = (
        root / "data/models/evaluation_queue"
        / f"job-{source_job_id:08d}.deployment.joblib"
    )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"reproducible-v22-source-artifact")
    parameters: dict[str, object] = {
        "source_job_id": source_job_id,
        "source_model_artifact": str(artifact.relative_to(root)),
        "source_model_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "training_from_date": "2024-01-01",
        "training_through_date": "2025-06-30",
        "outer_from_date": "2025-07-01",
        "outer_through_date": "2026-06-30",
        "max_snapshot_age_seconds": 65,
        "timeout_seconds": 86400,
    }
    parameters.update(overrides)
    return {
        "job_id": 91,
        "task_type": "four_head_v22_actual_annual",
        "model_key": "four-head-v22-actual-annual",
        "parameters": parameters,
    }, artifact


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_v22_annual_builds_existing_actual_data_cli_with_audit_inputs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "boat"
    job, artifact = _job(root)
    command, output = build_command(
        job, app_root=root, python=root / ".venv/bin/python", db="postgresql://test"
    )

    assert TASK_PROFILES["four_head_v22_actual_annual"] == {
        "category": "evaluation",
        "memory_mb": 21504,
        "disk_mb": 4096,
        "idle_cpu": 15.0,
        "max_parallel": 1,
    }
    assert command[:3] == [
        str(root / ".venv/bin/python"),
        "-m",
        "boatrace_ai.listwise.four_head_v22_evaluation",
    ]
    assert _option(command, "--source-model") == str(artifact.resolve())
    assert _option(command, "--training-from") == "2024-01-01"
    assert _option(command, "--training-through") == "2025-06-30"
    assert _option(command, "--outer-from") == "2025-07-01"
    assert _option(command, "--outer-through") == "2026-06-30"
    assert _option(command, "--max-snapshot-age-seconds") == "65.0"
    assert _option(command, "--output") == str(
        root / "data/models/evaluation_queue/job-00000091.json"
    )
    assert output == root / "data/models/evaluation_queue/job-00000091.json"
    assert not any("bankroll" in part for part in command)
    assert result_decision("four_head_v22_actual_annual", {}) == (
        "diagnostic_evaluation_complete"
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"outer_through_date": "2026-06-29"}, "exactly 365 calendar days"),
        ({"training_through_date": "2025-07-01"}, "strictly after training"),
        ({"max_snapshot_age_seconds": 301}, "must be in"),
        ({"timeout_seconds": 299}, "must be in"),
        ({"unexpected": True}, "unsupported"),
    ],
)
def test_v22_annual_rejects_non_reproducible_protocols(
    tmp_path: Path, overrides: dict[str, object], message: str
) -> None:
    root = tmp_path / "boat"
    job, _artifact = _job(root, **overrides)
    with pytest.raises(ValueError, match=message):
        build_command(
            job, app_root=root, python=root / ".venv/bin/python", db="postgresql://test"
        )


def test_v22_annual_rejects_artifact_identity_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "boat"
    job, artifact = _job(root)
    job["parameters"]["source_model_sha256"] = "0" * 64
    with pytest.raises(ObsoleteJob, match="SHA-256"):
        build_command(
            job, app_root=root, python=root / ".venv/bin/python", db="postgresql://test"
        )
    artifact.unlink()
    with pytest.raises(JobDependencyUnavailable, match="not available"):
        build_command(
            job, app_root=root, python=root / ".venv/bin/python", db="postgresql://test"
        )


def test_v22_annual_rejects_model_not_bound_to_source_job(tmp_path: Path) -> None:
    root = tmp_path / "boat"
    job, artifact = _job(root)
    unrelated = artifact.with_name("job-000027070.deployment.joblib")
    artifact.rename(unrelated)
    job["parameters"]["source_model_artifact"] = str(unrelated.relative_to(root))
    with pytest.raises(ValueError, match="source_job_id"):
        build_command(
            job, app_root=root, python=root / ".venv/bin/python", db="postgresql://test"
        )
