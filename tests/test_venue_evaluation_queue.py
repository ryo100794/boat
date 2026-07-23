from __future__ import annotations

from pathlib import Path

import pytest

from boatrace_ai.evaluation_queue import TASK_PROFILES, build_command


def test_venue_conditional_order_uses_fixed_model_and_module(tmp_path: Path) -> None:
    root = tmp_path / "boat"
    command, output = build_command(
        {
            "job_id": 42,
            "task_type": "venue_conditional_order",
            "model_key": "venue-context",
            "parameters": {
                "training_through": "2025-07-22",
                "evaluation_from": "2025-07-23",
                "evaluation_through": "2026-07-22",
            },
        },
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )

    assert command[1:3] == ["-m", "boatrace_ai.listwise.venue_conditional_order"]
    baseline_index = command.index("--baseline-model") + 1
    assert command[baseline_index] == str(
        root / "data/models/standardized_365d_v2/listwise_newton.joblib"
    )
    assert "--cache-prefix" not in command
    assert command[command.index("--cache-dir") + 1].startswith(
        "/tmp/boatrace-evaluation/job-00000042"
    )
    assert output == root / "data/models/evaluation_queue/job-00000042.json"
    assert TASK_PROFILES["venue_conditional_order"]["max_parallel"] == 1


def test_venue_conditional_order_rejects_overlapping_dates(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="chronological"):
        build_command(
            {
                "job_id": 42,
                "task_type": "venue_conditional_order",
                "model_key": "venue-context",
                "parameters": {
                    "training_through": "2025-07-23",
                    "evaluation_from": "2025-07-23",
                    "evaluation_through": "2026-07-22",
                },
            },
            app_root=tmp_path,
            python=tmp_path / "python",
            db="postgresql://test",
        )
