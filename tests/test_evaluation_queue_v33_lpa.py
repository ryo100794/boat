from __future__ import annotations

from pathlib import Path

import pytest

from boatrace_ai.evaluation_queue import TASK_PROFILES, build_command


def _job(parameters: dict, *, task_type: str = "learned_purchase_allocation_v33"):
    return {
        "job_id": 12001,
        "status": "running",
        "task_type": task_type,
        "model_key": "learned-purchase-allocation-v33",
        "parameters": parameters,
    }


def _parameters():
    return {
        "source_model": "data/models/evaluation_queue/job-00002707.joblib",
        "training_from": "2026-07-20",
        "training_through": "2026-07-30",
        "outer_from": "2026-07-31",
        "outer_through": "2026-08-01",
        "projection_dimensions": 8,
        "base_training_fraction": 0.6,
        "minimum_base_training_dates": 5,
        "minimum_lpa_teacher_dates": 4,
        "allocation_validation_fraction": 0.25,
        "allocation_max_iterations": 200,
        "bootstrap_samples": 2_000,
    }


def test_v33_lpa_command_is_fixed_and_saves_deployment_artifact(
    tmp_path: Path,
) -> None:
    root = tmp_path / "boat"
    command, output = build_command(
        _job(_parameters()),
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )

    assert command[1:3] == [
        "-m",
        "boatrace_ai.listwise.learned_purchase_allocation_v33_evaluation",
    ]
    assert command[command.index("--training-through") + 1] == "2026-07-30"
    assert command[command.index("--outer-from") + 1] == "2026-07-31"
    assert command[command.index("--base-training-fraction") + 1] == "0.6"
    assert command[command.index("--bootstrap-samples") + 1] == "2000"
    assert command[command.index("--model-output") + 1] == str(
        root / "data/models/evaluation_queue/job-00012001.joblib"
    )
    assert command[command.index("--data-cache") + 1].startswith(
        str(root / "data/models/evaluation_cache/four_head_v22")
    )
    assert output == root / "data/models/evaluation_queue/job-00012001.json"
    assert "learned_purchase_allocation_v33" in TASK_PROFILES


def test_v33_lpa_reuses_the_exact_v22_input_cache_identity(tmp_path: Path) -> None:
    root = tmp_path / "boat"
    v33, _ = build_command(
        _job(_parameters()),
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )
    v22_parameters = {
        key: value
        for key, value in _parameters().items()
        if key
        in {
            "source_model",
            "training_from",
            "training_through",
            "outer_from",
            "outer_through",
            "projection_dimensions",
        }
    }
    v22_parameters.update(
        purchase_teacher_version=20,
        purchase_loss="multinomial_market_offset_oof_scaled_payout_stacked_tweedie",
    )
    v22, _ = build_command(
        _job(v22_parameters, task_type="four_head_learned_value"),
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )

    assert v33[v33.index("--data-cache") + 1] == v22[v22.index("--data-cache") + 1]


@pytest.mark.parametrize(
    "change,match",
    [
        ({"training_through": "2026-08-01"}, "outer period"),
        ({"minimum_lpa_teacher_dates": 3}, "minimum_lpa_teacher_dates"),
        ({"allocation_validation_fraction": 0.9}, "allocation_validation_fraction"),
        ({"arbitrary_command": "curl"}, "unsupported"),
    ],
)
def test_v33_lpa_rejects_unsafe_parameters(
    tmp_path: Path, change: dict, match: str
) -> None:
    parameters = {**_parameters(), **change}
    with pytest.raises(ValueError, match=match):
        build_command(
            _job(parameters),
            app_root=tmp_path / "boat",
            python=tmp_path / "boat/.venv/bin/python",
            db="postgresql://test",
        )
