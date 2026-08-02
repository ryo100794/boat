from __future__ import annotations

from pathlib import Path

import pytest

from boatrace_ai.evaluation_queue import build_command


@pytest.mark.parametrize(
    ("purchase_loss", "teacher_version"),
    (
        ("multinomial_market_offset_oof_scaled_payout_closing", 16),
        ("multinomial_market_offset_oof_scaled_payout_tweedie", 17),
        ("multinomial_market_offset_oof_scaled_payout_factor_tweedie", 18),
        (
            "multinomial_market_offset_oof_scaled_payout_context_factor_tweedie",
            19,
        ),
    ),
)
def test_payout_models_require_matching_teacher_version(
    tmp_path: Path, purchase_loss: str, teacher_version: int
) -> None:
    root = tmp_path / "boat"
    parameters = {
        "source_model": "data/models/source.joblib",
        "training_from": "2026-07-20",
        "training_through": "2026-07-30",
        "outer_from": "2026-07-31",
        "outer_through": "2026-08-01",
        "purchase_loss": purchase_loss,
        "purchase_teacher_version": teacher_version,
    }
    job = {
        "job_id": teacher_version,
        "status": "running",
        "task_type": "four_head_learned_value",
        "model_key": f"payout-v{teacher_version}",
        "parameters": parameters,
    }

    command, _output = build_command(
        job,
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )

    assert command[command.index("--purchase-loss") + 1] == purchase_loss
    parameters["purchase_teacher_version"] = teacher_version - 1
    with pytest.raises(ValueError, match="does not match"):
        build_command(
            job,
            app_root=root,
            python=root / ".venv/bin/python",
            db="postgresql://test",
        )
