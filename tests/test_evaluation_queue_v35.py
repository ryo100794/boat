from pathlib import Path

from boatrace_ai.evaluation_queue import build_command


def test_market_residual_walk_forward_accepts_v35(tmp_path: Path) -> None:
    root = tmp_path / "boat"
    model = root / "data/models/evaluation_queue/job-00002707.joblib"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"artifact")
    job = {
        "job_id": 1,
        "task_type": "market_residual_walk_forward",
        "parameters": {
            "model_input": "data/models/evaluation_queue/job-00002707.joblib",
            "from_date": "2026-07-18",
            "through_date": "2026-07-31",
            "calibrator_strategy": (
                "odds_path_observed_closing_return_stable_policy_triple_head_v35"
            ),
        },
    }

    command, _ = build_command(
        job,
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )

    assert command[command.index("--calibrator-strategy") + 1] == (
        "odds_path_observed_closing_return_stable_policy_triple_head_v35"
    )
