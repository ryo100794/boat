from pathlib import Path

from boatrace_ai.evaluation_queue import build_command


def test_v34_odds_path_uses_distinct_cache_and_required_schema(tmp_path: Path) -> None:
    root = tmp_path / "boat"
    job = {
        "job_id": 12034,
        "status": "running",
        "task_type": "learned_purchase_allocation_v33",
        "model_key": "learned-purchase-allocation-odds-path-v34",
        "parameters": {
            "source_model": "data/models/evaluation_queue/job-00002707.joblib",
            "training_from": "2026-07-20",
            "training_through": "2026-07-30",
            "outer_from": "2026-07-31",
            "outer_through": "2026-08-01",
            "odds_path_schema": "t5_odds_path_v1",
        },
    }

    command, _output = build_command(
        job,
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )

    assert command[command.index("--odds-path-schema") + 1] == "t5_odds_path_v1"
    assert "/evaluation_cache/learned_allocation_v34/" in command[
        command.index("--data-cache") + 1
    ]
