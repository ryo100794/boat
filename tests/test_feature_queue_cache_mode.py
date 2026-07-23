from pathlib import Path

from boatrace_ai.evaluation_queue import TASK_PROFILES, build_command


def test_feature_search_does_not_persist_each_large_variant_cache(tmp_path: Path) -> None:
    command, _output = build_command(
        {
            "job_id": 8,
            "task_type": "listwise_feature_search",
            "model_key": "feature-search",
            "parameters": {"evaluation_date": "2026-07-22"},
        },
        app_root=tmp_path,
        python=tmp_path / ".venv/bin/python",
        db="postgresql://test",
    )

    assert command[command.index("--cache-write-mode") + 1] == "never"
    assert command[command.index("--as-of-date") + 1] == "2026-07-22"
    assert command[command.index("--variant-workers") + 1] == "1"
    assert command[command.index("--candidate-workers") + 1] == "2"
    assert "--selected-cache-dir" in command
    selected_cache = Path(command[command.index("--selected-cache-dir") + 1])
    assert selected_cache == (
        tmp_path / "data/models/evaluation_cache/job-00000008"
    )
    assert TASK_PROFILES["listwise_feature_search"]["memory_mb"] == 14336
    assert TASK_PROFILES["listwise_feature_search"]["max_parallel"] == 1


def test_standard_evaluation_uses_one_variant_and_two_candidate_workers() -> None:
    script = Path("scripts/run_standardized_365d_evaluations.sh").read_text(
        encoding="utf-8"
    )
    assert "--variant-workers 1 --candidate-workers 2" in script
