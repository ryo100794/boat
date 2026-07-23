from pathlib import Path

from boatrace_ai.evaluation_queue import TASK_PROFILES, build_command


def test_feature_search_does_not_persist_each_large_variant_cache(tmp_path: Path) -> None:
    command, _output = build_command(
        {
            "job_id": 8,
            "task_type": "listwise_feature_search",
            "model_key": "feature-search",
            "parameters": {},
        },
        app_root=tmp_path,
        python=tmp_path / ".venv/bin/python",
        db="postgresql://test",
    )

    assert command[command.index("--cache-write-mode") + 1] == "never"
    assert "--selected-cache-dir" in command
    selected_cache = Path(command[command.index("--selected-cache-dir") + 1])
    assert selected_cache == (
        tmp_path / "data/models/evaluation_cache/job-00000008"
    )
    assert TASK_PROFILES["listwise_feature_search"]["memory_mb"] == 12288
    assert TASK_PROFILES["listwise_feature_search"]["max_parallel"] == 1
