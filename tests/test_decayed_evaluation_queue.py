from __future__ import annotations

from pathlib import Path

import pytest

from boatrace_ai.evaluation_queue import build_command


def _job(task_type: str, parameters: dict[str, object]) -> dict[str, object]:
    return {
        "job_id": 9001,
        "status": "running",
        "task_type": task_type,
        "model_key": "decayed-history-experiment",
        "parameters": parameters,
    }


def test_listwise_queue_can_enable_decayed_history(tmp_path: Path) -> None:
    command, _output = build_command(
        _job(
            "listwise_feature_search",
            {
                "evaluation_date": "2026-07-29",
                "feature_variants": "full,drop_rolling_history",
                "include_decayed_history": True,
            },
        ),
        app_root=tmp_path,
        python=tmp_path / "python",
        db="postgresql://test",
    )

    assert "--include-decayed-history" in command
    assert command[command.index("--feature-variants") + 1] == (
        "full,drop_rolling_history"
    )


@pytest.mark.parametrize("value", [1, "true", None])
def test_decayed_history_queue_requires_boolean(value, tmp_path: Path) -> None:
    parameters: dict[str, object] = {
        "evaluation_date": "2026-07-29",
        "include_decayed_history": value,
    }
    if value is None:
        parameters["include_decayed_history"] = None

    with pytest.raises(ValueError, match="must be a boolean"):
        build_command(
            _job("listwise_feature_search", parameters),
            app_root=tmp_path,
            python=tmp_path / "python",
            db="postgresql://test",
        )


def test_combined_search_cannot_enable_listwise_decayed_history(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="supported only"):
        build_command(
            _job(
                "combined_feature_search",
                {
                    "evaluation_date": "2026-07-29",
                    "include_decayed_history": True,
                },
            ),
            app_root=tmp_path,
            python=tmp_path / "python",
            db="postgresql://test",
        )
