from __future__ import annotations

from pathlib import Path

from boatrace_ai.evaluation_queue import build_command
from boatrace_ai.listwise.feature_search import parse_feature_variants


VARIANT = "drop_base_pastlog_rolling_history"


def test_history_only_variant_drops_static_and_legacy_rolling_features() -> None:
    parsed = parse_feature_variants(VARIANT)

    assert parsed == (
        (VARIANT, ("base_pastlog", "rolling_history")),
    )


def test_queue_accepts_decayed_history_only_comparison(tmp_path: Path) -> None:
    job = {
        "job_id": 9200,
        "status": "running",
        "task_type": "listwise_feature_search",
        "model_key": "history-centered-decayed-ab",
        "parameters": {
            "evaluation_date": "2026-07-29",
            "targets": "winner",
            "alphas": "0.00001",
            "feature_variants": (
                f"drop_base_pastlog,{VARIANT}"
            ),
            "include_decayed_history": True,
        },
    }

    command, _output = build_command(
        job,
        app_root=tmp_path,
        python=tmp_path / "python",
        db="postgresql://test",
    )

    assert "--include-decayed-history" in command
    index = command.index("--feature-variants")
    assert command[index + 1] == f"drop_base_pastlog,{VARIANT}"
