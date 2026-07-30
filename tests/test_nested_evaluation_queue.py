from __future__ import annotations

import json
from pathlib import Path

import pytest

from boatrace_ai.evaluation_queue import (
    ObsoleteJob,
    TASK_PROFILES,
    build_command,
    result_decision,
    summarize_result,
)
from boatrace_ai.feature_schema import FEATURE_SCHEMA_VERSION


def _job(parameters: dict) -> dict:
    return {
        "job_id": 77,
        "task_type": "bankroll_policy_nested_annual",
        "model_key": "nested-test",
        "parameters": parameters,
    }


def _source(root: Path, *, job_id: int = 4829, schema: str = FEATURE_SCHEMA_VERSION):
    cache = root / "data/models/evaluation_cache/job-00004829-combined/selected"
    cache.parent.mkdir(parents=True)
    cache.with_suffix(".manifest.json").write_text("{}", encoding="utf-8")
    result = root / f"data/models/evaluation_queue/job-{job_id:08d}.json"
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps({
        "feature_schema_version": schema,
        "selected_cache_prefix": str(cache),
    }), encoding="utf-8")
    return cache


def test_nested_annual_queue_builds_strict_five_fold_command(tmp_path) -> None:
    root = tmp_path / "boat"
    cache = _source(root)

    command, output = build_command(
        _job({"source_job_id": 4829, "embargo_days": 1}),
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )

    assert TASK_PROFILES["bankroll_policy_nested_annual"] == {
        "category": "evaluation",
        "memory_mb": 21504,
        "disk_mb": 4096,
        "idle_cpu": 15.0,
        "max_parallel": 1,
    }
    assert command[:3] == [
        str(root / ".venv/bin/python"),
        "-m",
        "boatrace_ai.listwise.bankroll_policy_nested_evaluation",
    ]
    assert command[command.index("--folds") + 1] == "5"
    assert command[command.index("--selection-days") + 1] == "365"
    assert command[command.index("--outer-days") + 1] == "365"
    assert command[command.index("--embargo-days") + 1] == "1"
    assert command[command.index("--targets") + 1] == "winner,top3_pl"
    assert command[command.index("--cache-prefix") + 1] == str(cache)
    assert command[command.index("--checkpoint-dir") + 1] == str(
        root / "data/models/evaluation_cache/nested_annual/job-00000077"
    )
    assert output == root / "data/models/evaluation_queue/job-00000077.json"


def test_nested_queue_rejects_legacy_source_and_schema(tmp_path) -> None:
    root = tmp_path / "boat"
    with pytest.raises(ObsoleteJob, match="legacy job 3995"):
        build_command(
            _job({"source_job_id": 3995}),
            app_root=root,
            python=root / ".venv/bin/python",
            db="postgresql://test",
        )

    _source(root, schema="pastlog-listwise-hashed-v4")
    with pytest.raises(ObsoleteJob, match="feature schema is obsolete"):
        build_command(
            _job({"source_job_id": 4829}),
            app_root=root,
            python=root / ".venv/bin/python",
            db="postgresql://test",
        )


def test_nested_result_summary_preserves_fold_and_confidence_metrics() -> None:
    payload = {
        "model": "bankroll_policy_nested_annual_v1",
        "promotion_eligible": False,
        "evaluation": {
            "aggregate": {
                "fold_count": 5,
                "roi": 1.04,
                "profit_yen": 4000,
                "minimum_fold_roi": 0.82,
                "profitable_folds": 4,
                "largest_hit_excluded_roi": 0.97,
                "folds": [{"roi": value} for value in (1.1, 1.2, 0.82, 1.1, 1.0)],
                "bootstrap": {
                    "roi_ci95_lower": 0.91,
                    "roi_ci95_upper": 1.18,
                    "probability_roi_above_one": 0.72,
                },
            },
            "promotion_gate": {
                "five_outer_folds_completed": True,
                "all_fold_roi_above_one": False,
            },
        },
    }

    summary = summarize_result(payload)

    assert summary["roi"] == 1.04
    assert summary["fold_count"] == 5
    assert summary["fold_rois"] == [1.1, 1.2, 0.82, 1.1, 1.0]
    assert summary["roi_ci95_lower"] == 0.91
    assert summary["promotion_gate_passed"] == 1
    assert summary["promotion_gate_failed"] == ["all_fold_roi_above_one"]
    assert result_decision("bankroll_policy_nested_annual", summary) == (
        "nested_gate_failed"
    )
