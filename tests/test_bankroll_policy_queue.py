from __future__ import annotations

import json

import pytest

from boatrace_ai.evaluation_queue import TASK_PROFILES, build_command
from boatrace_ai.listwise.bankroll_policy_evaluation import (
    build_parser,
    prior_selection_key,
)


def _job(parameters):
    return {
        "job_id": 99,
        "task_type": "bankroll_policy_search",
        "model_key": "policy",
        "parameters": parameters,
    }


def _source(root, *, cache_prefix=None):
    result = root / "data/models/evaluation_queue/job-00003565.json"
    result.parent.mkdir(parents=True)
    cache = cache_prefix or (
        root
        / "data/models/evaluation_cache/job-00003565"
        / "listwise_search_8192_drop_research_correlates"
    )
    result.write_text(
        json.dumps({"selected_cache_prefix": str(cache)}),
        encoding="utf-8",
    )
    return result, cache


def test_bankroll_policy_search_profile_and_command(tmp_path) -> None:
    root = tmp_path / "boat"
    source, cache = _source(root)
    python = root / ".venv/bin/python"
    command, output = build_command(
        _job({
            "source_job_id": 3565,
            "learning_rate": 0.0075,
            "epochs": 3,
            "candidate_count": 24,
            "finalists": 6,
            "bootstrap_samples": 20000,
            "payout_prior_weights": "10,30,100",
            "timeout_seconds": 43200,
            "research_only": True,
        }),
        app_root=root,
        python=python,
        db="postgresql://test",
    )
    assert TASK_PROFILES["bankroll_policy_search"] == {
        "category": "evaluation",
        "memory_mb": 9216,
        "idle_cpu": 15.0,
        "max_parallel": 1,
        "disk_mb": 1024,
    }
    assert output == root / "data/models/evaluation_queue/job-00000099.json"
    assert command[:3] == [
        str(python),
        "-m",
        "boatrace_ai.listwise.bankroll_policy_evaluation",
    ]
    assert command[command.index("--search-result") + 1] == str(source)
    assert command[command.index("--cache-prefix") + 1] == str(cache)
    assert command[command.index("--candidate-count") + 1] == "24"
    assert command[command.index("--payout-prior-weights") + 1] == "10,30,100"
    assert command[command.index("--evaluation-days") + 1] == "365"
    assert command[command.index("--research-only") + 1] == "true"


def test_bankroll_policy_evaluation_defaults_to_standard_365_days() -> None:
    parser = build_parser()
    args = parser.parse_args(["--db", "x", "--search-result", "x", "--cache-prefix", "x", "--output", "x"])
    assert args.evaluation_days == 365


def test_prior_selection_prefers_temporal_stability_before_bootstrap_ci() -> None:
    def row(prior, *, minimum_roi, stable_score, ci_lower):
        return {
            "payout_prior_weight": prior,
            "selected": {
                "temporal_stability": {
                    "all_minimum_evidence": True,
                    "minimum_roi": minimum_roi,
                    "mean_roi_minus_std": stable_score,
                },
                "confidence": {
                    "roi_ci95_lower": ci_lower,
                    "probability_roi_above_one": 0.55,
                },
                "metrics": {"profit_yen": 1000, "hit_tickets": 80},
            },
        }

    unstable = row(
        5,
        minimum_roi=0.991,
        stable_score=0.989,
        ci_lower=0.803,
    )
    stable = row(
        100,
        minimum_roi=1.024,
        stable_score=1.024,
        ci_lower=0.795,
    )

    assert max([unstable, stable], key=prior_selection_key) is stable


@pytest.mark.parametrize(
    "parameters,message",
    [
        ({"source_job_id": 3565, "candidate_count": 7}, "candidate_count"),
        (
            {"source_job_id": 3565, "candidate_count": 8, "finalists": 9},
            "finalists",
        ),
        (
            {"source_job_id": 3565, "payout_prior_weights": "0,30"},
            "payout_prior_weights",
        ),
        ({"source_job_id": 3565, "command": "sh"}, "unsupported"),
    ],
)
def test_bankroll_policy_search_rejects_invalid_parameters(
    tmp_path, parameters, message
) -> None:
    root = tmp_path / "boat"
    _source(root)
    with pytest.raises(ValueError, match=message):
        build_command(
            _job(parameters),
            app_root=root,
            python=root / ".venv/bin/python",
            db="postgresql://test",
        )
