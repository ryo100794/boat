from __future__ import annotations

from pathlib import Path

import joblib
import pytest

from boatrace_ai.evaluation_queue import (
    build_command,
    result_decision,
    summarize_result,
)
from boatrace_ai.joint_scenario_evaluation import (
    run_joint_scenario_evaluation,
)


def _race(day: int, index: int) -> dict:
    actual = "A" if (day + index) % 2 else "B"
    return {
        "race_id": f"202607{day:02d}01{index:02d}",
        "race_date": f"2026-07-{day:02d}",
        "jcd": "01",
        "actual_combination": actual,
        "model_probabilities": {"A": 0.6, "B": 0.4},
        "market_probabilities": {"A": 0.55, "B": 0.45},
        "official_closing_odds": {"A": 1.8, "B": 2.4},
    }


def test_joint_scenario_evaluation_reports_exclusions_and_deltas(
    tmp_path: Path,
) -> None:
    races = [_race(day, index) for day in range(1, 8) for index in range(1, 3)]
    races.append({**_race(7, 3), "official_closing_odds": None})
    races[-1].pop("official_closing_odds")
    cache = tmp_path / "races.joblib"
    joblib.dump({"races": races}, cache)

    result = run_joint_scenario_evaluation(
        cache,
        terminal_min_training_days=2,
        joint_min_training_days=2,
        scenarios_per_race=8,
        rank=2,
        pooling_strength=4.0,
        seed=7,
        expected_outcomes=("A", "B"),
        learn_residual_scales=True,
    )

    assert result["coverage"]["excluded_races"] == 1
    assert result["metrics"]["evaluated_races"] == 6
    assert "generated_log_loss_delta_vs_decision_model" in result["metrics"]
    assert result["promotion_eligible"] is False


def test_joint_scenario_queue_command_is_path_restricted(tmp_path: Path) -> None:
    root = tmp_path / "boat"
    cache = root / "data/models/evaluation_cache/market_scored/races.joblib"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"cache")
    job = {
        "job_id": 19,
        "task_type": "joint_scenario_walk_forward",
        "parameters": {
            "scored_cache": str(cache.relative_to(root)),
            "terminal_min_training_days": 5,
            "joint_min_training_days": 3,
            "scenarios_per_race": 32,
            "rank": 8,
            "learn_residual_scales": True,
        },
    }
    command, output = build_command(
        job,
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )

    assert command[1:3] == ["-m", "boatrace_ai.joint_scenario_evaluation"]
    assert command[command.index("--scenarios-per-race") + 1] == "32"
    assert "--learn-residual-scales" in command
    assert output.name == "job-00000019.json"
    assert result_decision("joint_scenario_walk_forward", {}) == (
        "diagnostic_complete_not_policy_connected"
    )

    job["parameters"]["scored_cache"] = "data/models/outside.joblib"
    with pytest.raises(ValueError, match="inside market_scored"):
        build_command(
            job,
            app_root=root,
            python=root / ".venv/bin/python",
            db="postgresql://test",
        )


def test_joint_bankroll_queue_command_is_distinct_and_restricted(
    tmp_path: Path,
) -> None:
    root = tmp_path / "boat"
    cache = root / "data/models/evaluation_cache/market_scored/races.joblib"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"cache")
    job = {
        "job_id": 20,
        "task_type": "joint_bankroll_walk_forward",
        "parameters": {
            "scored_cache": str(cache.relative_to(root)),
            "outer_draws": 20,
            "scenarios_per_draw": 64,
            "initial_daily_bankroll_yen": 10_000,
            "population_size": 8,
            "generations": 3,
        },
    }

    command, output = build_command(
        job,
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )

    assert command[1:3] == ["-m", "boatrace_ai.joint_bankroll_evaluation"]
    assert command[command.index("--outer-draws") + 1] == "20"
    assert command[command.index("--initial-daily-bankroll-yen") + 1] == "10000"
    assert output.name == "job-00000020.json"
    assert result_decision("joint_bankroll_walk_forward", {}) == (
        "accumulate_sealed_bankroll_evidence"
    )


def test_joint_bankroll_result_uses_unified_probability_and_bankroll_metrics() -> None:
    summary = summarize_result({
        "model": "joint_bankroll_strict_walk_forward_v1",
        "status": "provisional_accumulate_sealed_days",
        "promotion_eligible": False,
        "evaluation_from": "2026-07-28",
        "evaluation_through": "2026-08-01",
        "evaluation_days": 5,
        "evaluated_races": 713,
        "probability_metrics": {
            "generated_winner_log_loss": 0.7,
            "generated_winner_top1_accuracy": 0.55,
            "generated_log_loss": 3.8,
            "generated_top5": 0.38,
        },
        "primary_bankroll": {
            "stake_yen": 20_000,
            "return_yen": 21_000,
            "profit_yen": 1_000,
            "roi": 1.05,
            "max_drawdown_yen": 3_000,
            "daily_cluster_bootstrap_roi_lower_95": 0.98,
        },
        "promotion_gate": {
            "minimum_30_complete_days": False,
            "positive_profit": True,
        },
        "daily": [{
            "races": [
                {"stake_yen": 10_000, "return_yen": 12_000},
                {"stake_yen": 10_000, "return_yen": 9_000},
            ],
        }],
    })

    assert summary["winner_log_loss"] == 0.7
    assert summary["winner_top1_accuracy"] == 0.55
    assert summary["model_trifecta_log_loss"] == 3.8
    assert summary["trifecta_top5_hit_rate"] == 0.38
    assert summary["roi"] == 1.05
    assert summary["daily_cluster_bootstrap_roi_lower_95"] == 0.98
    assert summary["largest_hit_return_share"] == pytest.approx(12_000 / 21_000)
    assert summary["roi_without_largest_hit"] == 0.45
    assert summary["promotion_gate_passed"] == 1
    assert summary["promotion_gate_total"] == 2
