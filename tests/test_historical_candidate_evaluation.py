from __future__ import annotations

from datetime import date
import json
import os
from pathlib import Path

import pytest

from boatrace_ai import historical_candidate_evaluation as evaluation
from boatrace_ai.evaluation_queue import (
    TASK_PROFILES,
    build_command,
    seed_default_jobs,
    summarize_result,
)


def test_evaluation_uses_frozen_holdout_and_shared_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    protocol = {
        "as_of_date_jst": "2026-07-23",
        "holdout_start": "2025-07-23",
        "holdout_end": "2026-07-22",
        "calendar_days": 365,
        "training_races": 1234,
        "prediction_races": 4000,
    }
    prediction = {
        "entry_log_loss": 0.41,
        "entry_brier": 0.12,
        "winner_top1_accuracy": 0.58,
        "trifecta_top1_hit_rate": 0.09,
        "trifecta_top5_hit_rate": 0.33,
    }
    daily = [{"race_date": "2026-07-22", "profit_yen": 500}]
    bankroll = {
        "model": "win_model_no_odds_v8",
        "race_days": 365,
        "roi": 1.05,
        "stake_yen": 10_000,
        "return_yen": 10_500,
        "profit_yen": 500,
        "daily": daily,
    }

    def fake_build_protocol(_conn: object, *, as_of_date: date):
        calls["as_of_date"] = as_of_date
        return protocol

    def fake_backtest(_conn: object, **kwargs: object):
        calls["prediction"] = kwargs
        calls["prediction_max_date"] = os.environ.get(
            "BOATRACE_EVAL_MAX_RACE_DATE"
        )
        return prediction

    def fake_bankroll(_conn: object, **kwargs: object):
        calls["bankroll"] = kwargs
        calls["bankroll_max_date"] = os.environ.get(
            "BOATRACE_EVAL_MAX_RACE_DATE"
        )
        return bankroll

    monkeypatch.setattr(
        evaluation.standard_evaluation,
        "build_protocol",
        fake_build_protocol,
    )
    monkeypatch.setattr(
        evaluation.historical_model,
        "backtest_model",
        fake_backtest,
    )
    monkeypatch.setattr(
        evaluation.operational_bankroll,
        "operational_adaptive_bankroll",
        fake_bankroll,
    )
    monkeypatch.setenv("BOATRACE_EVAL_MAX_RACE_DATE", "keep-me")
    output = tmp_path / "job-00000042.json"

    result = evaluation.evaluate_historical_candidate(
        object(),
        output_path=output,
        evaluation_date=date(2026, 7, 22),
    )

    assert calls["as_of_date"] == date(2026, 7, 23)
    assert calls["prediction_max_date"] == "2026-07-22"
    assert calls["bankroll_max_date"] == "2026-07-22"
    assert os.environ["BOATRACE_EVAL_MAX_RACE_DATE"] == "keep-me"

    prediction_args = calls["prediction"]
    bankroll_args = calls["bankroll"]
    assert isinstance(prediction_args, dict)
    assert isinstance(bankroll_args, dict)
    assert prediction_args["folds"] == 1
    assert prediction_args["min_train_races"] == 1234
    assert prediction_args["model_output_path"] == output.with_suffix(".joblib")
    assert bankroll_args["folds"] == 1
    assert bankroll_args["min_train_races"] == 1234
    assert bankroll_args["model_input_path"] == prediction_args["model_output_path"]
    assert bankroll_args["daily_budget_yen"] == 10_000

    assert result["protocol"] == protocol
    assert result["prediction"] == prediction
    assert result["bankroll"] == bankroll
    assert result["metrics"] == {
        "entry_log_loss": 0.41,
        "entry_brier": 0.12,
        "winner_top1_accuracy": 0.58,
        "trifecta_top1_hit_rate": 0.09,
        "trifecta_top5_hit_rate": 0.33,
        "roi": 1.05,
        "profit_yen": 500,
        "stake_yen": 10_000,
        "return_yen": 10_500,
        "evaluation_days": 365,
    }
    assert result["daily"] == daily
    assert result["promotion_eligible"] is False
    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert not output.with_name(f".{output.name}.tmp").exists()
    assert not output.with_name(f".{output.name}.prediction.json").exists()
    assert not output.with_name(f".{output.name}.bankroll.json").exists()


def test_historical_queue_command_and_resource_profile(tmp_path: Path) -> None:
    job = {
        "job_id": 42,
        "task_type": "historical_coverage_safe",
        "model_key": "no_odds_candidate",
        "parameters": {
            "evaluation_date": "2026-07-22",
            "timeout_seconds": 28_800,
        },
    }

    command, output = build_command(
        job,
        app_root=tmp_path,
        python=tmp_path / ".venv/bin/python",
        db="postgresql://test",
    )

    assert command == [
        str(tmp_path / ".venv/bin/python"),
        "-m",
        "boatrace_ai.historical_candidate_evaluation",
        "--db",
        "postgresql://test",
        "--output",
        str(tmp_path / "data/models/evaluation_queue/job-00000042.json"),
        "--evaluation-date",
        "2026-07-22",
    ]
    assert output == tmp_path / "data/models/evaluation_queue/job-00000042.json"
    assert TASK_PROFILES["historical_coverage_safe"] == {
        "category": "evaluation",
        "memory_mb": 16_384,
        "idle_cpu": 15.0,
        "max_parallel": 1,
        "disk_mb": 2_048,
    }


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"evaluation_date": "2026/07/22"}, "does not match format"),
        (
            {"evaluation_date": "2026-07-22", "timeout_seconds": 299},
            "timeout_seconds must be in",
        ),
        (
            {"evaluation_date": "2026-07-22", "command": "arbitrary"},
            "unsupported historical_coverage_safe parameters",
        ),
    ],
)
def test_historical_queue_rejects_invalid_parameters(
    tmp_path: Path,
    parameters: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_command(
            {
                "job_id": 42,
                "task_type": "historical_coverage_safe",
                "model_key": "candidate",
                "parameters": parameters,
            },
            app_root=tmp_path,
            python=tmp_path / "python",
            db="postgresql://test",
        )


def test_historical_metrics_are_summarized_without_promotion() -> None:
    summary = summarize_result(
        {
            "model": "win_model_no_odds_v8",
            "promotion_eligible": False,
            "metrics": {
                "entry_log_loss": 0.41,
                "entry_brier": 0.12,
                "winner_top1_accuracy": 0.58,
                "trifecta_top1_hit_rate": 0.09,
                "trifecta_top5_hit_rate": 0.33,
                "roi": 1.05,
                "profit_yen": 500,
            },
        }
    )

    assert summary["entry_brier"] == 0.12
    assert summary["trifecta_top1_hit_rate"] == 0.09
    assert summary["promotion_eligible"] is False


def test_default_seed_keeps_historical_task_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_enqueue(_conn: object, **kwargs: object) -> int:
        calls.append(kwargs)
        return len(calls)

    monkeypatch.setattr(
        "boatrace_ai.evaluation_queue.enqueue_job",
        fake_enqueue,
    )

    seed_default_jobs(object(), evaluation_date="2026-07-22")

    assert not any(
        call["task_type"] == "historical_coverage_safe" for call in calls
    )
    standardized = next(
        call for call in calls if call["task_type"] == "standardized_365d"
    )
    assert standardized["parameters"] == {
        "evaluation_date": "2026-07-22",
        "timeout_seconds": 28_800,
    }
