from __future__ import annotations

from datetime import date
import json
import os
from pathlib import Path

import pytest

from boatrace_ai import historical_model
from boatrace_ai import historical_research_evaluation as evaluation
from boatrace_ai import operational_bankroll
from boatrace_ai.evaluation_queue import TASK_PROFILES, build_command
from boatrace_ai.standard_evaluation import race_set_sha256


def _protocol() -> dict[str, object]:
    holdout = {"h0", "h1"}
    return {
        "training_races": 2,
        "prediction_races": 2,
        "holdout_start": "2025-07-24",
        "holdout_end": "2026-07-23",
        "race_set_sha256": race_set_sha256(holdout),
    }


def test_research_queue_command_and_resource_profile(tmp_path: Path) -> None:
    command, output = build_command(
        {
            "job_id": 88,
            "task_type": "historical_research_logit",
            "model_key": "no_odds_v9_research_logit",
            "parameters": {
                "evaluation_date": "2026-07-23",
                "timeout_seconds": 86400,
            },
        },
        app_root=tmp_path,
        python=tmp_path / ".venv/bin/python",
        db="postgresql://test",
    )

    assert command == [
        str(tmp_path / ".venv/bin/python"),
        "-m",
        "boatrace_ai.historical_research_evaluation",
        "--db",
        "postgresql://test",
        "--evaluation-date",
        "2026-07-23",
        "--model-dir",
        str(tmp_path / "data/models"),
        "--output",
        str(tmp_path / "data/models/evaluation_queue/job-00000088.json"),
    ]
    assert output == tmp_path / "data/models/evaluation_queue/job-00000088.json"
    assert TASK_PROFILES["historical_research_logit"] == {
        "category": "evaluation",
        "memory_mb": 14336,
        "idle_cpu": 15.0,
        "max_parallel": 1,
        "disk_mb": 4096,
    }


def test_research_iterator_explicitly_enables_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_examples(_conn, **kwargs):
        captured.update(kwargs)
        yield {"research_home_branch": 1}, 1, {
            "race_id": "h0", "lane": 1, "rank": 1,
        }

    monkeypatch.setattr(historical_model, "iter_training_examples", fake_examples)
    monkeypatch.setattr(
        historical_model.base,
        "positive_probs",
        lambda _pipeline, batch: [0.5 for _ in batch],
    )

    rows = list(historical_model.iter_scored_entries(
        object(),
        pipeline=object(),
        include_races={"h0"},
        include_research=True,
    ))

    assert len(rows) == 1
    assert captured["include_research"] is True
    assert captured["include_beforeinfo"] is False


def test_pretrained_research_bundle_requires_research_contract() -> None:
    training = {"t0", "t1"}
    bundle = {
        "pipeline": object(),
        "metadata": {
            "feature_set": historical_model.RESEARCH_FEATURE_SET,
            "train_races": 2,
            "train_race_set_sha256": race_set_sha256(training),
        },
    }

    operational_bankroll._validate_pretrained_bundle(
        bundle,
        train_races=training,
        expected_feature_set=historical_model.RESEARCH_FEATURE_SET,
    )
    with pytest.raises(ValueError, match="feature set mismatch"):
        operational_bankroll._validate_pretrained_bundle(
            bundle,
            train_races=training,
        )


def test_research_evaluation_uses_frozen_protocol_and_baseline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol = _protocol()
    model_dir = tmp_path / "models"
    raw = model_dir / "standardized_365d_v2/raw"
    raw.mkdir(parents=True)
    expected_hash = str(protocol["race_set_sha256"])
    baseline_prediction = {
        "entry_log_loss": 0.40,
        "winner_top1_accuracy": 0.56,
        "trifecta_top5_hit_rate": 0.31,
    }
    baseline_bankroll = {"roi": 0.80, "profit_yen": -100}
    (raw / "no_odds_v8_prediction.json").write_text(
        json.dumps(baseline_prediction), encoding="utf-8"
    )
    (raw / "no_odds_v8_bankroll.json").write_text(
        json.dumps(baseline_bankroll), encoding="utf-8"
    )
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        evaluation.standard_evaluation,
        "build_protocol",
        lambda _conn, **_kwargs: protocol,
    )

    def fake_prediction(_conn, **kwargs):
        calls["prediction"] = kwargs
        return {
            "evaluation_race_set_sha256": expected_hash,
            "evaluated_races": 2,
            "entry_log_loss": 0.39,
            "entry_brier": 0.10,
            "winner_top1_accuracy": 0.57,
            "trifecta_top1_hit_rate": 0.10,
            "trifecta_top5_hit_rate": 0.32,
        }

    def fake_bankroll(_conn, **kwargs):
        calls["bankroll"] = kwargs
        return {
            "evaluation_race_set_sha256": expected_hash,
            "evaluated_races": 2,
            "roi": 1.05,
            "profit_yen": 50,
            "stake_yen": 1000,
            "return_yen": 1050,
            "daily": [{"race_date": "2026-07-23", "profit_yen": 50}],
        }

    monkeypatch.setattr(
        evaluation.historical_model, "backtest_model", fake_prediction
    )
    monkeypatch.setattr(
        evaluation.operational_bankroll,
        "operational_adaptive_bankroll",
        fake_bankroll,
    )
    output = tmp_path / "candidate.json"
    previous = os.environ.get("BOATRACE_EVAL_MAX_RACE_DATE")

    result = evaluation.evaluate(
        object(),
        output_path=output,
        evaluation_date=date(2026, 7, 23),
        model_dir=model_dir,
    )

    assert calls["prediction"]["include_research"] is True
    assert calls["prediction"]["min_train_races"] == 2
    assert calls["bankroll"]["include_research"] is True
    assert result["promotion_eligible"] is True
    assert result["comparison"]["checks"] == {
        "comparison_ready": True,
        "race_set_matches_protocol": True,
        "entry_log_loss_not_worse": True,
        "winner_top1_not_worse": True,
        "trifecta_top5_not_worse": True,
        "roi_at_least_one": True,
        "profit_positive": True,
    }
    assert json.loads(output.read_text(encoding="utf-8"))["metrics"]["roi"] == 1.05
    assert os.environ.get("BOATRACE_EVAL_MAX_RACE_DATE") == previous
