from __future__ import annotations

from datetime import date
import json
import os
from pathlib import Path

import joblib
import pytest

from boatrace_ai import historical_candidate_evaluation as evaluation
from boatrace_ai import historical_model
from boatrace_ai.evaluation_queue import (
    TASK_PROFILES,
    build_command,
    seed_default_jobs,
    summarize_result,
)
from boatrace_ai.standard_evaluation import race_set_sha256


def _protocol(training: int = 2, holdout: int = 2) -> dict[str, object]:
    holdout_ids = {f"h{index}" for index in range(holdout)}
    return {
        "as_of_date_jst": "2026-07-23",
        "holdout_start": "2025-07-23",
        "holdout_end": "2026-07-22",
        "calendar_days": 365,
        "holdout_date_count": 365,
        "training_races": training,
        "prediction_races": holdout,
        "race_set_sha256": race_set_sha256(holdout_ids),
    }


def _source_bundle(path: Path, training_races: set[str]) -> None:
    joblib.dump(
        {
            "pipeline": {"fitted": True},
            "metadata": {
                "feature_set": evaluation.LEGACY_FEATURE_SET,
                "include_odds": False,
                "train_races": len(training_races),
                "train_race_set_sha256": race_set_sha256(training_races),
            },
        },
        path,
    )


def test_protocol_race_sets_use_prefix_and_validate_holdout_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _protocol()
    monkeypatch.setattr(
        evaluation,
        "load_complete_race_ids",
        lambda _conn: [
            ("t0", "2025-07-21", "01", 1),
            ("t1", "2025-07-22", "01", 2),
            ("h0", "2025-07-23", "01", 1),
            ("h1", "2026-07-22", "24", 12),
        ],
    )

    training, holdout = evaluation._protocol_race_sets(object(), protocol)

    assert training == {"t0", "t1"}
    assert holdout == {"h0", "h1"}

    protocol["race_set_sha256"] = race_set_sha256({"different"})
    with pytest.raises(ValueError, match="holdout race set hash mismatch"):
        evaluation._protocol_race_sets(object(), protocol)


def test_training_beforeinfo_count_is_limited_to_complete_training_races() -> None:
    captured: dict[str, object] = {}

    class Cursor:
        def fetchone(self):
            return (0,)

    class Conn:
        def execute(self, sql: str, params: tuple[str]):
            captured["sql"] = sql
            captured["params"] = params
            return Cursor()

    assert evaluation._training_beforeinfo_rows(
        Conn(), holdout_start="2025-07-23"
    ) == 0
    sql = " ".join(str(captured["sql"]).split()).lower()
    assert "join races r on r.race_id = b.race_id" in sql
    assert "r.race_date < ?" in sql
    assert "from entries" in sql
    assert "from race_results" in sql
    assert captured["params"] == ("2025-07-23",)


def test_reused_model_copy_rejects_beforeinfo_and_strict_metadata(
    tmp_path: Path,
) -> None:
    training_races = {"t0", "t1"}
    source = tmp_path / "source.joblib"
    output = tmp_path / "job.joblib"
    _source_bundle(source, training_races)

    with pytest.raises(ValueError, match="contain beforeinfo"):
        evaluation._copy_reused_model(
            source_path=source,
            output_path=output,
            training_races=training_races,
            training_beforeinfo_rows=1,
        )

    bundle = joblib.load(source)
    bundle["metadata"]["train_races"] = 1
    joblib.dump(bundle, source)
    with pytest.raises(ValueError, match="training race count mismatch"):
        evaluation._copy_reused_model(
            source_path=source,
            output_path=output,
            training_races=training_races,
            training_beforeinfo_rows=0,
        )

    _source_bundle(source, training_races)
    bundle = joblib.load(source)
    bundle["metadata"]["train_race_set_sha256"] = "wrong"
    joblib.dump(bundle, source)
    with pytest.raises(ValueError, match="training race set hash mismatch"):
        evaluation._copy_reused_model(
            source_path=source,
            output_path=output,
            training_races=training_races,
            training_beforeinfo_rows=0,
        )


def test_reused_model_copy_only_changes_equivalent_metadata(tmp_path: Path) -> None:
    training_races = {"t0", "t1"}
    source = tmp_path / "source.joblib"
    output = tmp_path / "job.joblib"
    _source_bundle(source, training_races)

    reused, source_metadata = evaluation._copy_reused_model(
        source_path=source,
        output_path=output,
        training_races=training_races,
        training_beforeinfo_rows=0,
    )

    copied = joblib.load(output)
    assert source_metadata["feature_set"] == evaluation.LEGACY_FEATURE_SET
    assert reused["pipeline"] == copied["pipeline"] == {"fitted": True}
    assert copied["metadata"]["feature_set"] == historical_model.FEATURE_SET
    assert copied["metadata"]["include_beforeinfo"] is False
    assert copied["metadata"]["training_beforeinfo_rows"] == 0
    assert copied["metadata"]["training_reuse_equivalent"] is True
    assert not output.with_name(f".{output.name}.tmp").exists()


def test_score_holdout_passes_dates_and_emits_existing_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    holdout = {"h0", "h1"}

    def fake_scored(_conn: object, **kwargs: object):
        captured.update(kwargs)
        for race_id in sorted(holdout):
            for lane in range(1, 7):
                yield 0.55 if lane == 1 else 0.09, {
                    "race_id": race_id,
                    "lane": lane,
                    "rank": lane,
                }

    monkeypatch.setattr(historical_model, "iter_scored_entries", fake_scored)
    result = evaluation._score_holdout(
        object(),
        bundle={"pipeline": "pipeline"},
        training_races={"t0", "t1"},
        holdout_races=holdout,
        holdout_start="2025-07-23",
        holdout_end="2026-07-22",
    )

    assert captured["include_races"] == holdout
    assert captured["from_date"] == "2025-07-23"
    assert captured["through_date"] == "2026-07-22"
    assert result["entry_log_loss"] > 0
    assert result["entry_brier"] > 0
    assert result["winner_top1_accuracy"] == 1.0
    assert result["trifecta_top1_hit_rate"] == 1.0
    assert result["trifecta_top5_hit_rate"] == 1.0
    assert result["evaluation_race_set_sha256"] == race_set_sha256(holdout)


def test_historical_iterator_propagates_date_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_examples(_conn: object, **kwargs: object):
        captured.update(kwargs)
        yield {"lane": 1}, 1, {"race_id": "h0", "lane": 1, "rank": 1}

    monkeypatch.setattr(historical_model, "iter_training_examples", fake_examples)
    monkeypatch.setattr(
        historical_model.base,
        "positive_probs",
        lambda _pipeline, batch: [0.5 for _ in batch],
    )

    rows = list(
        historical_model.iter_scored_entries(
            object(),
            pipeline=object(),
            include_races={"h0"},
            from_date="2025-07-23",
            through_date="2026-07-22",
        )
    )

    assert rows[0][0] == 0.5
    assert captured["from_date"] == "2025-07-23"
    assert captured["through_date"] == "2026-07-22"
    assert captured["include_beforeinfo"] is False


def test_evaluation_reuses_fixed_model_and_runs_bankroll_with_job_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    protocol = _protocol()
    training_races = {"t0", "t1"}
    holdout_races = {"h0", "h1"}
    source = tmp_path / "no_odds_v8.joblib"
    _source_bundle(source, training_races)
    prediction = {
        "entry_log_loss": 0.41,
        "entry_brier": 0.12,
        "winner_top1_accuracy": 0.58,
        "trifecta_top1_hit_rate": 0.09,
        "trifecta_top5_hit_rate": 0.33,
        "evaluation_race_set_sha256": race_set_sha256(holdout_races),
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

    def fake_protocol(_conn: object, *, as_of_date: date):
        calls["as_of_date"] = as_of_date
        return protocol

    def fake_score(_conn: object, **kwargs: object):
        calls["score"] = kwargs
        return prediction

    def fake_bankroll(_conn: object, **kwargs: object):
        calls["bankroll"] = kwargs
        calls["bankroll_max_date"] = os.environ.get(
            "BOATRACE_EVAL_MAX_RACE_DATE"
        )
        return bankroll

    monkeypatch.setattr(evaluation.standard_evaluation, "build_protocol", fake_protocol)
    monkeypatch.setattr(
        evaluation,
        "_protocol_race_sets",
        lambda _conn, _protocol: (training_races, holdout_races),
    )
    monkeypatch.setattr(evaluation, "_training_beforeinfo_rows", lambda *_a, **_k: 0)
    monkeypatch.setattr(evaluation, "_score_holdout", fake_score)
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
        model_input_path=source,
    )

    assert calls["as_of_date"] == date(2026, 7, 23)
    assert calls["bankroll_max_date"] == "2026-07-22"
    assert os.environ["BOATRACE_EVAL_MAX_RACE_DATE"] == "keep-me"
    score_args = calls["score"]
    bankroll_args = calls["bankroll"]
    assert isinstance(score_args, dict)
    assert isinstance(bankroll_args, dict)
    assert score_args["holdout_start"] == "2025-07-23"
    assert score_args["holdout_end"] == "2026-07-22"
    assert bankroll_args["folds"] == 1
    assert bankroll_args["min_train_races"] == 2
    assert bankroll_args["model_input_path"] == output.with_suffix(".joblib")
    assert bankroll_args["daily_budget_yen"] == 10_000
    assert result["model"] == "win_model_no_odds_v8_beforeinfo_excluded"
    assert result["source_model_path"] == str(source)
    assert result["source_train_hash"] == race_set_sha256(training_races)
    assert result["training_beforeinfo_rows"] == 0
    assert result["training_reuse_equivalent"] is True
    assert result["promotion_eligible"] is False
    assert result["daily"] == daily
    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert not output.with_name(f".{output.name}.tmp").exists()
    assert not output.with_name(f".{output.name}.bankroll.json").exists()


def test_evaluation_rejects_nonzero_training_beforeinfo_before_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evaluation.standard_evaluation,
        "build_protocol",
        lambda *_a, **_k: _protocol(),
    )
    monkeypatch.setattr(
        evaluation,
        "_protocol_race_sets",
        lambda *_a, **_k: ({"t0", "t1"}, {"h0", "h1"}),
    )
    monkeypatch.setattr(evaluation, "_training_beforeinfo_rows", lambda *_a, **_k: 1)
    source = tmp_path / "source.joblib"
    _source_bundle(source, {"t0", "t1"})

    with pytest.raises(ValueError, match="contain beforeinfo"):
        evaluation.evaluate_historical_candidate(
            object(),
            output_path=tmp_path / "result.json",
            evaluation_date=date(2026, 7, 22),
            model_input_path=source,
        )
    assert not (tmp_path / "result.joblib").exists()


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
        "--model-input",
        str(tmp_path / "data/models/standardized_365d_v2/no_odds_v8.joblib"),
        "--evaluation-date",
        "2026-07-22",
    ]
    assert output == tmp_path / "data/models/evaluation_queue/job-00000042.json"
    assert TASK_PROFILES["historical_coverage_safe"] == {
        "category": "evaluation",
        "memory_mb": 4_096,
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
            {"evaluation_date": "2026-07-22", "model_input": "/tmp/model"},
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
            "model": "win_model_no_odds_v8_beforeinfo_excluded",
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

    monkeypatch.setattr("boatrace_ai.evaluation_queue.enqueue_job", fake_enqueue)
    seed_default_jobs(object(), evaluation_date="2026-07-22")

    assert not any(
        call["task_type"] == "historical_coverage_safe" for call in calls
    )
    standardized = next(
        call for call in calls if call["task_type"] == "standardized_365d"
    )
    assert standardized["parameters"] == {
        "evaluation_date": "2026-07-22",
        "timeout_seconds": 86_400,
    }
