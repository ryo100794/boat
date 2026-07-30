from __future__ import annotations

from pathlib import Path

import pytest

import boatrace_ai.evaluation_queue as evaluation_queue
import boatrace_ai.listwise.market_calibration as market_calibration
import boatrace_ai.listwise.odds_path_role_integrated_v16 as integrated_v16
from boatrace_ai.evaluation_queue import (
    build_command,
    seed_daily_market_jobs,
    summarize_result,
)


STRATEGY = "odds_path_role_integrated_fixed_band_passthrough_v16"


def _job() -> dict[str, object]:
    return {
        "job_id": 16,
        "status": "running",
        "task_type": "market_residual_walk_forward",
        "model_key": "v16-candidate",
        "parameters": {
            "model_input": "data/models/source.joblib",
            "from_date": "2026-07-18",
            "through_date": "2026-08-01",
            "daily_budget_yen": 10_000,
            "min_calibration_days": 5,
            "calibrator_strategy": STRATEGY,
            "v12_closing_fallback_policy": "no_bet",
        },
    }


def test_v16_parser_and_queue_command_are_reproducible(tmp_path: Path) -> None:
    model = tmp_path / "data/models/source.joblib"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"artifact")
    python = tmp_path / ".venv/bin/python"

    first, first_output = build_command(
        _job(), app_root=tmp_path, python=python, db="postgresql://test"
    )
    second, second_output = build_command(
        _job(), app_root=tmp_path, python=python, db="postgresql://test"
    )

    assert first == second
    assert first_output == second_output
    assert first[first.index("--calibrator-strategy") + 1] == STRATEGY
    assert first[first.index("--v12-closing-fallback-policy") + 1] == "no_bet"
    parsed = market_calibration.build_parser().parse_args([
        "--from-date", "2026-07-18",
        "--calibrator-strategy", STRATEGY,
    ])
    assert parsed.calibrator_strategy == STRATEGY

    invalid = _job()
    invalid["parameters"] = {
        **invalid["parameters"],
        "v12_closing_fallback_policy": "v11",
    }
    with pytest.raises(
        ValueError, match="V16 requires v12_closing_fallback_policy=no_bet"
    ):
        build_command(
            invalid, app_root=tmp_path, python=python, db="postgresql://test"
        )


def test_v16_dispatcher_preserves_the_exact_evaluation_population(
    monkeypatch,
) -> None:
    fallback_policies: list[str] = []
    received_inputs: list[tuple[list[dict[str, str]], list[str]]] = []

    def fake_v12(input_races, **kwargs):
        fallback_policies.append(str(kwargs["closing_fallback_policy"]))
        received_inputs.append(
            (list(input_races), list(kwargs["evaluation_dates"]))
        )
        return {}

    monkeypatch.setattr(integrated_v16, "walk_forward_evaluate_v12", fake_v12)
    races = [
        {"race_date": "2026-07-29", "race_id": "partial-t300-complete"},
        {"race_date": "2026-07-30", "race_id": "2026-07-30-02-01"},
        {"race_date": "2026-07-30", "race_id": "2026-07-30-01-01"},
    ]
    direct = integrated_v16.walk_forward_evaluate_v16(
        races,
        daily_budget_yen=10_000,
        min_calibration_days=2,
        evaluation_dates=["2026-07-30"],
    )
    dispatched = market_calibration.walk_forward_evaluate(
        list(reversed(races)),
        daily_budget_yen=10_000,
        min_calibration_days=2,
        calibrator_strategy=STRATEGY,
        evaluation_dates=["2026-07-30"],
        v12_closing_fallback_policy="no_bet",
    )

    assert integrated_v16.MODEL_NAME == STRATEGY
    assert integrated_v16.STRATEGY_NAME == STRATEGY
    assert direct["evaluation_population_races"] == 3
    assert dispatched["evaluation_population_races"] == 3
    assert direct["evaluation_population_hash"] == dispatched[
        "evaluation_population_hash"
    ]
    assert direct["model"] == dispatched["model"] == STRATEGY
    assert direct["calibrator_strategy"] == STRATEGY
    assert dispatched["calibrator_strategy"] == STRATEGY
    assert fallback_policies == ["no_bet", "no_bet"]
    expected_training_ids = {race["race_id"] for race in races}
    assert all(
        {race["race_id"] for race in input_races} == expected_training_ids
        for input_races, _ in received_inputs
    )
    assert all(
        any(race["race_id"] == "partial-t300-complete" for race in input_races)
        for input_races, _ in received_inputs
    )
    assert all(
        evaluation_dates == ["2026-07-30"]
        for _, evaluation_dates in received_inputs
    )
    assert direct["calibration_input_scope"] == (
        "all_eligible_races_including_partial_market_days"
    )
    assert direct["evaluation_date_scope"] == (
        "formal_complete_market_days_only"
    )
    assert "partial market days" in direct["validation_design"]
    assert "holdout evaluation restricted" in direct["validation_design"]


def test_daily_v16_is_queued_ahead_of_v15(tmp_path: Path, monkeypatch) -> None:
    model_dir = tmp_path / "data/models/evaluation_queue"
    model_dir.mkdir(parents=True)
    source_path = model_dir / "source.json"
    source_path.with_suffix(".joblib").write_bytes(b"model")

    class FakeResult:
        def fetchone(self):
            return {"result_path": str(source_path)}

    class FakeConnection:
        def execute(self, _sql, _parameters):
            return FakeResult()

    calls: list[dict[str, object]] = []

    def fake_enqueue(_conn, **kwargs):
        calls.append(kwargs)
        return len(calls)

    monkeypatch.setattr(evaluation_queue, "enqueue_job", fake_enqueue)
    seed_daily_market_jobs(
        FakeConnection(), app_root=tmp_path, evaluation_date="2026-07-30"
    )

    by_strategy = {
        row["parameters"]["calibrator_strategy"]: row for row in calls
    }
    v16 = by_strategy[STRATEGY]
    v15 = by_strategy[
        "odds_path_role_integrated_selection_free_envelope_v15"
    ]
    assert v16["priority"] > v15["priority"]
    deployment_priorities = {
        strategy: by_strategy[strategy]["priority"]
        for strategy in evaluation_queue.DEPLOYMENT_DAILY_MARKET_PRIORITIES
    }
    assert sum(
        row["parameters"]["calibrator_strategy"] in deployment_priorities
        for row in calls
    ) == len(deployment_priorities)
    assert deployment_priorities == {
        "odds_path_role_integrated_t300_nonlinear_v12": 104,
        "odds_path_role_integrated_registered_band_lcb_v14": 103,
        "odds_path_role_integrated_fixed_band_passthrough_v16": 102,
        "odds_path_observed_closing_return_schedule_quota_v18": 101,
        "odds_path_observed_closing_return_schedule_quota_dual_head_v20": 100,
    }
    assert min(deployment_priorities.values()) > max(
        row["priority"]
        for strategy, row in by_strategy.items()
        if strategy not in deployment_priorities
    )
    assert v16["parameters"]["timeout_seconds"] == 14_400
    assert v16["parameters"]["min_calibration_days"] == 5
    v19 = by_strategy[
        "odds_path_observed_closing_return_schedule_quota_raw_nonregression_v19"
    ]
    assert v19["priority"] == 96
    assert v19["priority"] < min(deployment_priorities.values())
    v21 = by_strategy[
        "odds_path_observed_closing_return_schedule_quota_triple_head_v21"
    ]
    assert v21["priority"] == 97
    assert v21["priority"] < min(deployment_priorities.values())
    assert v16["parameters"]["v12_closing_fallback_policy"] == "no_bet"
    assert v16["parameters"]["model_input"] == v15["parameters"]["model_input"]
    assert v16["parameters"]["from_date"] == v15["parameters"]["from_date"]
    assert v16["parameters"]["through_date"] == v15["parameters"]["through_date"]
    assert calls.index(v16) < calls.index(v15)
    assert evaluation_queue.TASK_PROFILES[
        "market_residual_walk_forward"
    ]["max_parallel"] == 2
    assert STRATEGY not in market_calibration.CLEAN_DAY_CALIBRATOR_STRATEGIES
    all_races = [{"race_id": "complete"}, {"race_id": "incomplete"}]
    clean_races = [all_races[0]]
    assert market_calibration.select_calibrator_evaluation_races(
        STRATEGY,
        races=all_races,
        clean_races=clean_races,
    ) is all_races
    for clean_day_strategy in (
        "odds_path_role_integrated_registered_band_lcb_v14",
        "odds_path_role_integrated_selection_free_envelope_v15",
    ):
        assert market_calibration.select_calibrator_evaluation_races(
            clean_day_strategy,
            races=all_races,
            clean_races=clean_races,
        ) is clean_races


def test_v16_summary_preserves_web_report_metrics_and_diagnostics() -> None:
    closing_envelope = {
        "selection_free": True,
        "evaluation_folds": 6,
        "ready_folds": 1,
        "haircut_latest": 0.72,
    }
    prospective_envelope = {
        "selection_free": True,
        "evaluation_folds": 1,
        "ready_folds": 1,
        "haircut_latest": 0.74,
    }
    promotion_gate = {
        "sample_days_pass": False,
        "closing_envelope_ready_pass": True,
        "closing_envelope_no_missing_races_pass": True,
    }
    prospective = {
        "status": "evaluating",
        "registered_after": "2026-07-29",
        "evaluation_days": 1,
        "evaluated_races": 72,
        "tickets": 8,
        "hit_tickets": 2,
        "stake_yen": 800,
        "return_yen": 1_240,
        "profit_yen": 440,
        "roi": 1.55,
        "roi_without_largest_hit": 0.8,
        "profit_without_largest_hit_yen": -160,
        "daily_cluster_bootstrap_roi_lower_95": 0.8,
        "effective_hit_count": 1.7,
        "largest_hit_return_share": 0.61,
        "max_drawdown_yen": 300,
        "selected_races": 4,
        "hit_races": 2,
        "profitable_days": 1,
        "profitable_day_fraction": 1.0,
        "race_selection_rate": 4 / 72,
        "promotion_eligible": False,
        "closing_envelope_conformal": prospective_envelope,
        "promotion_gate": promotion_gate,
    }

    fixed_band_diagnostics = {
        "real_betting_enabled": False,
        "post_hoc_best_rule_is_promotion_evidence": False,
        "folds": [{"fold": 1, "rules": {}}],
        "rules": {"safe_ev_desc": {"days": 1, "roi": 1.1}},
    }
    summary = summarize_result({
        "model": STRATEGY,
        "closing_envelope_conformal": closing_envelope,
        "prospective_role_integrated_v16_walk_forward": prospective,
        "fixed_band_ranking_diagnostics": fixed_band_diagnostics,
    })

    assert summary["closing_envelope_conformal"] == closing_envelope
    for key, value in prospective.items():
        if key in {"closing_envelope_conformal", "promotion_gate"}:
            continue
        assert summary[f"prospective_v16_{key}"] == value
    assert summary["prospective_v16_closing_envelope_conformal"] == (
        prospective_envelope
    )
    assert summary["prospective_v16_promotion_gate"] == promotion_gate
    assert summary["fixed_band_ranking_diagnostics"] == fixed_band_diagnostics
