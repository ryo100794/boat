from __future__ import annotations

from datetime import date, timedelta

from boatrace_ai.listwise import nested_stacked_value_v43 as subject


def _race(day: date, index: int) -> dict:
    race_date = day.isoformat()
    return {
        "race_id": f"{race_date}-{index}",
        "race_date": race_date,
        "actual_combination": "1-2-3",
        "actual_payout_yen": 300,
        "odds": {"1-2-3": 3.0, "1-3-2": 4.0},
        "model_probabilities": {"1-2-3": 0.6, "1-3-2": 0.4},
        "market_probabilities": {"1-2-3": 0.6, "1-3-2": 0.4},
        "snapshot_id": index,
    }


def test_v43_keeps_model_value_and_outer_periods_disjoint(monkeypatch) -> None:
    observed = {}

    def fake_fit(model_training, evaluation, *, num_threads):
        observed["training"] = model_training
        assert evaluation == []
        assert num_threads == 2
        return {
            "base_training_through": "2026-01-17",
            "stack_validation_from": "2026-01-18",
            "selected_stack": "market",
            "selected_weights": {"market": 1.0, "linear": 0.0, "nonlinear": 0.0},
            "component_selection": {},
            "artifact": {"artifact_sha256": "a" * 64},
        }

    monkeypatch.setattr(subject, "fit_temporal_stacked_market_residual", fake_fit)
    monkeypatch.setattr(
        subject,
        "_score",
        lambda races, artifact: [dict(race) for race in races],
    )
    monkeypatch.setattr(
        subject,
        "stacked_metrics",
        lambda races, artifact: {"evaluated_races": len(races)},
    )
    start = date(2026, 1, 1)
    calibration = [
        _race(start + timedelta(days=day), day * 10 + race)
        for day in range(52)
        for race in range(10)
    ]
    evaluation = [
        _race(start + timedelta(days=52 + day), 1000 + day * 10 + race)
        for day in range(5)
        for race in range(10)
    ]
    result = subject.evaluate_nested_stacked_value_v43(
        calibration,
        evaluation,
        daily_budget_yen=10_000,
        num_threads=2,
    )

    assert len(observed["training"]) == 220
    assert result["model_training_days"] == 22
    assert result["value_calibration_days"] == 30
    assert result["evaluation_from"] == "2026-02-22"
    assert result["calibration_ledger_candidates"] == 600
    assert result["evaluation_ledger_candidates"] == 100
    assert result["probability_selection"]["selected_stack"] == "market"
    assert result["real_betting_enabled"] is False
    assert result["promotion_eligible"] is False


def test_v43_refuses_less_than_50_nested_days() -> None:
    start = date(2026, 1, 1)
    races = [_race(start + timedelta(days=day), day) for day in range(49)]
    result = subject.evaluate_nested_stacked_value_v43(
        races, [], daily_budget_yen=10_000
    )
    assert result["status"] == "insufficient_nested_days"
    assert result["required_days"] == 50
