from __future__ import annotations

from datetime import date, timedelta

from boatrace_ai.listwise import nested_nonlinear_value_v40 as subject


def _race(day: date, index: int) -> dict:
    race_date = day.isoformat()
    return {
        "race_id": f"{race_date}-01-{index:03d}",
        "race_date": race_date,
        "jcd": "01",
        "rno": index % 12 + 1,
        "actual_combination": "1-2-3",
        "actual_payout_yen": 300,
        "odds": {"1-2-3": 3.0, "1-3-2": 4.0},
        "model_probabilities": {"1-2-3": 0.6, "1-3-2": 0.4},
        "market_probabilities": {"1-2-3": 0.6, "1-3-2": 0.4},
        "snapshot_id": index,
        "captured_at": f"{race_date}T10:00:00+09:00",
        "odds_deadline_at": f"{race_date}T10:01:00+09:00",
    }


def test_v40_uses_disjoint_model_value_and_outer_periods(monkeypatch) -> None:
    observed = {}

    def fake_fit(model_training, evaluation, *, num_threads):
        observed["model_training"] = model_training
        assert evaluation == []
        assert num_threads == 2
        return {
            "inner_fit_through": "2026-01-17",
            "inner_validation_from": "2026-01-18",
            "selected_tree_preset": "compact",
            "selected_shrinkage": 0.5,
            "artifact": {"booster_sha256": "a" * 64},
        }

    monkeypatch.setattr(
        subject, "fit_temporal_nonlinear_market_residual", fake_fit
    )
    monkeypatch.setattr(
        subject,
        "_score",
        lambda races, _artifact: [dict(race) for race in races],
    )
    monkeypatch.setattr(
        subject,
        "nonlinear_residual_metrics",
        lambda races, _artifact, *, shrinkage: {
            "evaluated_races": len(races),
            "selected_shrinkage": shrinkage,
        },
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
    result = subject.evaluate_nested_nonlinear_value_v40(
        calibration,
        evaluation,
        daily_budget_yen=10_000,
        num_threads=2,
    )

    assert result["model_training_days"] == 22
    assert result["model_training_through"] == "2026-01-22"
    assert result["value_calibration_days"] == 30
    assert result["value_calibration_from"] == "2026-01-23"
    assert result["value_calibration_through"] == "2026-02-21"
    assert result["evaluation_from"] == "2026-02-22"
    assert len(observed["model_training"]) == 220
    assert result["calibration_ledger_candidates"] == 600
    assert result["empirical_ev_calibration"]["ready"] is True
    assert result["bankroll"]["evaluation_days"] == 5
    assert result["real_betting_enabled"] is False
    assert result["promotion_eligible"] is False


def test_v40_refuses_less_than_50_nested_days() -> None:
    start = date(2026, 1, 1)
    races = [_race(start + timedelta(days=day), day) for day in range(49)]
    result = subject.evaluate_nested_nonlinear_value_v40(
        races,
        [],
        daily_budget_yen=10_000,
    )
    assert result["status"] == "insufficient_nested_days"
    assert result["required_days"] == 50
    assert result["promotion_eligible"] is False
