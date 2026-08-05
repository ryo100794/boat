from __future__ import annotations

from datetime import date, timedelta

from boatrace_ai.listwise import decision_stacked_market_v44 as subject


def _race(day: date, index: int) -> dict:
    return {
        "race_id": f"{day.isoformat()}-{index}",
        "race_date": day.isoformat(),
        "jcd": "01",
        "rno": index + 1,
        "actual_combination": "1-2-3",
        "actual_payout_yen": 500,
        "odds": {"1-2-3": 5.0, "1-3-2": 7.0},
        "model_probabilities": {"1-2-3": 0.6, "1-3-2": 0.4},
        "market_probabilities": {"1-2-3": 0.55, "1-3-2": 0.45},
        "lane_context": [{"lane": lane} for lane in range(1, 7)],
        "snapshot_id": index,
        "captured_at": f"{day.isoformat()}T10:00:00+09:00",
        "odds_deadline_at": f"{day.isoformat()}T10:01:00+09:00",
        "input_snapshot_age_seconds": 60.0,
        "closing_odds": {"1-2-3": 99.0},
    }


def test_v44_uses_only_decision_fields_and_keeps_outer_holdout(monkeypatch) -> None:
    start = date(2026, 7, 20)
    races = [
        _race(start + timedelta(days=day), day * 120 + race)
        for day in range(12)
        for race in range(2)
    ]
    observed = {}

    def fake_fit(calibration, evaluation, *, num_threads):
        observed["calibration"] = calibration
        observed["evaluation"] = evaluation
        assert num_threads == 2
        assert all("closing_odds" not in race for race in calibration + evaluation)
        return {
            "market_is_exact_nested_null": True,
            "base_training_through": calibration[-2]["race_date"],
            "stack_validation_from": calibration[-1]["race_date"],
            "stack_candidates": [],
            "selected_stack": "market50_linear50",
            "selected_weights": {
                "market": 0.5,
                "linear": 0.5,
                "nonlinear": 0.0,
            },
            "component_selection": {},
            "artifact": {"artifact_sha256": "a" * 64},
            "metrics": {
                "evaluated_days": 2,
                "evaluated_races": 4,
                "days_better_than_market": 2,
                "log_loss_delta_vs_market": -0.01,
                "trifecta_top5_hit_rate": 0.4,
                "market_trifecta_top5_hit_rate": 0.4,
            },
        }

    monkeypatch.setattr(subject, "fit_temporal_stacked_market_residual", fake_fit)
    result = subject.fit_decision_time_stacked_market(
        races,
        calibration_through=(start + timedelta(days=9)).isoformat(),
        minimum_training_days=10,
        minimum_training_races=20,
        num_threads=2,
    )

    assert len(observed["calibration"]) == 20
    assert len(observed["evaluation"]) == 4
    assert result["official_closing_fields_used"] is False
    assert result["market_probability_source"] == "decision_snapshot_odds"
    assert result["selected_stack"] == "market50_linear50"


def test_v44_challenger_requires_nonmarket_weight_and_seven_holdout_days() -> None:
    payload = {
        "training_status": "ready",
        "official_closing_fields_used": False,
        "market_is_exact_nested_null": True,
        "selected_weights": {"market": 0.5, "linear": 0.5, "nonlinear": 0.0},
        "artifact": {"artifact_sha256": "b" * 64},
        "holdout_metrics": {
            "evaluated_days": 7,
            "days_better_than_market": 4,
            "log_loss_delta_vs_market": -0.001,
            "trifecta_top5_hit_rate": 0.37,
            "market_trifecta_top5_hit_rate": 0.371,
        },
    }
    assert subject.decision_v44_challenger_eligible(payload) is True

    payload["selected_weights"] = {
        "market": 1.0,
        "linear": 0.0,
        "nonlinear": 0.0,
    }
    assert subject.decision_v44_challenger_eligible(payload) is False


def test_v44_waits_for_training_history() -> None:
    start = date(2026, 7, 20)
    result = subject.fit_decision_time_stacked_market(
        [_race(start + timedelta(days=day), day) for day in range(9)],
        calibration_through=(start + timedelta(days=8)).isoformat(),
        minimum_training_days=10,
        minimum_training_races=1,
    )
    assert result["status"] == "insufficient_training_history"
    assert result["ready_reasons"] == ["training_days_below_minimum"]
    assert result["decision_time_boundary_all_passed"] is True
    assert result["decision_time_boundary_violations"] == 0
    assert result["maximum_input_snapshot_age_seconds"] == 60.0
    assert result["allowed_input_snapshot_age_seconds"] == 65.0
