from __future__ import annotations

from datetime import date, timedelta
from itertools import permutations

from boatrace_ai.listwise import decision_stacked_terminal_market_v46 as subject


COMBINATIONS = tuple(
    "-".join(str(lane) for lane in lanes)
    for lanes in permutations(range(1, 7), 3)
)


def _race(day: date, index: int, *, closing_scale: float = 1.0) -> dict:
    odds = {
        combination: 5.0 + position
        for position, combination in enumerate(COMBINATIONS)
    }
    official = {
        combination: value * closing_scale
        for combination, value in odds.items()
    }
    return {
        "race_id": f"{day.isoformat()}-{index}",
        "race_date": day.isoformat(),
        "jcd": "01",
        "rno": 1,
        "actual_combination": COMBINATIONS[0],
        "actual_payout_yen": 500,
        "odds": odds,
        "official_closing_odds": official,
        "model_probabilities": {
            combination: 1.0 / len(COMBINATIONS)
            for combination in COMBINATIONS
        },
        "market_probabilities": {
            combination: 1.0 / len(COMBINATIONS)
            for combination in COMBINATIONS
        },
        "odds_path": [],
        "snapshot_id": index,
        "captured_at": f"{day.isoformat()}T10:00:00+09:00",
        "odds_deadline_at": f"{day.isoformat()}T10:01:00+09:00",
        "betting_deadline_at": f"{day.isoformat()}T10:06:00+09:00",
        "decision_lead_seconds": 300.0,
        "input_snapshot_age_seconds": 60.0,
    }


def _probability_result() -> dict:
    return {
        "status": "ready",
        "model": "decision_time_stacked_market_residual_v44",
        "training_days": 10,
        "training_races": 10,
        "selected_stack": "linear50_nonlinear50",
        "selected_weights": {
            "market": 0.0,
            "linear": 0.5,
            "nonlinear": 0.5,
        },
        "artifact": {
            "model": "stacked_market_residual_v42",
            "artifact_sha256": "a" * 64,
        },
    }


def test_v46_price_teacher_is_prior_and_outer_cannot_change_artifact(
    monkeypatch,
) -> None:
    start = date(2026, 7, 20)
    cutoff = start + timedelta(days=9)
    races = [_race(start + timedelta(days=day), day) for day in range(12)]
    observed: list[list[dict]] = []

    monkeypatch.setattr(
        subject,
        "fit_decision_time_stacked_market",
        lambda *args, **kwargs: _probability_result(),
    )

    def fake_price(calibration):
        observed.append(calibration)
        return {
            "model_type": subject.PRICE_MODEL_TYPE,
            "calibration_method": "leave_one_training_day_out_cross_conformal",
            "residual_q10": -0.2,
            "residual_q50": 0.0,
            "residual_q90": 0.2,
        }

    monkeypatch.setattr(subject, "_fit_price_model", fake_price)
    monkeypatch.setattr(
        subject,
        "closing_odds_quantile_metrics",
        lambda evaluation, model: {"evaluation_races": len(evaluation)},
    )
    monkeypatch.setattr(
        subject,
        "stacked_probabilities",
        lambda race, artifact: {
            COMBINATIONS[0]: 0.6,
            COMBINATIONS[1]: 0.4,
        },
    )
    monkeypatch.setattr(
        subject,
        "forecast_closing_odds_quantiles",
        lambda race, model: {
            "q10": {key: float(value) * 0.8 for key, value in race["odds"].items()},
            "q50": {key: float(value) for key, value in race["odds"].items()},
            "q90": {key: float(value) * 1.2 for key, value in race["odds"].items()},
        },
    )

    first = subject.fit_decision_time_stacked_terminal_market(
        races,
        calibration_through=cutoff.isoformat(),
        minimum_training_days=10,
        minimum_training_races=10,
        num_threads=2,
    )
    changed = [
        (
            _race(start + timedelta(days=day), day, closing_scale=9.0)
            if day > 9 else race
        )
        for day, race in enumerate(races)
    ]
    second = subject.fit_decision_time_stacked_terminal_market(
        changed,
        calibration_through=cutoff.isoformat(),
        minimum_training_days=10,
        minimum_training_races=10,
        num_threads=2,
    )

    assert len(observed) == 2
    assert all(len(value) == 10 for value in observed)
    assert all(
        max(str(race["race_date"]) for race in value) == cutoff.isoformat()
        for value in observed
    )
    assert all(
        race["closing_odds"] == race["official_closing_odds"]
        for value in observed for race in value
    )
    assert first["artifact"] == second["artifact"]
    assert first["price_model_outer_period_used_for_selection"] is False
    diagnostic = first["terminal_value_candidate_diagnostic"]
    assert diagnostic["purchase_gate"] == (
        "disabled_pending_strict_prior_realized_roi_lcb"
    )
    assert diagnostic["outer_used_for_threshold_or_model_selection"] is False
    assert first["promotion_eligible"] is False
    assert first["real_betting_enabled"] is False


def test_v46_does_not_fit_price_when_probability_history_is_not_ready(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        subject,
        "fit_decision_time_stacked_market",
        lambda *args, **kwargs: {
            "status": "insufficient_training_history",
            "model": "decision_time_stacked_market_residual_v44",
        },
    )
    result = subject.fit_decision_time_stacked_terminal_market(
        [],
        calibration_through="2026-08-01",
        minimum_training_days=30,
        minimum_training_races=3000,
    )
    assert result["status"] == "insufficient_training_history"
    assert result["price_training_status"] == (
        "not_started_probability_not_ready"
    )
