from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta

import pytest

from boatrace_ai.fast_math import TRIFECTA_COMBINATIONS
from boatrace_ai.listwise import market_calibration
from boatrace_ai.listwise.closing_odds import decision_odds


COMBINATIONS = tuple(
    "-".join(str(lane) for lane in combination)
    for combination in TRIFECTA_COMBINATIONS
)


def _race(race_date: str, *, closing_scale: float = 0.8) -> dict[str, object]:
    probability = 1.0 / len(COMBINATIONS)
    odds = {
        combination: 20.0 + index / 10.0
        for index, combination in enumerate(COMBINATIONS)
    }
    return {
        "race_id": f"{race_date}-01-01",
        "race_date": race_date,
        "jcd": "01",
        "rno": 1,
        "actual_combination": COMBINATIONS[0],
        "actual_payout_yen": 1_600,
        "model_probabilities": {
            combination: probability for combination in COMBINATIONS
        },
        "market_probabilities": {
            combination: probability for combination in COMBINATIONS
        },
        "odds": odds,
        "closing_odds": {
            combination: value * closing_scale
            for combination, value in odds.items()
        },
        "closing_source_changed": True,
        "closing_odds_changed": True,
        "historical_return_multipliers": {},
        "snapshot_id": race_date,
    }


def _races(days: int) -> list[dict[str, object]]:
    start = date(2026, 1, 1)
    return [
        _race((start + timedelta(days=index)).isoformat())
        for index in range(days)
    ]


def _selection_spy(monkeypatch) -> list[tuple[str, ...]]:
    calls: list[tuple[str, ...]] = []

    def select(teachers):
        training_dates = tuple(sorted({str(race["race_date"]) for race in teachers}))
        calls.append(training_dates)
        return {
            "selected": "baseline",
            "baseline_model": {
                "intercept": 0.0,
                "log_odds_coefficient": 1.0,
                "expected_odds_multiplier": 0.75,
            },
            "momentum_model": None,
        }

    monkeypatch.setattr(
        market_calibration,
        "closing_odds_training_ready",
        lambda teachers, **_kwargs: bool(teachers),
    )
    monkeypatch.setattr(market_calibration, "select_closing_odds_model", select)
    return calls


def test_prequential_policy_prices_use_strictly_earlier_days(monkeypatch) -> None:
    calls = _selection_spy(monkeypatch)
    races = _races(3)

    result = market_calibration.prequential_closing_odds_policy_inputs(races)
    attached = result["races_by_id"]

    first = attached[str(races[0]["race_id"])]
    assert first["closing_odds_policy_input"] == "observed_t5_fallback"
    assert "estimated_final_odds" not in first

    for race in races[1:]:
        prediction_date = str(race["race_date"])
        item = attached[str(race["race_id"])]
        audit = result["folds"][prediction_date]
        assert item["closing_odds_policy_input"] == (
            "oof_forecast_final_from_real_t5"
        )
        assert item["closing_odds_model_trained_through_date"] < prediction_date
        assert all(day < prediction_date for day in audit["training_dates"])
        assert decision_odds(item) == item["estimated_final_odds"]
        assert audit["evaluation"]["evaluation_races"] == 1

    assert calls == [("2026-01-01",), ("2026-01-01", "2026-01-02")]


def test_future_closing_result_cannot_change_earlier_policy_price(monkeypatch) -> None:
    _selection_spy(monkeypatch)
    races = _races(3)
    baseline = market_calibration.prequential_closing_odds_policy_inputs(races)

    changed = deepcopy(races)
    changed[-1]["closing_odds"] = {
        combination: value * 10.0
        for combination, value in changed[-1]["closing_odds"].items()
    }
    mutated = market_calibration.prequential_closing_odds_policy_inputs(changed)

    second_id = str(races[1]["race_id"])
    assert (
        mutated["races_by_id"][second_id]["estimated_final_odds"]
        == baseline["races_by_id"][second_id]["estimated_final_odds"]
    )
    assert mutated["folds"]["2026-01-02"] == baseline["folds"]["2026-01-02"]


def test_unavailable_forecast_explicitly_falls_back_to_observed_t5(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        market_calibration,
        "closing_odds_training_ready",
        lambda teachers, **_kwargs: False,
    )
    races = _races(2)
    races[1]["estimated_final_odds"] = {
        combination: 999.0 for combination in COMBINATIONS
    }

    result = market_calibration.prequential_closing_odds_policy_inputs(races)
    item = result["races_by_id"][str(races[1]["race_id"])]

    assert item["closing_odds_policy_input"] == "observed_t5_fallback"
    assert item["closing_odds_policy_fallback"] is True
    assert item["closing_odds_policy_fallback_reason"] == (
        "insufficient_strictly_prior_closing_odds_teachers"
    )
    assert "estimated_final_odds" not in item
    assert decision_odds(item) == item["odds"]


def test_conformal_lower_policy_input_is_attached_without_lookahead() -> None:
    races = _races(2)
    inputs = {
        "policy_forecasts_by_race_id": {
            str(races[1]["race_id"]): {
                "estimated_final_odds": {
                    combination: value * 0.7
                    for combination, value in races[1]["odds"].items()
                },
                "closing_odds_forecast_target": (
                    "adaptive_conformal_lower_bound"
                ),
                "closing_odds_model_trained_through_date": "2026-01-01",
            }
        }
    }

    attached = (
        market_calibration.apply_prequential_conformal_lower_odds_policy_inputs(
            races, inputs
        )
    )

    assert attached[0]["closing_odds_policy_fallback"] is True
    assert attached[0]["closing_odds_policy_fallback_reason"] == (
        "insufficient_conformal_closing_odds_teachers"
    )
    assert attached[1]["closing_odds_policy_input"] == (
        "oof_adaptive_conformal_lower_from_real_t5"
    )
    assert attached[1]["closing_odds_model_trained_through_date"] < (
        attached[1]["race_date"]
    )
    assert decision_odds(attached[1]) == attached[1]["estimated_final_odds"]


def test_trend_point_policy_input_is_attached_without_lookahead() -> None:
    races = _races(2)
    inputs = {
        "point_policy_forecasts_by_race_id": {
            str(races[1]["race_id"]): {
                "estimated_final_odds": {
                    combination: value * 0.9
                    for combination, value in races[1]["odds"].items()
                },
                "closing_odds_forecast_target": "conditional_median",
                "closing_odds_model_type": "ridge_log_location_odds_path_v2",
                "closing_odds_model_trained_through_date": "2026-01-01",
            }
        }
    }

    attached = market_calibration.apply_prequential_trend_point_odds_policy_inputs(
        races, inputs
    )

    assert attached[0]["closing_odds_policy_fallback"] is True
    assert attached[0]["closing_odds_policy_fallback_reason"] == (
        "insufficient_trend_closing_odds_teachers"
    )
    assert attached[1]["closing_odds_policy_input"] == (
        "oof_trend_conditional_median_from_real_t5"
    )
    assert attached[1]["closing_odds_model_trained_through_date"] < (
        attached[1]["race_date"]
    )
    assert decision_odds(attached[1]) == attached[1]["estimated_final_odds"]


def test_walk_forward_connects_oof_prices_to_calibration_and_holdout(
    monkeypatch,
) -> None:
    _selection_spy(monkeypatch)
    races = _races(3)
    captured: dict[str, list[dict[str, object]]] = {}

    monkeypatch.setattr(
        market_calibration,
        "select_calibrator",
        lambda races: ({"model_weight": 1.0, "temperature": 1.0}, []),
    )

    def select_policy(policy_races, **kwargs):
        captured["calibration"] = policy_races
        return {"name": "no_bet", "no_bet": True}, []

    def simulate_policy(policy_races, **kwargs):
        captured["holdout"] = policy_races
        raise RuntimeError("captured policy inputs")

    monkeypatch.setattr(market_calibration, "select_policy", select_policy)
    monkeypatch.setattr(market_calibration, "simulate_policy", simulate_policy)

    with pytest.raises(RuntimeError, match="captured policy inputs"):
        market_calibration.walk_forward_evaluate(
            races,
            min_calibration_days=2,
            calibrator_strategy="grid",
        )

    calibration_by_date = {
        str(race["race_date"]): race for race in captured["calibration"]
    }
    assert calibration_by_date["2026-01-01"]["closing_odds_policy_input"] == (
        "observed_t5_fallback"
    )
    assert calibration_by_date["2026-01-02"]["closing_odds_policy_input"] == (
        "oof_forecast_final_from_real_t5"
    )
    holdout = captured["holdout"][0]
    assert holdout["race_date"] == "2026-01-03"
    assert holdout["closing_odds_model_trained_through_date"] == "2026-01-02"
    assert decision_odds(holdout) == holdout["estimated_final_odds"]
