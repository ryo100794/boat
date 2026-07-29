from __future__ import annotations

import json
from typing import Any

import pytest

from boatrace_ai.listwise import market_calibration
from boatrace_ai.listwise import odds_path_role_integrated_v12 as v12


COMBINATIONS = [
    f"{first}{second}{third}"
    for first in range(1, 7)
    for second in range(1, 7)
    if second != first
    for third in range(1, 7)
    if third not in (first, second)
]


def _race(race_date: str, race_no: int = 1) -> dict[str, Any]:
    odds = {
        combination: 5.0 + index * 0.1
        for index, combination in enumerate(COMBINATIONS)
    }
    return {
        "race_id": f"{race_date.replace("-", "")}01{race_no:02d}",
        "race_date": race_date,
        "jcd": 1,
        "rno": race_no,
        "odds": odds,
        "actual_combination": "123",
        "actual_payout_yen": 1_500,
    }


def _v12_artifact(*, ready: bool, adopted: bool) -> dict[str, Any]:
    return {
        "model_name": v12.V12_CLOSING_MODEL_NAME,
        "ready": ready,
        "challenger_adopted": adopted,
        "selection_reason": "test",
        "trained_through_date": "2026-07-01",
    }


def _v11_artifact(*, ready: bool) -> dict[str, Any]:
    return {
        "model_name": "closing_odds_multihorizon_v11",
        "ready": ready,
        "trained_through_date": "2026-07-01",
    }


def test_initial_selection_observations_are_copied_and_strict_prior() -> None:
    source = [{
        "race_date": "2026-07-01",
        "race_id": "r1",
        "closing_ratio": 1.0,
    }]

    copied = v12._copy_initial_selection_observations(source)
    copied[0]["closing_ratio"] = 0.8

    assert copied is not source
    assert copied[0] is not source[0]
    assert source[0]["closing_ratio"] == 1.0
    v12._assert_strict_prior_selection_observations(
        copied, evaluation_date="2026-07-02"
    )


@pytest.mark.parametrize("race_date", ["2026-07-02", "2026-07-03"])
def test_selection_observations_reject_same_or_future_day(race_date: str) -> None:
    with pytest.raises(ValueError, match="strict-prior days"):
        v12._assert_strict_prior_selection_observations(
            [{"race_date": race_date}], evaluation_date="2026-07-02"
        )


def test_closing_contract_prefers_adopted_v12_then_explicit_v11_fallback() -> None:
    adopted = v12._closing_contract(
        _v12_artifact(ready=True, adopted=True),
        _v11_artifact(ready=True),
        fallback_policy=v12.CLOSING_FALLBACK_V11,
    )
    assert adopted["selected_model"] == v12.V12_CLOSING_MODEL_NAME
    assert adopted["selection_reason"] == "v12_ready_and_adopted"
    assert adopted["ready_for_purchase"] is True

    fallback = v12._closing_contract(
        _v12_artifact(ready=True, adopted=False),
        _v11_artifact(ready=True),
        fallback_policy=v12.CLOSING_FALLBACK_V11,
    )
    assert fallback["selected_model"] == "closing_odds_multihorizon_v11"
    assert fallback["v12_ready"] is True
    assert fallback["v12_adopted"] is False
    assert fallback["fallback_policy"] == "v11"


def test_closing_contract_no_bet_is_explicit_when_v12_is_not_adopted() -> None:
    explicit = v12._closing_contract(
        _v12_artifact(ready=True, adopted=False),
        _v11_artifact(ready=True),
        fallback_policy=v12.CLOSING_FALLBACK_NO_BET,
    )
    assert explicit["selected_model"] == "no_bet"
    assert explicit["ready_for_purchase"] is False
    assert explicit["selection_reason"] == (
        "v12_not_ready_or_not_adopted_no_bet_contract"
    )

    unavailable = v12._closing_contract(
        _v12_artifact(ready=False, adopted=False),
        _v11_artifact(ready=False),
        fallback_policy=v12.CLOSING_FALLBACK_V11,
    )
    assert unavailable["selected_model"] == "no_bet"
    assert unavailable["selection_reason"].endswith("fallback_not_ready")


def test_strict_prior_forecast_uses_only_t300_and_identifies_v12(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    race = _race("2026-07-02")
    seen: list[str] = []

    def forecast(row, model, *, prediction_date):
        seen.append(prediction_date)
        return {
            "ready": True,
            "lower_final_odds": dict(row["odds"]),
            "used_checkpoint_offsets": [300],
            "future_checkpoint_offsets_used": [],
        }

    monkeypatch.setattr(v12, "forecast_closing_odds_t300_nonlinear_v12", forecast)
    forecasts, audit = v12._strict_prior_forecasts(
        [race],
        _v12_artifact(ready=True, adopted=True),
        _v11_artifact(ready=True),
        evaluation_date="2026-07-02",
        fallback_policy=v12.CLOSING_FALLBACK_V11,
    )

    assert seen == ["2026-07-02"]
    assert forecasts[race["race_id"]] == race["odds"]
    assert audit["ready_races"] == 1
    assert audit["future_checkpoint_violations"] == 0
    assert audit["closing_model_identity"]["selected_model"] == (
        v12.V12_CLOSING_MODEL_NAME
    )


def test_future_checkpoint_reaching_v12_stack_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    race = _race("2026-07-02")
    monkeypatch.setattr(
        v12,
        "forecast_closing_odds_t300_nonlinear_v12",
        lambda *args, **kwargs: {
            "ready": True,
            "lower_final_odds": dict(race["odds"]),
            "used_checkpoint_offsets": [300, 120],
            "future_checkpoint_offsets_used": [120],
        },
    )
    with pytest.raises(ValueError, match="future checkpoint"):
        v12._strict_prior_forecasts(
            [race],
            _v12_artifact(ready=True, adopted=True),
            None,
            evaluation_date="2026-07-02",
            fallback_policy=v12.CLOSING_FALLBACK_NO_BET,
        )


def test_no_bet_contract_does_not_invoke_any_closing_forecaster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("closing forecaster must not run for no-bet contract")

    monkeypatch.setattr(v12, "forecast_closing_odds_t300_nonlinear_v12", forbidden)
    monkeypatch.setattr(v12, "forecast_closing_odds_multihorizon_v11", forbidden)

    forecasts, audit = v12._strict_prior_forecasts(
        [_race("2026-07-02")],
        _v12_artifact(ready=True, adopted=False),
        _v11_artifact(ready=True),
        evaluation_date="2026-07-02",
        fallback_policy=v12.CLOSING_FALLBACK_NO_BET,
    )

    assert forecasts == {}
    assert audit["closing_model_identity"]["selected_model"] == "no_bet"
    assert audit["ready_races"] == 0


def test_walk_forward_keeps_strict_prior_purchase_then_settlement_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    races = [_race(f"2026-07-0{day}") for day in range(1, 4)]
    events: list[tuple[str, str]] = []
    estimator = object()
    forecast_models: list[dict[str, Any]] = []

    def trained_through(rows):
        return max(str(row["race_date"]) for row in rows) if rows else None

    monkeypatch.setattr(
        v12,
        "fit_odds_path_probability_v8",
        lambda rows: {
            "ready": True,
            "trained_through_date": trained_through(rows),
        },
    )
    monkeypatch.setattr(
        v12,
        "attach_odds_path_probability_v8",
        lambda rows, model: list(rows),
    )
    monkeypatch.setattr(v12, "_crossfit_probability_rows", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        v12,
        "fit_probability_lcb",
        lambda rows: {"ready": True, "trained_through_date": "2026-07-02"},
    )

    def fit_v12(rows, *, prediction_date, **kwargs):
        boundary = trained_through(rows)
        assert boundary is None or boundary < prediction_date
        return {
            **_v12_artifact(ready=True, adopted=True),
            "prediction_date": prediction_date,
            "trained_through_date": boundary,
            "point_model": {
                "engine": "test",
                "estimator": estimator,
            },
        }

    def fit_v11(rows, *, prediction_date, **kwargs):
        boundary = trained_through(rows)
        assert boundary is None or boundary < prediction_date
        return {
            **_v11_artifact(ready=True),
            "prediction_date": prediction_date,
            "trained_through_date": boundary,
        }

    monkeypatch.setattr(v12, "fit_closing_odds_t300_nonlinear_v12", fit_v12)
    monkeypatch.setattr(v12, "fit_closing_odds_multihorizon_v11", fit_v11)

    def forecast(race, model, **kwargs):
        assert model["point_model"]["estimator"] is estimator
        forecast_models.append(model)
        return {
            "ready": True,
            "lower_final_odds": dict(race["odds"]),
            "future_checkpoint_offsets_used": [],
        }

    monkeypatch.setattr(
        v12,
        "forecast_closing_odds_t300_nonlinear_v12",
        forecast,
    )

    def fit_conformal(observations, *, evaluation_date, **kwargs):
        assert all(str(row["race_date"]) < evaluation_date for row in observations)
        return {
            "ready": True,
            "haircut": 1.0,
            "trained_through_date": (
                max(str(row["race_date"]) for row in observations)
                if observations
                else None
            ),
        }

    monkeypatch.setattr(v12, "fit_selection_conformal_haircut", fit_conformal)

    def simulate(rows, **kwargs):
        evaluation_date = str(rows[0]["race_date"])
        events.append(("purchase", evaluation_date))
        return (
            {
                "daily": [{
                    "race_date": evaluation_date,
                    "evaluated_races": 1,
                    "tickets": 0,
                    "hit_tickets": 0,
                    "stake_yen": 0,
                    "return_yen": 0,
                    "profit_yen": 0,
                    "races_bet": 0,
                    "hit_races": 0,
                }],
                "selection_conformal": dict(kwargs["selection_conformal"]),
            },
            {},
        )

    monkeypatch.setattr(v12, "_simulate_selection_conformal_policy", simulate)

    def observe(observations, rows, *, evaluation_date, **kwargs):
        assert events[-1] == ("purchase", evaluation_date)
        events.append(("settlement", evaluation_date))
        observations.append({
            "race_date": evaluation_date,
            "race_id": rows[0]["race_id"],
            "combination": "123",
            "closing_ratio": 1.0,
        })
        return 1

    monkeypatch.setattr(v12, "_append_selection_observations", observe)
    probability = {
        "evaluated_races": 1,
        "calibrated_trifecta_log_loss": 3.0,
        "model_trifecta_log_loss": 3.1,
        "market_trifecta_log_loss": 3.2,
        "calibrated_trifecta_top5_hit_rate": 0.3,
        "model_trifecta_top5_hit_rate": 0.2,
        "market_trifecta_top5_hit_rate": 0.2,
        "winner_log_loss": 1.0,
        "winner_top1_accuracy": 0.4,
        "model_winner_log_loss": 1.1,
        "model_winner_top1_accuracy": 0.3,
        "market_winner_log_loss": 1.2,
        "market_winner_top1_accuracy": 0.2,
    }
    monkeypatch.setattr(v12, "probability_metrics", lambda rows: dict(probability))
    monkeypatch.setattr(
        v12,
        "_weighted_probability_metrics",
        lambda folds: dict(probability),
    )
    monkeypatch.setattr(
        v12,
        "closing_odds_t300_nonlinear_v12_metrics",
        lambda rows, model: {
            "model_name": v12.V12_CLOSING_MODEL_NAME,
            "evaluation_races": 1,
            "evaluation_tickets": 120,
            "lower_bound_coverage": 0.8,
        },
    )
    monkeypatch.setattr(
        v12,
        "_summarize_bankroll",
        lambda daily, **kwargs: {
            "tickets": 0,
            "stake_yen": 0,
            "return_yen": 0,
            "profit_yen": 0,
            "roi": 0.0,
            "daily": daily,
        },
    )
    monkeypatch.setattr(
        v12,
        "_aggregate_selection_conformal",
        lambda folds: {"evaluation_folds": len(folds), "ready_folds": len(folds)},
    )
    monkeypatch.setattr(v12, "_selection_coverage_gate", lambda summary: {})
    monkeypatch.setattr(
        v12,
        "_prospective_summary",
        lambda *args, **kwargs: {"promotion_gate": {}, "promotion_eligible": False},
    )

    result = v12.walk_forward_evaluate_v12(
        races,
        daily_budget_yen=10_000,
        min_calibration_days=2,
        evaluation_dates=["2026-07-03"],
    )

    json.dumps(result)
    assert result["evaluation_days"] == 1
    fold = result["folds"][0]
    assert forecast_models
    assert "estimator" not in fold["closing_model"]["point_model"]
    assert "estimator" not in fold["closing_v12_model"]["point_model"]
    assert "estimator" not in result["deployment_configuration"][
        "closing_t300_v12_model"
    ]["point_model"]
    assert fold["closing_model_identity"]["selected_model"] == (
        v12.V12_CLOSING_MODEL_NAME
    )
    assert fold["closing_t300_v12_metrics"]["evaluation_tickets"] == 120
    assert fold["closing_multihorizon_v11_metrics"] is None
    assert fold["leakage_guard"]["as_of_offset_seconds"] == 300
    assert fold["leakage_guard"]["future_checkpoint_imputation"] is False
    assert fold["leakage_guard"]["settlement_after_purchase_decision"] is True
    assert fold["leakage_guard"]["pass"] is True
    assert events.index(("purchase", "2026-07-03")) < events.index(
        ("settlement", "2026-07-03")
    )
    identity = result["closing_model_identity"]
    assert identity["v12_adopted_folds"] == 1
    assert identity["v11_fallback_folds"] == 0
    assert result["deployment_configuration"]["selected_policy"] == {
        "name": "no_bet",
        "no_bet": True,
    }


def test_market_calibration_dispatches_v12_and_keeps_v11_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = {"model": v12.MODEL_NAME}
    monkeypatch.setattr(
        v12,
        "walk_forward_evaluate_v12",
        lambda *args, **kwargs: {**sentinel, "kwargs": kwargs},
    )

    result = market_calibration.walk_forward_evaluate(
        [],
        calibrator_strategy=v12.STRATEGY_NAME,
        daily_budget_yen=10_000,
        min_calibration_days=2,
        v12_closing_fallback_policy="no_bet",
    )

    assert result["model"] == v12.MODEL_NAME
    assert result["kwargs"]["closing_fallback_policy"] == "no_bet"
    assert market_calibration.odds_path_model_name(v12.STRATEGY_NAME) == v12.MODEL_NAME
    choices = next(
        action.choices
        for action in market_calibration.build_parser()._actions
        if action.dest == "calibrator_strategy"
    )
    assert v12.STRATEGY_NAME in choices
    assert "odds_path_role_integrated_multihorizon_v11" in choices
