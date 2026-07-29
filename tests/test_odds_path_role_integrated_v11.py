from __future__ import annotations

from itertools import permutations
from typing import Any

import pytest

import boatrace_ai.discrete_log_allocation as allocation
from boatrace_ai.listwise import market_calibration
from boatrace_ai.listwise import odds_path_role_integrated_v11 as v11
from boatrace_ai.listwise.odds_path_selection_conformal_v10 import (
    _simulate_selection_conformal_policy,
)


COMBINATIONS = tuple(
    "-".join(map(str, values))
    for values in permutations(range(1, 7), 3)
)


def _probabilities(primary: str = "1-2-3") -> dict[str, float]:
    remainder = 0.80 / (len(COMBINATIONS) - 1)
    return {
        combination: 0.20 if combination == primary else remainder
        for combination in COMBINATIONS
    }


def _race(race_date: str, rno: int = 1) -> dict[str, Any]:
    odds = {combination: 10.0 for combination in COMBINATIONS}
    return {
        "race_id": f"{race_date}-01-{rno:02d}",
        "race_date": race_date,
        "jcd": "01",
        "rno": rno,
        "model_probabilities": _probabilities(),
        "market_probabilities": _probabilities("2-1-3"),
        "odds": odds,
        "closing_odds": dict(odds),
        "official_closing_odds": dict(odds),
        "actual_combination": "1-2-3",
        "actual_payout_yen": 1_000,
    }


def _forecast(*, ready: bool, future: list[int] | None = None) -> dict[str, Any]:
    return {
        "checkpoint_access_audit": {
            "future_checkpoint_offsets_used": list(future or []),
        },
        "predictions": {
            "t300": {
                "ready": ready,
                "lower_final_odds": {
                    combination: 10.0 for combination in COMBINATIONS
                }
                if ready
                else {},
                "future_checkpoint_offsets_used": [],
            }
        },
    }


def test_t300_forecast_is_explicit_and_missing_checkpoint_is_no_forecast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_forecast(race, model, **kwargs):
        calls.append(kwargs)
        return _forecast(ready=race["rno"] == 1)

    monkeypatch.setattr(v11, "forecast_closing_odds_multihorizon_v11", fake_forecast)
    races = [_race("2026-07-29", 1), _race("2026-07-29", 2)]

    forecasts, audit = v11._strict_prior_forecasts(
        races,
        {"model_name": "stub"},
        evaluation_date="2026-07-29",
    )

    assert set(forecasts) == {races[0]["race_id"]}
    assert audit["ready_races"] == 1
    assert audit["missing_t300_races"] == 1
    assert all(
        call["as_of_offset_seconds"] == v11.DECISION_OFFSET_SECONDS
        for call in calls
    )
    assert all(call["prediction_date"] == "2026-07-29" for call in calls)


def test_future_checkpoint_use_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        v11,
        "forecast_closing_odds_multihorizon_v11",
        lambda *args, **kwargs: _forecast(ready=True, future=[120]),
    )

    with pytest.raises(ValueError, match="future checkpoint"):
        v11._strict_prior_forecasts(
            [_race("2026-07-29")],
            {"model_name": "stub"},
            evaluation_date="2026-07-29",
        )


def test_missing_t300_produces_zero_bet() -> None:
    race = _race("2026-07-29")
    bankroll, diagnostic = _simulate_selection_conformal_policy(
        [race],
        closing_forecasts={},
        probability_lcb={
            "ready": True,
            "factors": {
                "top2": 1.0,
                "top5": 1.0,
                "top20": 1.0,
                "rest": 1.0,
            },
        },
        daily_budget_yen=10_000,
        selection_conformal={"ready": True, "haircut": 0.8},
    )

    assert bankroll["tickets"] == 0
    assert bankroll["stake_yen"] == 0
    assert diagnostic["closing_forecast_missing_races"] == 1


def test_v11_purchase_optimizer_never_receives_settlement_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[str, Any]] = []
    original = allocation._enumerate_race_options

    def inspect(race_id, candidates, **kwargs):
        for candidate in candidates:
            seen.append(dict(candidate.source))
            assert "actual_combination" not in candidate.source
            assert "actual_payout_yen" not in candidate.source
            assert "hit" not in candidate.source
        return original(race_id, candidates, **kwargs)

    monkeypatch.setattr(allocation, "_enumerate_race_options", inspect)
    race = _race("2026-07-29")
    bankroll, _diagnostic = _simulate_selection_conformal_policy(
        [race],
        closing_forecasts={race["race_id"]: dict(race["odds"])},
        probability_lcb={
            "ready": True,
            "factors": {
                "top2": 1.0,
                "top5": 1.0,
                "top20": 1.0,
                "rest": 1.0,
            },
        },
        daily_budget_yen=10_000,
        selection_conformal={
            "ready": True,
            "haircut": 1.0,
            "trained_through_date": "2026-07-28",
        },
    )

    assert seen
    assert bankroll["tickets"] > 0
    assert bankroll["return_yen"] > 0


def test_outer_days_fit_strict_prior_and_observe_after_purchase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = ["2026-07-26", "2026-07-27", "2026-07-28", "2026-07-29"]
    races = [_race(value) for value in dates]
    events: list[tuple[str, str]] = []

    def fit_probability(training):
        trained = max(str(row["race_date"]) for row in training)
        return {"ready": True, "trained_through_date": trained}

    monkeypatch.setattr(v11, "fit_odds_path_probability_v8", fit_probability)
    monkeypatch.setattr(
        v11,
        "attach_odds_path_probability_v8",
        lambda holdout, model: [dict(row) for row in holdout],
    )
    monkeypatch.setattr(v11, "_crossfit_probability_rows", lambda rows, **kwargs: rows)
    monkeypatch.setattr(
        v11,
        "fit_probability_lcb",
        lambda rows: {
            "ready": True,
            "trained_through_date": max(str(row["race_date"]) for row in rows),
            "factors": {
                "top2": 1.0,
                "top5": 1.0,
                "top20": 1.0,
                "rest": 1.0,
            },
        },
    )

    def fit_closing(training, *, prediction_date, **kwargs):
        training_dates = {str(row["race_date"]) for row in training}
        assert training_dates
        assert max(training_dates) < prediction_date
        events.append(("fit", prediction_date))
        return {
            "ready": True,
            "prediction_date": prediction_date,
            "trained_through_date": max(training_dates),
        }

    monkeypatch.setattr(v11, "fit_closing_odds_multihorizon_v11", fit_closing)
    monkeypatch.setattr(
        v11,
        "_strict_prior_forecasts",
        lambda transformed, model, *, evaluation_date: (
            {
                str(row["race_id"]): {
                    combination: 10.0 for combination in COMBINATIONS
                }
                for row in transformed
            },
            {
                "races": len(transformed),
                "ready_races": len(transformed),
                "missing_t300_races": 0,
                "incomplete_t300_races": 0,
                "future_checkpoint_violations": 0,
            },
        ),
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
            "selection_evaluation_candidates": 0,
        }

    monkeypatch.setattr(v11, "fit_selection_conformal_haircut", fit_conformal)

    def simulate(transformed, **kwargs):
        evaluation_date = str(transformed[0]["race_date"])
        artifact = kwargs["selection_conformal"]
        is_research = bool(artifact.get("research_only_non_deployable"))
        if is_research:
            assert artifact["ready"] is True
            assert artifact["haircut"] == 1.0
            assert (
                artifact["method"]
                == "fixed_identity_pre_selection_conformal_upper_bound"
            )
        events.append(("research" if is_research else "purchase", evaluation_date))
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

    monkeypatch.setattr(v11, "_simulate_selection_conformal_policy", simulate)

    def observe(observations, transformed, **kwargs):
        evaluation_date = kwargs["evaluation_date"]
        assert events[-1] == ("research", evaluation_date)
        events.append(("observe", evaluation_date))
        observations.append({
            "race_date": evaluation_date,
            "race_id": transformed[0]["race_id"],
            "combination": "1-2-3",
            "closing_ratio": 1.0,
        })
        return 1

    monkeypatch.setattr(v11, "_append_selection_observations", observe)
    probability_row = {
        "evaluated_races": 1,
        "calibrated_trifecta_log_loss": 1.0,
        "model_trifecta_log_loss": 1.1,
        "market_trifecta_log_loss": 1.2,
        "calibrated_trifecta_top5_hit_rate": 0.3,
        "model_trifecta_top5_hit_rate": 0.2,
        "market_trifecta_top5_hit_rate": 0.2,
        "winner_log_loss": 0.5,
        "winner_top1_accuracy": 0.5,
        "model_winner_log_loss": 0.6,
        "model_winner_top1_accuracy": 0.4,
        "market_winner_log_loss": 0.6,
        "market_winner_top1_accuracy": 0.4,
    }
    monkeypatch.setattr(v11, "probability_metrics", lambda rows: dict(probability_row))
    monkeypatch.setattr(
        v11,
        "closing_odds_multihorizon_v11_metrics",
        lambda *args, **kwargs: {
            "evaluation_races": 1,
            "evaluation_tickets": 120,
            "lower_bound_coverage": 0.8,
        },
    )
    monkeypatch.setattr(
        v11,
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
        v11,
        "_aggregate_selection_conformal",
        lambda folds: {"evaluation_folds": len(folds), "ready_folds": len(folds)},
    )
    monkeypatch.setattr(v11, "_selection_coverage_gate", lambda summary: {})

    result = v11.walk_forward_evaluate_v11(
        races,
        daily_budget_yen=10_000,
        min_calibration_days=2,
        evaluation_dates=["2026-07-28", "2026-07-29"],
    )

    assert result["evaluation_days"] == 2
    assert all(fold["leakage_guard"]["pass"] for fold in result["folds"])
    assert all(
        fold["leakage_guard"]["as_of_offset_seconds"] == 300
        for fold in result["folds"]
    )
    for evaluation_date in ("2026-07-28", "2026-07-29"):
        assert events.index(("purchase", evaluation_date)) < events.index(
            ("observe", evaluation_date)
        )
        assert events.index(("research", evaluation_date)) < events.index(
            ("observe", evaluation_date)
        )

    assert result["tickets"] == 0
    assert result["profit_yen"] == 0
    assert result["daily"] == [
        {
            "race_date": value,
            "evaluated_races": 1,
            "tickets": 0,
            "hit_tickets": 0,
            "stake_yen": 0,
            "return_yen": 0,
            "profit_yen": 0,
            "races_bet": 0,
            "hit_races": 0,
            "cumulative_profit_yen": 0,
        }
        for value in ("2026-07-28", "2026-07-29")
    ]
    assert all(
        fold["selected_policy"]["name"] != "research" for fold in result["folds"]
    )
    assert (
        result["deployment_configuration"]["selected_policy"]
        == {"name": "no_bet", "no_bet": True}
    )
    research = result["research_preconformal_upper_bound"]
    assert research["status"] == "research_only_non_deployable"
    assert research["research_only_non_deployable"] is True
    assert research["deployable"] is False
    assert research["included_in_promotion_gate"] is False
    assert research["included_in_deployment_selected_policy"] is False
    assert research["included_in_operational_decision"] is False
    assert research["eligible_dates"] == ["2026-07-28", "2026-07-29"]
    assert research["fixed_selection_conformal"]["ready"] is True
    assert research["fixed_selection_conformal"]["haircut"] == 1.0


def test_research_preconformal_summary_reports_bankroll_and_largest_hit_exclusion(
) -> None:
    daily = [{
        "race_date": "2026-07-29",
        "evaluated_races": 2,
        "tickets": 2,
        "hit_tickets": 1,
        "stake_yen": 1_000,
        "return_yen": 3_000,
        "profit_yen": 2_000,
        "races_bet": 1,
        "hit_races": 1,
        "largest_hit_return_yen": 3_000,
        "hit_return_square_sum_yen2": 9_000_000,
    }]

    result = v11._research_preconformal_summary(
        daily,
        evaluated_races=2,
        eligible_dates=["2026-07-29"],
        skipped_dates=[],
        diagnostics_by_date={},
    )

    assert result["daily"][0]["cumulative_profit_yen"] == 2_000
    assert result["roi"] == 3.0
    assert result["profit_yen"] == 2_000
    assert result["tickets"] == 2
    assert result["largest_hit_return_yen"] == 3_000
    assert result["roi_without_largest_hit"] == 0.0
    assert result["profit_without_largest_hit_yen"] == -1_000


def test_market_calibration_routes_v11_without_changing_v10(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = {"model": v11.MODEL_NAME}
    monkeypatch.setattr(v11, "walk_forward_evaluate_v11", lambda *args, **kwargs: sentinel)

    result = market_calibration.walk_forward_evaluate(
        [],
        calibrator_strategy=v11.STRATEGY_NAME,
        daily_budget_yen=10_000,
        min_calibration_days=2,
    )

    assert result is sentinel
    assert market_calibration.odds_path_model_name(v11.STRATEGY_NAME) == v11.MODEL_NAME
    choices = next(
        action.choices
        for action in market_calibration.build_parser()._actions
        if action.dest == "calibrator_strategy"
    )
    assert v11.STRATEGY_NAME in choices
    assert (
        market_calibration.odds_path_model_name(
            "odds_path_market_offset_selection_conformal_discrete_ev_v10"
        )
        == "odds_path_market_offset_selection_conformal_discrete_ev_v10"
    )


def test_actual_and_payout_changes_cannot_change_purchase_decisions() -> None:
    base = _race("2026-07-29")
    changed = {
        **base,
        "actual_combination": "6-5-4",
        "actual_payout_yen": 999_900,
    }
    probability_lcb = {
        "ready": True,
        "factors": {
            "top2": 1.0,
            "top5": 1.0,
            "top20": 1.0,
            "rest": 1.0,
        },
    }
    conformal = {
        "ready": True,
        "haircut": 1.0,
        "trained_through_date": "2026-07-28",
    }

    def decisions(race: dict[str, Any]) -> list[tuple[str, int, float]]:
        bankroll, _diagnostic = _simulate_selection_conformal_policy(
            [race],
            closing_forecasts={race["race_id"]: dict(race["odds"])},
            probability_lcb=probability_lcb,
            daily_budget_yen=10_000,
            selection_conformal=conformal,
        )
        return [
            (
                str(row["combination"]),
                int(row["stake_yen"]),
                float(row["estimated_ev"]),
            )
            for row in bankroll["daily"][0]["selected_sample"]
        ]

    assert decisions(base) == decisions(changed)
