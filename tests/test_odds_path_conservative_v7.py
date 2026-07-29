from __future__ import annotations

from itertools import permutations

import pytest

import boatrace_ai.listwise.odds_path_conservative_v7 as v7


COMBINATIONS = tuple(
    "-".join(map(str, values))
    for values in permutations(range(1, 7), 3)
)


def _distribution(
    primary: str, probability: float
) -> dict[str, float]:
    remainder = (1.0 - probability) / (len(COMBINATIONS) - 1)
    return {
        combination: (
            probability if combination == primary else remainder
        )
        for combination in COMBINATIONS
    }


def _race(
    race_date: str,
    rno: int,
    *,
    winner: str = "1-2-3",
    with_closing: bool = True,
) -> dict:
    base = _distribution(winner, 0.30)
    market = _distribution(winner, 0.20)
    odds = {
        combination: 1.0 / probability / 1.25
        for combination, probability in market.items()
    }
    path = [
        {
            "minutes_before_decision": 10.0,
            "market_probabilities": dict(market),
        },
        {
            "minutes_before_decision": 5.0,
            "market_probabilities": {
                combination: (
                    probability * (1.02 if combination == winner else 1.0)
                )
                for combination, probability in market.items()
            },
        },
    ]
    race = {
        "race_id": f"{race_date}-01-{rno:02d}",
        "race_date": race_date,
        "jcd": "01",
        "rno": rno,
        "actual_combination": winner,
        "actual_payout_yen": 500,
        "model_probabilities": base,
        "market_probabilities": market,
        "odds": odds,
        "odds_path": path,
        "odds_path_points": len(path),
        "snapshot_id": rno,
    }
    if with_closing:
        race["closing_odds"] = {
            combination: value * (
                0.90 if combination == winner else 0.95
            )
            for combination, value in odds.items()
        }
        race["closing_odds_changed"] = True
    return race


def test_v7_probability_model_is_pure_residual_without_return_features() -> None:
    races = [
        _race("2026-07-20", 1),
        _race("2026-07-21", 1, winner="2-1-3"),
    ]

    model = v7.fit_t5_residual_probability_model(races)
    attached = v7.attach_t5_residual_probabilities(
        [_race("2026-07-22", 1)], model
    )[0]

    assert model["base_offset_prior"] == 0.0
    assert model["market_offset_prior"] == 0.0
    assert model["uses_return_multiplier"] is False
    assert model["uses_historical_hit_lift"] is False
    assert "historical_hit_lift" not in model["feature_names"]
    assert "return_multiplier" not in " ".join(model["feature_names"])
    assert "historical_return_multipliers" not in attached
    assert sum(attached["model_probabilities"].values()) == pytest.approx(1.0)
    assert len(attached["model_probabilities"]) == 120


def test_v7_closing_model_predicts_crossfit_log_ratio_q20() -> None:
    races = [
        _race(race_date, rno)
        for race_date in ("2026-07-20", "2026-07-21")
        for rno in range(1, 3)
    ]

    model = v7.fit_closing_log_ratio_q20_model(races)
    forecast = v7.forecast_closing_q20(
        _race("2026-07-22", 1), model
    )

    assert model["teacher"] == "log(closing_odds / t5_odds)"
    assert model["quantile"] == 0.20
    assert model["crossfit_days"] == 2
    assert model["trained_through_date"] == "2026-07-21"
    assert len(forecast) == 120
    assert all(value > 0.0 for value in forecast.values())


def test_v7_uses_no_bet_for_every_fold_before_closing_readiness() -> None:
    races = [
        _race(race_date, rno)
        for race_date in (
            "2026-07-20",
            "2026-07-21",
            "2026-07-22",
            "2026-07-23",
        )
        for rno in range(1, 3)
    ]

    result = v7.walk_forward_evaluate_v7(
        races,
        daily_budget_yen=10_000,
        min_calibration_days=2,
    )

    assert result["evaluation_days"] == 2
    assert all(fold["closing_ready"] is False for fold in result["folds"])
    assert all(
        fold["selected_policy"] == {"name": "no_bet", "no_bet": True}
        for fold in result["folds"]
    )
    assert result["tickets"] == 0
    assert result["stake_yen"] == 0


def test_v7_fixed_policy_enforces_two_ticket_and_exposure_caps() -> None:
    race = _race("2026-07-30", 1)
    probabilities = _distribution("1-2-3", 0.20)
    probabilities["1-3-2"] = 0.15
    race["model_probabilities"] = probabilities
    closing = {combination: 2.0 for combination in COMBINATIONS}
    closing["1-2-3"] = 10.0
    closing["1-3-2"] = 10.0
    closing["2-1-3"] = 10.0

    result = v7.simulate_fixed_safe_ev_policy(
        [race],
        closing_forecasts={race["race_id"]: closing},
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
    )

    assert result["tickets"] == 2
    assert result["stake_yen"] == 200
    assert result["daily"][0]["max_stake_yen"] == 100
    assert result["daily"][0]["budget_used_fraction"] <= 0.20
    assert result["selected_races"] == 1


def test_v7_purchase_diagnostics_explain_missing_inputs() -> None:
    missing_closing = _race("2026-07-30", 1)
    complete_closing = _race("2026-07-30", 2)
    closing = {combination: 10.0 for combination in COMBINATIONS}

    result = v7.simulate_fixed_safe_ev_policy(
        [missing_closing, complete_closing],
        closing_forecasts={complete_closing["race_id"]: closing},
        probability_lcb={"ready": False, "factors": {}},
        daily_budget_yen=10_000,
    )

    diagnostic = result["purchase_decision_diagnostics"]
    assert diagnostic["total_races"] == 2
    assert diagnostic["closing_forecast_missing_races"] == 1
    assert diagnostic["lcb_not_ready_races"] == 2
    assert diagnostic["evaluated_combinations"] == 0
    assert diagnostic["safe_ev_max"] is None
    assert diagnostic["safe_ev_p50"] is None
    assert diagnostic["safe_ev_p90"] is None
    assert diagnostic["safe_ev_p95"] is None
    assert diagnostic["safe_ev_p99"] is None
    assert diagnostic["safe_ev_at_least"] == {
        "1.00": 0,
        "1.02": 0,
        "1.05": 0,
        "1.10": 0,
    }
    assert diagnostic["threshold_pass_candidates"] == 0
    assert diagnostic["candidates_after_race_cap"] == 0
    assert diagnostic["purchases_after_allocation"] == 0
    assert result["tickets"] == 0
    assert result["stake_yen"] == 0


def test_v7_purchase_diagnostics_measure_every_safe_ev_stage() -> None:
    race = _race("2026-07-30", 1)
    residual_probability = 0.20 / (len(COMBINATIONS) - 2)
    race["model_probabilities"] = {
        combination: (
            0.45
            if index == 0
            else 0.35
            if index == 1
            else residual_probability
        )
        for index, combination in enumerate(COMBINATIONS)
    }
    targets = (
        [1.101] * 20
        + [1.051] * 10
        + [1.021] * 10
        + [1.001] * 20
        + [0.99] * 60
    )
    closing = {
        combination: target / race["model_probabilities"][combination]
        for combination, target in zip(COMBINATIONS, targets, strict=True)
    }

    result = v7.simulate_fixed_safe_ev_policy(
        [race],
        closing_forecasts={race["race_id"]: closing},
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
    )

    diagnostic = result["purchase_decision_diagnostics"]
    assert diagnostic["total_races"] == 1
    assert diagnostic["closing_forecast_missing_races"] == 0
    assert diagnostic["lcb_not_ready_races"] == 0
    assert diagnostic["evaluated_combinations"] == 120
    assert diagnostic["safe_ev_max"] == pytest.approx(1.101)
    assert diagnostic["safe_ev_p50"] == pytest.approx(0.9955)
    assert diagnostic["safe_ev_p90"] == pytest.approx(1.101)
    assert diagnostic["safe_ev_p95"] == pytest.approx(1.101)
    assert diagnostic["safe_ev_p99"] == pytest.approx(1.101)
    assert diagnostic["safe_ev_at_least"] == {
        "1.00": 60,
        "1.02": 40,
        "1.05": 30,
        "1.10": 20,
    }
    assert diagnostic["threshold_pass_candidates"] == 30
    assert diagnostic["candidates_after_race_cap"] == 2
    assert diagnostic["purchases_after_allocation"] == 2
    assert result["tickets"] == 2
    assert result["stake_yen"] == 200


def test_v7_outer_models_and_lcb_never_cross_holdout_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(v7, "MIN_CLOSING_TRAINING_DAYS", 2)
    monkeypatch.setattr(v7, "MIN_CLOSING_TRAINING_RACES", 4)
    races = [
        _race(race_date, rno)
        for race_date in (
            "2026-07-28",
            "2026-07-29",
            "2026-07-30",
            "2026-07-31",
        )
        for rno in range(1, 3)
    ]

    result = v7.walk_forward_evaluate_v7(
        races,
        daily_budget_yen=10_000,
        min_calibration_days=2,
    )

    assert result["model"] == v7.MODEL_NAME
    assert all(
        fold["leakage_guard"]["pass"] is True
        for fold in result["folds"]
    )
    assert result["folds"][0]["operational_model"][
        "trained_through_date"
    ] == "2026-07-29"
    assert result["folds"][0]["closing_model"][
        "trained_through_date"
    ] == "2026-07-29"
    prospective = result[
        "prospective_crossfit_conservative_ev_v7_walk_forward"
    ]
    assert prospective["registered_after"] == "2026-07-29"
    assert prospective["evaluation_days"] == 2
    assert prospective["promotion_eligible"] is False
    assert set(prospective["promotion_gate"]) >= {
        "sample_days_pass",
        "sample_races_pass",
        "sample_tickets_pass",
        "largest_hit_excluded_roi_pass",
        "cluster_bootstrap_roi_pass",
        "effective_hits_pass",
        "profitable_day_fraction_pass",
        "largest_hit_share_pass",
        "probability_log_loss_pass",
        "quantile_coverage_pass",
        "no_lookahead_pass",
    }
    fold_diagnostics = [
        fold["bankroll"]["purchase_decision_diagnostics"]
        for fold in result["folds"]
    ]
    aggregate = result["purchase_decision_diagnostics"]
    prospective_diagnostic = prospective[
        "purchase_decision_diagnostics"
    ]
    assert [row["total_races"] for row in fold_diagnostics] == [2, 2]
    assert aggregate["total_races"] == 4
    assert aggregate["evaluated_combinations"] == sum(
        row["evaluated_combinations"] for row in fold_diagnostics
    )
    assert aggregate["threshold_pass_candidates"] == sum(
        row["threshold_pass_candidates"] for row in fold_diagnostics
    )
    assert aggregate["candidates_after_race_cap"] == sum(
        row["candidates_after_race_cap"] for row in fold_diagnostics
    )
    assert aggregate["purchases_after_allocation"] == result["tickets"]
    assert prospective_diagnostic == aggregate
