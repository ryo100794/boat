from __future__ import annotations

import numpy as np
import pytest

import boatrace_ai.listwise.direct_bankroll as direct_bankroll
from boatrace_ai.listwise.direct_bankroll import (
    COMBINATION_LABELS,
    _select_conditional_payout_policy,
    bootstrap_daily_bankroll,
    direct_candidates,
    simulate_conditional_payout_walk_forward,
    simulate_direct_bankroll,
    standard_direct_policy,
)


def test_daily_bankroll_bootstrap_reports_absolute_and_paired_confidence() -> None:
    candidate = [
        {
            "race_date": f"2026-07-{day:02d}",
            "stake_yen": 1_000,
            "return_yen": 1_500,
        }
        for day in range(1, 6)
    ]
    baseline = [
        {
            "race_date": f"2026-07-{day:02d}",
            "stake_yen": 1_000,
            "return_yen": 1_100,
        }
        for day in range(1, 6)
    ]

    result = bootstrap_daily_bankroll(
        candidate,
        baseline_daily=baseline,
        samples=500,
    )

    assert result["days"] == 5
    assert result["roi"] == 1.5
    assert result["roi_ci95_lower"] == 1.5
    assert result["profit_ci95_lower_yen"] == 2_500
    assert np.isclose(result["roi_delta"], 0.4)
    assert result["roi_delta_ci95_lower"] > 0.39
    assert result["probability_roi_above_one"] == 1.0
    assert result["probability_roi_delta_above_zero"] == 1.0


def test_daily_bankroll_bootstrap_handles_no_bet_days() -> None:
    candidate = [
        {"race_date": "2026-07-01", "stake_yen": 0, "return_yen": 0},
        {"race_date": "2026-07-02", "stake_yen": 1_000, "return_yen": 1_400},
    ]
    baseline = [
        {"race_date": "2026-07-01", "stake_yen": 0, "return_yen": 0},
        {"race_date": "2026-07-02", "stake_yen": 1_000, "return_yen": 1_100},
    ]

    result = bootstrap_daily_bankroll(
        candidate,
        baseline_daily=baseline,
        samples=500,
    )

    assert np.isfinite(result["roi_ci95_lower"])
    assert np.isfinite(result["roi_delta_ci95_lower"])
    assert 0.0 < result["probability_roi_above_one"] < 1.0


def test_daily_bankroll_bootstrap_returns_zero_roi_when_no_bets_exist() -> None:
    daily = [
        {"race_date": "2026-07-01", "stake_yen": 0, "return_yen": 0},
        {"race_date": "2026-07-02", "stake_yen": 0, "return_yen": 0},
    ]

    result = bootstrap_daily_bankroll(
        daily,
        baseline_daily=daily,
        samples=500,
    )

    assert result["roi"] == 0.0
    assert result["roi_ci95_lower"] == 0.0
    assert result["roi_delta"] == 0.0
    assert result["probability_roi_above_one"] == 0.0


def _payout_model() -> dict[str, dict[str, float]]:
    return {
        combination: {
            "estimated_odds": 10.0,
            "estimated_payout_yen": 1_000.0,
            "history_count": 100.0,
        }
        for combination in COMBINATION_LABELS
    }


def test_direct_candidates_use_exact_trifecta_probabilities() -> None:
    probabilities = np.zeros(120, dtype=np.float64)
    target = COMBINATION_LABELS.index("1-2-3")
    probabilities[target] = 0.20
    probabilities[COMBINATION_LABELS.index("1-3-2")] = 0.80

    candidates = direct_candidates(
        probabilities,
        race_key=("race-1", "2026-07-20", "01", 1),
        actual={"combination": "1-2-3", "payout_yen": 1_000},
        payout_model=_payout_model(),
        ev_threshold=1.20,
    )

    by_combination = {row["combination"]: row for row in candidates}
    assert by_combination["1-2-3"]["probability"] == 0.20
    assert by_combination["1-2-3"]["estimated_ev"] == 2.0
    assert by_combination["1-2-3"]["hit"] is True
    assert by_combination["1-3-2"]["hit"] is False


def test_direct_bankroll_uses_fixed_daily_policy_and_settles_returns() -> None:
    race_keys = [
        ("train", "2026-07-19", "01", 1),
        ("test", "2026-07-20", "01", 1),
    ]
    payouts = {
        "train": {
            "combination": "1-2-3",
            "payout_yen": 1_000,
        },
        "test": {
            "combination": "1-2-3",
            "payout_yen": 1_000,
        },
    }
    probabilities = np.full((2, 120), 1e-9, dtype=np.float64)
    probabilities[:, COMBINATION_LABELS.index("1-2-3")] = 1.0

    result = simulate_direct_bankroll(
        probabilities[1:],
        race_keys=race_keys[1:],
        payouts=payouts,
        training_races={"train"},
    )

    assert result["policy"] == standard_direct_policy()
    assert result["evaluated_races"] == 1
    assert result["selected_tickets"] == 1
    assert result["hit_tickets"] == 1
    assert result["stake_yen"] == 300
    assert result["return_yen"] == 3_000
    assert result["roi"] == 10.0
    attribution = result["ticket_roi_attribution"]
    dimensions = {row["dimension"]: row for row in attribution["dimensions"]}
    assert dimensions["first_lane"]["buckets"][0]["bucket"] == "1"
    assert dimensions["first_lane"]["buckets"][0]["tickets"] == 1
    assert attribution["fold_stability"]["folds"] == 1



def test_conditional_payout_policy_selection_uses_pre_evaluation_days() -> None:
    target_index = COMBINATION_LABELS.index("1-2-3")
    calibration_keys = [
        (f"cal-{day}-{race}", f"2026-06-{day:02d}", "01", race % 12 + 1)
        for day in range(1, 5)
        for race in range(30)
    ]
    probabilities = np.full((120, 120), 0.9 / 119.0)
    probabilities[:, target_index] = 0.1
    payouts = {
        race_key[0]: {"combination": "1-2-3", "payout_yen": 2_000}
        for race_key in calibration_keys
    }

    selected = _select_conditional_payout_policy(
        probabilities,
        probabilities,
        calibration_keys,
        payouts,
        selection_days=2,
        base_policy=standard_direct_policy(),
        fallback_ridge=10.0,
        ridge_candidates=(10.0,),
        correction_candidates=(0.0,),
        threshold_candidates=(1.20,),
        minimum_tickets=10,
        minimum_hits=5,
        minimum_winning_days=1,
        minimum_roi=1.05,
    )
    assert len(selected) == 6
    ridge, correction, threshold, source, diagnostics, period = selected

    assert ridge == 10.0
    assert correction == 0.0
    assert threshold == 1.20
    assert source == "pre_evaluation_adaptive_selection"
    assert {
        row["min_daily_exposure_fraction"] for row in diagnostics
    } == {0.0, 0.1, 0.2, 0.4}
    assert diagnostics[0]["tickets"] >= 10
    assert diagnostics[0]["roi"] > 1.0
    assert diagnostics[0]["tail_eligible_candidates"] == 3_600
    assert diagnostics[0]["tail_ineligible_candidates"] == 3_600
    assert diagnostics[0]["tail_eligibility_daily"] == [
        {
            "race_date": "2026-06-03",
            "eligible_candidates": 0,
            "ineligible_candidates": 3_600,
        },
        {
            "race_date": "2026-06-04",
            "eligible_candidates": 3_600,
            "ineligible_candidates": 0,
        },
    ]
    assert period["fit_through"] == "2026-06-02"
    assert period["selection_from"] == "2026-06-03"


def test_conditional_payout_walk_forward_adds_results_only_after_each_day() -> None:
    target_index = COMBINATION_LABELS.index("1-2-3")
    calibration_keys = [
        (
            f"cal-{index}",
            "2026-06-01",
            f"{index % 24 + 1:02d}",
            index % 12 + 1,
        )
        for index in range(60)
    ]
    calibration_probabilities = np.full((60, 120), 0.8 / 119.0)
    target_probabilities = np.linspace(0.05, 0.20, 60)
    calibration_probabilities[:, target_index] = target_probabilities
    calibration_probabilities /= calibration_probabilities.sum(axis=1, keepdims=True)
    race_keys = [
        ("day-1", "2026-07-01", "01", 1),
        ("day-2", "2026-07-02", "01", 1),
    ]
    probabilities = np.full((2, 120), 0.9 / 119.0)
    probabilities[:, target_index] = 0.1
    payouts = {
        race_id: {
            "combination": "1-2-3",
            "payout_yen": int(round(200.0 / probability)),
        }
        for (race_id, _date, _jcd, _rno), probability in zip(
            calibration_keys,
            target_probabilities,
        )
    }
    payouts.update(
        {
            "day-1": {"combination": "1-2-3", "payout_yen": 2_000},
            "day-2": {"combination": "1-2-3", "payout_yen": 2_000},
        }
    )
    market_probabilities = np.full((2, 120), 0.95 / 119.0)
    market_probabilities[:, target_index] = 0.05

    result = simulate_conditional_payout_walk_forward(
        probabilities,
        race_keys=race_keys,
        payouts=payouts,
        calibration_probabilities=calibration_probabilities,
        calibration_race_keys=calibration_keys,
        market_reference_probabilities=market_probabilities,
        calibration_market_reference_probabilities=calibration_probabilities,
    )

    assert result["payout_training_samples_initial"] == 60
    assert result["payout_training_samples_final"] == 62
    assert [row["payout_training_samples"] for row in result["daily"]] == [60, 61]
    assert result["evaluated_races"] == 2
    assert result["policy"]["market_reference"] == "fixed baseline probability"
    assert result["policy_selection"]["source"] == "fallback_fixed_policy"
    diagnostics = result["payout_diagnostics"]
    assert diagnostics["feature_schema"] == "conditional_payout_additive_v1"
    assert diagnostics["feature_count"] == 54
    assert diagnostics["candidate_combinations"] == 240
    assert np.isfinite(diagnostics["max_estimated_ev"])
    assert diagnostics["max_estimated_ev"] > 1.2
    counts = diagnostics["estimated_ev_at_least"]
    assert counts["0.80"] >= counts["0.90"] >= counts["1.00"]
    assert counts["1.00"] >= counts["1.05"] >= counts["1.10"] >= counts["1.20"]
    assert diagnostics["residual_variance_initial"] >= 0.0
    assert diagnostics["residual_variance_final"] >= 0.0


def _tail_calibration_case(
    *,
    first_evaluation_payout_yen: int = 2_000,
    same_day_reversed: bool = False,
    policy: dict[str, object] | None = None,
) -> dict[str, object]:
    target = COMBINATION_LABELS.index("1-2-3")
    calibration_keys = [
        (f"cal-{day}", f"2026-06-{day:02d}", "01", 1)
        for day in range(1, 5)
    ]
    calibration_probabilities = np.full((4, 120), 0.85 / 119.0)
    calibration_probabilities[:, target] = 0.15
    race_keys = [
        ("eval-a", "2026-07-01", "01", 1),
        ("eval-b", "2026-07-01", "02", 1),
        ("eval-c", "2026-07-02", "01", 1),
    ]
    probabilities = np.full((3, 120), 0.85 / 119.0)
    probabilities[:, target] = 0.15
    if same_day_reversed:
        race_keys = [race_keys[1], race_keys[0], race_keys[2]]
        probabilities = probabilities[[1, 0, 2]]
    payouts = {
        key[0]: {"combination": "1-2-3", "payout_yen": 2_000}
        for key in calibration_keys
    }
    payouts.update(
        {
            "eval-a": {
                "combination": "1-2-3",
                "payout_yen": first_evaluation_payout_yen,
            },
            "eval-b": {"combination": "1-2-3", "payout_yen": 2_000},
            "eval-c": {"combination": "1-2-3", "payout_yen": 2_000},
        }
    )
    return simulate_conditional_payout_walk_forward(
        probabilities,
        race_keys=race_keys,
        payouts=payouts,
        calibration_probabilities=calibration_probabilities,
        calibration_race_keys=calibration_keys,
        policy=policy,
        policy_selection_days=2,
        minimum_selection_tickets=10_000,
    )


def test_conditional_payout_rejects_decreasing_dates() -> None:
    probabilities = np.full((2, 120), 1.0 / 120.0)
    calibration_keys = [
        ("cal-1", "2026-06-01", "01", 1),
        ("cal-2", "2026-06-02", "01", 1),
    ]
    payouts = {
        key[0]: {"combination": "1-2-3", "payout_yen": 2_000}
        for key in calibration_keys
    }
    with pytest.raises(ValueError, match="race_keys dates must be non-decreasing"):
        simulate_conditional_payout_walk_forward(
            probabilities,
            race_keys=[
                ("eval-2", "2026-07-02", "01", 1),
                ("eval-1", "2026-07-01", "01", 1),
            ],
            payouts=payouts,
            calibration_probabilities=probabilities,
            calibration_race_keys=calibration_keys,
        )


def test_conditional_payout_day_result_only_changes_following_day_prediction() -> None:
    high = _tail_calibration_case(first_evaluation_payout_yen=20_000)
    low = _tail_calibration_case(first_evaluation_payout_yen=110)

    high_day, high_next = high["daily"]
    low_day, low_next = low["daily"]
    assert high_day["tickets"] == low_day["tickets"]
    assert high_day["stake_yen"] == low_day["stake_yen"]
    assert high_day["tail_calibration_bin_factors_initial"] == (
        low_day["tail_calibration_bin_factors_initial"]
    )
    assert high_next["tail_calibration_bin_factors_initial"] != (
        low_next["tail_calibration_bin_factors_initial"]
    )
    selection_keys = (
        "selected_ridge",
        "selected_ev_threshold",
        "selected_min_daily_exposure_fraction",
    )
    assert {
        key: high["policy_selection"][key] for key in selection_keys
    } == {
        key: low["policy_selection"][key] for key in selection_keys
    }


def test_conditional_payout_same_day_order_does_not_change_purchases() -> None:
    original = _tail_calibration_case()
    reversed_rows = _tail_calibration_case(same_day_reversed=True)

    original_day = original["daily"][0]
    reversed_day = reversed_rows["daily"][0]
    for key in ("tickets", "stake_yen", "return_yen", "profit_yen"):
        assert original_day[key] == reversed_day[key]
    assert original_day["selected_sample"] == reversed_day["selected_sample"]


def _selection_diagnostic(
    *,
    roi: float,
    profit_yen: int,
    exposure: float,
    tickets: int,
) -> dict[str, object]:
    return {
        "ridge": 10.0,
        "mean_correction_factor": 0.0,
        "ev_threshold": 1.2,
        "min_daily_exposure_fraction": exposure,
        "tickets": tickets,
        "selected_races": 1,
        "hits": 1,
        "stake_yen": 100,
        "return_yen": 100 + profit_yen,
        "profit_yen": profit_yen,
        "roi": roi,
        "winning_days": 1,
        "losing_days": 0,
        "max_drawdown_yen": 0,
    }


@pytest.mark.parametrize(
    ("diagnostics", "expected_exposure"),
    [
        (
            [
                _selection_diagnostic(
                    roi=1.2, profit_yen=10_000, exposure=0.0, tickets=999
                ),
                _selection_diagnostic(
                    roi=1.3, profit_yen=1, exposure=0.4, tickets=1
                ),
            ],
            0.4,
        ),
        (
            [
                _selection_diagnostic(
                    roi=1.3, profit_yen=100, exposure=0.0, tickets=999
                ),
                _selection_diagnostic(
                    roi=1.3, profit_yen=200, exposure=0.4, tickets=1
                ),
            ],
            0.4,
        ),
        (
            [
                _selection_diagnostic(
                    roi=1.3, profit_yen=200, exposure=0.0, tickets=1
                ),
                _selection_diagnostic(
                    roi=1.3, profit_yen=200, exposure=0.4, tickets=999
                ),
            ],
            0.0,
        ),
    ],
)
def test_conditional_policy_tie_break_uses_roi_profit_then_low_exposure(
    monkeypatch: pytest.MonkeyPatch,
    diagnostics: list[dict[str, object]],
    expected_exposure: float,
) -> None:
    def fake_selection_walk_forward(*args: object, **kwargs: object):
        return (
            diagnostics,
            direct_bankroll.ConditionalPayoutStatistics.empty(),
            direct_bankroll.ConditionalPayoutTailCalibrator.empty(),
        )

    monkeypatch.setattr(
        direct_bankroll,
        "_selection_walk_forward_for_ridge",
        fake_selection_walk_forward,
    )
    race_keys = [
        (f"cal-{day}", f"2026-06-{day:02d}", "01", 1)
        for day in range(1, 5)
    ]
    probabilities = np.full((4, 120), 1.0 / 120.0)

    selected = direct_bankroll._select_conditional_payout_policy_state(
        probabilities,
        probabilities,
        race_keys,
        {},
        selection_days=2,
        base_policy=standard_direct_policy(),
        fallback_ridge=10.0,
        ridge_candidates=(10.0,),
        correction_candidates=(0.0,),
        threshold_candidates=(1.2,),
        minimum_tickets=0,
        minimum_hits=0,
        minimum_winning_days=0,
        minimum_roi=0.0,
    )

    assert selected[8] == expected_exposure


def test_conditional_policy_overrides_legacy_forced_exposure() -> None:
    legacy_policy = standard_direct_policy()
    legacy_policy["min_daily_exposure_fraction"] = 0.4

    result = _tail_calibration_case(policy=legacy_policy)

    assert result["policy"]["min_daily_exposure_fraction"] == 0.0
    assert (
        result["policy_selection"]["selected_min_daily_exposure_fraction"]
        == 0.0
    )


def test_min_zero_does_not_expand_weak_edges_to_four_thousand_yen() -> None:
    policy = standard_direct_policy()
    policy["min_daily_exposure_fraction"] = 0.0
    candidates = [
        {
            "race_id": f"weak-{index}",
            "combination": "1-2-3",
            "probability": 0.14,
            "estimated_odds": 10.0,
            "estimated_ev": 1.4,
            "actual_payout_yen": 0,
            "hit": False,
        }
        for index in range(14)
    ]

    result = direct_bankroll.allocate_adaptive_day(
        "2026-07-20",
        candidates,
        {str(row["race_id"]) for row in candidates},
        daily_budget_yen=int(policy["daily_budget_yen"]),
        fractional_kelly=float(policy["fractional_kelly"]),
        max_daily_exposure_fraction=float(policy["max_daily_exposure_fraction"]),
        min_daily_exposure_fraction=float(policy["min_daily_exposure_fraction"]),
        race_cap_fraction=float(policy["race_cap_fraction"]),
        ticket_cap_fraction=float(policy["ticket_cap_fraction"]),
        max_daily_tickets=int(policy["max_daily_tickets"]),
        allocation_mode=str(policy["allocation_mode"]),
        stake_granularity_yen=int(policy["stake_granularity_yen"]),
        min_stake_yen=int(policy["min_stake_yen"]),
    )

    assert policy["min_daily_exposure_fraction"] == 0.0
    assert result["stake_yen"] == 1_400
    assert result["stake_yen"] < 4_000


def _adaptive_selection_with_evaluation_payout(
    evaluation_payout_yen: int,
) -> dict[str, object]:
    target = COMBINATION_LABELS.index("1-2-3")
    calibration_keys = [
        (f"cal-{day}-{race}", f"2026-06-{day:02d}", "01", race % 12 + 1)
        for day in range(1, 5)
        for race in range(30)
    ]
    calibration_probabilities = np.full((120, 120), 0.9 / 119.0)
    calibration_probabilities[:, target] = 0.1
    evaluation_probabilities = calibration_probabilities[:1].copy()
    payouts = {
        race_key[0]: {"combination": "1-2-3", "payout_yen": 2_000}
        for race_key in calibration_keys
    }
    payouts["evaluation"] = {
        "combination": "1-2-3",
        "payout_yen": evaluation_payout_yen,
    }

    return simulate_conditional_payout_walk_forward(
        evaluation_probabilities,
        race_keys=[("evaluation", "2026-07-01", "01", 1)],
        payouts=payouts,
        calibration_probabilities=calibration_probabilities,
        calibration_race_keys=calibration_keys,
        ridge_candidates=(10.0,),
        threshold_candidates=(1.2,),
        policy_selection_days=2,
        minimum_selection_tickets=10,
        minimum_selection_hits=5,
        minimum_selection_winning_days=1,
        minimum_selection_roi=1.05,
    )


def test_exposure_selection_uses_only_pre_evaluation_period() -> None:
    low = _adaptive_selection_with_evaluation_payout(110)
    high = _adaptive_selection_with_evaluation_payout(20_000)

    assert low["policy_selection"]["source"] == (
        "pre_evaluation_adaptive_selection"
    )
    assert low["policy_selection"]["period"]["selection_through"] == (
        "2026-06-04"
    )
    selection_keys = (
        "selected_ridge",
        "selected_ev_threshold",
        "selected_min_daily_exposure_fraction",
    )
    assert {
        key: low["policy_selection"][key] for key in selection_keys
    } == {
        key: high["policy_selection"][key] for key in selection_keys
    }


def test_direct_candidates_skip_ineligible_tail_and_default_to_eligible() -> None:
    probabilities = np.zeros(120, dtype=np.float64)
    target = COMBINATION_LABELS.index("1-2-3")
    probabilities[target] = 1.0
    payout_model = _payout_model()
    payout_model["1-2-3"]["tail_eligible"] = False
    race_key = ("race-1", "2026-07-20", "01", 1)
    actual = {"combination": "1-2-3", "payout_yen": 1_000}

    ineligible = direct_candidates(
        probabilities,
        race_key=race_key,
        actual=actual,
        payout_model=payout_model,
        ev_threshold=1.2,
    )
    del payout_model["1-2-3"]["tail_eligible"]
    legacy = direct_candidates(
        probabilities,
        race_key=race_key,
        actual=actual,
        payout_model=payout_model,
        ev_threshold=1.2,
    )

    assert ineligible == []
    assert [row["combination"] for row in legacy] == ["1-2-3"]
    assert legacy[0]["tail_eligible"] is True


def _tail_eligibility_walk_forward_case(
    monkeypatch: pytest.MonkeyPatch,
    *,
    selection_samples: int,
    include_first_evaluation_result: bool,
    evaluation_target_probability: float = 0.2,
) -> dict[str, object]:
    target = COMBINATION_LABELS.index("1-2-3")

    def fixed_odds(*args: object, **kwargs: object) -> np.ndarray:
        probabilities = np.asarray(args[1], dtype=np.float64)
        return np.full(probabilities.shape, 200.0)

    monkeypatch.setattr(direct_bankroll, "predict_conditional_odds", fixed_odds)
    calibration_keys = [("fit", "2026-06-01", "01", 1)] + [
        (f"selection-{index}", "2026-06-02", "01", index % 12 + 1)
        for index in range(selection_samples)
    ]
    calibration_probabilities = np.full(
        (len(calibration_keys), 120),
        0.8 / 119.0,
    )
    calibration_probabilities[:, target] = 0.2
    evaluation_probabilities = np.full(
        (2, 120),
        (1.0 - evaluation_target_probability) / 119.0,
    )
    evaluation_probabilities[:, target] = evaluation_target_probability
    evaluation_keys = [
        ("evaluation-d", "2026-07-01", "01", 1),
        ("evaluation-d-plus-1", "2026-07-02", "01", 1),
    ]
    payouts = {
        race_key[0]: {"combination": "1-2-3", "payout_yen": 40_000}
        for race_key in calibration_keys
    }
    payouts["evaluation-d-plus-1"] = {
        "combination": "1-2-3",
        "payout_yen": 40_000,
    }
    if include_first_evaluation_result:
        payouts["evaluation-d"] = {
            "combination": "1-2-3",
            "payout_yen": 40_000,
        }
    policy = standard_direct_policy()
    policy["fractional_kelly"] = 1.0

    return simulate_conditional_payout_walk_forward(
        evaluation_probabilities,
        race_keys=evaluation_keys,
        payouts=payouts,
        calibration_probabilities=calibration_probabilities,
        calibration_race_keys=calibration_keys,
        policy=policy,
        ridge_candidates=(10.0,),
        threshold_candidates=(1.2,),
        min_daily_exposure_candidates=(0.0,),
        policy_selection_days=1,
        minimum_selection_tickets=10_000,
    )


def test_tail_result_changes_eligibility_only_on_following_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with_result = _tail_eligibility_walk_forward_case(
        monkeypatch,
        selection_samples=19,
        include_first_evaluation_result=True,
    )
    without_result = _tail_eligibility_walk_forward_case(
        monkeypatch,
        selection_samples=19,
        include_first_evaluation_result=False,
    )

    with_day, with_next = with_result["daily"]
    without_day, without_next = without_result["daily"]
    for key in (
        "tail_eligible_candidates",
        "tail_ineligible_candidates",
        "tickets",
        "stake_yen",
    ):
        assert with_day[key] == without_day[key]
    assert with_day["tail_eligible_candidates"] == 0
    assert with_day["tail_ineligible_candidates"] == 120
    assert with_day["tickets"] == 0
    assert with_next["tail_eligible_candidates"] == 120
    assert without_next["tail_eligible_candidates"] == 0
    assert with_next["tickets"] > 0
    assert without_next["tickets"] == 0


def test_tail_insufficient_support_produces_no_purchase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _tail_eligibility_walk_forward_case(
        monkeypatch,
        selection_samples=19,
        include_first_evaluation_result=False,
    )

    assert result["selected_tickets"] == 0
    assert result["stake_yen"] == 0
    assert result["payout_diagnostics"]["tail_eligible_candidates"] == 0
    assert result["payout_diagnostics"]["tail_ineligible_candidates"] == 240


def test_tail_global_support_allows_sparse_bin_purchase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _tail_eligibility_walk_forward_case(
        monkeypatch,
        selection_samples=20,
        include_first_evaluation_result=True,
        evaluation_target_probability=0.01999,
    )

    first_day = result["daily"][0]
    assert result["tail_calibration_samples_initial"] == 20
    assert result["policy_selection"]["tail_initial"]["bin_counts"] == [
        20,
        0,
        0,
        0,
    ]
    assert first_day["tail_eligible_candidates"] == 120
    assert first_day["tail_ineligible_candidates"] == 0
    assert first_day["tickets"] > 0
    assert result["payout_diagnostics"]["tail_eligible_candidates"] == 240
    assert result["payout_diagnostics"]["tail_ineligible_candidates"] == 0
