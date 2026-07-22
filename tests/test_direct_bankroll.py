from __future__ import annotations

import numpy as np

from boatrace_ai.listwise.direct_bankroll import (
    COMBINATION_LABELS,
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

    result = simulate_conditional_payout_walk_forward(
        probabilities,
        race_keys=race_keys,
        payouts=payouts,
        calibration_probabilities=calibration_probabilities,
        calibration_race_keys=calibration_keys,
    )

    assert result["payout_training_samples_initial"] == 60
    assert result["payout_training_samples_final"] == 62
    assert [row["payout_training_samples"] for row in result["daily"]] == [60, 61]
    assert result["evaluated_races"] == 2
