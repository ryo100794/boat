from __future__ import annotations

from itertools import permutations

import pytest

from boatrace_ai.listwise import odds_path_selection_conformal_v10 as v10
from boatrace_ai.listwise.selection_conformal import (
    fit_selection_conformal_haircut,
)


COMBINATIONS = tuple(
    "-".join(map(str, values))
    for values in permutations(range(1, 7), 3)
)


def _distribution(primary: str, probability: float) -> dict[str, float]:
    remainder = (1.0 - probability) / (len(COMBINATIONS) - 1)
    return {
        combination: probability if combination == primary else remainder
        for combination in COMBINATIONS
    }


def _race(
    race_date: str,
    rno: int,
    *,
    primary: str | None = None,
    primary_probability: float = 0.07,
    predicted_closing: float = 20.0,
    actual_closing_ratio: float = 0.5,
) -> tuple[dict, dict[str, float]]:
    primary = primary or COMBINATIONS[rno % len(COMBINATIONS)]
    probabilities = _distribution(primary, primary_probability)
    forecast = {combination: 2.0 for combination in COMBINATIONS}
    forecast[primary] = predicted_closing
    actual = COMBINATIONS[(rno + 17) % len(COMBINATIONS)]
    race = {
        "race_id": f"{race_date}-01-{rno:02d}",
        "race_date": race_date,
        "jcd": "01",
        "rno": rno,
        "actual_combination": actual,
        "actual_payout_yen": 10_000,
        "model_probabilities": probabilities,
        "closing_odds": {
            combination: (
                predicted_closing * actual_closing_ratio
                if combination == primary
                else odds
            )
            for combination, odds in forecast.items()
        },
    }
    return race, forecast


def _lcb() -> dict:
    return {
        "ready": True,
        "factors": {
            "top2": 1.0,
            "top5": 1.0,
            "top20": 1.0,
            "rest": 1.0,
        },
    }


def _artifact(haircut: float = 0.5) -> dict:
    return {
        "ready": True,
        "method": "selected_top2_daily_cluster_finite_sample_lower_conformal_v2",
        "haircut": haircut,
        "target_coverage": 0.8,
        "training_days": 5,
        "training_candidates": 30,
        "trained_through_date": "2026-07-29",
    }


def test_daily_cluster_finite_sample_guard_uses_prior_days_as_units() -> None:
    daily_ratios = (
        (0.20, 0.22, 0.24, 0.26, 0.28, 1.10),
        (0.30, 0.32, 0.34, 0.36, 0.38, 1.10),
        (0.40, 0.42, 0.44, 0.46, 0.48, 1.10),
        (0.50, 0.52, 0.54, 0.56, 0.58, 1.10),
        (0.60, 0.62, 0.64, 0.66, 0.68, 1.10),
    )
    observations = [
        {
            "race_date": f"2026-07-{20 + day_index:02d}",
            "race_id": f"r{day_index}-{candidate_index}",
            "combination": COMBINATIONS[candidate_index],
            "closing_ratio": ratio,
        }
        for day_index, ratios in enumerate(daily_ratios)
        for candidate_index, ratio in enumerate(ratios)
    ]

    artifact = fit_selection_conformal_haircut(
        observations, evaluation_date="2026-07-25"
    )

    assert artifact["ready"] is True
    assert artifact["training_days"] == 5
    assert artifact["training_candidates"] == 30
    assert artifact["finite_sample_unit"] == "prior_day"
    assert artifact["finite_sample_rank"] == 1
    assert artifact["finite_sample_coverage"] == pytest.approx(5 / 6)
    assert artifact["daily_candidate_counts"] == {
        f"2026-07-{20 + index:02d}": 6 for index in range(5)
    }
    assert artifact["haircut"] == pytest.approx(0.20)


def test_conditional_guard_uses_only_post_haircut_allocator_inputs() -> None:
    races = []
    forecasts = {}
    ratios = (0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 1.10)
    for rno, ratio in enumerate(ratios, start=1):
        race, forecast = _race(
            "2026-07-31",
            rno,
            primary_probability=0.30,
            actual_closing_ratio=ratio,
        )
        races.append(race)
        forecasts[race["race_id"]] = forecast

    _bankroll, diagnostic = v10._simulate_selection_conformal_policy(
        races,
        closing_forecasts=forecasts,
        probability_lcb=_lcb(),
        daily_budget_yen=10_000,
        selection_conformal=_artifact(0.22),
    )
    metrics = diagnostic["selection_conformal"]

    assert diagnostic["raw_selected_candidates"] == 9
    assert diagnostic["guarded_threshold_candidates"] == 9
    assert metrics["selection_evaluation_candidates"] == 9
    assert metrics["selection_raw_closing_coverage"] == pytest.approx(1 / 9)
    assert metrics["selection_guarded_closing_coverage"] == pytest.approx(8 / 9)
    assert metrics["selection_closing_missing_candidates"] == 0
    assert metrics["selection_closing_complete"] is True


def test_pre_haircut_candidate_rejected_by_guard_is_not_in_coverage() -> None:
    accepted, accepted_forecast = _race(
        "2026-07-31", 1, primary_probability=0.30
    )
    rejected, rejected_forecast = _race(
        "2026-07-31", 2, primary_probability=0.055
    )
    _bankroll, diagnostic = v10._simulate_selection_conformal_policy(
        [accepted, rejected],
        closing_forecasts={
            accepted["race_id"]: accepted_forecast,
            rejected["race_id"]: rejected_forecast,
        },
        probability_lcb=_lcb(),
        daily_budget_yen=10_000,
        selection_conformal=_artifact(0.5),
    )

    assert diagnostic["raw_selected_candidates"] == 2
    assert diagnostic["guarded_threshold_candidates"] == 1
    assert diagnostic["selection_conformal"][
        "selection_evaluation_candidates"
    ] == 1


def test_missing_closing_stays_in_denominator_and_fails_gate() -> None:
    race, forecast = _race(
        "2026-07-31", 1, primary_probability=0.30
    )
    selected = max(race["model_probabilities"], key=race["model_probabilities"].get)
    race["closing_odds"] = {
        key: value for key, value in race["closing_odds"].items() if key != selected
    }
    _bankroll, diagnostic = v10._simulate_selection_conformal_policy(
        [race],
        closing_forecasts={race["race_id"]: forecast},
        probability_lcb=_lcb(),
        daily_budget_yen=10_000,
        selection_conformal=_artifact(0.5),
    )
    metrics = diagnostic["selection_conformal"]
    gate = v10._selection_coverage_gate(metrics)

    assert metrics["selection_evaluation_candidates"] == 1
    assert metrics["selection_observed_closing_candidates"] == 0
    assert metrics["selection_closing_missing_candidates"] == 1
    assert metrics["selection_guarded_closing_coverage"] == 0.0
    assert metrics["selection_closing_complete"] is False
    assert gate["selection_conditional_coverage_pass"] is False
    assert gate["selection_conditional_complete_pass"] is False


def test_same_or_future_day_calibration_is_rejected() -> None:
    with pytest.raises(ValueError, match="prior-day only"):
        fit_selection_conformal_haircut(
            [
                {
                    "race_date": "2026-07-31",
                    "race_id": "same-day",
                    "combination": "1-2-3",
                    "closing_ratio": 0.5,
                }
            ],
            evaluation_date="2026-07-31",
        )


@pytest.mark.parametrize(
    ("days", "candidates"),
    ((4, 30), (5, 29)),
)
def test_insufficient_prior_selection_sample_forces_no_bet(
    days: int, candidates: int
) -> None:
    observations = [
        {
            "race_date": f"2026-07-{20 + index % days:02d}",
            "race_id": f"r{index}",
            "combination": COMBINATIONS[index],
            "closing_ratio": 0.5,
        }
        for index in range(candidates)
    ]
    artifact = fit_selection_conformal_haircut(
        observations, evaluation_date="2026-07-31"
    )
    race, forecast = _race("2026-07-31", 1)

    bankroll, diagnostic = v10._simulate_selection_conformal_policy(
        [race],
        closing_forecasts={race["race_id"]: forecast},
        probability_lcb=_lcb(),
        daily_budget_yen=10_000,
        selection_conformal=artifact,
    )

    assert artifact["ready"] is False
    assert bankroll["tickets"] == 0
    assert diagnostic["zero_reason_counts"] == {
        "selection_conformal_not_ready": 1
    }


def test_haircut_is_applied_before_ev_recheck_and_allocation() -> None:
    race, forecast = _race(
        "2026-07-31",
        1,
        primary_probability=0.055,
        predicted_closing=20.0,
    )

    bankroll, diagnostic = v10._simulate_selection_conformal_policy(
        [race],
        closing_forecasts={race["race_id"]: forecast},
        probability_lcb=_lcb(),
        daily_budget_yen=10_000,
        selection_conformal=_artifact(0.5),
    )

    assert diagnostic["raw_selected_candidates"] == 1
    assert diagnostic["guarded_threshold_candidates"] == 0
    assert bankroll["tickets"] == 0
    assert diagnostic["zero_reason_counts"] == {
        "no_candidate_after_selection_conformal": 1
    }


def test_purchase_selection_does_not_depend_on_result_or_payout() -> None:
    race, forecast = _race(
        "2026-07-31",
        1,
        primary_probability=0.07,
        predicted_closing=20.0,
    )
    selected_combination = max(
        race["model_probabilities"], key=race["model_probabilities"].get
    )
    race["actual_combination"] = selected_combination
    first, first_diagnostic = v10._simulate_selection_conformal_policy(
        [race],
        closing_forecasts={race["race_id"]: forecast},
        probability_lcb=_lcb(),
        daily_budget_yen=10_000,
        selection_conformal=_artifact(0.8),
    )
    changed = {
        **race,
        "actual_combination": COMBINATIONS[-1],
        "actual_payout_yen": 1_000_000,
    }
    second, second_diagnostic = v10._simulate_selection_conformal_policy(
        [changed],
        closing_forecasts={race["race_id"]: forecast},
        probability_lcb=_lcb(),
        daily_budget_yen=10_000,
        selection_conformal=_artifact(0.8),
    )

    first_selection = [
        (row["race_id"], row["combination"], row["stake_yen"])
        for row in first["daily"][0]["selected_sample"]
    ]
    second_selection = [
        (row["race_id"], row["combination"], row["stake_yen"])
        for row in second["daily"][0]["selected_sample"]
    ]
    assert first_selection == second_selection
    assert first["tickets"] == 1
    assert first_diagnostic["guarded_threshold_candidates"] == 1
    assert second_diagnostic["guarded_threshold_candidates"] == 1
    assert first["return_yen"] != second["return_yen"]
