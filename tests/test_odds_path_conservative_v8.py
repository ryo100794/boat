from __future__ import annotations

from itertools import permutations

import pytest

import boatrace_ai.listwise.odds_path_conservative_v7 as v7
import boatrace_ai.listwise.odds_path_conservative_v8 as v8
from boatrace_ai.listwise.odds_path_probability_v8 import MODEL_NAME as PROBABILITY_MODEL


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


def _race(race_date: str, rno: int) -> dict:
    actual = COMBINATIONS[(rno * 5) % len(COMBINATIONS)]
    base = _distribution(COMBINATIONS[(rno * 7 + 1) % len(COMBINATIONS)], 0.15)
    market = _distribution(COMBINATIONS[(rno * 11 + 2) % len(COMBINATIONS)], 0.12)
    odds = {
        combination: 1.0 / probability / 1.25
        for combination, probability in market.items()
    }
    earlier = dict(market)
    earlier[actual] *= 0.94
    total = sum(earlier.values())
    earlier = {key: value / total for key, value in earlier.items()}
    return {
        "race_id": f"{race_date}-01-{rno:02d}",
        "race_date": race_date,
        "jcd": "01",
        "rno": rno,
        "actual_combination": actual,
        "actual_payout_yen": 10_000,
        "model_probabilities": base,
        "market_probabilities": market,
        "odds": odds,
        "closing_odds": {
            combination: value * 0.95
            for combination, value in odds.items()
        },
        "closing_source_changed": True,
        "closing_odds_changed": True,
        "odds_path": [
            {
                "minutes_before_decision": 10.0,
                "market_probabilities": earlier,
            },
            {
                "minutes_before_decision": 0.0,
                "market_probabilities": market,
            },
        ],
    }


def _races() -> list[dict]:
    return [
        _race(race_date, rno)
        for race_date in (
            "2026-07-28",
            "2026-07-29",
            "2026-07-30",
            "2026-07-31",
        )
        for rno in range(1, 3)
    ]


def test_v7_and_v8_share_outer_population_policy_and_closing_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(v7, "MIN_CLOSING_TRAINING_DAYS", 2)
    monkeypatch.setattr(v7, "MIN_CLOSING_TRAINING_RACES", 4)
    races = _races()
    formal_dates = ("2026-07-30", "2026-07-31")

    legacy = v7.walk_forward_evaluate_v7(
        races,
        daily_budget_yen=10_000,
        min_calibration_days=2,
        evaluation_dates=formal_dates,
    )
    candidate = v8.walk_forward_evaluate_v8(
        races,
        daily_budget_yen=10_000,
        min_calibration_days=2,
        evaluation_dates=formal_dates,
    )

    assert candidate["model"] == v8.MODEL_NAME
    assert candidate["calibrator_strategy"] == v8.STRATEGY_NAME
    assert candidate["available_races"] == legacy["available_races"]
    assert candidate["evaluation_races"] == legacy["evaluation_races"]
    assert candidate["evaluation_days"] == legacy["evaluation_days"]
    assert [fold["evaluation_date"] for fold in candidate["folds"]] == list(
        formal_dates
    )
    assert candidate["fixed_policy"] == legacy["fixed_policy"]
    assert set(candidate["promotion_gate"]) == set(legacy["promotion_gate"])
    assert [
        (
            fold["calibration_dates"],
            fold["evaluation_date"],
            fold["calibration_races"],
            fold["evaluation_races"],
            fold["closing_q20_metrics"]["closing_q20_evaluation_races"],
            fold["closing_q20_metrics"]["closing_q20_evaluation_tickets"],
        )
        for fold in candidate["folds"]
    ] == [
        (
            fold["calibration_dates"],
            fold["evaluation_date"],
            fold["calibration_races"],
            fold["evaluation_races"],
            fold["closing_q20_metrics"]["closing_q20_evaluation_races"],
            fold["closing_q20_metrics"]["closing_q20_evaluation_tickets"],
        )
        for fold in legacy["folds"]
    ]
    assert all(
        fold["operational_model"]["model_type"] == PROBABILITY_MODEL
        for fold in candidate["folds"]
    )


def test_v8_outer_and_nested_dates_never_cross_holdout() -> None:
    result = v8.walk_forward_evaluate_v8(
        _races(), daily_budget_yen=10_000, min_calibration_days=2
    )

    assert all(fold["leakage_guard"]["pass"] for fold in result["folds"])
    for fold in result["folds"]:
        outer_date = fold["evaluation_date"]
        model = fold["operational_model"]
        assert model["trained_through_date"] < outer_date
        assert all(date < outer_date for date in model["training_dates"])
        selection = model["regularization_selection"]
        assert all(date < outer_date for date in selection["training_dates"])
        assert all(
            nested["trained_through_date"] < nested["validation_date"] < outer_date
            for candidate in selection["candidates"]
            for nested in candidate["folds"]
        )


def test_v7_public_contract_remains_legacy_only() -> None:
    result = v7.walk_forward_evaluate_v7(
        _races(), daily_budget_yen=10_000, min_calibration_days=2
    )

    assert result["model"] == v7.MODEL_NAME
    assert result["calibrator_strategy"] == v7.STRATEGY_NAME
    assert result["comparison_role"] == (
        "real_t5_crossfit_q20_fixed_safe_ev_shadow"
    )
    assert "prospective_crossfit_conservative_ev_v7_walk_forward" in result
    assert v8.PROSPECTIVE_OUTPUT_KEY not in result
    assert result["deployment_configuration"]["calibrator_strategy"] == (
        v7.STRATEGY_NAME
    )
