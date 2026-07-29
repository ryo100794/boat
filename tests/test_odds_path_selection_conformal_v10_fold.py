from __future__ import annotations

from itertools import permutations

import pytest

import boatrace_ai.listwise.odds_path_conservative_v7 as v7
from boatrace_ai.listwise import odds_path_selection_conformal_v10 as v10


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
            combination: value * 0.80
            for combination, value in odds.items()
        },
        "closing_source_changed": True,
        "closing_odds_changed": True,
        "odds_path": [
            {"minutes_before_decision": 10.0, "market_probabilities": earlier},
            {"minutes_before_decision": 0.0, "market_probabilities": market},
        ],
    }


def _artifact(evaluation_date: str, trained_through: str | None) -> dict:
    return {
        "ready": True,
        "method": "selected_top2_finite_sample_lower_rank_conformal_v1",
        "haircut": 0.75,
        "target_coverage": 0.8,
        "training_days": 3,
        "training_candidates": 9,
        "evaluation_date": evaluation_date,
        "trained_through_date": trained_through,
    }


def test_walk_forward_emits_selection_conditional_fold_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = (
        "2026-07-28",
        "2026-07-29",
        "2026-07-30",
        "2026-07-31",
    )
    races = [
        _race(race_date, rno)
        for race_date in dates
        for rno in range(1, 3)
    ]
    monkeypatch.setattr(v7, "MIN_CLOSING_TRAINING_DAYS", 2)
    monkeypatch.setattr(v7, "MIN_CLOSING_TRAINING_RACES", 4)
    artifacts = {
        date: _artifact(date, dates[index - 1] if index else None)
        for index, date in enumerate(dates)
    }
    monkeypatch.setattr(
        v10,
        "build_prequential_selection_conformal",
        lambda *args, **kwargs: {
            "artifacts_by_date": artifacts,
            "deployment_artifact": _artifact("9999-12-31", dates[-1]),
            "observations": [],
        },
    )

    result = v10.walk_forward_evaluate_v10(
        races,
        daily_budget_yen=10_000,
        min_calibration_days=2,
        evaluation_dates=("2026-07-30", "2026-07-31"),
    )

    assert result["model"] == v10.MODEL_NAME
    assert result["calibrator_strategy"] == v10.STRATEGY_NAME
    assert result["selection_conformal"]["evaluation_folds"] == 2
    assert result["selection_conformal"]["ready_folds"] == 2
    assert result["haircut_latest"] == pytest.approx(0.75)
    assert all(fold["leakage_guard"]["pass"] for fold in result["folds"])
    for fold in result["folds"]:
        guard = fold["selection_conformal"]
        assert guard["haircut"] == pytest.approx(0.75)
        assert guard["training_days"] == 3
        assert guard["training_candidates"] == 9
        assert guard["trained_through_date"] < fold["evaluation_date"]
        assert "selection_raw_closing_coverage" in guard
        assert "selection_guarded_closing_coverage" in guard
