from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from boatrace_ai.listwise.nonlinear_market_residual_v38 import (
    _softmax_with_market_offset,
    fit_nonlinear_market_residual,
    fit_temporal_nonlinear_market_residual,
    nonlinear_residual_probabilities,
)


COMBINATIONS = [
    f"{first}-{second}-{third}"
    for first in range(1, 7)
    for second in range(1, 7)
    if second != first
    for third in range(1, 7)
    if third not in (first, second)
][:12]


def _race(day: date, index: int) -> dict:
    raw_market = {
        combination: 1.0 / (8.0 + ticket)
        for ticket, combination in enumerate(COMBINATIONS)
    }
    total = sum(raw_market.values())
    market = {key: value / total for key, value in raw_market.items()}
    # The winner alternates between a market favourite and a context-dependent
    # ticket so a tree has a real residual signal in the fixture.
    winner = 0 if index % 2 == 0 else 5
    race_date = day.isoformat()
    return {
        "race_id": f"{race_date}-01-{index:02d}",
        "race_date": race_date,
        "jcd": "01",
        "rno": index % 12 + 1,
        "actual_combination": COMBINATIONS[winner],
        "actual_payout_yen": 1500,
        "odds": {
            combination: 8.0 + ticket
            for ticket, combination in enumerate(COMBINATIONS)
        },
        "model_probabilities": market,
        "market_probabilities": market,
        "lane_context": {
            str(lane): {
                "class_rank": float(lane),
                "national_win_rate": float(8 - lane),
                "research_local_vs_national_win": float((index % 2) * lane),
                "research_home_branch": float(lane == 1),
                "motor_2_rate": float(35 + lane),
                "boat_2_rate": float(32 + lane),
                "hist_racer_win_rate_s": float(7 - lane) / 10,
                "hist_racer_venue_win_rate_s": float(6 - lane) / 10,
                "hist_motor_win_rate_s": float(lane) / 10,
                "hist_boat_win_rate_s": float(6 - lane) / 10,
            }
            for lane in range(1, 7)
        },
    }


def test_zero_shrinkage_is_exact_market_distribution() -> None:
    market = np.asarray([0.6, 0.3, 0.1])
    correction = np.asarray([100.0, -50.0, 5.0])
    np.testing.assert_allclose(
        _softmax_with_market_offset(market, correction, 0.0), market
    )


def test_custom_market_offset_booster_round_trips() -> None:
    start = date(2026, 1, 1)
    races = [_race(start + timedelta(days=index // 4), index) for index in range(40)]
    artifact = fit_nonlinear_market_residual(
        races,
        tree_preset={
            "name": "tiny",
            "num_leaves": 7,
            "max_depth": 3,
            "min_child_samples": 5,
        },
        num_threads=1,
        num_boost_round=8,
    )
    probabilities = nonlinear_residual_probabilities(
        races[-1], artifact, shrinkage=0.5
    )
    assert artifact["objective"] == (
        "grouped_multinomial_logloss_with_fixed_market_offset"
    )
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert set(probabilities) == set(COMBINATIONS)

    broken = {**artifact, "booster_sha256": "0" * 64}
    with pytest.raises(ValueError, match="digest mismatch"):
        nonlinear_residual_probabilities(races[-1], broken, shrinkage=0.5)


def test_temporal_selection_includes_exact_market_null_and_stays_prior() -> None:
    start = date(2026, 1, 1)
    calibration = [_race(start + timedelta(days=index // 3), index) for index in range(30)]
    evaluation = [
        _race(start + timedelta(days=10 + index // 3), 30 + index)
        for index in range(9)
    ]
    result = fit_temporal_nonlinear_market_residual(
        calibration,
        evaluation,
        tree_presets=(
            {
                "name": "tiny",
                "num_leaves": 7,
                "max_depth": 3,
                "min_child_samples": 5,
            },
        ),
        shrinkages=(0.0, 0.5),
        num_threads=1,
    )
    assert result["market_is_exact_nested_null"] is True
    assert {row["shrinkage"] for row in result["candidates"]} == {0.0, 0.5}
    assert result["inner_fit_through"] < result["inner_validation_from"]
    assert result["inner_validation_from"] < evaluation[0]["race_date"]
    assert result["metrics"]["evaluated_races"] == len(evaluation)
