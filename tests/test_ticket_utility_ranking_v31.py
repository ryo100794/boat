from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from boatrace_ai.listwise.ticket_utility_ranking_v31 import (
    evaluate_temporal_ticket_utility_roles,
    fit_ticket_utility_ranker,
    ticket_feature_matrix,
    ticket_ranking,
    ticket_ranking_metrics,
)


COMBINATIONS = [
    f"{first}-{second}-{third}"
    for first in range(1, 7)
    for second in range(1, 7)
    if second != first
    for third in range(1, 7)
    if third not in (first, second)
][:12]


def _race(day: date, *, winner_index: int = 0, payout: int = 1800) -> dict:
    race_date = day.isoformat()
    raw_market = {
        combination: 1.0 / (5.0 + index)
        for index, combination in enumerate(COMBINATIONS)
    }
    market_total = sum(raw_market.values())
    market = {key: value / market_total for key, value in raw_market.items()}
    raw_model = {
        combination: value * (1.15 if index % 3 == 0 else 0.95)
        for index, (combination, value) in enumerate(market.items())
    }
    model_total = sum(raw_model.values())
    model = {key: value / model_total for key, value in raw_model.items()}
    return {
        "race_id": f"{race_date}-01-01",
        "race_date": race_date,
        "jcd": "01",
        "rno": 1,
        "captured_at": f"{race_date}T10:00:00+09:00",
        "odds_deadline_at": f"{race_date}T10:00:00+09:00",
        "actual_combination": COMBINATIONS[winner_index % len(COMBINATIONS)],
        "actual_payout_yen": payout,
        "odds": {
            combination: 5.0 + index
            for index, combination in enumerate(COMBINATIONS)
        },
        "model_probabilities": model,
        "market_probabilities": market,
        "lane_context": {
            str(lane): {
                "class_rank": float(lane),
                "national_win_rate": float(8 - lane),
                "research_local_vs_national_win": float(lane - 3) / 10,
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


def test_ticket_features_do_not_read_payout_or_winner_teacher() -> None:
    race = _race(date(2026, 1, 1))
    combinations, features = ticket_feature_matrix(race)
    changed = {**race, "actual_combination": COMBINATIONS[-1], "actual_payout_yen": 999_900}
    changed_combinations, changed_features = ticket_feature_matrix(changed)

    assert combinations == changed_combinations
    np.testing.assert_array_equal(features, changed_features)
    assert features.shape[0] == len(COMBINATIONS)
    assert features.shape[1] > 100


def test_ticket_ranker_serializes_and_returns_a_complete_order() -> None:
    start = date(2026, 1, 1)
    races = [
        _race(start + timedelta(days=index), winner_index=index, payout=1000 + 100 * index)
        for index in range(16)
    ]
    artifact = fit_ticket_utility_ranker(
        races,
        label_scheme="payout_bucket",
        tree_preset={"name": "tiny", "num_leaves": 7, "max_depth": 3},
        num_threads=1,
    )
    ranked = ticket_ranking(races[-1], artifact)
    metrics = ticket_ranking_metrics(races, artifact)

    assert artifact["role"] == "ticket_utility_ranking_only"
    assert artifact["training_tickets"] == len(races) * len(COMBINATIONS)
    assert set(ranked) == set(COMBINATIONS)
    assert len(ranked) == len(COMBINATIONS)
    assert metrics["evaluated_races"] == len(races)
    assert set(metrics["by_top_k"]) == {"1", "3", "5"}

    broken = {**artifact, "booster_sha256": "0" * 64}
    with pytest.raises(ValueError, match="digest mismatch"):
        ticket_ranking(races[-1], broken)


def test_temporal_ticket_roles_keep_all_teacher_windows_prior() -> None:
    start = date(2026, 1, 1)
    calibration = [
        _race(
            start + timedelta(days=index),
            winner_index=index,
            payout=8000 if index % 7 == 0 else 1400,
        )
        for index in range(42)
    ]
    evaluation = [_race(start + timedelta(days=42), winner_index=5)]
    result = evaluate_temporal_ticket_utility_roles(
        calibration,
        evaluation,
        daily_budget_yen=10_000,
        label_schemes=("winner",),
        tree_presets=({"name": "tiny", "num_leaves": 7, "max_depth": 3},),
        bootstrap_samples=100,
    )

    assert result["status"] == "completed"
    assert result["ranking_training_through"] < result["policy_calibration_from"]
    assert result["policy_calibration_through"] < result["evaluation_from"]
    assert result["probability_metrics"]["evaluated_races"] == 1
    assert result["ranking_metrics"]["evaluated_races"] == 1
    assert result["bankroll"]["evaluation_days"] == 1
