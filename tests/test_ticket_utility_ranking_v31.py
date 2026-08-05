from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from boatrace_ai.listwise.ticket_utility_ranking_v31 import (
    _candidate_score,
    _ranking_teacher_weights,
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


def test_payout_teacher_weights_races_without_changing_features() -> None:
    start = date(2026, 1, 1)
    races = [
        _race(start + timedelta(days=index), payout=payout)
        for index, payout in enumerate((500, 1000, 5000, 50_000))
    ]
    winner = _ranking_teacher_weights(races, "winner")
    weighted = _ranking_teacher_weights(races, "payout_weighted")
    np.testing.assert_array_equal(winner, np.ones(len(races)))
    assert np.mean(weighted) == pytest.approx(1.0)
    assert float(np.max(weighted)) > float(np.min(weighted))


def test_payout_teacher_changes_fitted_ranking_model() -> None:
    start = date(2026, 1, 1)
    races = [
        _race(
            start + timedelta(days=index),
            winner_index=index,
            payout=50_000 if index % 6 == 5 else 700,
        )
        for index in range(48)
    ]
    preset = {"name": "tiny", "num_leaves": 7, "max_depth": 3}
    winner = fit_ticket_utility_ranker(
        races, label_scheme="winner", tree_preset=preset, num_threads=1
    )
    weighted = fit_ticket_utility_ranker(
        races, label_scheme="payout_weighted", tree_preset=preset, num_threads=1
    )
    assert winner["booster_sha256"] != weighted["booster_sha256"]


def test_poisson_head_directly_targets_capped_realized_gross_return() -> None:
    start = date(2026, 1, 1)
    races = [
        _race(
            start + timedelta(days=index),
            winner_index=index,
            payout=80_000 if index % 6 == 5 else 700,
        )
        for index in range(48)
    ]
    preset = {"name": "tiny", "num_leaves": 7, "max_depth": 3}
    winner = fit_ticket_utility_ranker(
        races, label_scheme="winner", tree_preset=preset, num_threads=1
    )
    poisson = fit_ticket_utility_ranker(
        races,
        label_scheme="gross_return_poisson_c50",
        tree_preset=preset,
        num_threads=1,
    )

    assert poisson["learner_objective"] == "poisson_expected_gross_return"
    assert poisson["gross_return_cap"] == 50.0
    assert poisson["booster_sha256"] != winner["booster_sha256"]
    assert set(ticket_ranking(races[-1], poisson)) == set(COMBINATIONS)


def test_ticket_ranker_serializes_and_returns_a_complete_order() -> None:
    start = date(2026, 1, 1)
    races = [
        _race(start + timedelta(days=index), winner_index=index, payout=1000 + 100 * index)
        for index in range(16)
    ]
    artifact = fit_ticket_utility_ranker(
        races,
        label_scheme="payout_weighted",
        tree_preset={"name": "tiny", "num_leaves": 7, "max_depth": 3},
        num_threads=1,
    )
    ranked = ticket_ranking(races[-1], artifact)
    metrics = ticket_ranking_metrics(races, artifact)

    assert artifact["role"] == "ticket_utility_ranking_only"
    assert artifact["training_tickets"] == len(races) * len(COMBINATIONS)
    assert artifact["teacher_weighting"] == "payout_weighted"
    assert set(ranked) == set(COMBINATIONS)
    assert len(ranked) == len(COMBINATIONS)
    assert metrics["evaluated_races"] == len(races)
    assert set(metrics["by_top_k"]) == {"1", "3", "5"}
    top1 = metrics["by_top_k"]["1"]
    assert top1["roi_excluding_largest_hit"] <= top1["roi"]
    assert top1["temporal_block_count"] == 3
    assert len(top1["temporal_block_rois"]) == 3

    broken = {**artifact, "booster_sha256": "0" * 64}
    with pytest.raises(ValueError, match="digest mismatch"):
        ticket_ranking(races[-1], broken)


def test_candidate_selection_rejects_single_hit_roi_spike() -> None:
    jackpot = {
        "top_k": 1,
        "selected_top_k_metrics": {
            "roi": 1.40,
            "roi_ci95_lower": 1.10,
            "roi_excluding_largest_hit": 0.52,
            "minimum_temporal_block_roi": 0.20,
        },
    }
    stable = {
        "top_k": 1,
        "selected_top_k_metrics": {
            "roi": 0.98,
            "roi_ci95_lower": 0.90,
            "roi_excluding_largest_hit": 0.91,
            "minimum_temporal_block_roi": 0.88,
        },
    }

    assert _candidate_score(stable) > _candidate_score(jackpot)


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
    assert result["model"] == "ticket_utility_robust_temporal_ranking_v32"
    assert result["inner_validation_days"] >= 5
    assert "largest-hit-excluded" in result["selection_rule"]
    assert set(result["selection_robustness_gate"]) == {
        "day_block_roi_lcb95_above_one",
        "largest_hit_excluded_roi_above_one",
        "every_temporal_block_roi_above_one",
        "effective_hit_count_at_least_five",
    }
    assert result["ranking_training_through"] < result["policy_calibration_from"]
    assert result["policy_calibration_through"] < result["evaluation_from"]
    assert result["probability_metrics"]["evaluated_races"] == 1
    assert result["ranking_metrics"]["evaluated_races"] == 1
    assert result["empirical_ev_calibration"]["tickets"] == (
        30 * int(result["selected_candidate"]["top_k"])
    )
    assert result["bankroll"]["evaluation_days"] == 1
