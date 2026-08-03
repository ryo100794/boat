from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import pytest

from boatrace_ai.joint_bankroll_evaluation import (
    _day_block_roi_interval,
    _probability_metrics,
    _rank_candidate_tickets,
    _realized_receipt,
    _release_matured_receipts,
    run_joint_bankroll_evaluation,
)
from boatrace_ai.joint_market_value import JointMarketScenario


def test_realized_receipt_uses_official_100_yen_payout() -> None:
    assert _realized_receipt(
        {"1-2-3": 300, "2-1-3": 100},
        actual_combination="1-2-3",
        actual_payout_yen=2_530,
    ) == 7_590
    assert _realized_receipt(
        {"1-2-3": 300},
        actual_combination="2-1-3",
        actual_payout_yen=2_530,
    ) == 0
    with pytest.raises(ValueError, match="100-yen"):
        _realized_receipt(
            {"1-2-3": 150},
            actual_combination="1-2-3",
            actual_payout_yen=2_530,
        )


def test_receipt_is_not_released_before_conservative_settlement_time() -> None:
    available = datetime.fromisoformat("2026-07-01T10:15:00+09:00")
    pending = [(available, 2_530)]

    remaining, receipt = _release_matured_receipts(
        pending,
        asof=datetime.fromisoformat("2026-07-01T10:14:59+09:00"),
    )
    assert remaining == pending
    assert receipt == 0

    remaining, receipt = _release_matured_receipts(
        pending,
        asof=datetime.fromisoformat("2026-07-01T10:15:00+09:00"),
    )
    assert remaining == []
    assert receipt == 2_530


def test_candidate_preselection_uses_joint_probability_price_paths() -> None:
    paths = [[
        JointMarketScenario(
            probabilities={"A": 0.4, "B": 0.6},
            market_state={"final_market_shares": {"A": 0.1, "B": 0.9}},
        ),
        JointMarketScenario(
            probabilities={"A": 0.2, "B": 0.8},
            market_state={"final_market_shares": {"A": 0.1, "B": 0.9}},
        ),
    ]]
    assert _rank_candidate_tickets(paths, ("A", "B"), limit=1) == ("A",)


def test_probability_metrics_match_existing_logloss_brier_top5_definitions() -> None:
    result = _probability_metrics(
        {"A": 0.7, "B": 0.3},
        {"A": 0.6, "B": 0.4},
        "A",
        ("A", "B"),
    )
    assert result["generated_log_loss"] < result["decision_model_log_loss"]
    assert result["generated_brier"] == pytest.approx(0.18)
    assert result["decision_model_brier"] == pytest.approx(0.32)
    assert result["generated_top5"] == 1.0
    assert result["generated_winner_log_loss"] == pytest.approx(
        result["generated_log_loss"]
    )
    assert result["generated_winner_top1_accuracy"] == 1.0


def test_day_block_roi_interval_resamples_complete_days_deterministically() -> None:
    days = [
        {"stake_yen": 1_000, "return_yen": 1_500},
        {"stake_yen": 1_000, "return_yen": 500},
        {"stake_yen": 0, "return_yen": 0},
    ]
    first = _day_block_roi_interval(days, samples=500, seed=7)
    second = _day_block_roi_interval(days, samples=500, seed=7)

    assert first == second
    assert first["block"] == "complete_operating_day"
    assert first["quantile_method"] == "inverted_cdf"
    assert first["roi_lower"] <= 1.0 <= first["roi_upper"]


def test_strict_walk_forward_runs_joint_paths_through_daily_bankroll(
    tmp_path: Path,
) -> None:
    races = []
    for day in range(1, 9):
        for index in range(3):
            actual = "A" if (day + index) % 2 else "B"
            races.append({
                "race_id": f"202607{day:02d}01{index + 1:02d}",
                "race_date": f"2026-07-{day:02d}",
                "jcd": "01",
                "actual_combination": actual,
                "actual_payout_yen": 150,
                "model_probabilities": {"A": 0.55, "B": 0.45},
                "market_probabilities": {"A": 0.5, "B": 0.5},
                "official_closing_odds": {"A": 1.5, "B": 1.5},
                "odds": {"A": 1.5, "B": 1.5},
                "captured_at": f"2026-07-{day:02d}T10:{index:02d}:00+09:00",
                "odds_deadline_at": (
                    f"2026-07-{day:02d}T10:{index + 5:02d}:00+09:00"
                ),
            })
    cache = tmp_path / "races.joblib"
    joblib.dump({"races": races}, cache)
    progress = []

    result = run_joint_bankroll_evaluation(
        cache,
        terminal_min_training_days=2,
        joint_min_training_days=2,
        outer_draws=2,
        scenarios_per_draw=10,
        rank=2,
        pooling_strength=4.0,
        learn_residual_scales=False,
        candidate_ticket_count=2,
        initial_daily_bankroll_yen=1_000,
        maximum_portfolio_stake_yen=1_000,
        maximum_ticket_stake_yen=1_000,
        maximum_selected_tickets=2,
        buy_margin=10.0,
        inner_tail_fraction=1.0,
        population_size=4,
        generations=1,
        bootstrap_samples=100,
        seed=11,
        expected_outcomes=("A", "B"),
        progress_callback=progress.append,
    )

    assert result["evaluated_days"] == 4
    assert result["evaluated_races"] == 12
    assert result["evaluation_from"] == "2026-07-05"
    assert result["evaluation_through"] == "2026-07-08"
    assert result["primary_bankroll"]["stake_yen"] == 0
    assert result["primary_bankroll"]["roi"] is None
    assert result["promotion_eligible"] is False
    ledger = result["calibration_ledger"]
    assert ledger["version"] == "joint_edge_calibration_ledger_v1"
    assert ledger["candidate_portfolios"] > 0
    assert ledger["stake_yen"] > 0
    assert ledger["role"] == (
        "evaluation_only_never_used_by_same_period_purchase_gate"
    )

    assert "generated_log_loss" in result["probability_metrics"]
    purchase_value = result["joint_purchase_value"]
    assert purchase_value["selected_portfolios"] == 0
    assert purchase_value["minimum"] is None
    assert purchase_value["all_above_safety_margin"] is False
    confidence = result["bankroll_confidence"]
    assert confidence["condition"]["formal_gate"] == (
        "Q0.05_ROI_greater_than_1"
    )
    assert len(confidence["condition_id"]) == 64
    assert confidence["formal_gate_passed"] is False
    assert confidence["probability_roi_above_one_is_diagnostic_only"] is True
    assert confidence["sensitivity"]["day_venue"]["block"] == (
        "independent_day_venue_sensitivity"
    )
    assert confidence["sensitivity"]["venue_meeting"]["block"] == (
        "consecutive_venue_meeting_sensitivity"
    )
    assert result["promotion_gate"][
        "joint_purchase_value_above_safety_margin"
    ] is False
    diagnostic_race = result["daily"][0]["races"][0]
    assert diagnostic_race["actual_combination"] in {"A", "B"}
    assert diagnostic_race["actual_payout_yen"] == 150
    assert diagnostic_race["selected_bets_yen"] == {}
    assert diagnostic_race["feasible_candidates_found"] == 0
    assert diagnostic_race["best_search_constraint_violation"] is not None
    completed = [
        row for row in progress
        if row["event"] == "joint_bankroll_day_completed"
    ]
    assert [row["completed_days"] for row in completed] == [1, 2, 3, 4]
    assert all(row["total_evaluation_days"] == 4 for row in completed)
    assert {
        row["evaluation_date"]
        for row in progress
        if row["event"] == "joint_bankroll_race_progress"
    } == {
        "2026-07-05", "2026-07-06", "2026-07-07", "2026-07-08"
    }
