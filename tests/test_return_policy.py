from __future__ import annotations

import numpy as np

from boatrace_ai.listwise.direct_bankroll import COMBINATION_INDEX, standard_direct_policy
from boatrace_ai.listwise.return_bankroll import _adaptive_threshold_diagnostics
from boatrace_ai.listwise.return_policy import (
    calibration_policy_split,
    flat_threshold_diagnostics,
    select_policy_threshold,
)


def test_calibration_policy_split_uses_latest_full_days() -> None:
    race_keys = [
        (f"r-{day}-{race}", f"2026-06-{day:02d}", "01", race)
        for day in range(1, 6)
        for race in range(1, 3)
    ]

    assert calibration_policy_split(race_keys, selection_days=2) == 6
    assert calibration_policy_split(race_keys, selection_days=5) is None


def test_threshold_selection_uses_only_profitable_supported_rows() -> None:
    race_keys = [
        ("r1", "2026-06-01", "01", 1),
        ("r2", "2026-06-02", "01", 1),
    ]
    expected_returns = np.asarray([[1.30, 1.10], [1.30, 1.10]])
    payouts = {
        "r1": {"combination": "1-2-3", "payout_yen": 300},
        "r2": {"combination": "1-2-3", "payout_yen": 300},
    }
    diagnostics = flat_threshold_diagnostics(
        expected_returns,
        race_keys,
        payouts,
        {"1-2-3": 0, "2-1-3": 1},
        (1.05, 1.20, 1.35),
    )

    assert diagnostics[0]["tickets"] == 4
    assert diagnostics[0]["roi"] == 1.5
    assert diagnostics[1]["tickets"] == 2
    assert diagnostics[1]["roi"] == 3.0
    threshold, source = select_policy_threshold(
        diagnostics,
        fallback=1.20,
        minimum_tickets=2,
        minimum_roi=1.05,
        minimum_probability_roi_above_one=0.0,
    )
    assert threshold == 1.20
    assert source == "pre_evaluation_risk_adjusted_temporal_selection"


def test_threshold_diagnostics_exclude_races_without_results() -> None:
    diagnostics = flat_threshold_diagnostics(
        np.asarray([[1.30, 1.10], [1.30, 1.10]]),
        [("complete", "2026-06-01", "01", 1), ("missing", "2026-06-01", "01", 2)],
        {"complete": {"combination": "1-2-3", "payout_yen": 300}},
        {"1-2-3": 0, "2-1-3": 1},
        (1.05,),
    )

    assert diagnostics[0]["tickets"] == 2
    assert diagnostics[0]["stake_yen"] == 200
    assert diagnostics[0]["roi"] == 1.5


def test_adaptive_threshold_diagnostics_use_operational_allocator() -> None:
    probabilities = np.full((2, 120), 0.4 / 118.0)
    expected_returns = np.zeros((2, 120))
    winner_index = COMBINATION_INDEX["1-2-3"]
    extra_index = COMBINATION_INDEX["1-3-2"]
    probabilities[:, winner_index] = 0.4
    probabilities[:, extra_index] = 0.2
    expected_returns[:, winner_index] = 1.30
    expected_returns[:, extra_index] = 1.10
    race_keys = [
        ("r1", "2026-06-01", "01", 1),
        ("r2", "2026-06-01", "01", 2),
    ]
    payouts = {
        race_id: {"combination": "1-2-3", "payout_yen": 300}
        for race_id, _race_date, _jcd, _rno in race_keys
    }

    diagnostics = _adaptive_threshold_diagnostics(
        probabilities,
        expected_returns,
        race_keys,
        payouts,
        (1.05, 1.20),
        standard_direct_policy(),
        1_000,
    )

    # A lower-EV ticket must not be forced into the allocation solely to
    # satisfy a minimum daily exposure target.
    assert diagnostics[0]["tickets"] == 2
    assert diagnostics[1]["tickets"] == 2
    assert diagnostics[1]["roi"] == diagnostics[0]["roi"]
    assert diagnostics[1]["profit_yen"] == diagnostics[0]["profit_yen"]
    assert diagnostics[1]["profit_yen"] > 0
    assert diagnostics[1]["selection_roi_ci95_lower"] > 1.0
    assert diagnostics[1]["selection_probability_roi_above_one"] == 1.0
    assert diagnostics[1]["roi_without_largest_hit"] > 1.0
    assert diagnostics[1]["effective_hit_count"] == 2.0


def test_threshold_selection_requires_bootstrap_roi_lower_bound() -> None:
    threshold, source = select_policy_threshold(
        [
            {
                "ev_threshold": 1.05,
                "tickets": 200,
                "hits": 20,
                "winning_days": 12,
                "roi": 1.8,
                "roi_without_largest_hit": 1.4,
                "effective_hit_count": 16.0,
                "selection_roi_ci95_lower": 0.7,
                "selection_probability_roi_above_one": 0.99,
                "profit_yen": 8_000,
            },
            {
                "ev_threshold": 1.10,
                "tickets": 180,
                "hits": 18,
                "winning_days": 12,
                "roi": 1.2,
                "roi_without_largest_hit": 1.1,
                "effective_hit_count": 12.0,
                "selection_roi_ci95_lower": 1.06,
                "selection_probability_roi_above_one": 0.96,
                "profit_yen": 2_000,
            },
        ],
        fallback=1.20,
        minimum_tickets=100,
        minimum_roi=1.05,
        minimum_hits=10,
        minimum_winning_days=8,
    )

    assert threshold == 1.10
    assert source == "pre_evaluation_risk_adjusted_temporal_selection"


def test_threshold_selection_rejects_concentrated_returns() -> None:
    threshold, source = select_policy_threshold(
        [{
            "ev_threshold": 1.10,
            "tickets": 180,
            "hits": 18,
            "effective_hit_count": 9.9,
            "winning_days": 12,
            "roi": 1.4,
            "roi_without_largest_hit": 1.01,
            "selection_roi_ci95_lower": 1.10,
            "selection_probability_roi_above_one": 0.98,
            "profit_yen": 4_000,
        }],
        fallback=1.20,
        minimum_tickets=100,
        minimum_roi=1.05,
        minimum_hits=10,
        minimum_winning_days=8,
    )

    assert threshold == 1.20
    assert source == "fallback_fixed_threshold"


def test_threshold_selection_requires_roi_probability() -> None:
    row = {
        "ev_threshold": 1.10,
        "tickets": 180,
        "hits": 18,
        "effective_hit_count": 12.0,
        "winning_days": 12,
        "roi": 1.4,
        "roi_without_largest_hit": 1.2,
        "selection_roi_ci95_lower": 1.10,
        "selection_probability_roi_above_one": 0.949999,
        "profit_yen": 4_000,
    }

    threshold, source = select_policy_threshold(
        [row],
        fallback=1.20,
        minimum_tickets=100,
        minimum_roi=1.05,
        minimum_hits=10,
        minimum_winning_days=8,
    )

    assert threshold == 1.20
    assert source == "fallback_fixed_threshold"


def test_threshold_selection_rejects_sparse_wins() -> None:
    threshold, source = select_policy_threshold(
        [
            {
                "ev_threshold": 1.25,
                "tickets": 102,
                "hits": 3,
                "winning_days": 3,
                "roi": 1.21,
                "profit_yen": 5_000,
            }
        ],
        fallback=1.20,
        minimum_tickets=100,
        minimum_roi=1.05,
        minimum_hits=10,
        minimum_winning_days=8,
    )

    assert threshold == 1.20
    assert source == "fallback_fixed_threshold"


def test_threshold_selection_falls_back_without_evidence() -> None:
    threshold, source = select_policy_threshold(
        [{"ev_threshold": 1.05, "tickets": 99, "roi": 2.0, "profit_yen": 100}],
        fallback=1.20,
        minimum_tickets=100,
        minimum_roi=1.05,
    )

    assert threshold == 1.20
    assert source == "fallback_fixed_threshold"
