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
    )
    assert threshold == 1.05
    assert source == "pre_evaluation_temporal_selection"


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

    assert diagnostics[0]["tickets"] == 4
    assert diagnostics[1]["tickets"] == 2
    assert diagnostics[1]["roi"] > diagnostics[0]["roi"]
    assert diagnostics[1]["profit_yen"] > 0


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
