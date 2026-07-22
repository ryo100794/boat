from __future__ import annotations

import numpy as np

from boatrace_ai.listwise.direct_bankroll import (
    COMBINATION_LABELS,
    direct_candidates,
    simulate_direct_bankroll,
    standard_direct_policy,
)


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
