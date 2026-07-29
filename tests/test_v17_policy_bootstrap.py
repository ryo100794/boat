from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

import boatrace_ai.listwise.market_calibration as market_calibration
from boatrace_ai.bankroll_bootstrap import bootstrap_daily_roi
from boatrace_ai.listwise.market_calibration import (
    _v17_batch_bootstrap_roi_lowers,
    select_policy_v17,
)


COMBINATIONS = tuple(
    "-".join(map(str, values)) for values in permutations(range(1, 7), 3)
)


def _daily(*values: tuple[str, int, int]) -> list[dict[str, object]]:
    return [
        {"race_date": day, "stake_yen": stake, "return_yen": returned}
        for day, stake, returned in values
    ]


def _race(day: str, rno: int, *, payout: int = 500) -> dict[str, object]:
    market = {key: 0.8 / (len(COMBINATIONS) - 1) for key in COMBINATIONS}
    model = {key: 0.65 / (len(COMBINATIONS) - 1) for key in COMBINATIONS}
    market["1-2-3"] = 0.2
    model["1-2-3"] = 0.35
    return {
        "race_id": f"{day}-01-{rno:02d}",
        "race_date": day,
        "jcd": "01",
        "rno": rno,
        "actual_combination": "1-2-3",
        "actual_payout_yen": payout,
        "model_probabilities": model,
        "market_probabilities": market,
        "odds": {key: 0.8 / probability for key, probability in market.items()},
        "snapshot_id": rno,
    }


def test_v17_batch_bootstrap_is_bit_exact_with_individual_evaluation() -> None:
    policies = [
        _daily(
            ("2026-07-18", 100, 0),
            ("2026-07-19", 200, 500),
            ("2026-07-20", 0, 0),
        ),
        _daily(
            ("2026-07-18", 0, 0),
            ("2026-07-19", 100, 1_230),
            ("2026-07-20", 100, 0),
        ),
        _daily(
            ("2026-07-18", 0, 0),
            ("2026-07-19", 0, 0),
            ("2026-07-20", 0, 0),
        ),
        _daily(
            ("2026-07-18", 100, 0),
            ("2026-07-19", 200, 500),
            ("2026-07-20", 0, 0),
        ),
    ]
    expected = [
        bootstrap_daily_roi(daily, samples=2_000)["roi_ci95_lower"]
        for daily in policies
    ]

    first = _v17_batch_bootstrap_roi_lowers(policies)
    second = _v17_batch_bootstrap_roi_lowers(policies)

    assert first == expected
    assert second == expected


def test_v17_batch_bootstrap_rejects_different_strict_prior_boundaries() -> None:
    with pytest.raises(ValueError, match="same date boundary"):
        _v17_batch_bootstrap_roi_lowers([
            _daily(("2026-07-18", 100, 200)),
            _daily(("2026-07-19", 100, 200)),
        ])


def test_v17_batch_bootstrap_matches_individual_for_varied_daily_arrays() -> None:
    rng = np.random.default_rng(731)
    days = tuple(f"2026-07-{day:02d}" for day in range(18, 26))
    policies = []
    for _ in range(32):
        stakes = rng.integers(0, 31, size=len(days)) * 100
        returns = rng.integers(0, 301, size=len(days)) * 100
        policies.append(_daily(*zip(days, stakes.tolist(), returns.tolist())))

    expected = [
        bootstrap_daily_roi(daily, samples=2_000)["roi_ci95_lower"]
        for daily in policies
    ]

    assert _v17_batch_bootstrap_roi_lowers(policies) == expected


def test_select_policy_v17_batches_bootstrap_once(monkeypatch) -> None:
    calls = 0
    original = market_calibration._v17_batch_bootstrap_roi_lowers

    def counted(rows):
        nonlocal calls
        calls += 1
        return original(rows)

    monkeypatch.setattr(
        market_calibration,
        "_v17_batch_bootstrap_roi_lowers",
        counted,
    )
    monkeypatch.setattr(
        market_calibration,
        "bootstrap_daily_roi",
        lambda *args, **kwargs: pytest.fail("per-policy bootstrap was called"),
    )
    policies = [
        {"name": "no_bet", "no_bet": True},
        {
            "name": "candidate",
            "ev_threshold": 1.0,
            "max_estimated_ev": None,
            "max_odds": None,
            "max_tickets_per_race": 1,
            "min_model_market_ratio": 1.0,
            "staking_mode": "kelly_025",
        },
    ]
    races = [_race("2026-07-18", rno) for rno in range(1, 13)]

    selected, rows = select_policy_v17(
        races,
        calibrator={"model_weight": 1.0, "temperature": 1.0},
        daily_budget_yen=10_000,
        policies=policies,
    )

    assert selected["name"] in {"no_bet", "candidate"}
    assert calls == 1
    assert len(rows) == len(policies)
    assert all("daily_cluster_bootstrap_roi_lower_95" in row for row in rows)
