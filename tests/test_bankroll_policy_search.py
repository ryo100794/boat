from __future__ import annotations

import pytest

from boatrace_ai.bankroll_policy_search import (
    CONSERVATIVE_POLICY_ANCHORS,
    SPARSE_POLICY_ANCHORS,
    TAIL_POLICY_ANCHORS,
    canonicalize_policy_candidates,
    policy_candidates,
    promotion_gate,
    recent_allocation_diagnostics,
    slice_day_range,
    successive_halving_search,
)
from boatrace_ai.packed_bankroll import pack_candidates


POLICY = {
    "daily_budget_yen": 10_000,
    "ev_threshold": 1.0,
    "min_ticket_probability": 0.0,
    "max_estimated_odds": None,
    "payout_prior_weight": 30.0,
    "fractional_kelly": 0.25,
    "max_daily_exposure_fraction": 0.60,
    "min_daily_exposure_fraction": 0.40,
    "race_cap_fraction": 0.10,
    "ticket_cap_fraction": 0.03,
    "max_daily_tickets": 30,
    "allocation_mode": "normalized_kelly",
    "stake_granularity_yen": 100,
    "min_stake_yen": 100,
}


def _packed_days():
    candidates_by_date = {}
    evaluated = {}
    for day in range(1, 9):
        date = f"2026-07-{day:02d}"
        candidates_by_date[date] = [
            {
                "race_id": f"r{day}",
                "estimated_odds": 8.0,
                "estimated_ev": 1.6,
                "probability": 0.2,
                "actual_payout_yen": 800,
                "hit": day % 2 == 0,
            },
            {
                "race_id": f"r{day}",
                "estimated_odds": 10.0,
                "estimated_ev": 1.1,
                "probability": 0.11,
                "actual_payout_yen": 1000,
                "hit": False,
            },
        ]
        evaluated[date] = 1
    return pack_candidates(candidates_by_date, evaluated)


def test_policy_candidates_are_unique_and_reproducible() -> None:
    first = policy_candidates(POLICY, count=16, seed=7)
    second = policy_candidates(POLICY, count=16, seed=7)
    assert first == second
    assert first[0] == {**POLICY, "min_estimated_odds": None}
    assert len({
        tuple(candidate[key] for key in (
            "ev_threshold", "min_ticket_probability", "min_estimated_odds",
            "max_estimated_odds",
            "fractional_kelly", "max_daily_exposure_fraction",
            "min_daily_exposure_fraction", "race_cap_fraction",
            "ticket_cap_fraction", "max_daily_tickets",
        ))
        for candidate in first
    }) == 16


def test_legacy_base_policy_is_normalized_to_current_search_schema() -> None:
    legacy = {
        key: value
        for key, value in POLICY.items()
        if key not in {"min_ticket_probability", "max_estimated_odds"}
    }

    candidate = policy_candidates(legacy, count=1)[0]

    assert candidate["min_ticket_probability"] == 0.0
    assert candidate["min_estimated_odds"] is None
    assert candidate["max_estimated_odds"] is None


def test_candidate_registry_has_canonical_order_sensitive_hash() -> None:
    first = policy_candidates(POLICY, count=4, seed=7)
    reordered_keys = [{key: row[key] for key in reversed(row)} for row in first]

    normalized, digest = canonicalize_policy_candidates(first)
    same_normalized, same_digest = canonicalize_policy_candidates(reordered_keys)

    assert normalized == same_normalized
    assert digest == same_digest
    assert len(digest) == 64
    _, reverse_digest = canonicalize_policy_candidates(tuple(reversed(first)))
    assert reverse_digest != digest


def test_external_candidates_are_used_without_regeneration(monkeypatch) -> None:
    registered = policy_candidates(POLICY, count=4, seed=9)

    def unexpected_generation(*args, **kwargs):
        raise AssertionError("external registry must prevent candidate generation")

    monkeypatch.setattr(
        "boatrace_ai.bankroll_policy_search.policy_candidates",
        unexpected_generation,
    )
    result = successive_halving_search(
        _packed_days(),
        POLICY,
        finalists=2,
        bootstrap_samples=100,
        candidates=registered,
        seed=7,
    )

    assert result["candidate_count"] == 4
    assert result["policy_candidates_sha256"] == canonicalize_policy_candidates(
        registered
    )[1]


def test_external_candidates_reject_duplicates_and_invalid_values() -> None:
    candidate = policy_candidates(POLICY, count=1, seed=7)[0]
    with pytest.raises(ValueError, match="duplicate"):
        successive_halving_search(
            _packed_days(), POLICY, finalists=1, candidates=[candidate, candidate]
        )

    invalid = {**candidate, "fractional_kelly": float("nan")}
    with pytest.raises(ValueError, match="invalid policy candidate"):
        successive_halving_search(
            _packed_days(), POLICY, finalists=1, candidates=[invalid]
        )

    missing = dict(candidate)
    del missing["daily_budget_yen"]
    with pytest.raises(ValueError, match="missing required fields"):
        canonicalize_policy_candidates([missing])


def test_successive_halving_bootstraps_only_finalists() -> None:
    result = successive_halving_search(
        _packed_days(),
        POLICY,
        candidate_count=9,
        finalists=4,
        bootstrap_samples=100,
        seed=7,
    )
    assert [row["evaluated_candidates"] for row in result["stages"]] == [9, 4, 4]
    assert len(result["finalists"]) == 4
    finalist_policies = [row["policy"] for row in result["finalists"]]
    assert [
        row["protected_anchor_count"] for row in result["stages"]
    ] == [1, 1, 0]
    assert result["selected"] == result["finalists"][0]
    assert "roi_ci95_lower" in result["selected"]["confidence"]
    assert len(result["selected"]["temporal_stability"]["folds"]) == 3
    assert (
        "minimum_temporal_roi_above_one"
        in result["selected"]["promotion_gate"]
    )


def test_policy_candidates_preserve_conservative_anchors() -> None:
    candidates = policy_candidates(POLICY, count=10, seed=7)

    for overrides in CONSERVATIVE_POLICY_ANCHORS:
        assert {**POLICY, "min_estimated_odds": None, **overrides} in candidates
    for overrides in TAIL_POLICY_ANCHORS:
        assert {**POLICY, "min_estimated_odds": None, **overrides} in candidates
    for overrides in SPARSE_POLICY_ANCHORS:
        assert {**POLICY, "min_estimated_odds": None, **overrides} in candidates

    assert max(candidate["ev_threshold"] for candidate in candidates) >= 2.0

    assert any(candidate["min_estimated_odds"] == 6.0 for candidate in candidates)
    assert any(
        candidate["min_estimated_odds"] is not None
        and candidate["min_estimated_odds"] >= 100.0
        for candidate in candidates
    )


def test_eight_finalists_are_not_occupied_by_forced_anchor_hypotheses() -> None:
    result = successive_halving_search(
        _packed_days(),
        POLICY,
        candidate_count=12,
        finalists=8,
        bootstrap_samples=100,
        seed=7,
    )

    assert [
        row["protected_anchor_count"] for row in result["stages"]
    ] == [2, 2, 0]
    assert len(result["finalists"]) == 8


def test_candidate_registry_canonicalizes_legacy_minimum_odds() -> None:
    legacy = dict(POLICY)
    normalized, digest = canonicalize_policy_candidates([legacy])
    explicit, explicit_digest = canonicalize_policy_candidates([
        {**legacy, "min_estimated_odds": None}
    ])

    assert normalized[0]["min_estimated_odds"] is None
    assert normalized == explicit
    assert digest == explicit_digest


def test_candidate_registry_rejects_inverted_odds_band() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        canonicalize_policy_candidates([{
            **POLICY,
            "min_estimated_odds": 101.0,
            "max_estimated_odds": 100.0,
        }])


def test_slice_day_range_rebases_offsets() -> None:
    packed = _packed_days()
    sliced = slice_day_range(packed, 2, 5)
    assert sliced.dates == ("2026-07-03", "2026-07-04", "2026-07-05")
    assert sliced.offsets.tolist() == [0, 2, 4, 6]
    assert sliced.tickets == 6
    assert sliced.evaluated_races.tolist() == [1, 1, 1]


def test_recent_allocation_diagnostics_flags_purchase_spike() -> None:
    daily = [
        {"stake_yen": 100, "tickets": 1}
        for _ in range(21)
    ] + [
        {"stake_yen": 800, "tickets": 8}
        for _ in range(7)
    ]

    diagnostics = recent_allocation_diagnostics(daily)

    assert diagnostics["stake_multiplier"] == 8.0
    assert diagnostics["ticket_multiplier"] == 8.0
    assert diagnostics["stable"] is False


def test_recent_allocation_diagnostics_accepts_normal_variation() -> None:
    daily = [
        {"stake_yen": 200, "tickets": 2}
        for _ in range(21)
    ] + [
        {"stake_yen": 300, "tickets": 3}
        for _ in range(7)
    ]

    assert recent_allocation_diagnostics(daily)["stable"] is True


def test_promotion_gate_requires_stable_diverse_bounded_evidence() -> None:
    row = {
        "metrics": {
            "tickets": 300,
            "hit_tickets": 30,
            "selected_races": 100,
            "days_with_bets": 60,
            "stake_yen": 100_000,
            "max_drawdown_yen": 50_000,
            "roi": 1.10,
        },
        "confidence": {
            "roi_ci95_lower": 1.01,
            "probability_roi_above_one": 0.96,
        },
        "temporal_stability": {
            "all_minimum_evidence": True,
            "minimum_roi": 1.01,
        },
        "recent_allocation": {"stable": True},
    }

    assert all(promotion_gate(row).values())

    for key, value in (
        ("selected_races", 99),
        ("days_with_bets", 59),
        ("max_drawdown_yen", 50_001),
    ):
        failed = {**row, "metrics": {**row["metrics"], key: value}}
        assert all(promotion_gate(failed).values()) is False

    unstable = {
        **row,
        "temporal_stability": {
            "all_minimum_evidence": True,
            "minimum_roi": 0.99,
        },
    }
    assert promotion_gate(unstable)["minimum_temporal_roi_above_one"] is False
