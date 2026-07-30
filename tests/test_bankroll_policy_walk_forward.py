from __future__ import annotations

from datetime import date, timedelta

import pytest

from boatrace_ai.bankroll_policy_search import (
    canonicalize_policy_candidates,
    policy_candidates,
)
from boatrace_ai.listwise import bankroll_policy_walk_forward as walk_forward
from boatrace_ai.listwise.bankroll_policy_walk_forward import (
    aggregate_outer_folds,
    build_annual_walk_forward_folds,
    evaluate_annual_walk_forward,
    nested_promotion_gate,
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


def _dates(start: str, count: int) -> list[str]:
    first = date.fromisoformat(start)
    return [(first + timedelta(days=index)).isoformat() for index in range(count)]


def _packed(dates: list[str]):
    candidates = {
        race_date: [
            {
                "race_id": f"{race_date}-01-01",
                "estimated_odds": 4.0,
                "estimated_ev": 1.2,
                "probability": 0.3,
                "actual_payout_yen": 400,
                "hit": True,
            }
        ]
        for race_date in dates
    }
    return pack_candidates(candidates, {value: 1 for value in dates})


def test_build_annual_folds_uses_contiguous_non_overlapping_outer_days() -> None:
    folds = build_annual_walk_forward_folds(
        _dates("2020-01-01", 10),
        selection_days=4,
        outer_days=2,
        folds=3,
    )

    assert [row["holdout_dates"] for row in folds] == [
        ("2020-01-05", "2020-01-06"),
        ("2020-01-07", "2020-01-08"),
        ("2020-01-09", "2020-01-10"),
    ]
    assert folds[0]["selection_dates"] == (
        "2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"
    )
    assert folds[1]["selection_dates"] == (
        "2020-01-03", "2020-01-04", "2020-01-05", "2020-01-06"
    )
    assert all(row["boundary_audit"]["passed"] for row in folds)
    all_holdout = [day for row in folds for day in row["holdout_dates"]]
    assert len(all_holdout) == len(set(all_holdout))


def test_build_annual_folds_audits_calendar_gaps_and_embargo() -> None:
    dates = _dates("2020-01-01", 11)
    dates.remove("2020-01-06")
    folds = build_annual_walk_forward_folds(
        dates,
        selection_days=4,
        outer_days=2,
        folds=2,
        embargo_days=2,
    )

    assert folds[0]["embargo_dates"] == ("2020-01-05", "2020-01-07")
    assert all(row["boundary_audit"]["source_window_contiguous"] is False for row in folds)
    assert all(row["boundary_audit"]["passed"] is False for row in folds)


def test_nested_evaluation_generates_registry_once_and_isolates_holdout(
    monkeypatch,
) -> None:
    inputs = []
    start = date(2026, 1, 1)
    for index in range(5):
        holdout_date = start + timedelta(days=4 + index)
        selection_dates = [
            (holdout_date - timedelta(days=offset)).isoformat()
            for offset in range(4, 0, -1)
        ]
        inputs.append(
            {
                "fold": index + 1,
                "selection": _packed(selection_dates),
                "holdout": _packed([holdout_date.isoformat()]),
                "boundary_audit": {"passed": True},
            }
        )

    generated = policy_candidates(POLICY, count=4, seed=3)
    generated_count = 0
    search_inputs = []
    registry_ids = []
    evaluated_inputs = []

    def generate_once(*args, **kwargs):
        nonlocal generated_count
        generated_count += 1
        return generated

    def search(selection, base_policy, **kwargs):
        search_inputs.append(selection)
        registry_ids.append(id(kwargs["candidates"]))
        digest = canonicalize_policy_candidates(kwargs["candidates"])[1]
        return {
            "policy_candidates_sha256": digest,
            "selected": {"policy": dict(kwargs["candidates"][0])},
        }

    def evaluate(holdout, policy):
        evaluated_inputs.append(holdout)
        daily = [
            {
                "race_date": value,
                "stake_yen": 6_000,
                "return_yen": 12_000,
            }
            for value in holdout.dates
        ]
        return {
            "evaluated_races": 20,
            "selected_races": 20,
            "tickets": 60,
            "hit_tickets": 4,
            "hit_races": 4,
            "stake_yen": 6_000,
            "return_yen": 12_000,
            "profit_yen": 6_000,
            "roi": 2.0,
            "ticket_hit_rate": 4 / 60,
            "race_hit_rate": 0.2,
            "max_drawdown_yen": 0,
            "days_with_bets": 1,
            "winning_days": 1,
            "losing_days": 0,
            "daily": daily,
        }

    monkeypatch.setattr(walk_forward, "policy_candidates", generate_once)
    monkeypatch.setattr(walk_forward, "successive_halving_search", search)
    monkeypatch.setattr(walk_forward, "evaluate_packed_policy", evaluate)
    monkeypatch.setattr(walk_forward, "_largest_selected_hit_return", lambda *args: 1_000)

    result = evaluate_annual_walk_forward(
        inputs,
        POLICY,
        candidate_count=4,
        finalists=2,
        outer_days=1,
        selection_days=4,
        aggregate_bootstrap_samples=100,
    )

    assert generated_count == 1
    assert search_inputs == [row["selection"] for row in inputs]
    assert evaluated_inputs == [row["holdout"] for row in inputs]
    assert len(evaluated_inputs) == 5
    assert len(set(registry_ids)) == 1
    assert result["candidate_count"] == 4
    assert all(
        row["policy_candidates_sha256"] == result["policy_candidates_sha256"]
        for row in result["folds"]
    )


def test_nested_evaluation_rejects_job_3995_before_selection(monkeypatch) -> None:
    called = False

    def search(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("legacy source must be rejected before search")

    monkeypatch.setattr(walk_forward, "successive_halving_search", search)
    with pytest.raises(ValueError, match="legacy job 3995"):
        evaluate_annual_walk_forward(
            [
                {
                    "fold": 1,
                    "selection": _packed(_dates("2026-01-01", 4)),
                    "holdout": _packed(["2026-01-05"]),
                    "provenance": {"policy_source_job_id": 3995},
                }
            ],
            POLICY,
            candidates=policy_candidates(POLICY, count=2, seed=3),
            finalists=1,
        )
    assert called is False


def test_aggregate_and_nested_gate_require_all_profit_and_evidence() -> None:
    selected_policy = policy_candidates(POLICY, count=1, seed=3)[0]
    selected_hash = canonicalize_policy_candidates([selected_policy])[1]
    rows = []
    for fold in range(5):
        first = date(2020, 1, 1) + timedelta(days=fold * 12)
        daily = [
            {
                "race_date": (first + timedelta(days=index)).isoformat(),
                "stake_yen": 500,
                "return_yen": 1_000,
            }
            for index in range(12)
        ]
        metrics = {
            "evaluated_races": 100,
            "selected_races": 20,
            "tickets": 60,
            "hit_tickets": 4,
            "hit_races": 4,
            "stake_yen": 6_000,
            "return_yen": 12_000,
            "roi": 2.0,
            "days_with_bets": 12,
        }
        rows.append(
            {
                "fold": fold + 1,
                "boundary_audit": {"passed": True},
                "selected_policy_sha256": selected_hash,
                "selected_policy": selected_policy,
                "holdout_metrics": metrics,
                "holdout_daily": daily,
                "largest_hit_return_yen": 1_000,
                "minimum_evidence": {
                    "minimum_tickets": True,
                    "minimum_hits": True,
                    "minimum_selected_races": True,
                    "minimum_purchase_days": True,
                },
            }
        )

    aggregate = aggregate_outer_folds(rows, bootstrap_samples=200, seed=7)
    gate = nested_promotion_gate(aggregate)

    assert aggregate["roi"] == 2.0
    assert aggregate["minimum_fold_roi"] == 2.0
    assert aggregate["profitable_folds"] == 5
    assert aggregate["largest_hit_excluded_roi"] == pytest.approx(59_000 / 30_000)
    assert aggregate["bootstrap"]["roi_ci95_lower"] == 2.0
    assert aggregate["bootstrap"]["probability_roi_above_one"] == 1.0
    assert aggregate["selection_frequency"][0]["count"] == 5
    assert all(gate.values())

    rows[3]["holdout_daily"] = [
        {**value, "return_yen": 495}
        for value in rows[3]["holdout_daily"]
    ]
    rows[3]["holdout_metrics"] = {
        **rows[3]["holdout_metrics"],
        "return_yen": 5_940,
    }
    failed = aggregate_outer_folds(rows, bootstrap_samples=50, seed=7)
    assert nested_promotion_gate(failed)["all_fold_roi_above_one"] is False
