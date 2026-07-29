from __future__ import annotations

from copy import deepcopy

from boatrace_ai.chronological_bankroll import (
    decision_information_fingerprint,
    simulate_chronological_bankroll_day,
)


DATE = "2026-07-29"


def _candidate(race_id: str, at: str) -> dict:
    return {
        "race_id": race_id,
        "race_date": DATE,
        "combination": "1-2-3",
        "probability": 0.20,
        "estimated_odds": 10.0,
        "decision_at": f"{DATE}T{at}:00+09:00",
    }


def _event(race_id: str, at: str, payout: int) -> dict:
    return {
        "race_id": race_id,
        "result_available_at": f"{DATE}T{at}:00+09:00",
        "result_available_at_source": "result_available_at",
        "payouts": {"1-2-3": payout},
    }


def _simulate(candidates: list[dict], events: list[dict]) -> dict:
    return simulate_chronological_bankroll_day(
        DATE,
        candidates,
        {str(row["race_id"]) for row in candidates},
        settlement_events=events,
        initial_bankroll_yen=10_000,
        max_decision_exposure_fraction=1.0,
        race_cap_fraction=1.0,
        ticket_cap_fraction=1.0,
        max_tickets_per_race=1,
    )


def _decisions(result: dict) -> list[dict]:
    return [row for row in result["ledger"] if row["event"] == "decision"]


def test_unsettled_stake_is_not_reused_and_settled_profit_is_reinvested() -> None:
    candidates = [
        _candidate("race-1", "12:00"),
        _candidate("race-2", "12:05"),
        _candidate("race-3", "12:20"),
    ]
    result = _simulate(candidates, [
        _event("race-1", "12:10", 1_000),
        _event("race-2", "12:25", 1_000),
        _event("race-3", "12:30", 1_000),
    ])
    decisions = _decisions(result)
    settlements = [row for row in result["ledger"] if row["event"] == "settlement"]
    assert settlements[0]["outstanding_stake_yen"] == decisions[1]["stake_yen"]
    assert decisions[0]["cash_before_yen"] == 10_000
    assert decisions[1]["cash_before_yen"] < decisions[0]["cash_before_yen"]
    assert decisions[1]["outstanding_stake_yen"] > decisions[0]["stake_yen"]
    assert decisions[2]["cash_before_yen"] > decisions[0]["cash_before_yen"]
    assert result["closing_bankroll_yen"] > result["initial_bankroll_yen"]
    assert result["outstanding_stake_yen"] == 0
    assert result["profit_reinvestment"] is True
    assert result["real_betting_enabled"] is False
    assert all(row["stake_yen"] % 100 == 0 for row in decisions)


def test_non_round_settlement_keeps_cash_remainder_and_bets_in_units() -> None:
    candidates = [
        _candidate("race-1", "12:00"),
        _candidate("race-2", "12:20"),
    ]
    result = _simulate(candidates, [
        _event("race-1", "12:10", 490),
        _event("race-2", "12:30", 490),
    ])
    decisions = _decisions(result)
    assert decisions[1]["cash_before_yen"] % 100 == 90
    assert all(row["stake_yen"] % 100 == 0 for row in decisions)
    assert result["closing_bankroll_yen"] == (
        10_000 - result["stake_yen"] + result["return_yen"]
    )


def test_zero_tickets_is_a_valid_decision() -> None:
    candidate = _candidate("race-1", "12:00")
    candidate.update(probability=0.01, estimated_odds=2.0)
    result = _simulate([candidate], [_event("race-1", "12:10", 500)])
    assert result["tickets"] == 0
    assert result["stake_yen"] == 0
    assert result["closing_bankroll_yen"] == 10_000
    assert _decisions(result)[0]["selections"] == []


def test_result_payload_cannot_change_decision_information_or_unsettled_bets() -> None:
    candidates = [
        _candidate("race-1", "12:00"),
        _candidate("race-2", "12:05"),
    ]
    contaminated = deepcopy(candidates)
    contaminated[0].update({
        "actual_combination": "6-5-4",
        "actual_payout_yen": 9_999_900,
        "hit": False,
        "payout_yen": 9_999_900,
        "result_available_at": f"{DATE}T12:01:00+09:00",
    })
    low = [_event("race-1", "12:10", 500), _event("race-2", "12:15", 500)]
    high = [
        _event("race-1", "12:10", 50_000),
        _event("race-2", "12:15", 50_000),
    ]
    first = _simulate(candidates, low)
    second = _simulate(contaminated, high)
    assert decision_information_fingerprint(candidates) == (
        decision_information_fingerprint(contaminated)
    )
    assert first["decision_information_sha256"] == second[
        "decision_information_sha256"
    ]
    first_decisions = _decisions(first)
    second_decisions = _decisions(second)
    assert [row["selections"] for row in first_decisions] == [
        row["selections"] for row in second_decisions
    ]
    assert [row["decision_information_sha256"] for row in first_decisions] == [
        row["decision_information_sha256"] for row in second_decisions
    ]
    assert first["return_yen"] != second["return_yen"]
