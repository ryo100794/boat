from __future__ import annotations

from copy import deepcopy

from boatrace_ai.chronological_bankroll import simulate_chronological_bankroll_day
from boatrace_ai.listwise import market_calibration


DATE = "2026-07-30"


def _candidate(index: int) -> dict:
    return {
        "race_id": f"race-{index}",
        "race_date": DATE,
        "rno": index,
        "combination": "1-2-3",
        "probability": 0.25,
        "estimated_odds": 8.0,
        "estimated_ev": 2.0,
        "decision_at": f"{DATE}T12:{index * 10:02d}:00+09:00",
    }


def _event(index: int, payout: int) -> dict:
    return {
        "race_id": f"race-{index}",
        "result_available_at": f"{DATE}T12:{index * 10 + 5:02d}:00+09:00",
        "payouts": {"1-2-3": payout},
    }


def _one_ticket_allocator(
    race_date, candidates, evaluated_races, *, settlements, **kwargs
):
    candidate = dict(candidates[0])
    payout = int(
        settlements.get((candidate["race_id"], candidate["combination"]), 0)
    )
    return {
        "allocation_candidate_tickets": 1,
        "selected_sample": [{
            **candidate,
            "stake_yen": 100,
            "return_yen": payout,
            "hit": payout > 0,
        }],
    }


def _simulate(limit: int, payouts: list[int]) -> dict:
    candidates = [_candidate(index) for index in range(4)]
    return simulate_chronological_bankroll_day(
        DATE,
        candidates,
        {row["race_id"] for row in candidates},
        settlement_events=[
            _event(index, payout) for index, payout in enumerate(payouts)
        ],
        schedule=deepcopy(candidates),
        max_daily_tickets=limit,
        initial_bankroll_yen=10_000,
        max_decision_exposure_fraction=1.0,
        race_cap_fraction=1.0,
        ticket_cap_fraction=1.0,
        allocate_day=_one_ticket_allocator,
    )


def test_v18_learns_lower_quartile_from_v17_prior_daily_tickets() -> None:
    control = market_calibration.learn_v18_daily_ticket_control([
        {"tickets": value} for value in (16, 14, 30, 30, 15, 10)
    ])

    assert control["prior_daily_ticket_counts"] == [10, 14, 15, 16, 30, 30]
    assert control["learned_daily_ticket_limit"] == 14
    assert control["stake_granularity_yen"] == 100
    assert control["result_or_payout_fields_used"] is False


def test_v18_schedule_quota_spreads_learned_limit_over_known_day() -> None:
    result = _simulate(2, [0, 0, 0, 0])
    decisions = [row for row in result["ledger"] if row["event"] == "decision"]

    assert [row["cumulative_ticket_quota"] for row in decisions] == [0, 1, 1, 2]
    assert [row["tickets"] for row in decisions] == [0, 1, 0, 1]
    assert result["tickets"] == 2
    assert result["learned_daily_ticket_limit"] == 2
    assert result["schedule_races_total"] == 4
    assert result["stake_yen"] == 200
    assert result["stake_granularity_yen"] == 100
    assert result["real_betting_enabled"] is False
    assert result["gross_stake_allowance_rule"] == (
        "initial_allowance_plus_positive_part_of_cumulative_net_realized_profit"
    )


def test_v18_zero_ticket_day_is_valid() -> None:
    result = _simulate(0, [99_900, 99_900, 99_900, 99_900])

    assert result["tickets"] == 0
    assert result["stake_yen"] == 0
    assert result["return_yen"] == 0
    assert result["closing_bankroll_yen"] == 10_000


def test_v18_schedule_selections_do_not_use_result_or_payout() -> None:
    losing = _simulate(2, [0, 0, 0, 0])
    winning = _simulate(2, [50_000, 50_000, 50_000, 50_000])

    losing_decisions = [
        row for row in losing["ledger"] if row["event"] == "decision"
    ]
    winning_decisions = [
        row for row in winning["ledger"] if row["event"] == "decision"
    ]
    assert [row["selections"] for row in losing_decisions] == [
        row["selections"] for row in winning_decisions
    ]
    assert losing["decision_information_sha256"] == (
        winning["decision_information_sha256"]
    )
    assert losing["return_yen"] != winning["return_yen"]


def test_v18_selector_attaches_control_without_changing_v17_policy(
    monkeypatch,
) -> None:
    selected = {
        "name": "v17-selected",
        "ev_threshold": 1.0,
        "max_estimated_ev": None,
        "max_odds": None,
        "max_tickets_per_race": 1,
        "min_model_market_ratio": 1.0,
        "staking_mode": "kelly_025",
    }
    monkeypatch.setattr(
        market_calibration,
        "select_policy_v17",
        lambda *args, **kwargs: (dict(selected), [{"policy": dict(selected)}]),
    )
    monkeypatch.setattr(
        market_calibration,
        "simulate_policy",
        lambda *args, **kwargs: {
            "daily": [{"tickets": value} for value in (16, 14, 30, 30, 15, 10)]
        },
    )
    races = [{
        "race_id": "prior-1",
        "race_date": "2026-07-29",
        "model_probabilities": {"1-2-3": 0.5, "1-3-2": 0.5},
        "market_probabilities": {"1-2-3": 0.5, "1-3-2": 0.5},
    }]

    v18, rows = market_calibration.select_policy_v18(
        races,
        calibrator={"model_weight": 1.0, "temperature": 1.0},
        daily_budget_yen=10_000,
    )

    assert {key: v18[key] for key in selected} == selected
    assert v18["v18_ticket_control"]["learned_daily_ticket_limit"] == 14
    assert rows == [{"policy": selected}]


def test_v18_identity_parser_and_primary_mode() -> None:
    strategy = market_calibration.V18_STRATEGY_NAME
    args = market_calibration.build_parser().parse_args([
        "--from-date", "2026-07-18",
        "--calibrator-strategy", strategy,
    ])

    assert args.calibrator_strategy == strategy
    assert market_calibration.odds_path_model_name(strategy) == strategy
    assert strategy in market_calibration.ROBUST_POLICY_STRATEGIES
    assert strategy != market_calibration.V17_STRATEGY_NAME
