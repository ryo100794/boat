from __future__ import annotations

from copy import deepcopy

from boatrace_ai.chronological_bankroll import simulate_chronological_bankroll_day
from boatrace_ai.listwise import market_calibration, market_residual


DATE = "2026-07-30"


def _candidate(index: int, *, estimated_ev: float = 2.0) -> dict:
    return {
        "race_id": f"race-{index}",
        "race_date": DATE,
        "rno": index,
        "combination": "1-2-3",
        "probability": 0.25,
        "estimated_odds": 8.0,
        "estimated_ev": estimated_ev,
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


def _simulate(
    limit: int,
    payouts: list[int],
    *,
    rounding: str = "floor",
    opportunity: dict | None = None,
    estimated_evs: list[float] | None = None,
) -> dict:
    candidates = [
        _candidate(
            index,
            estimated_ev=(estimated_evs or [2.0] * 4)[index],
        )
        for index in range(4)
    ]
    return simulate_chronological_bankroll_day(
        DATE,
        candidates,
        {row["race_id"] for row in candidates},
        settlement_events=[
            _event(index, payout) for index, payout in enumerate(payouts)
        ],
        schedule=deepcopy(candidates),
        schedule_quota_rounding=rounding,
        schedule_quota_opportunity=opportunity,
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


def test_ceil_schedule_quota_releases_each_slot_before_day_end() -> None:
    result = _simulate(2, [0, 0, 0, 0], rounding="ceil")
    decisions = [row for row in result["ledger"] if row["event"] == "decision"]

    assert [row["cumulative_ticket_quota"] for row in decisions] == [1, 1, 2, 2]
    assert [row["tickets"] for row in decisions] == [1, 0, 1, 0]
    assert result["schedule_quota_rounding"] == "ceil"
    assert result["schedule_quota_rule"].startswith("ceil(")


def test_opportunity_quota_releases_last_slot_for_late_high_score() -> None:
    result = _simulate(
        2,
        [0, 0, 0, 0],
        opportunity={
            "after_fraction": 0.75,
            "score_quantile": 0.75,
            "reserve_slots": 1,
            "minimum_score": 1.0,
        },
        estimated_evs=[1.2, 1.4, 2.0, 1.1],
    )
    decisions = [row for row in result["ledger"] if row["event"] == "decision"]

    assert [row["tickets"] for row in decisions] == [0, 1, 1, 0]
    assert decisions[2]["opportunity_quota_released"] is True
    assert decisions[2]["opportunity_candidate_score"] == 2.0
    assert decisions[2]["opportunity_score_threshold"] == 1.4
    assert result["schedule_quota_opportunity"]["result_or_payout_fields_used"] is False


def test_opportunity_quota_keeps_last_slot_for_late_low_score() -> None:
    result = _simulate(
        2,
        [0, 0, 0, 0],
        opportunity={
            "after_fraction": 0.75,
            "score_quantile": 0.75,
            "reserve_slots": 1,
            "minimum_score": 1.0,
        },
        estimated_evs=[2.0, 1.8, 1.1, 1.0],
    )
    decisions = [row for row in result["ledger"] if row["event"] == "decision"]

    assert [row["tickets"] for row in decisions] == [0, 1, 0, 1]
    assert decisions[2]["opportunity_quota_released"] is False


def test_online_reserve_preserves_capacity_for_later_high_scores() -> None:
    result = _simulate(
        4,
        [0, 0, 0, 0],
        rounding="ceil",
        opportunity={
            "quota_mode": "online_reserve",
            "after_fraction": 0.50,
            "score_quantile": 0.75,
            "reserve_slots": 2,
            "minimum_score": 1.0,
        },
        estimated_evs=[1.2, 1.1, 3.0, 4.0],
    )
    decisions = [
        row for row in result["ledger"] if row["event"] == "decision"
    ]

    assert [row["tickets"] for row in decisions] == [0, 1, 1, 1]
    assert [row["opportunity_quota_released"] for row in decisions] == [
        False, False, True, True,
    ]
    assert result["tickets"] == 3
    assert (
        result["schedule_quota_opportunity"]["quota_mode"]
        == "online_reserve"
    )


def test_opportunity_quota_decisions_do_not_use_payouts() -> None:
    config = {
        "after_fraction": 0.75,
        "score_quantile": 0.75,
        "reserve_slots": 1,
        "minimum_score": 1.0,
    }
    losing = _simulate(2, [0, 0, 0, 0], opportunity=config)
    winning = _simulate(
        2,
        [50_000, 50_000, 50_000, 50_000],
        opportunity=config,
    )

    losing_decisions = [
        row for row in losing["ledger"] if row["event"] == "decision"
    ]
    winning_decisions = [
        row for row in winning["ledger"] if row["event"] == "decision"
    ]
    assert [row["selections"] for row in losing_decisions] == [
        row["selections"] for row in winning_decisions
    ]
    assert losing["return_yen"] != winning["return_yen"]


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
    def simulate(*args, **kwargs):
        if not kwargs.get("include_chronological"):
            return {
                "daily": [
                    {"tickets": value}
                    for value in (16, 14, 30, 30, 15, 10)
                ]
            }
        rounding = kwargs["policy"]["v18_ticket_control"][
            "schedule_quota_rounding"
        ]
        returned = 200 if rounding == "ceil" else 100
        day = {
            "race_date": "2026-07-29",
            "evaluated_races": 1,
            "tickets": 1,
            "races_bet": 1,
            "hit_races": int(returned > 100),
            "hit_tickets": int(returned > 100),
            "largest_hit_return_yen": returned,
            "hit_return_square_sum_yen2": returned * returned,
            "stake_yen": 100,
            "return_yen": returned,
            "profit_yen": returned - 100,
            "roi": returned / 100,
            "max_drawdown_yen": 0,
        }
        return {
            "chronological_bankroll": {
                "race_days": 1,
                "tickets": 1,
                "hit_tickets": day["hit_tickets"],
                "stake_yen": 100,
                "return_yen": returned,
                "profit_yen": returned - 100,
                "roi": returned / 100,
                "winning_days": int(returned > 100),
                "daily": [day],
            }
        }

    monkeypatch.setattr(market_calibration, "simulate_policy", simulate)
    races = [{
        "race_id": "prior-1",
        "race_date": "2026-07-29",
        "model_probabilities": {"1-2-3": 0.5, "1-3-2": 0.5},
        "market_probabilities": {"1-2-3": 0.5, "1-3-2": 0.5},
        "odds": {"1-2-3": 2.0, "1-3-2": 2.0},
    }]

    v18, rows = market_calibration.select_policy_v18(
        races,
        calibrator={"model_weight": 1.0, "temperature": 1.0},
        daily_budget_yen=10_000,
    )

    assert {key: v18[key] for key in selected} == selected
    control = v18["v18_ticket_control"]
    assert control["learned_daily_ticket_limit"] == 14
    assert control["schedule_quota_rounding"] == "ceil"
    assert control["schedule_quota_rounding_selection"]["selected"] == "ceil"
    assert len(control["schedule_quota_rounding_selection"]["candidates"]) == 2
    assert rows == [{"policy": selected}]


def test_opportunity_policy_selector_uses_prior_bankroll_only(
    monkeypatch,
) -> None:
    def simulate(*args, **kwargs):
        control = kwargs["policy"]["v18_ticket_control"]
        opportunity = control.get("schedule_quota_opportunity") or {}
        selected = (
            float(opportunity.get("after_fraction") or 0.0) == 0.95
            and float(opportunity.get("score_quantile") or 0.0) == 0.90
        )
        returned = 300 if selected else 100
        day = {
            "race_date": "2026-07-29",
            "evaluated_races": 1,
            "tickets": 1,
            "races_bet": 1,
            "hit_races": int(selected),
            "hit_tickets": int(selected),
            "largest_hit_return_yen": returned,
            "hit_return_square_sum_yen2": returned * returned,
            "stake_yen": 100,
            "return_yen": returned,
            "profit_yen": returned - 100,
            "roi": returned / 100,
            "max_drawdown_yen": 0,
        }
        return {
            "chronological_bankroll": {
                "race_days": 1,
                "tickets": 1,
                "hit_tickets": day["hit_tickets"],
                "stake_yen": 100,
                "return_yen": returned,
                "profit_yen": returned - 100,
                "roi": returned / 100,
                "winning_days": int(selected),
                "daily": [day],
            }
        }

    monkeypatch.setattr(market_calibration, "simulate_policy", simulate)
    control, diagnostics = market_calibration.select_v18_schedule_quota_policy(
        [],
        calibrator={"model_weight": 1.0, "temperature": 1.0},
        policy={"ev_threshold": 1.05},
        ticket_control={"learned_daily_ticket_limit": 10},
        daily_budget_yen=10_000,
    )

    assert control["schedule_quota_rounding"] == "floor"
    assert control["schedule_quota_opportunity"] == {
        "after_fraction": 0.95,
        "score_quantile": 0.90,
        "reserve_slots": 1,
        "minimum_score": 1.05,
    }
    assert len(diagnostics) == 14
    assert any(
        row["name"] == "online_reserve_f25_q75_r25"
        and row["schedule_quota_opportunity"]["reserve_slots"] == 3
        for row in diagnostics
    )
    assert all(
        "actual_combination" not in row and "actual_payout_yen" not in row
        for row in diagnostics
    )


def test_schedule_quota_selector_can_choose_strict_prior_maximum_limit(
    monkeypatch,
) -> None:
    def simulate(*args, **kwargs):
        control = kwargs["policy"]["v18_ticket_control"]
        selected = control["learned_daily_ticket_limit"] == 30
        returned = 300 if selected else 100
        day = {
            "race_date": "2026-07-29",
            "evaluated_races": 1,
            "tickets": 1,
            "races_bet": 1,
            "hit_races": int(selected),
            "hit_tickets": int(selected),
            "largest_hit_return_yen": returned,
            "hit_return_square_sum_yen2": returned * returned,
            "stake_yen": 100,
            "return_yen": returned,
            "profit_yen": returned - 100,
            "roi": returned / 100,
            "max_drawdown_yen": 0,
        }
        return {
            "chronological_bankroll": {
                "race_days": 1,
                "tickets": 1,
                "hit_tickets": day["hit_tickets"],
                "stake_yen": 100,
                "return_yen": returned,
                "profit_yen": returned - 100,
                "roi": returned / 100,
                "winning_days": int(selected),
                "daily": [day],
            }
        }

    monkeypatch.setattr(market_calibration, "simulate_policy", simulate)
    control, diagnostics = market_calibration.select_v18_schedule_quota_policy(
        [],
        calibrator={"model_weight": 1.0, "temperature": 1.0},
        policy={"ev_threshold": 1.05},
        ticket_control={
            "learned_daily_ticket_limit": 10,
            "prior_daily_ticket_counts": [10, 20, 30],
        },
        daily_budget_yen=10_000,
    )

    assert control["learned_daily_ticket_limit"] == 30
    assert control["schedule_quota_rounding"] == "ceil"
    assert control["schedule_quota_opportunity"] is None
    assert any(
        row["name"] == "prior_max_ceil"
        and row["learned_daily_ticket_limit"] == 30
        for row in diagnostics
    )


def test_v18_keeps_original_calibrator_and_v19_opts_in(
    monkeypatch,
) -> None:
    calls = []

    def selected(races, *, enforce_raw_nonregression=False):
        calls.append(enforce_raw_nonregression)
        return {
            "final_calibrator": {"model_weight": 0.25, "temperature": 0.9},
            "candidates": [],
        }

    monkeypatch.setattr(
        market_residual, "select_regularization_prequential", selected
    )
    races = [
        {"race_date": "2026-07-28"},
        {"race_date": "2026-07-29"},
    ]

    v18 = market_calibration.fit_market_residual_calibrator(
        races,
        calibrator_strategy=market_calibration.V18_STRATEGY_NAME,
    )
    v19 = market_calibration.fit_market_residual_calibrator(
        races,
        calibrator_strategy=market_calibration.V19_STRATEGY_NAME,
    )

    assert calls == [False, True]
    assert v18["final_calibrator"] == {
        "model_weight": 0.25,
        "temperature": 0.9,
    }
    assert v19["final_calibrator"] == v18["final_calibrator"]
    assert market_calibration.V19_STRATEGY_NAME in (
        market_calibration.SCHEDULE_QUOTA_STRATEGIES
    )


def test_v19_has_distinct_parser_and_model_identity() -> None:
    strategy = market_calibration.V19_STRATEGY_NAME
    args = market_calibration.build_parser().parse_args([
        "--from-date", "2026-07-18",
        "--calibrator-strategy", strategy,
    ])

    assert args.calibrator_strategy == strategy
    assert market_calibration.odds_path_model_name(strategy) == strategy
    assert strategy != market_calibration.V18_STRATEGY_NAME
    assert strategy in market_calibration.ROBUST_POLICY_STRATEGIES


def test_v20_builds_both_heads_from_the_same_strict_prior_rows(
    monkeypatch,
) -> None:
    calls = []

    def fitted(races, *, calibrator_strategy):
        calls.append((calibrator_strategy, [race["race_id"] for race in races]))
        weight = 1.0 if calibrator_strategy == market_calibration.V19_STRATEGY_NAME else 0.25
        return {
            "dates": sorted({race["race_date"] for race in races}),
            "final_calibrator": {"model_weight": weight, "temperature": 1.0},
            "candidates": [],
        }

    monkeypatch.setattr(
        market_calibration, "fit_market_residual_calibrator", fitted
    )
    prior = [
        {"race_id": "prior-1", "race_date": "2026-07-28"},
        {"race_id": "prior-2", "race_date": "2026-07-29"},
    ]

    dual = market_calibration.fit_v20_dual_head_calibrators(prior)

    assert calls == [
        (market_calibration.V19_STRATEGY_NAME, ["prior-1", "prior-2"]),
        (market_calibration.V18_STRATEGY_NAME, ["prior-1", "prior-2"]),
    ]
    assert dual["outer_holdout_used"] is False
    assert dual["training_dates"] == ["2026-07-28", "2026-07-29"]
    assert dual["probability_head"]["raw_nonregression_enforced"] is True
    assert dual["purchase_head"]["raw_nonregression_enforced"] is False
    assert dual["probability_head"]["role"] == (
        "probability_reporting_and_promotion_calibration"
    )
    assert dual["purchase_head"]["role"] == (
        "purchase_policy_and_chronological_bankroll"
    )


def test_v20_is_distinct_evaluation_only_schedule_quota_model() -> None:
    strategy = market_calibration.V20_STRATEGY_NAME
    args = market_calibration.build_parser().parse_args([
        "--from-date", "2026-07-18",
        "--calibrator-strategy", strategy,
    ])

    assert args.calibrator_strategy == strategy
    assert market_calibration.odds_path_model_name(strategy) == strategy
    assert strategy in market_calibration.SCHEDULE_QUOTA_STRATEGIES
    assert strategy in market_calibration.EVALUATION_ONLY_STRATEGIES
    assert strategy in market_calibration.CHRONOLOGICAL_BANKROLL_STRATEGIES
    assert strategy not in market_calibration.ROBUST_POLICY_STRATEGIES


def test_v21_builds_three_roles_from_strict_prior_rows(monkeypatch) -> None:
    calls = []

    def fitted(races, *, calibrator_strategy):
        calls.append((calibrator_strategy, [race["race_id"] for race in races]))
        weight = (
            1.0
            if calibrator_strategy == market_calibration.V19_STRATEGY_NAME
            else 0.25
        )
        return {
            "final_calibrator": {"model_weight": weight, "temperature": 1.0},
            "candidates": [],
        }

    monkeypatch.setattr(
        market_calibration, "fit_market_residual_calibrator", fitted
    )
    prior = [
        {"race_id": "prior-1", "race_date": "2026-07-28"},
        {"race_id": "prior-2", "race_date": "2026-07-29"},
    ]

    triple = market_calibration.fit_v21_triple_head_calibrators(prior)

    assert calls == [
        (market_calibration.V19_STRATEGY_NAME, ["prior-1", "prior-2"]),
        (market_calibration.V18_STRATEGY_NAME, ["prior-1", "prior-2"]),
    ]
    assert triple["outer_holdout_used"] is False
    assert triple["probability_head"]["raw_nonregression_enforced"] is True
    assert triple["ranking_head"]["raw_nonregression_enforced"] is False
    assert triple["purchase_head"]["raw_nonregression_enforced"] is False
    assert (
        triple["ranking_head"]["selection"]
        is triple["purchase_head"]["selection"]
    )
    assert (
        triple["ranking_head"]["calibrator"]
        == triple["purchase_head"]["calibrator"]
    )
    assert triple["ranking_purchase_share_v18_selection"] is True


def test_v21_split_metrics_use_probability_and_ranking_heads() -> None:
    race = {
        "race_id": "split-head",
        "race_date": "2026-07-30",
        "actual_combination": "1-2-3",
        "model_probabilities": {
            "1-2-3": 0.50,
            "1-3-2": 0.10,
            "2-1-3": 0.10,
            "2-3-1": 0.10,
            "3-1-2": 0.10,
            "3-2-1": 0.10,
        },
        "market_probabilities": {
            "1-2-3": 0.01,
            "1-3-2": 0.25,
            "2-1-3": 0.20,
            "2-3-1": 0.19,
            "3-1-2": 0.18,
            "3-2-1": 0.17,
        },
    }
    probability = {"model_weight": 0.0, "temperature": 1.0}
    ranking = {"model_weight": 1.0, "temperature": 1.0}

    probability_only = market_calibration.probability_metrics(
        [race], calibrator=probability
    )
    ranking_only = market_calibration.probability_metrics(
        [race], calibrator=ranking
    )
    split = market_calibration.split_head_probability_metrics(
        [race],
        probability_calibrator=probability,
        ranking_calibrator=ranking,
    )
    loss_differences, top5_differences = (
        market_calibration.split_head_paired_market_differences(
            [race],
            probability_calibrator=probability,
            ranking_calibrator=ranking,
        )
    )

    assert split["calibrated_trifecta_log_loss"] == (
        probability_only["calibrated_trifecta_log_loss"]
    )
    assert split["calibrated_winner_log_loss"] == (
        probability_only["calibrated_winner_log_loss"]
    )
    assert split["calibrated_trifecta_top5_hit_rate"] == (
        ranking_only["calibrated_trifecta_top5_hit_rate"]
    )
    assert loss_differences == [0.0]
    assert top5_differences == [1.0]


def test_v21_is_distinct_evaluation_only_triple_head_model() -> None:
    strategy = market_calibration.V21_STRATEGY_NAME
    args = market_calibration.build_parser().parse_args([
        "--from-date", "2026-07-18",
        "--calibrator-strategy", strategy,
    ])

    assert args.calibrator_strategy == strategy
    assert market_calibration.odds_path_model_name(strategy) == strategy
    assert strategy in market_calibration.SCHEDULE_QUOTA_STRATEGIES
    assert strategy in market_calibration.EVALUATION_ONLY_STRATEGIES
    assert strategy in market_calibration.CHRONOLOGICAL_BANKROLL_STRATEGIES
    assert strategy not in market_calibration.ROBUST_POLICY_STRATEGIES


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
