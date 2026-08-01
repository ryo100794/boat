from __future__ import annotations

from copy import deepcopy

import pytest

from boatrace_ai.listwise import market_calibration


def _races(days: int) -> list[dict]:
    return [
        {
            "race_id": f"2026-07-{day:02d}-01-01",
            "race_date": f"2026-07-{day:02d}",
            "model_probabilities": {"1-2-3": 1.0},
            "market_probabilities": {"1-2-3": 1.0},
        }
        for day in range(1, days + 1)
    ]


def _bankroll(daily_profits: list[int]) -> dict:
    daily = [
        {
            "race_date": f"2026-07-{day:02d}",
            "stake_yen": 100,
            "return_yen": 100 + profit,
            "profit_yen": profit,
        }
        for day, profit in enumerate(daily_profits, start=1)
    ]
    stake = 100 * len(daily)
    returns = sum(row["return_yen"] for row in daily)
    return {
        "tickets": len(daily),
        "hit_tickets": sum(row["return_yen"] > 0 for row in daily),
        "stake_yen": stake,
        "return_yen": returns,
        "profit_yen": returns - stake,
        "roi": returns / stake if stake else 0.0,
        "max_drawdown_yen": max(0, -(returns - stake)),
        "winning_days": sum(row["profit_yen"] > 0 for row in daily),
        "race_days": len(daily),
        "daily": daily,
    }


def _install_policy_simulator(monkeypatch, profits_by_name: dict[str, list[int]]) -> None:
    monkeypatch.setattr(
        market_calibration,
        "prepare_policy_matrix",
        lambda *_args, **_kwargs: {},
    )

    def fake_simulate(_races, *, policy, include_chronological, **_kwargs):
        if not include_chronological:
            return {
                "daily": [
                    {"race_date": race["race_date"], "tickets": 10}
                    for race in _races
                ]
            }
        profits = profits_by_name[policy["name"]]
        return {"chronological_bankroll": _bankroll(profits)}

    monkeypatch.setattr(market_calibration, "simulate_policy", fake_simulate)


def test_v35_candidates_change_only_one_anchor_axis() -> None:
    candidates = market_calibration.v35_registered_policy_candidates()
    anchor = candidates[0]

    assert 1 < len(candidates) <= 8
    for candidate in candidates[1:]:
        changed = {
            key
            for key in anchor
            if key != "name" and candidate.get(key) != anchor.get(key)
        }
        assert len(changed) == 1


def test_v35_before_seven_prior_days_keeps_fixed_anchor(monkeypatch) -> None:
    policies = market_calibration.v35_registered_policy_candidates()[:2]
    _install_policy_simulator(monkeypatch, {policies[0]["name"]: [0] * 6})

    selected, rows = market_calibration.select_policy_v35(
        _races(6),
        calibrator={"model_weight": 1.0, "temperature": 1.0},
        daily_budget_yen=10_000,
        policies=policies,
    )

    diagnostics = selected["v35_selection_diagnostics"]
    assert selected["name"] == policies[0]["name"]
    assert len(rows) == 1
    assert diagnostics["selection_regime"] == "fixed_anchor_warmup"
    assert diagnostics["fallback_reason"] == "strict_prior_days_below_7"
    assert selected["v18_ticket_control"]["learned_daily_ticket_limit"] == 10
    assert selected["v18_ticket_control"]["schedule_quota_rounding"] == "floor"


def test_v35_selects_only_stable_positive_paired_improvement(monkeypatch) -> None:
    policies = market_calibration.v35_registered_policy_candidates()[:2]
    _install_policy_simulator(
        monkeypatch,
        {
            policies[0]["name"]: [0] * 7,
            policies[1]["name"]: [100] * 7,
        },
    )

    selected, rows = market_calibration.select_policy_v35(
        _races(7),
        calibrator={"model_weight": 1.0, "temperature": 1.0},
        daily_budget_yen=10_000,
        policies=policies,
    )

    assert selected["name"] == policies[1]["name"]
    assert rows[1]["paired_profit_lcb_yen"] == pytest.approx(100.0)
    assert selected["v35_selection_diagnostics"]["fallback_reason"] is None


def test_v35_rejects_unstable_positive_average(monkeypatch) -> None:
    policies = market_calibration.v35_registered_policy_candidates()[:2]
    _install_policy_simulator(
        monkeypatch,
        {
            policies[0]["name"]: [0] * 7,
            policies[1]["name"]: [1000, -100, -100, -100, -100, -100, -100],
        },
    )

    selected, rows = market_calibration.select_policy_v35(
        _races(7),
        calibrator={"model_weight": 1.0, "temperature": 1.0},
        daily_budget_yen=10_000,
        policies=policies,
    )

    assert selected["name"] == policies[0]["name"]
    assert rows[1]["paired_profit_lcb_yen"] < 0
    assert (
        selected["v35_selection_diagnostics"]["fallback_reason"]
        == "no_candidate_positive_bonferroni_lcb"
    )


def test_v35_quota_is_fixed_before_30_days_and_bounded_afterward() -> None:
    daily = [{"tickets": 50} for _ in range(30)]

    early = market_calibration._v35_ticket_control(
        deepcopy(daily[:29]), prior_days=29
    )
    mature = market_calibration._v35_ticket_control(daily, prior_days=30)

    assert early["learned_daily_ticket_limit"] == 10
    assert early["schedule_quota_rounding"] == "floor"
    assert mature["learned_daily_ticket_limit"] == 20
    assert 5 <= mature["learned_daily_ticket_limit"] <= 20
