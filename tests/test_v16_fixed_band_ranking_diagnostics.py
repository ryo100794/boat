from __future__ import annotations

import math
from copy import deepcopy
from itertools import permutations

import pytest

from boatrace_ai.listwise.strict_prior_t300_divergence_passthrough_v16 import (
    fit_strict_prior_t300_divergence_passthrough_v16,
)
from boatrace_ai.listwise.v16_fixed_band_ranking_diagnostics import (
    RULES,
    build_fixed_band_diagnostic_inputs,
    compare_v16_fixed_band_ranking_rules,
    select_diagnostic_portfolio,
)


COMBINATIONS = tuple(
    "".join(map(str, values)) for values in permutations(range(1, 7), 3)
)


def _race(
    race_id: str,
    *,
    race_date: str = "2026-07-27",
    winner: str = "123",
    payout: int = 47_200,
    divergence_shift: float = 0.75,
) -> dict:
    t300 = {combination: 120.0 for combination in COMBINATIONS}
    market = 1.0 / 120.0
    band_count = 30
    band_factor = math.exp(divergence_shift)
    remainder_factor = (120.0 - band_count * band_factor) / (120 - band_count)
    probabilities = {
        combination: market * (
            band_factor if index < band_count else remainder_factor
        )
        for index, combination in enumerate(COMBINATIONS)
    }
    point = {
        combination: 8.0 + (index % 50)
        for index, combination in enumerate(COMBINATIONS)
    }
    return {
        "race_id": race_id,
        "race_date": race_date,
        "jcd": int(race_id[-4:-2]),
        "rno": int(race_id[-2:]),
        "model_probabilities": probabilities,
        "point_final_odds": point,
        "odds_checkpoints": {
            "t300": {
                "target_offset_seconds": 300,
                "captured_age_seconds": 300,
                "odds": t300,
            }
        },
        "actual_combination": winner,
        "actual_payout_yen": payout,
    }


def _diagnostic_races() -> list[dict]:
    return [
        _race("202607270101", winner="123", payout=47_200),
        _race("202607270202", winner="234", payout=8_600),
        _race("202607280101", race_date="2026-07-28", winner="125", payout=21_000),
    ]


def test_builds_every_fixed_band_candidate_without_result_fields() -> None:
    inputs = build_fixed_band_diagnostic_inputs(_diagnostic_races())

    assert inputs.decision_candidates
    assert all(
        0.5 <= row["log_model_market_divergence"] < 1.0
        for row in inputs.decision_candidates
    )
    assert all(
        "actual_combination" not in row and "actual_payout_yen" not in row
        for row in inputs.decision_candidates
    )
    assert inputs.settlements[("202607270101", "123")] == 47_200


@pytest.mark.parametrize("rule", RULES)
def test_each_rule_is_deterministic_budgeted_and_allows_zero(rule: str) -> None:
    candidates = build_fixed_band_diagnostic_inputs(
        _diagnostic_races()[:2]
    ).decision_candidates

    first = select_diagnostic_portfolio(candidates, rule=rule)
    second = select_diagnostic_portfolio(reversed(candidates), rule=rule)

    assert first == second
    assert len(first) <= 100
    assert select_diagnostic_portfolio([], rule=rule) == ()
    assert select_diagnostic_portfolio(
        candidates, rule=rule, daily_budget_yen=0
    ) == ()


def test_round_robin_diversifies_before_second_ticket_per_race() -> None:
    candidates = build_fixed_band_diagnostic_inputs(
        _diagnostic_races()[:2]
    ).decision_candidates
    selected = select_diagnostic_portfolio(
        candidates,
        rule="per_race_round_robin_diversified",
        daily_budget_yen=200,
    )

    assert len(selected) == 2
    assert len({row["race_id"] for row in selected}) == 2


def test_result_changes_do_not_change_population_decisions_or_selection() -> None:
    races = _diagnostic_races()
    changed = deepcopy(races)
    for race in changed:
        race["actual_combination"] = "654"
        race["actual_payout_yen"] = 9_999_900

    first = compare_v16_fixed_band_ranking_rules(races)
    second = compare_v16_fixed_band_ranking_rules(changed)

    assert (
        first["candidate_population_fingerprint"]
        == second["candidate_population_fingerprint"]
    )
    assert (
        first["decision_information_fingerprint"]
        == second["decision_information_fingerprint"]
    )
    for rule in RULES:
        left = first["rules"][rule]
        right = second["rules"][rule]
        assert (
            left["aggregate"]["selected_portfolio_fingerprint"]
            == right["aggregate"]["selected_portfolio_fingerprint"]
        )
        assert left["aggregate"]["stake_yen"] == right["aggregate"]["stake_yen"]
    assert any(
        first["rules"][rule]["aggregate"]["return_yen"]
        != second["rules"][rule]["aggregate"]["return_yen"]
        for rule in RULES
    )


def test_output_has_daily_aggregate_tail_metrics_and_research_guard() -> None:
    result = compare_v16_fixed_band_ranking_rules(_diagnostic_races())

    assert result["real_betting_enabled"] is False
    assert result["post_hoc_best_rule_is_promotion_evidence"] is False
    assert result["candidate_population_fingerprint"]
    assert result["decision_information_fingerprint"]
    for rule in RULES:
        output = result["rules"][rule]
        assert len(output["daily"]) == 2
        assert output["aggregate"]["tickets"] <= 200
        assert output["aggregate"]["stake_yen"] == (
            output["aggregate"]["tickets"] * 100
        )
        assert "roi" in output["aggregate"]
        assert "roi_excluding_largest_hit" in output["aggregate"]
        assert "hits" in output["aggregate"]


def test_invalid_or_result_bearing_selection_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="100-yen multiple"):
        select_diagnostic_portfolio([], rule="safe_ev_desc", daily_budget_yen=10_050)
    with pytest.raises(ValueError, match="result or payout"):
        select_diagnostic_portfolio(
            [{"race_id": "r", "combination": "123", "actual_payout_yen": 100}],
            rule="safe_ev_desc",
        )


def test_callback_forecasts_and_probability_artifact_are_primary_inputs() -> None:
    race = _race("202607270101")
    race["market_probabilities"] = {key: 1.0 / 120.0 for key in COMBINATIONS}
    forecasts = {race["race_id"]: dict(race.pop("point_final_odds"))}
    artifact = fit_strict_prior_t300_divergence_passthrough_v16([])

    inputs = build_fixed_band_diagnostic_inputs(
        [race],
        closing_forecasts=forecasts,
        probability_lcb=artifact,
    )
    result = compare_v16_fixed_band_ranking_rules(
        [race],
        closing_forecasts=forecasts,
        probability_lcb=artifact,
    )

    assert inputs.decision_candidates
    assert result["closing_forecast_source"] == "callback_mapping"
    assert result["probability_source"] == "probability_lcb_artifact_callback"
    assert all(
        row["estimated_closing_odds"]
        == forecasts[race["race_id"]][row["combination"]]
        for row in inputs.decision_candidates
    )
