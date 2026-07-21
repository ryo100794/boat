from __future__ import annotations

from itertools import permutations

import pytest

from boatrace_ai.listwise.market_calibration import (
    blend_probabilities,
    normalized_market_probabilities,
    policy_calibration_eligible,
    load_scored_cache,
    select_calibrator,
    select_policy,
    write_scored_cache,
    walk_forward_evaluate,
)


COMBINATIONS = tuple(
    "-".join(map(str, values)) for values in permutations(range(1, 7), 3)
)


def _distribution(primary: str, probability: float) -> dict[str, float]:
    remainder = (1.0 - probability) / (len(COMBINATIONS) - 1)
    return {
        combination: probability if combination == primary else remainder
        for combination in COMBINATIONS
    }


def _race(race_date: str, rno: int, *, winner: str = "1-2-3") -> dict:
    market = _distribution(winner, 0.20)
    model = _distribution(winner, 0.35)
    odds = {combination: 1.0 / probability / 1.25 for combination, probability in market.items()}
    return {
        "race_id": f"{race_date}-01-{rno:02d}",
        "race_date": race_date,
        "jcd": "01",
        "rno": rno,
        "actual_combination": winner,
        "actual_payout_yen": 500,
        "model_probabilities": model,
        "market_probabilities": market,
        "odds": odds,
        "snapshot_id": rno,
    }


def test_market_probabilities_remove_overround() -> None:
    probabilities = normalized_market_probabilities({"1-2-3": 2.0, "2-1-3": 4.0})
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert probabilities["1-2-3"] == pytest.approx(2.0 / 3.0)


def test_geometric_blend_has_exact_endpoints() -> None:
    model = {"a": 0.8, "b": 0.2}
    market = {"a": 0.3, "b": 0.7}
    assert blend_probabilities(
        model, market, model_weight=0.0, temperature=1.0
    ) == pytest.approx(market)
    assert blend_probabilities(
        model, market, model_weight=1.0, temperature=1.0
    ) == pytest.approx(model)


def test_calibrator_is_selected_without_bankroll_outcomes() -> None:
    races = [_race("2026-07-18", index) for index in range(1, 13)]
    selected, candidates = select_calibrator(races)
    assert len(candidates) == 15
    assert selected["model_weight"] == 1.0
    assert selected["temperature"] == 0.75


def test_policy_falls_back_to_no_bet_when_every_candidate_loses() -> None:
    races = []
    for index in range(1, 21):
        race = _race("2026-07-18", index)
        race["actual_combination"] = "6-5-4"
        race["actual_payout_yen"] = 1_000
        races.append(race)
    selected, _ = select_policy(
        races,
        calibrator={"model_weight": 1.0, "temperature": 1.0},
        daily_budget_yen=10_000,
        policies=[
            {"name": "no_bet", "no_bet": True},
            {
                "name": "loser",
                "ev_threshold": 1.0,
                "max_odds": None,
                "max_tickets_per_race": 1,
                "min_model_market_ratio": 1.0,
            },
        ],
    )
    assert selected == {"name": "no_bet", "no_bet": True}


def test_walk_forward_uses_only_strictly_earlier_dates_for_selection() -> None:
    races = [
        _race(race_date, rno)
        for race_date in ("2026-07-18", "2026-07-19", "2026-07-20", "2026-07-21")
        for rno in range(1, 13)
    ]
    result = walk_forward_evaluate(races, min_calibration_days=2)
    assert result["evaluation_days"] == 2
    assert result["evaluation_races"] == 24
    assert [row["evaluation_date"] for row in result["folds"]] == [
        "2026-07-20",
        "2026-07-21",
    ]
    assert result["folds"][0]["calibration_dates"] == ["2026-07-18", "2026-07-19"]
    assert result["folds"][1]["calibration_dates"] == [
        "2026-07-18",
        "2026-07-19",
        "2026-07-20",
    ]
    assert result["promotion_gate"]["sample_size_pass"] is False
    assert result["promotion_eligible"] is False


def test_scored_cache_requires_exact_contract(tmp_path) -> None:
    path = tmp_path / "scores.joblib"
    contract = {
        "version": 1,
        "model_sha256": "a" * 64,
        "trained_through": ("race", "2026-05-09", "24", 12),
        "from_date": "2026-07-18",
        "through_date": "2026-07-21",
    }
    races = [_race("2026-07-18", 1)]
    dataset = {"target_complete_races": 1, "eligible_real_odds_races": 1}
    write_scored_cache(
        path,
        contract=contract,
        races=races,
        dataset=dataset,
    )
    assert load_scored_cache(path, contract=contract) == (races, dataset)
    assert load_scored_cache(
        path,
        contract={**contract, "through_date": "2026-07-22"},
    ) is None


def test_policy_calibration_requires_repeatable_winning_days() -> None:
    result = {
        "race_days": 3,
        "winning_days": 1,
        "tickets": 30,
        "stake_yen": 3_000,
        "return_yen": 5_000,
        "profit_yen": 2_000,
        "roi": 5 / 3,
        "max_drawdown_yen": 1_000,
    }
    assert not policy_calibration_eligible(
        result,
        minimum_tickets=20,
        minimum_stake_yen=2_000,
    )
    assert policy_calibration_eligible(
        {**result, "winning_days": 2},
        minimum_tickets=20,
        minimum_stake_yen=2_000,
    )
