from __future__ import annotations

import json
from copy import deepcopy

import pytest

from boatrace_ai.paired_market_comparison import compare_market_results, main


def _day(
    race_date: str,
    *,
    first_stake: int,
    second_stake: int,
    return_yen: int,
    largest_hit: int,
) -> dict:
    selections_1 = (
        [{"combination": "1-2-3", "stake_yen": first_stake}]
        if first_stake else []
    )
    selections_2 = (
        [{"combination": "2-1-3", "stake_yen": second_stake}]
        if second_stake else []
    )
    stake = first_stake + second_stake
    return {
        "race_date": race_date,
        "evaluated_races": 2,
        "tickets": len(selections_1) + len(selections_2),
        "stake_yen": stake,
        "return_yen": return_yen,
        "profit_yen": return_yen - stake,
        "largest_hit_return_yen": largest_hit,
        "ledger": [
            {
                "event": "decision",
                "race_id": f"{race_date}-01-01",
                "tickets": len(selections_1),
                "stake_yen": first_stake,
                "selections": selections_1,
            },
            {"event": "settlement", "race_id": f"{race_date}-01-01"},
            {
                "event": "decision",
                "race_id": f"{race_date}-01-02",
                "tickets": len(selections_2),
                "stake_yen": second_stake,
                "selections": selections_2,
            },
        ],
    }


def _result(*, candidate: bool = False) -> dict:
    if candidate:
        daily = [
            _day(
                "2026-07-30",
                first_stake=100,
                second_stake=100,
                return_yen=400,
                largest_hit=400,
            ),
            _day(
                "2026-07-31",
                first_stake=100,
                second_stake=0,
                return_yen=200,
                largest_hit=200,
            ),
        ]
    else:
        daily = [
            _day(
                "2026-07-30",
                first_stake=100,
                second_stake=100,
                return_yen=300,
                largest_hit=300,
            ),
            _day(
                "2026-07-31",
                first_stake=100,
                second_stake=0,
                return_yen=100,
                largest_hit=100,
            ),
        ]
    stake = sum(day["stake_yen"] for day in daily)
    returned = sum(day["return_yen"] for day in daily)
    return {
        "evaluation_version": 32,
        "calibrator_strategy": "odds_path_role_integrated_v21",
        "benchmark_dates": ["2026-07-30", "2026-07-31"],
        "odds_data_signature": {"version": 4, "sha256": "same-source"},
        "evaluation_races": 4,
        "chronological_bankroll": {
            "daily": daily,
            "tickets": sum(day["tickets"] for day in daily),
            "stake_yen": stake,
            "return_yen": returned,
            "profit_yen": returned - stake,
            "roi": returned / stake,
        },
    }


def test_compare_market_results_reports_paired_metrics_and_gate() -> None:
    result = compare_market_results(
        _result(), _result(candidate=True), samples=500, seed=7
    )

    assert result["tickets"] == {
        "anchor_count": 3,
        "candidate_count": 3,
        "common_count": 3,
        "union_count": 3,
        "jaccard": 1.0,
        "turnover": 0.0,
    }
    assert result["stakes"]["turnover"] == 0.0
    assert result["anchor"]["roi"] == pytest.approx(4 / 3)
    assert result["candidate"]["roi"] == pytest.approx(2.0)
    assert result["anchor"]["largest_hit_excluded_roi"] == pytest.approx(1 / 3)
    assert result["candidate"]["largest_hit_excluded_roi"] == pytest.approx(2 / 3)
    assert [row["profit_difference_yen"] for row in result["daily_profit_differences"]] == [100, 100]
    assert result["daily_cluster_bootstrap"]["one_sided_5pct_lower_yen"] == 100
    assert result["daily_cluster_bootstrap"]["probability_difference_greater_than_or_equal_to_zero"] == 1.0
    assert result["gate"]["pass"] is True


def test_ticket_and_stake_turnover_include_changed_stake_and_membership() -> None:
    candidate = _result(candidate=True)
    first = candidate["chronological_bankroll"]["daily"][0]
    first["ledger"][0]["selections"][0]["stake_yen"] = 200
    first["ledger"][0]["stake_yen"] = 200
    first["ledger"][2]["selections"][0]["combination"] = "3-1-2"
    first["stake_yen"] = 300
    first["profit_yen"] = 100
    bankroll = candidate["chronological_bankroll"]
    bankroll["stake_yen"] = 400
    bankroll["profit_yen"] = bankroll["return_yen"] - 400
    bankroll["roi"] = bankroll["return_yen"] / 400

    result = compare_market_results(_result(), candidate, samples=100, seed=3)

    assert result["tickets"]["common_count"] == 2
    assert result["tickets"]["union_count"] == 4
    assert result["tickets"]["jaccard"] == 0.5
    assert result["tickets"]["turnover"] == 0.5
    assert result["stakes"]["absolute_difference_yen"] == 300
    assert result["stakes"]["union_maximum_yen"] == 500
    assert result["stakes"]["turnover"] == pytest.approx(0.6)
    assert result["gate"]["checks"]["ticket_jaccard"] is False
    assert result["gate"]["checks"]["stake_turnover"] is False
    assert result["gate"]["pass"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.__setitem__("evaluation_version", 33), "evaluation_version mismatch"),
        (lambda row: row.__setitem__("calibrator_strategy", "other"), "calibrator_strategy mismatch"),
        (lambda row: row.__setitem__("benchmark_dates", ["2026-07-30"]), "benchmark_dates mismatch"),
        (lambda row: row.__setitem__("odds_data_signature", {"sha256": "other"}), "odds_data_signature mismatch"),
        (lambda row: row.__setitem__("evaluation_races", 3), "evaluation_races mismatch"),
        (
            lambda row: row["chronological_bankroll"]["daily"][0]["ledger"][0].__setitem__("race_id", "different-race"),
            "evaluation races mismatch",
        ),
        (
            lambda row: row["chronological_bankroll"]["daily"][0].__setitem__("profit_yen", 999),
            "profit_yen is inconsistent",
        ),
        (
            lambda row: row["chronological_bankroll"]["daily"][0]["ledger"][0]["selections"].append({"combination": "1-2-3", "stake_yen": 100}),
            "duplicate ticket",
        ),
    ],
)
def test_contract_and_malformed_fixtures_fail_closed(mutation, message: str) -> None:
    candidate = deepcopy(_result(candidate=True))
    mutation(candidate)
    with pytest.raises(ValueError, match=message):
        compare_market_results(_result(), candidate, samples=100)


def test_cli_prints_reproducible_json(tmp_path, capsys) -> None:
    anchor_path = tmp_path / "anchor.json"
    candidate_path = tmp_path / "candidate.json"
    anchor_path.write_text(json.dumps(_result()), encoding="utf-8")
    candidate_path.write_text(json.dumps(_result(candidate=True)), encoding="utf-8")

    assert main([
        "--anchor", str(anchor_path),
        "--candidate", str(candidate_path),
        "--samples", "250",
        "--seed", "19",
    ]) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["daily_cluster_bootstrap"]["samples"] == 250
    assert output["daily_cluster_bootstrap"]["seed"] == 19
    assert output["contract"]["evaluation_race_count"] == 4
    assert len(output["contract"]["evaluation_races_sha256"]) == 64
