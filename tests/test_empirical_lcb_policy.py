from __future__ import annotations

from dataclasses import dataclass

import pytest

from boatrace_ai.listwise.empirical_lcb_policy import (
    empirical_bankroll_promotion_eligible,
    policy_edge_records,
    simulate_empirical_lcb_policy,
)


def _blend(model, market, *, model_weight, temperature):
    assert model_weight == pytest.approx(0.75)
    assert temperature == pytest.approx(1.1)
    return dict(model)


def _race(race_id, *, race_date="2026-07-01", actual="1-2-3", payout=800,
          probabilities=None, multipliers=None):
    probabilities = probabilities or {
        "1-2-3": 0.40,
        "1-3-2": 0.35,
        "2-1-3": 0.25,
    }
    return {
        "race_id": race_id, "race_date": race_date, "jcd": "01", "rno": 1,
        "model_probabilities": probabilities,
        "market_probabilities": dict(probabilities),
        "odds": {key: 4.0 for key in probabilities},
        "historical_return_multipliers": multipliers or {},
        "actual_combination": actual, "actual_payout_yen": payout,
        "snapshot_id": 42,
    }


@dataclass
class _Artifact:
    ready: bool = True
    point: float = 1.20
    lcb: float = 1.05
    trained_through_date: str | None = "2026-06-30"

    def predict(self, raw_ev, probability_rank=None, forecast_odds=None):
        return {"empirical_ev": self.point, "empirical_ev_lcb95": self.lcb}

    def as_dict(self):
        return {"ready": self.ready, "test_artifact": True}


CALIBRATOR = {"model_weight": 0.75, "temperature": 1.1}


def test_edge_records_include_return_multiplier_and_realized_return():
    records = policy_edge_records(
        [_race("r1", multipliers={"1-2-3": 1.5})], CALIBRATOR, _blend
    )
    winner = next(row for row in records if row["combination"] == "1-2-3")
    loser = next(row for row in records if row["combination"] == "1-3-2")
    assert winner["raw_estimated_ev"] == pytest.approx(0.40 * 4.0 * 1.5)
    assert winner["gross_return_per_yen"] == pytest.approx(8.0)
    assert winner["probability_rank"] == 1
    assert loser["gross_return_per_yen"] == 0.0


@pytest.mark.parametrize(
    ("artifact", "expected_status"),
    [
        (_Artifact(ready=False), "calibration_not_ready"),
        (_Artifact(lcb=1.0), "ready"),
        (_Artifact(point=1.0, lcb=1.1), "ready"),
    ],
)
def test_not_ready_or_non_strict_edge_never_bets(artifact, expected_status):
    result = simulate_empirical_lcb_policy(
        [_race("r1")], CALIBRATOR, _blend, artifact, 10_000
    )
    assert result["status"] == expected_status
    assert result["tickets"] == 0
    assert result["stake_yen"] == 0
    assert result["roi"] is None
    assert result["daily"][0]["candidate_tickets"] == 0


def test_lcb_drives_kelly_and_point_estimate_is_audited():
    result = simulate_empirical_lcb_policy(
        [_race("r1", multipliers={"1-2-3": 3.0})],
        CALIBRATOR, _blend, _Artifact(point=1.20, lcb=1.05), 10_000,
    )
    audit = result["daily"][0]["eligible_candidate_audit"]
    winner = next(row for row in audit if row["combination"] == "1-2-3")
    assert winner["raw_estimated_ev"] == pytest.approx(4.8)
    assert winner["empirical_ev"] == pytest.approx(1.20)
    assert winner["empirical_ev_lcb95"] == pytest.approx(1.05)
    assert winner["allocation_ev"] == pytest.approx(1.05)
    selected = result["daily"][0]["selected_sample"]
    assert not selected or selected[0]["estimated_ev"] == pytest.approx(1.05)


def test_normalized_kelly_can_measure_minimum_stake_portfolio_diagnostic():
    race = _race("r1")
    default = simulate_empirical_lcb_policy(
        [race], CALIBRATOR, _blend, _Artifact(point=1.20, lcb=1.05), 10_000
    )
    normalized = simulate_empirical_lcb_policy(
        [race],
        CALIBRATOR,
        _blend,
        _Artifact(point=1.20, lcb=1.05),
        10_000,
        allocation_mode="normalized_kelly",
        min_daily_exposure_fraction=0.10,
    )

    assert default["tickets"] == 0
    assert normalized["tickets"] > 0
    assert normalized["allocation_policy"] == {
        "allocation_mode": "normalized_kelly",
        "min_daily_exposure_fraction": 0.10,
        "max_daily_exposure_fraction": 0.30,
    }


class _RankArtifact(_Artifact):
    def predict(self, raw_ev, probability_rank=None, forecast_odds=None):
        return {
            "empirical_ev": 1.1 + raw_ev / 100.0,
            "empirical_ev_lcb95": 1.01 + (2.0 - raw_ev) / 10.0,
        }


def test_limits_three_per_race_and_thirty_per_day_in_lcb_order():
    combinations = {f"c{index:02d}": 0.30 - index * 0.005 for index in range(8)}
    races = [_race(f"r{index:02d}", probabilities=combinations) for index in range(12)]
    result = simulate_empirical_lcb_policy(
        races, CALIBRATOR, _blend, _RankArtifact(), 10_000
    )
    audit = result["daily"][0]["eligible_candidate_audit"]
    assert len(audit) == 30
    assert all(
        sum(row["race_id"] == race["race_id"] for row in audit) <= 3
        for race in races
    )
    ordering = [
        (row["empirical_ev_lcb95"], row["empirical_ev"], row["probability"])
        for row in audit
    ]
    assert ordering == sorted(ordering, reverse=True)


def test_policy_passes_probability_rank_and_forecast_odds_to_artifact():
    calls = []

    class _ContextArtifact(_Artifact):
        def predict(self, raw_ev, probability_rank=None, forecast_odds=None):
            calls.append((raw_ev, probability_rank, forecast_odds))
            return super().predict(raw_ev, probability_rank, forecast_odds)

    simulate_empirical_lcb_policy(
        [_race("r1")], CALIBRATOR, _blend, _ContextArtifact(), 10_000
    )

    assert [rank for _raw_ev, rank, _odds in calls] == [1, 2, 3]
    assert all(odds == pytest.approx(4.0) for _raw_ev, _rank, odds in calls)


def test_policy_rejects_candidate_when_local_lcb_is_unavailable():
    class _OutOfRangeArtifact(_Artifact):
        def predict(self, raw_ev, probability_rank=None, forecast_odds=None):
            return {
                "empirical_ev": 1.20,
                "empirical_ev_lcb95": 1.05,
                "purchase_lcb95_available": False,
                "input_in_local_block_range": False,
            }

    result = simulate_empirical_lcb_policy(
        [_race("r1")], CALIBRATOR, _blend, _OutOfRangeArtifact(), 10_000
    )

    assert result["eligible_tickets"] == 0
    assert result["tickets"] == 0
    assert result["stake_yen"] == 0
    audit = result["daily"][0]["candidate_decision_audit"]
    assert len(audit) == 3
    assert all(row["purchase_gate_approved"] is False for row in audit)
    assert all(
        row["denial_reason"] == "outside_local_block_range"
        for row in audit
    )


def test_candidate_audit_carries_calibration_stability_metadata():
    class _StableArtifact(_Artifact):
        def predict(self, raw_ev, probability_rank=None, forecast_odds=None):
            return {
                "empirical_ev": 1.20,
                "empirical_ev_lcb95": 1.05,
                "calibration_level": "rank_group",
                "positive_return_days": 4,
                "return_hhi": 0.30,
                "rank_support": 120,
                "rank_support_days": 6,
            }

    result = simulate_empirical_lcb_policy(
        [_race("r1")], CALIBRATOR, _blend, _StableArtifact(), 10_000
    )
    audit = result["daily"][0]["eligible_candidate_audit"][0]

    assert audit["calibration_level"] == "rank_group"
    assert audit["positive_return_days"] == 4
    assert audit["return_hhi"] == pytest.approx(0.30)
    assert audit["rank_support"] == 120
    assert audit["rank_support_days"] == 6
    decision = result["daily"][0]["candidate_decision_audit"][0]
    assert decision["purchase_gate_approved"] is True
    assert decision["approval_reason"] == (
        "local_support_ready_and_calibrated_roi_lcb95_above_one"
    )
    assert decision["buy_threshold"] == 1.0


def test_simulator_contract_has_no_calibration_records_argument():
    with pytest.raises(TypeError):
        simulate_empirical_lcb_policy(
            [_race("r1")], CALIBRATOR, _blend, _Artifact(), 10_000,
            calibration_records=[],
        )


def test_policy_rejects_same_day_or_future_trained_artifact():
    with pytest.raises(ValueError, match="strictly before"):
        simulate_empirical_lcb_policy(
            [_race("r1")],
            CALIBRATOR,
            _blend,
            _Artifact(trained_through_date="2026-07-01"),
            10_000,
        )


def test_ready_policy_rejects_artifact_without_training_boundary():
    with pytest.raises(ValueError, match="trained_through_date"):
        simulate_empirical_lcb_policy(
            [_race("r1")],
            CALIBRATOR,
            _blend,
            _Artifact(trained_through_date=None),
            10_000,
        )


def test_external_ranking_changes_rank_without_changing_probability() -> None:
    race = _race("r1")
    order = ["2-1-3", "1-3-2", "1-2-3"]
    records = policy_edge_records(
        [race],
        CALIBRATOR,
        _blend,
        ranking_provider=lambda _race, _probabilities: order,
    )

    by_combination = {row["combination"]: row for row in records}
    assert by_combination["2-1-3"]["probability_rank"] == 1
    assert by_combination["2-1-3"]["probability"] == pytest.approx(0.25)
    assert by_combination["1-2-3"]["probability_rank"] == 3
    assert by_combination["1-2-3"]["probability"] == pytest.approx(0.40)


def test_external_ranking_cutoff_limits_calibration_and_purchase_candidates() -> None:
    race = _race("r1")
    order = ["2-1-3", "1-3-2", "1-2-3"]
    records = policy_edge_records(
        [race],
        CALIBRATOR,
        _blend,
        ranking_provider=lambda _race, _probabilities: order,
        max_rank=1,
    )
    result = simulate_empirical_lcb_policy(
        [race],
        CALIBRATOR,
        _blend,
        _Artifact(),
        10_000,
        ranking_provider=lambda _race, _probabilities: order,
        max_rank=1,
    )

    assert [row["combination"] for row in records] == ["2-1-3"]
    assert [
        row["combination"]
        for row in result["daily"][0]["eligible_candidate_audit"]
    ] == ["2-1-3"]


@pytest.mark.parametrize("max_rank", [0, -1, True, 1.5])
def test_external_ranking_cutoff_requires_a_positive_integer(max_rank) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        policy_edge_records(
            [_race("r1")], CALIBRATOR, _blend, max_rank=max_rank
        )


@pytest.mark.parametrize(
    "invalid_order",
    [
        ["1-2-3", "1-3-2"],
        ["1-2-3", "1-3-2", "not-a-ticket"],
        ["1-2-3", "1-2-3", "2-1-3"],
    ],
)
def test_external_ranking_must_be_a_complete_permutation(invalid_order) -> None:
    with pytest.raises(ValueError, match="every probability combination once"):
        policy_edge_records(
            [_race("r1")], CALIBRATOR, _blend,
            ranking_provider=lambda _race, _probabilities: invalid_order,
        )


def test_promotion_requires_thirty_days_and_roi_confidence() -> None:
    passing = {
        "status": "ready",
        "evaluation_days": 30,
        "tickets": 50,
        "roi": 1.05,
        "roi_ci95_lower": 1.0001,
        "probability_roi_above_one": 0.95,
    }
    assert empirical_bankroll_promotion_eligible(passing)

    failures = (
        ("status", "calibration_not_ready"),
        ("evaluation_days", 29),
        ("tickets", 49),
        ("roi", 1.0499),
        ("roi_ci95_lower", 1.0),
        ("probability_roi_above_one", 0.9499),
    )
    for key, value in failures:
        candidate = dict(passing)
        candidate[key] = value
        assert not empirical_bankroll_promotion_eligible(candidate)
