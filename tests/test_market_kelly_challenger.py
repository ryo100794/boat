from __future__ import annotations

import itertools
import math
from types import SimpleNamespace

import pytest

from boatrace_ai.listwise import market_kelly_challenger as challenger


COMBINATIONS = tuple(
    "-".join(map(str, lanes))
    for lanes in itertools.permutations(range(1, 7), 3)
)


def _probabilities(favourite: str | None = None) -> dict[str, float]:
    if favourite is None:
        return {key: 1.0 / 120.0 for key in COMBINATIONS}
    remainder = 0.1 / 119.0
    return {
        key: 0.9 if key == favourite else remainder
        for key in COMBINATIONS
    }


def _race(
    day: str,
    race_id: str,
    *,
    favourite: str | None = None,
    actual: str = "1-2-3",
    race_time: str = "10:00:00",
    payout: int = 200,
) -> dict[str, object]:
    probabilities = _probabilities(favourite)
    return {
        "race_id": race_id,
        "race_date": day,
        "deadline_time": race_time,
        "jcd": 1,
        "rno": int(race_id.rsplit("-", 1)[-1]) if race_id[-1].isdigit() else 1,
        "model_probabilities": dict(probabilities),
        "market_probabilities": dict(probabilities),
        "odds": {
            key: (2.0 if key == favourite else 1.0)
            for key in COMBINATIONS
        },
        "actual_combination": actual,
        "actual_payout_yen": payout,
    }


def _prior_pool() -> list[dict[str, object]]:
    races = []
    for index in range(500):
        day = f"2026-07-{index % 7 + 1:02d}"
        races.append(_race(day, f"teacher-{index}"))
    return races


def test_offset_fit_is_strictly_prior_and_audited(monkeypatch) -> None:
    fit_calls = []

    class Artifact:
        fitted = True
        converged = True
        fallback_reason = None

        def __init__(self, prediction_date, records):
            self.training_dates = tuple(sorted({row["race_date"] for row in records}))
            self.training_races = len(records)
            self.trained_through_date = self.training_dates[-1]
            self.prediction_date = prediction_date

        def predict(self, model, market, odds, *, prediction_date):
            assert prediction_date == self.prediction_date
            return SimpleNamespace(probabilities=dict(market))

    def fake_fit(records, *, prediction_date, **kwargs):
        copied = list(records)
        fit_calls.append((prediction_date, copied, kwargs))
        return Artifact(prediction_date, copied)

    monkeypatch.setattr(challenger, "fit_market_offset_calibration", fake_fit)
    races = _prior_pool() + [
        _race("2026-07-08", "holdout-8"),
        _race("2026-07-09", "holdout-9"),
    ]
    annotated, summary = challenger.attach_prequential_market_offsets(races)

    assert [call[0] for call in fit_calls] == ["2026-07-08", "2026-07-09"]
    for prediction_day, teachers, kwargs in fit_calls:
        assert teachers
        assert all(row["race_date"] < prediction_day for row in teachers)
        assert kwargs["min_training_races"] == 500
    holdout = next(row for row in annotated if row["race_id"] == "holdout-8")
    audit = holdout["_market_kelly_calibration"]
    assert audit["ready"] is True
    assert audit["trained_through_date"] == "2026-07-07"
    assert audit["training_days"] == 7
    assert audit["training_races"] == 500
    assert summary["ready_days"] == 2
    assert summary["ready_races"] == 2


def test_insufficient_or_nonconverged_fit_explicitly_falls_back(monkeypatch) -> None:
    insufficient = _prior_pool()[:-1] + [_race("2026-07-08", "holdout-8")]
    annotated, summary = challenger.attach_prequential_market_offsets(insufficient)
    holdout = next(row for row in annotated if row["race_id"] == "holdout-8")
    audit = holdout["_market_kelly_calibration"]
    assert audit["mode"] == "market_only_fallback"
    assert audit["fallback_reason"] == "insufficient_strictly_prior_races"
    assert holdout["_policy_calibrated_probabilities"] == pytest.approx(
        holdout["market_probabilities"]
    )
    assert summary["ready_days"] == 0

    artifact = SimpleNamespace(
        fitted=True,
        converged=False,
        fallback_reason=None,
        training_dates=tuple(f"2026-07-{day:02d}" for day in range(1, 8)),
        training_races=500,
        trained_through_date="2026-07-07",
    )
    monkeypatch.setattr(
        challenger,
        "fit_market_offset_calibration",
        lambda *args, **kwargs: artifact,
    )
    annotated, _ = challenger.attach_prequential_market_offsets(
        _prior_pool() + [_race("2026-07-08", "holdout-8")]
    )
    holdout = next(row for row in annotated if row["race_id"] == "holdout-8")
    assert holdout["_market_kelly_calibration"]["fallback_reason"] == "not_converged"


def test_races_are_settled_in_time_order_and_profit_is_reinvested() -> None:
    races = [
        _race(
            "2026-07-20",
            "race-2",
            favourite="1-2-3",
            actual="1-2-3",
            race_time="10:10:00",
        ),
        _race(
            "2026-07-20",
            "race-1",
            favourite="1-2-3",
            actual="1-2-3",
            race_time="10:00:00",
        ),
    ]
    result = challenger.evaluate_market_kelly_challenger(races)
    decisions = result["daily"][0]["decisions"]

    assert [row["race_id"] for row in decisions] == ["race-1", "race-2"]
    assert decisions[0]["bankroll_before_yen"] == 10_000
    assert decisions[0]["stake_yen"] == 500
    assert decisions[0]["return_yen"] == 1_000
    assert decisions[1]["bankroll_before_yen"] == 10_500
    assert result["daily"][0]["ending_bankroll_yen"] == 11_000


def test_exact_kelly_can_choose_zero_bet() -> None:
    race = _race("2026-07-20", "race-1", favourite=None)
    race["odds"] = {key: 100.0 for key in COMBINATIONS}
    result = challenger.evaluate_market_kelly_challenger([race])

    assert result["tickets"] == 0
    assert result["stake_yen"] == 0
    assert result["return_yen"] == 0
    assert result["roi"] == 0.0
    assert result["profitable_day_fraction"] == 0.0
    assert result["daily"][0]["ending_bankroll_yen"] == 10_000
    assert result["reliability"]["selected_races"] == 0


def test_odds_safety_factor_requires_a_larger_forecast_edge() -> None:
    race = _race(
        "2026-07-20",
        "race-1",
        favourite="1-2-3",
        actual="1-2-3",
    )

    baseline = challenger.evaluate_market_kelly_challenger([race])
    conservative = challenger.evaluate_market_kelly_challenger(
        [race],
        odds_safety_factor=2.0,
    )

    assert baseline["tickets"] == 1
    assert baseline["daily"][0]["decisions"][0]["allocations"][0] == {
        "selection": "1-2-3",
        "units": 5,
        "stake_yen": 500,
        "probability": 0.9,
        "forecast_odds": 2.0,
        "kelly_effective_odds": 2.0,
    }
    assert conservative["policy"]["odds_safety_factor"] == 2.0
    assert conservative["tickets"] == 0
    with pytest.raises(ValueError, match="odds_safety_factor"):
        challenger.evaluate_market_kelly_challenger(
            [race], odds_safety_factor=0.99
        )


def test_minimum_race_number_is_a_fixed_candidate_generation_rule() -> None:
    races = [
        _race("2026-07-20", "race-4", favourite="1-2-3"),
        _race("2026-07-20", "race-5", favourite="1-2-3"),
    ]

    result = challenger.evaluate_market_kelly_challenger(
        races,
        minimum_race_number=5,
    )

    decisions = result["daily"][0]["decisions"]
    assert result["policy"]["minimum_race_number"] == 5
    assert decisions[0]["rno"] == 4
    assert decisions[0]["stake_yen"] == 0
    assert decisions[0]["allocations"] == []
    assert decisions[1]["rno"] == 5
    assert decisions[1]["stake_yen"] == 500
    with pytest.raises(ValueError, match="minimum_race_number"):
        challenger.evaluate_market_kelly_challenger(
            races,
            minimum_race_number=13,
        )


def test_maximum_race_number_is_a_fixed_candidate_generation_rule() -> None:
    races = [
        _race("2026-07-20", "race-8", favourite="1-2-3"),
        _race("2026-07-20", "race-9", favourite="1-2-3"),
    ]

    result = challenger.evaluate_market_kelly_challenger(
        races,
        minimum_race_number=5,
        maximum_race_number=8,
    )

    decisions = result["daily"][0]["decisions"]
    assert result["policy"]["maximum_race_number"] == 8
    assert decisions[0]["rno"] == 8
    assert decisions[0]["stake_yen"] == 500
    assert decisions[1]["rno"] == 9
    assert decisions[1]["stake_yen"] == 0
    assert decisions[1]["allocations"] == []
    with pytest.raises(ValueError, match="must not exceed"):
        challenger.evaluate_market_kelly_challenger(
            races,
            minimum_race_number=9,
            maximum_race_number=8,
        )


def test_required_ticket_count_executes_only_matching_allocations() -> None:
    first, second = COMBINATIONS[:2]
    probabilities = {key: 0.1 / 118.0 for key in COMBINATIONS}
    probabilities[first] = 0.45
    probabilities[second] = 0.45
    race = _race(
        "2026-07-20",
        "race-1",
        actual=first,
        payout=250,
    )
    race["model_probabilities"] = dict(probabilities)
    race["market_probabilities"] = dict(probabilities)
    race["odds"] = {
        key: 2.5 if key in {first, second} else 1.0
        for key in COMBINATIONS
    }

    matching = challenger.evaluate_market_kelly_challenger(
        [race], required_ticket_count=2
    )
    rejected = challenger.evaluate_market_kelly_challenger(
        [race], required_ticket_count=1
    )

    assert matching["policy"]["required_ticket_count"] == 2
    assert matching["tickets"] == 2
    assert matching["stake_yen"] == 500
    assert rejected["tickets"] == 0
    assert rejected["stake_yen"] == 0
    assert rejected["daily"][0]["decisions"][0]["allocations"] == []
    with pytest.raises(ValueError, match="required_ticket_count"):
        challenger.evaluate_market_kelly_challenger(
            [race], required_ticket_count=0
        )


def test_reversed_place_pair_requires_same_winner_and_swapped_places() -> None:
    first = "1-2-3"
    reversed_places = "1-3-2"
    non_reversed = "1-2-4"

    def evaluate(second: str) -> dict[str, object]:
        probabilities = {key: 0.1 / 118.0 for key in COMBINATIONS}
        probabilities[first] = 0.45
        probabilities[second] = 0.45
        race = _race("2026-07-20", second, actual=first, payout=250)
        race["model_probabilities"] = dict(probabilities)
        race["market_probabilities"] = dict(probabilities)
        race["odds"] = {
            key: 2.5 if key in {first, second} else 1.0
            for key in COMBINATIONS
        }
        return challenger.evaluate_market_kelly_challenger(
            [race],
            required_ticket_count=2,
            require_reversed_place_pair=True,
        )

    matching = evaluate(reversed_places)
    rejected = evaluate(non_reversed)
    above_forecast_odds_limit = challenger.evaluate_market_kelly_challenger(
        [dict(
            _race("2026-07-20", "limited", actual=first, payout=250),
            model_probabilities={
                key: (0.45 if key in {first, reversed_places} else 0.1 / 118.0)
                for key in COMBINATIONS
            },
            market_probabilities={
                key: (0.45 if key in {first, reversed_places} else 0.1 / 118.0)
                for key in COMBINATIONS
            },
            odds={
                key: (2.5 if key in {first, reversed_places} else 1.0)
                for key in COMBINATIONS
            },
        )],
        required_ticket_count=2,
        require_reversed_place_pair=True,
        maximum_forecast_odds=2.4,
    )

    assert matching["tickets"] == 2
    assert matching["policy"]["require_reversed_place_pair"] is True
    assert rejected["tickets"] == 0
    assert above_forecast_odds_limit["tickets"] == 0
    assert above_forecast_odds_limit["policy"]["maximum_forecast_odds"] == 2.4
    with pytest.raises(ValueError, match="requires required_ticket_count=2"):
        challenger.evaluate_market_kelly_challenger(
            [_race("2026-07-20", "bad")],
            require_reversed_place_pair=True,
        )
    with pytest.raises(ValueError, match="maximum_forecast_odds"):
        challenger.evaluate_market_kelly_challenger(
            [_race("2026-07-20", "bad")],
            maximum_forecast_odds=1.0,
        )


def test_actual_return_uses_actual_stake_and_payout_per_100_yen() -> None:
    race = _race(
        "2026-07-20",
        "race-1",
        favourite="1-2-3",
        actual="1-2-3",
        payout=1_230,
    )
    result = challenger.evaluate_market_kelly_challenger([race])
    decision = result["daily"][0]["decisions"][0]

    assert decision["actual_stake_yen"] == 500
    assert decision["return_yen"] == 6_150
    assert decision["bankroll_after_yen"] == 15_650
    assert result["return_yen"] == 6_150
    assert result["profit_yen"] == 5_650
    assert result["reliability"]["largest_hit_return_yen"] == 6_150
    diagnostics = result["edge_diagnostics"]
    assert diagnostics["races"] == 1
    assert diagnostics["positive_ev_races"] == 1
    assert diagnostics["positive_ev_combinations"] == 1
    assert diagnostics["max_estimated_ev"] == pytest.approx(1.8)
    assert result["log_loss"]["challenger"] == pytest.approx(
        -math.log(0.9)
    )


def test_evaluation_dates_keep_prior_teachers_but_exclude_their_bankroll() -> None:
    races = [
        _race("2026-07-20", "teacher-1", favourite="1-2-3"),
        _race("2026-07-21", "holdout-1", favourite="1-2-3"),
    ]
    result = challenger.evaluate_market_kelly_challenger(
        races,
        evaluation_dates=["2026-07-21"],
    )

    assert result["evaluation_dates"] == ["2026-07-21"]
    assert result["evaluation_days"] == 1
    assert result["evaluated_races"] == 1
    assert result["profitable_day_fraction"] == 1.0
    assert result["daily"][0]["race_date"] == "2026-07-21"
    assert result["calibration"]["days"][1]["training_races"] == 1
    assert "races" not in result
    assert result["promotion_gate"]["sample_size_pass"] is False
    assert result["promotion_gate"]["evaluated_races_pass"] is False
    assert result["promotion_gate"]["effective_hit_count_pass"] is False
    assert result["promotion_gate"]["largest_hit_return_share_pass"] is False
    assert result["promotion_gate"]["profitable_day_fraction_pass"] is True
    assert result["promotion_gate"]["market_log_loss_confidence_pass"] is False
    assert result["promotion_gate"]["market_top5_confidence_pass"] is False
    assert result["promotion_gate"]["selected_probability_not_overconfident"] is True
    assert result["promotion_gate"]["no_lookahead_pass"] is True
    assert result["promotion_gate"]["operational_data_errors_zero"] is False
    assert result["data_quality"]["market_calibration_fallback_races"] == 1
    assert result["promotion_gate"]["pass"] is False


def test_evaluation_dates_exclude_prior_fallbacks_from_formal_data_quality() -> None:
    prior = _race("2026-08-03", "prior-1", favourite="1-2-3")
    prior["_market_kelly_calibration"] = {
        "ready": False,
        "trained_through_date": None,
    }
    prospective = _race("2026-08-04", "prospective-1", favourite="1-2-3")
    prospective["_market_kelly_calibration"] = {
        "ready": True,
        "trained_through_date": "2026-08-03",
    }
    prospective["_policy_calibrated_probabilities"] = dict(
        prospective["market_probabilities"]
    )

    result = challenger.evaluate_attached_market_kelly_challenger(
        [prior, prospective],
        calibration={},
        evaluation_dates=["2026-08-04"],
    )

    assert result["evaluation_dates"] == ["2026-08-04"]
    assert result["data_quality"] == {
        "evaluated_races": 1,
        "duplicate_race_ids": 0,
        "market_calibration_fallback_races": 0,
        "closing_policy_fallback_races": 0,
        "lookahead_violations": 0,
        "operational_data_errors": 0,
        "pass": True,
    }
    assert result["promotion_gate"]["operational_data_errors_zero"] is True


def test_bootstrap_gate_distinguishes_no_bet_and_profitable_days() -> None:
    no_bet = _race("2026-07-20", "no-bet", favourite=None)
    no_bet["odds"] = {key: 100.0 for key in COMBINATIONS}
    empty = challenger.evaluate_market_kelly_challenger([no_bet])

    assert empty["bootstrap"]["roi_ci95_lower"] is None
    assert empty["promotion_gate"]["bootstrap_status"] == "undefined_no_stake"
    assert empty["promotion_gate"]["pass"] is False

    profitable = challenger.evaluate_market_kelly_challenger([
        _race(
            "2026-07-20",
            "winner",
            favourite="1-2-3",
            actual="1-2-3",
            payout=1_230,
        )
    ])
    assert profitable["bootstrap"]["roi_ci95_lower"] == pytest.approx(12.3)
    assert profitable["bootstrap"]["probability_roi_above_one"] == 1.0
    assert profitable["hit_tickets"] == 1
    assert profitable["promotion_gate"]["minimum_hits"] == 20
    assert profitable["promotion_gate"]["minimum_clean_evaluation_days"] == 30
    assert profitable["promotion_gate"]["minimum_evaluated_races"] == 1_000
    assert profitable["promotion_gate"]["minimum_tickets"] == 200
    assert profitable["promotion_gate"]["minimum_effective_hits"] == 20.0
    assert profitable["promotion_gate"][
        "maximum_largest_hit_return_share"
    ] == 0.15
    assert profitable["promotion_gate"][
        "largest_hit_return_share_pass"
    ] is False
    assert profitable["promotion_gate"]["minimum_profitable_day_fraction"] == 0.60
    assert profitable["promotion_gate"]["minimum_market_confidence"] == 0.95
    assert profitable["promotion_gate"]["clean_evaluation_days_pass"] is False
    assert profitable["promotion_gate"][
        "minimum_selected_probability_calibration_pvalue"
    ] == 0.05
    assert profitable["promotion_gate"]["bootstrap_lower_95_pass"] is True
    assert profitable["promotion_gate"]["sample_size_pass"] is False


def test_selected_probability_calibration_rejects_clear_overconfidence() -> None:
    race = _race(
        "2026-07-20",
        "overconfident-miss",
        favourite="1-2-3",
        actual="1-2-4",
    )
    probabilities = {
        key: (0.99 if key == "1-2-3" else 0.01 / 119.0)
        for key in COMBINATIONS
    }
    race["model_probabilities"] = dict(probabilities)
    race["market_probabilities"] = dict(probabilities)

    result = challenger.evaluate_market_kelly_challenger([race])

    calibration = result["purchase_probability_calibration"]
    assert calibration["selected_races"] == 1
    assert calibration["observed_hits"] == 0
    assert calibration["expected_hits"] == pytest.approx(0.99)
    assert calibration["probability_at_most_observed_hits"] == pytest.approx(0.01)
    assert result["promotion_gate"]["selected_probability_not_overconfident"] is False


def test_walk_forward_can_select_regularization_from_prior_days(monkeypatch) -> None:
    from boatrace_ai.listwise import market_offset_selection

    selection_calls = []
    fit_calls = []

    def select(records, *, prediction_date):
        rows = list(records)
        selection_calls.append((prediction_date, rows))
        return {
            "selected_regularization": 0.1,
            "validation_date": "2026-07-07",
            "fallback_reason": None,
        }

    class Artifact:
        fitted = True
        converged = True
        fallback_reason = None

        def __init__(self, prediction_date, records, regularization):
            self.prediction_date = prediction_date
            self.training_dates = tuple(
                sorted({str(row["race_date"]) for row in records})
            )
            self.training_races = len(records)
            self.trained_through_date = self.training_dates[-1]
            self.regularization = regularization

        def predict(self, model, market, odds, *, prediction_date):
            del model, odds
            assert prediction_date == self.prediction_date
            return SimpleNamespace(probabilities=dict(market))

    def fit(records, *, prediction_date, regularization, **kwargs):
        rows = list(records)
        fit_calls.append((prediction_date, regularization, kwargs, rows))
        return Artifact(prediction_date, rows, regularization)

    monkeypatch.setattr(
        market_offset_selection,
        "select_market_offset_regularization",
        select,
    )
    monkeypatch.setattr(challenger, "fit_market_offset_calibration", fit)
    races = _prior_pool() + [_race("2026-07-08", "holdout-8")]

    annotated, summary = challenger.attach_prequential_market_offsets(
        races,
        select_regularization=True,
    )

    assert len(selection_calls) == 1
    assert all(row["race_date"] < "2026-07-08" for row in selection_calls[0][1])
    assert fit_calls[0][1] == 0.1
    audit = next(row for row in annotated if row["race_id"] == "holdout-8")[
        "_market_kelly_calibration"
    ]
    assert audit["regularization_selection"]["selected_regularization"] == 0.1
    assert summary["ready_days"] == 1
