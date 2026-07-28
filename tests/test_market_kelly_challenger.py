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
    assert result["daily"][0]["ending_bankroll_yen"] == 10_000
    assert result["reliability"]["selected_races"] == 0


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
    assert result["daily"][0]["race_date"] == "2026-07-21"
    assert result["calibration"]["days"][1]["training_races"] == 1
    assert "races" not in result
    assert result["promotion_gate"]["sample_size_pass"] is False
    assert result["promotion_gate"]["pass"] is False


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
    assert profitable["promotion_gate"]["bootstrap_lower_95_pass"] is True
    assert profitable["promotion_gate"]["sample_size_pass"] is False
