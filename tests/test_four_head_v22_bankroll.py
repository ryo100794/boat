from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import boatrace_ai.listwise.four_head_v22_bankroll as module
from boatrace_ai.fast_math import TRIFECTA_COMBINATIONS
from boatrace_ai.listwise.four_head_nested_v22 import (
    DecisionRace,
    FourHeadArtifact,
    LabeledRace,
    LinearHead,
    RaceOutcome,
    RacePrediction,
)
from boatrace_ai.listwise.four_head_v22_bankroll import (
    V22BankrollSettlement,
    evaluate_four_head_v22_bankroll,
)


def _head(name: str, size: int) -> LinearHead:
    return LinearHead(name, "test_teacher", (0.0,) * size, 0.0)


def _artifact() -> FourHeadArtifact:
    return FourHeadArtifact(
        model_key="four_head_nested_v22",
        artifact_version=1,
        trained_through_date="2026-07-28",
        choice_count=120,
        feature_count=1,
        probability_head=_head("probability_head", 2),
        ranking_head=_head("ranking_head", 2),
        closing_odds_head=_head("closing_odds_head", 2),
        purchase_head=_head("purchase_head", 6),
        purchase_threshold=0.0,
        inner_oof_folds=(),
        inner_oof_race_ids=(),
        inner_oof_prediction_sha256="inner",
        purchase_oof_folds=(),
        purchase_oof_race_ids=(),
        purchase_oof_score_sha256="scores",
        purchase_threshold_input_sha256="threshold",
        purchase_oof_score_sha256_by_date=(),
        purchase_threshold_input_sha256_by_date=(),
        training_race_ids=("train",),
    )


def _race(race_id: str, winner: int = 0) -> LabeledRace:
    closing = tuple(7.0 if index == winner else 20.0 for index in range(120))
    order = (winner, *(index for index in range(120) if index != winner))
    return LabeledRace(
        DecisionRace(
            race_id=race_id,
            race_date="2026-07-29",
            features=tuple((float(index),) for index in range(120)),
            current_odds=tuple(5.0 for _ in range(120)),
        ),
        RaceOutcome(winner, closing, order),
    )


def _settlement(race: LabeledRace, *, payout: int = 700) -> V22BankrollSettlement:
    number = int(race.decision.race_id[-1])
    return V22BankrollSettlement(
        race_id=race.decision.race_id,
        deadline_at=f"2026-07-29T0{number}:10:00+00:00",
        odds_captured_at=f"2026-07-29T0{number}:04:30+00:00",
        result_available_at=f"2026-07-29T0{number}:08:00+00:00",
        official_winner_index=race.outcome.winner_index,
        official_closing_odds=race.outcome.closing_odds,
        official_payout_yen=payout,
        snapshot_id=number,
    )


def _prediction(race: DecisionRace, selected: tuple[int, ...]) -> RacePrediction:
    probability = np.full(120, 0.5 / 119.0)
    probability[0] = 0.5
    ranking = np.arange(120, 0, -1, dtype=float)
    return RacePrediction(
        race_id=race.race_id,
        race_date=race.race_date,
        probabilities=tuple(probability),
        ranking_scores=tuple(ranking),
        predicted_closing_odds=tuple(10.0 for _ in range(120)),
        purchase_scores=tuple(1.0 for _ in range(120)),
        selected_indices=selected,
    )


def test_uses_existing_chronological_allocator_and_official_settlement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    races = (_race("race1"), _race("race2"))
    monkeypatch.setattr(
        module, "predict_race", lambda _artifact, race: _prediction(race, (0,))
    )

    result = evaluate_four_head_v22_bankroll(
        _artifact(), races, [_settlement(races[0]), _settlement(races[1])]
    )

    assert result["stake_yen"] == 400
    assert result["return_yen"] == 2_800
    assert result["profit_yen"] == 2_400
    assert result["roi"] == 7.0
    assert result["bankroll"]["profit_reinvestment"] is True
    decisions = [
        row
        for row in result["daily"][0]["ledger"]
        if row["event"] == "decision"
    ]
    assert decisions[1]["gross_stake_allowance_yen"] > 10_000
    assert result["policy"]["allocation_api"].startswith("simulate_chronological")
    assert result["diagnostic_unit_roi_is_formal_roi"] is False


def test_allows_zero_tickets_without_forcing_a_purchase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    race = _race("race1")
    monkeypatch.setattr(
        module, "predict_race", lambda _artifact, source: _prediction(source, ())
    )

    result = evaluate_four_head_v22_bankroll(
        _artifact(), [race], [_settlement(race)]
    )

    assert result["bankroll"]["evaluated_races"] == 1
    assert result["stake_yen"] == result["return_yen"] == 0
    assert result["daily"][0]["tickets"] == 0


def test_reports_formal_daily_loss_and_max_drawdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    race = _race("race1", winner=1)
    monkeypatch.setattr(
        module, "predict_race", lambda _artifact, source: _prediction(source, (0,))
    )

    result = evaluate_four_head_v22_bankroll(
        _artifact(), [race], [_settlement(race)]
    )

    assert result["stake_yen"] == 200
    assert result["return_yen"] == 0
    assert result["profit_yen"] == -200
    assert result["max_drawdown_yen"] == 200
    assert result["daily"][0]["profit_yen"] == -200


def test_metrics_include_logloss_winner_top1_and_trifecta_top5(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    race = _race("race1")
    monkeypatch.setattr(
        module, "predict_race", lambda _artifact, source: _prediction(source, (0,))
    )

    result = evaluate_four_head_v22_bankroll(
        _artifact(), [race], [_settlement(race)]
    )

    assert result["winner_log_loss"] == pytest.approx(-np.log(0.5 + 0.5 * 19 / 119))
    assert result["winner_top1_accuracy"] == 1.0
    assert result["trifecta_log_loss"] == pytest.approx(-np.log(0.5))
    assert result["trifecta_top1_accuracy"] == 1.0
    assert result["trifecta_top5_hit_rate"] == 1.0


def test_predictions_and_selection_are_frozen_before_settlement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    race = _race("race1")
    monkeypatch.setattr(
        module, "predict_race", lambda _artifact, source: _prediction(source, (0,))
    )

    low = evaluate_four_head_v22_bankroll(
        _artifact(), [race], [_settlement(race, payout=700)]
    )
    high = evaluate_four_head_v22_bankroll(
        _artifact(), [race], [_settlement(race, payout=1_400)]
    )

    assert low["frozen_prediction_sha256"] == high["frozen_prediction_sha256"]
    low_decision = low["daily"][0]["ledger"][0]
    high_decision = high["daily"][0]["ledger"][0]
    assert (
        low_decision["decision_information_sha256"]
        == high_decision["decision_information_sha256"]
    )
    assert low_decision["selections"] == high_decision["selections"]
    assert low["return_yen"] * 2 == high["return_yen"]
    assert low["outer_outcomes_used_for_fit_selection_or_threshold"] is False


@pytest.mark.parametrize(
    "change, message",
    [
        ({"odds_captured_at": "2026-07-29T01:05:01+00:00"}, "unsafe T-5"),
        ({"odds_captured_at": "2026-07-29T00:59:59+00:00"}, "unsafe T-5"),
        (
            {"result_available_at": "2026-07-29T01:05:00+00:00"},
            "settlement precedes",
        ),
        ({"official_winner_index": 1}, "official winner mismatch"),
        ({"official_payout_yen": 0}, "invalid official payout"),
    ],
)
def test_fails_closed_on_unsafe_t5_or_settlement(
    monkeypatch: pytest.MonkeyPatch,
    change: dict[str, object],
    message: str,
) -> None:
    race = _race("race1")
    monkeypatch.setattr(
        module, "predict_race", lambda _artifact, source: _prediction(source, (0,))
    )
    settlement = replace(_settlement(race), **change)

    with pytest.raises(ValueError, match=message):
        evaluate_four_head_v22_bankroll(_artifact(), [race], [settlement])


def test_rejects_non_outer_races_and_nonformal_money_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    race = _race("race1")
    monkeypatch.setattr(
        module, "predict_race", lambda _artifact, source: _prediction(source, (0,))
    )
    overlapping = replace(
        race,
        decision=replace(race.decision, race_id="train"),
    )

    with pytest.raises(ValueError, match="overlap"):
        evaluate_four_head_v22_bankroll(
            _artifact(), [overlapping], [replace(_settlement(race), race_id="train")]
        )
    with pytest.raises(ValueError, match="JPY10000"):
        evaluate_four_head_v22_bankroll(
            _artifact(), [race], [_settlement(race)], initial_bankroll_yen=20_000
        )
    with pytest.raises(ValueError, match="JPY100"):
        evaluate_four_head_v22_bankroll(
            _artifact(), [race], [_settlement(race)], stake_unit_yen=200
        )


def test_uses_t5_odds_not_predicted_or_closing_odds_for_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    race = _race("race1")
    prediction = replace(
        _prediction(race.decision, (0,)), predicted_closing_odds=(100.0,) * 120
    )
    monkeypatch.setattr(module, "predict_race", lambda _artifact, _source: prediction)

    result = evaluate_four_head_v22_bankroll(
        _artifact(), [race], [_settlement(race)]
    )
    selection = next(
        row for row in result["daily"][0]["ledger"] if row["event"] == "decision"
    )

    assert selection["stake_yen"] == 200
    assert result["policy"]["decision_odds"].endswith("T-5")
