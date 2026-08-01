from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

import boatrace_ai.listwise.four_head_nested_v22 as v22

from boatrace_ai.listwise.four_head_nested_v22 import (
    MODEL_KEY,
    DecisionRace,
    LabeledRace,
    RaceOutcome,
    artifact_fingerprint,
    evaluate_outer_outcomes,
    fit_four_head_nested_v22,
    predict_race,
    prediction_fingerprint,
)


def labeled_races(
    *, start_day: int, days: int, races_per_day: int = 2, choices: int = 6
) -> tuple[LabeledRace, ...]:
    rows: list[LabeledRace] = []
    for day_offset in range(days):
        day = start_day + day_offset
        for race_number in range(races_per_day):
            features: list[tuple[float, ...]] = []
            latent: list[float] = []
            for choice in range(choices):
                form = ((choice * 7 + day * 3 + race_number * 5) % 17) / 8.0 - 1.0
                lane = (choices - choice) / choices
                score = 1.4 * form + 0.7 * lane + 0.05 * day
                features.append((form, lane, race_number / 10.0))
                latent.append(score)
            probability = np.exp(np.asarray(latent) - max(latent))
            probability /= probability.sum()
            current = tuple(float(max(1.05, 0.96 / value)) for value in probability)
            closing = tuple(
                float(max(1.05, odds * (0.92 + 0.03 * ((index + day) % 5))))
                for index, odds in enumerate(current)
            )
            ranking = tuple(int(index) for index in np.argsort(-np.asarray(latent)))
            rows.append(
                LabeledRace(
                    DecisionRace(
                        race_id=f"2026-07-{day:02d}-{race_number:02d}",
                        race_date=f"2026-07-{day:02d}",
                        features=tuple(features),
                        current_odds=current,
                    ),
                    RaceOutcome(
                        winner_index=ranking[0],
                        closing_odds=closing,
                        ranking_order=ranking,
                    ),
                )
            )
    return tuple(rows)


def fitted():
    return fit_four_head_nested_v22(
        labeled_races(start_day=1, days=7),
        minimum_inner_training_dates=2,
        alpha=1e-2,
    )


def test_four_heads_execute_and_artifact_is_fixed_and_deterministic() -> None:
    training = labeled_races(start_day=1, days=7)
    first = fit_four_head_nested_v22(
        training, minimum_inner_training_dates=2, alpha=1e-2
    )
    second = fit_four_head_nested_v22(
        training, minimum_inner_training_dates=2, alpha=1e-2
    )

    assert first.model_key == MODEL_KEY == "four_head_nested_v22"
    assert first.outer_outcomes_used is False
    assert first.fixed_after_fit is True
    assert first.purchase_teacher_source == "strict_prior_base_head_oof_predictions"
    assert first.purchase_threshold_source == "learned_unit_return_break_even_zero"
    assert first.purchase_threshold == 0.0
    assert {
        first.probability_head.name,
        first.ranking_head.name,
        first.closing_odds_head.name,
        first.purchase_head.name,
    } == {
        "probability_head",
        "ranking_head",
        "closing_odds_head",
        "purchase_head",
    }
    assert artifact_fingerprint(first) == artifact_fingerprint(second)
    assert len(artifact_fingerprint(first)) == 64
    with pytest.raises(FrozenInstanceError):
        first.purchase_threshold = 0.0  # type: ignore[misc]

    prediction = predict_race(first, labeled_races(start_day=8, days=1)[0].decision)
    assert sum(prediction.probabilities) == pytest.approx(1.0)
    assert all(value > 1.0 for value in prediction.predicted_closing_odds)
    assert len(prediction.purchase_scores) == first.choice_count


def test_purchase_head_uses_each_eligible_race_once_from_strict_prior_inner_oof() -> None:
    training = labeled_races(start_day=1, days=7, races_per_day=3)
    artifact = fit_four_head_nested_v22(
        training, minimum_inner_training_dates=2, alpha=1e-2
    )
    expected = tuple(
        race.decision.race_id
        for race in training
        if race.decision.race_date >= "2026-07-03"
    )

    assert artifact.inner_oof_race_ids == expected
    assert len(set(artifact.inner_oof_race_ids)) == len(expected)
    assert len(artifact.inner_oof_prediction_sha256) == 64
    for fold in artifact.inner_oof_folds:
        assert fold.trained_through_date < fold.validation_date
        assert fold.validation_race_ids
        assert not set(fold.training_race_ids) & set(fold.validation_race_ids)
        assert all(
            race_id < fold.validation_race_ids[0]
            for race_id in fold.training_race_ids
        )


def test_purchase_level_oof_is_unique_and_uses_only_earlier_base_oof_dates() -> None:
    training = labeled_races(start_day=1, days=8, races_per_day=3)
    artifact = fit_four_head_nested_v22(
        training,
        minimum_inner_training_dates=2,
        minimum_purchase_training_dates=2,
        alpha=1e-2,
    )
    expected = tuple(
        race.decision.race_id
        for race in training
        if race.decision.race_date >= "2026-07-05"
    )

    assert artifact.purchase_oof_race_ids == expected
    assert len(set(artifact.purchase_oof_race_ids)) == len(expected)
    assert len(artifact.purchase_oof_score_sha256) == 64
    assert len(artifact.purchase_threshold_input_sha256) == 64
    assert tuple(date for date, _sha in artifact.purchase_oof_score_sha256_by_date) == (
        "2026-07-05",
        "2026-07-06",
        "2026-07-07",
        "2026-07-08",
    )
    assert tuple(
        date for date, _sha in artifact.purchase_threshold_input_sha256_by_date
    ) == tuple(date for date, _sha in artifact.purchase_oof_score_sha256_by_date)
    for fold in artifact.purchase_oof_folds:
        assert fold.trained_through_date < fold.validation_date
        assert fold.validation_race_ids
        assert not set(fold.training_base_oof_race_ids) & set(
            fold.validation_race_ids
        )
        assert all(
            race_id < fold.validation_race_ids[0]
            for race_id in fold.training_base_oof_race_ids
        )


def test_later_purchase_validation_labels_cannot_change_earlier_oof_inputs() -> None:
    training = labeled_races(start_day=1, days=9, races_per_day=2)
    changed_rows: list[LabeledRace] = []
    for race in training:
        if race.decision.race_date != "2026-07-09":
            changed_rows.append(race)
            continue
        closing = list(race.outcome.closing_odds)
        closing[race.outcome.winner_index] *= 7.0
        changed_rows.append(
            replace(
                race,
                outcome=replace(race.outcome, closing_odds=tuple(closing)),
            )
        )

    original = fit_four_head_nested_v22(
        training, minimum_inner_training_dates=2, minimum_purchase_training_dates=2
    )
    changed = fit_four_head_nested_v22(
        tuple(changed_rows),
        minimum_inner_training_dates=2,
        minimum_purchase_training_dates=2,
    )
    original_scores = dict(original.purchase_oof_score_sha256_by_date)
    changed_scores = dict(changed.purchase_oof_score_sha256_by_date)
    original_inputs = dict(original.purchase_threshold_input_sha256_by_date)
    changed_inputs = dict(changed.purchase_threshold_input_sha256_by_date)

    assert original.purchase_oof_score_sha256 == changed.purchase_oof_score_sha256
    for date in sorted(original_inputs):
        if date < "2026-07-09":
            assert original_scores[date] == changed_scores[date]
            assert original_inputs[date] == changed_inputs[date]
    assert original_scores["2026-07-09"] == changed_scores["2026-07-09"]
    assert original_inputs["2026-07-09"] != changed_inputs["2026-07-09"]
    assert (
        original.purchase_threshold_input_sha256
        != changed.purchase_threshold_input_sha256
    )


def test_purchase_head_learns_unbiased_capped_unit_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    original = v22._fit_ridge

    def capture(matrix, targets, *, alpha, sample_weight=None):
        captured["sample_weight"] = sample_weight
        return original(
            matrix, targets, alpha=alpha, sample_weight=sample_weight
        )

    monkeypatch.setattr(v22, "_fit_ridge", capture)
    head = v22._fit_purchase_head(
        [np.ones((4, 2))],
        [np.asarray([-1.0, -1.0, 4.0, -1.0])],
        alpha=0.01,
        purchase_loss="ridge_capped_net",
    )

    assert head.teacher.startswith("capped_realized_unit_return")
    assert captured["sample_weight"] is None


def test_poisson_purchase_head_learns_nonnegative_expected_gross_return() -> None:
    matrix = np.asarray(
        [[-2.0], [-1.0], [0.0], [1.0], [2.0]], dtype=np.float64
    )
    returns = np.asarray([-1.0, -1.0, -1.0, 2.0, 8.0])
    head = v22._fit_purchase_head(
        [matrix], [returns], alpha=0.01,
        purchase_loss="poisson_capped_gross",
    )
    gross = np.exp(v22._scores(head, matrix))

    assert head.teacher.startswith("poisson_expected_capped_gross")
    assert np.isfinite(gross).all()
    assert (gross > 0.0).all()
    assert gross[-1] > gross[0]


def test_nested_poisson_artifact_predicts_finite_net_returns() -> None:
    artifact = fit_four_head_nested_v22(
        labeled_races(start_day=1, days=8, races_per_day=3),
        minimum_inner_training_dates=2,
        minimum_purchase_training_dates=2,
        alpha=0.01,
        purchase_loss="poisson_capped_gross",
    )
    prediction = predict_race(
        artifact, labeled_races(start_day=9, days=1)[0].decision
    )

    assert artifact.purchase_head.teacher.startswith(
        "poisson_expected_capped_gross"
    )
    assert np.isfinite(prediction.purchase_scores).all()
    assert all(-1.0 < value <= 50.0 for value in prediction.purchase_scores)
    assert prediction.selected_indices == tuple(
        index
        for index, value in enumerate(prediction.purchase_scores)
        if value >= 0.0
    )


def test_purchase_selection_uses_learned_return_break_even_not_oof_roi_search() -> None:
    artifact = fit_four_head_nested_v22(
        labeled_races(start_day=1, days=8, races_per_day=3),
        minimum_inner_training_dates=2,
        minimum_purchase_training_dates=2,
    )

    assert artifact.purchase_threshold == 0.0
    assert artifact.purchase_threshold_source == "learned_unit_return_break_even_zero"


def test_outer_outcomes_cannot_change_predictions_or_artifact() -> None:
    artifact = fitted()
    outer = labeled_races(start_day=8, days=2)
    changed = tuple(
        replace(
            race,
            outcome=replace(
                race.outcome,
                winner_index=(race.outcome.winner_index + 1) % artifact.choice_count,
                ranking_order=(
                    (race.outcome.winner_index + 1) % artifact.choice_count,
                    *tuple(
                        choice
                        for choice in race.outcome.ranking_order
                        if choice
                        != (race.outcome.winner_index + 1) % artifact.choice_count
                    ),
                ),
                closing_odds=tuple(value * 1.8 for value in race.outcome.closing_odds),
            ),
        )
        for race in outer
    )
    fingerprint_before = artifact_fingerprint(artifact)
    predictions_before = tuple(predict_race(artifact, race.decision) for race in outer)
    predictions_after = tuple(predict_race(artifact, race.decision) for race in changed)
    first_metrics = evaluate_outer_outcomes(artifact, outer)
    changed_metrics = evaluate_outer_outcomes(artifact, changed)

    assert prediction_fingerprint(predictions_before) == prediction_fingerprint(
        predictions_after
    )
    assert first_metrics["frozen_prediction_sha256"] == changed_metrics[
        "frozen_prediction_sha256"
    ]
    assert first_metrics["artifact_sha256"] == fingerprint_before
    assert changed_metrics["artifact_sha256"] == fingerprint_before
    assert first_metrics["outer_outcomes_used_for_fit_or_selection"] is False
    assert first_metrics["outer_outcomes_role"] == (
        "evaluation_only_after_predictions_frozen"
    )
    assert (
        first_metrics["probability_log_loss"]
        != changed_metrics["probability_log_loss"]
    )
    assert first_metrics["closing_odds_log_mae"] != changed_metrics[
        "closing_odds_log_mae"
    ]


def test_outer_evaluation_is_strictly_disjoint_and_future_only() -> None:
    training = labeled_races(start_day=1, days=7)
    artifact = fit_four_head_nested_v22(training, minimum_inner_training_dates=2)
    with pytest.raises(ValueError, match="overlap"):
        evaluate_outer_outcomes(artifact, (training[-1],))

    old_with_new_id = replace(
        training[-1],
        decision=replace(training[-1].decision, race_id="old-but-unseen-id"),
    )
    with pytest.raises(ValueError, match="strictly after"):
        evaluate_outer_outcomes(artifact, (old_with_new_id,))


def test_labels_are_not_part_of_decision_input_and_invalid_order_is_rejected() -> None:
    race = labeled_races(start_day=1, days=1)[0]
    assert not hasattr(race.decision, "winner_index")
    assert not hasattr(race.decision, "closing_odds")
    reversed_rows = tuple(reversed(labeled_races(start_day=1, days=4)))
    with pytest.raises(ValueError, match="uniquely sorted"):
        fit_four_head_nested_v22(reversed_rows, minimum_inner_training_dates=1)

    invalid_ranking = tuple(
        replace(row, outcome=replace(row.outcome, ranking_order=(0,) * 6))
        if index == 0 else row
        for index, row in enumerate(labeled_races(start_day=1, days=5))
    )
    with pytest.raises(ValueError, match="ranking_order must be a permutation"):
        fit_four_head_nested_v22(
            invalid_ranking,
            minimum_inner_training_dates=1,
            minimum_purchase_training_dates=1,
        )


def test_outer_metrics_cover_all_four_heads() -> None:
    artifact = fitted()
    metrics = evaluate_outer_outcomes(
        artifact, labeled_races(start_day=8, days=2, races_per_day=3)
    )
    assert metrics["races"] == 6
    assert metrics["probability_log_loss"] >= 0.0
    assert 0.0 <= metrics["ranking_top5_hit_rate"] <= 1.0
    assert metrics["closing_odds_log_mae"] >= 0.0
    assert metrics["production_bankroll_evaluated"] is False
    value = metrics["purchase_value_diagnostics"]
    assert value["schema_version"] == 1
    assert value["tickets"] == 6 * artifact.choice_count
    assert len(value["calibration_deciles"]) == 10
    assert 0.0 <= value["positive_predicted_fraction"] <= 1.0
    assert value["calibration_mae"] >= 0.0
    diagnostic = metrics["diagnostic_unit_stake"]
    assert diagnostic["label"] == (
        "equal_one_unit_per_selected_ticket_not_production_bankroll"
    )
    assert diagnostic["stake_units"] >= diagnostic["hits"] >= 0
    if diagnostic["stake_units"]:
        assert diagnostic["roi"] is not None
