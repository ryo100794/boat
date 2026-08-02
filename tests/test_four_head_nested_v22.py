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


def test_nested_tweedie_artifact_predicts_finite_net_returns() -> None:
    artifact = fit_four_head_nested_v22(
        labeled_races(start_day=1, days=8, races_per_day=3),
        minimum_inner_training_dates=2,
        minimum_purchase_training_dates=2,
        alpha=0.01,
        purchase_loss="tweedie_capped_gross",
    )
    prediction = predict_race(
        artifact, labeled_races(start_day=9, days=1)[0].decision
    )

    assert artifact.purchase_head.teacher.startswith(
        "tweedie_expected_capped_gross"
    )
    assert np.isfinite(prediction.purchase_scores).all()
    assert all(-1.0 < value <= 50.0 for value in prediction.purchase_scores)


def test_pairwise_purchase_head_learns_payout_weighted_ticket_order() -> None:
    matrices = [
        np.asarray([[-2.0], [-1.0], [2.0]], dtype=np.float64),
        np.asarray([[-3.0], [0.0], [3.0]], dtype=np.float64),
    ]
    returns = [
        np.asarray([-1.0, -1.0, 8.0]),
        np.asarray([-1.0, -1.0, 3.0]),
    ]

    head, payout = v22._fit_purchase_heads(
        matrices,
        returns,
        alpha=0.01,
        purchase_loss="pairwise_contextual_rank_calibrated",
    )

    assert payout is None
    assert head.teacher.startswith("payout_weighted_winner_over_loser")
    assert v22._scores(head, matrices[0])[-1] > v22._scores(head, matrices[0])[0]


def test_nested_pairwise_purchase_rank_is_oof_calibrated() -> None:
    artifact = fit_four_head_nested_v22(
        labeled_races(start_day=1, days=8, races_per_day=3),
        minimum_inner_training_dates=2,
        minimum_purchase_training_dates=2,
        alpha=0.01,
        purchase_loss="pairwise_contextual_rank_calibrated",
    )
    prediction = predict_race(
        artifact, labeled_races(start_day=9, days=1)[0].decision
    )

    assert artifact.purchase_feature_map == "decision_context_v2"
    assert artifact.purchase_payout_head is None
    assert artifact.purchase_calibration_head is not None
    assert artifact.purchase_calibration_head.teacher.startswith(
        "poisson_calibration_of_strict_purchase_head_oof"
    )
    assert np.isfinite(prediction.purchase_scores).all()




def test_offset_tail_heads_preserve_race_probability_and_uncapped_payout() -> None:
    base_probability = np.asarray([0.55, 0.30, 0.15], dtype=np.float64)
    matrices = [
        np.column_stack(
            (
                base_probability,
                np.asarray([1.0, 0.0, -1.0]),
                np.log(np.asarray([8.0, 15.0, 40.0])),
            )
        ),
        np.column_stack(
            (
                base_probability[::-1],
                np.asarray([-1.0, 0.0, 1.0]),
                np.log(np.asarray([45.0, 18.0, 7.0])),
            )
        ),
    ]
    returns = [
        np.asarray([119.0, -1.0, -1.0]),
        np.asarray([-1.0, -1.0, 89.0]),
    ]

    hit_head, payout_head = v22._fit_purchase_heads(
        matrices,
        returns,
        alpha=0.1,
        purchase_loss="multinomial_offset_uncapped_lognormal",
    )
    scores = v22._purchase_net_scores(hit_head, payout_head, matrices[0])
    conditional_payout = np.exp(
        matrices[0][:, 2] + v22._scores(payout_head, matrices[0])
    )
    learned_probability = (scores + 1.0) / conditional_payout

    assert hit_head.teacher.startswith("multinomial_probability_residual")
    assert payout_head.teacher.startswith("uncapped_log_payout_residual")
    assert learned_probability.sum() == pytest.approx(1.0)
    assert (learned_probability > 0.0).all()
    assert conditional_payout.max() > 51.0
    assert np.isfinite(scores).all()


def test_nested_offset_tail_artifact_uses_context_without_oof_calibration() -> None:
    artifact = fit_four_head_nested_v22(
        labeled_races(start_day=1, days=8, races_per_day=3),
        minimum_inner_training_dates=2,
        minimum_purchase_training_dates=2,
        alpha=0.01,
        purchase_loss="multinomial_offset_uncapped_lognormal",
    )
    prediction = predict_race(
        artifact, labeled_races(start_day=9, days=1)[0].decision
    )

    assert artifact.purchase_feature_map == "decision_context_v2"
    assert artifact.purchase_payout_head is not None
    assert artifact.purchase_calibration_head is None
    assert artifact.purchase_head.teacher.startswith(
        "multinomial_probability_residual"
    )
    assert artifact.purchase_payout_head.teacher.startswith(
        "uncapped_log_payout_residual"
    )
    assert np.isfinite(prediction.purchase_scores).all()
    assert prediction.selected_indices == tuple(
        index
        for index, value in enumerate(prediction.purchase_scores)
        if value >= 0.0
    )


def test_all_choice_closing_head_uses_every_ticket_teacher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrices = [
        np.column_stack(
            (
                np.asarray([0.6, 0.3, 0.1]),
                np.asarray([1.0, 0.0, -1.0]),
                np.log(np.asarray([8.0, 16.0, 40.0])),
            )
        ),
        np.column_stack(
            (
                np.asarray([0.2, 0.3, 0.5]),
                np.asarray([-1.0, 0.0, 1.0]),
                np.log(np.asarray([50.0, 20.0, 7.0])),
            )
        ),
    ]
    gross = [
        np.asarray([10.0, 18.0, 44.0]),
        np.asarray([55.0, 22.0, 8.0]),
    ]
    captured: dict[str, np.ndarray] = {}

    def fake_ridge(
        matrix: np.ndarray,
        target: np.ndarray,
        *,
        alpha: float,
        sample_weight: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        captured["matrix"] = matrix.copy()
        captured["target"] = target.copy()
        return np.zeros(matrix.shape[1]), 0.0

    monkeypatch.setattr(v22, "_fit_ridge", fake_ridge)
    head = v22._fit_all_choice_closing_residual_head(
        matrices, gross, alpha=0.01
    )

    assert captured["matrix"].shape == (6, 3)
    assert captured["target"] == pytest.approx(
        np.concatenate(
            [
                np.log(gross[0]) - matrices[0][:, 2],
                np.log(gross[1]) - matrices[1][:, 2],
            ]
        )
    )
    assert head.teacher.startswith("all_choice_log_closing_odds_residual")


def test_nested_all_choice_closing_artifact_routes_full_closing_targets() -> None:
    artifact = fit_four_head_nested_v22(
        labeled_races(start_day=1, days=8, races_per_day=3),
        minimum_inner_training_dates=2,
        minimum_purchase_training_dates=2,
        alpha=0.01,
        purchase_loss="multinomial_offset_all_choice_closing",
    )
    prediction = predict_race(
        artifact, labeled_races(start_day=9, days=1)[0].decision
    )

    assert artifact.purchase_feature_map == "decision_context_v2"
    assert artifact.purchase_head.teacher.startswith(
        "multinomial_probability_residual"
    )
    assert artifact.purchase_payout_head is not None
    assert artifact.purchase_payout_head.teacher.startswith(
        "all_choice_log_closing_odds_residual"
    )
    assert np.isfinite(prediction.purchase_scores).all()


def test_multinomial_temperature_softens_overconfident_oof_probabilities() -> None:
    temperature = v22._fit_multinomial_temperature(
        [
            np.asarray([0.99, 0.01]),
            np.asarray([0.99, 0.01]),
        ],
        [0, 1],
        alpha=1e-3,
    )

    assert temperature > 1.0


def test_nested_temperature_artifact_learns_from_strict_purchase_oof() -> None:
    artifact = fit_four_head_nested_v22(
        labeled_races(start_day=1, days=8, races_per_day=3),
        minimum_inner_training_dates=2,
        minimum_purchase_training_dates=2,
        alpha=0.01,
        purchase_loss="multinomial_offset_all_choice_closing_temperature",
    )
    prediction = predict_race(
        artifact, labeled_races(start_day=9, days=1)[0].decision
    )

    assert artifact.purchase_probability_temperature > 0.0
    assert artifact.purchase_oof_folds
    assert artifact.purchase_payout_head is not None
    assert artifact.purchase_payout_head.teacher.startswith(
        "all_choice_log_closing_odds_residual"
    )
    assert np.isfinite(prediction.purchase_scores).all()


def test_t5_market_probability_is_normalized_from_current_odds() -> None:
    matrix = np.zeros((3, 6), dtype=np.float64)
    matrix[:, 5] = np.log(np.asarray([2.0, 4.0, 8.0]))

    probability = v22._market_probability_from_purchase_matrix(matrix)

    assert probability == pytest.approx(np.asarray([4 / 7, 2 / 7, 1 / 7]))
    assert probability.sum() == pytest.approx(1.0)


def test_nested_market_offset_artifact_learns_residual_from_t5_odds() -> None:
    artifact = fit_four_head_nested_v22(
        labeled_races(start_day=1, days=8, races_per_day=3),
        minimum_inner_training_dates=2,
        minimum_purchase_training_dates=2,
        alpha=0.01,
        purchase_loss="multinomial_market_offset_all_choice_closing",
    )
    prediction = predict_race(
        artifact, labeled_races(start_day=9, days=1)[0].decision
    )

    assert "_from_t5_market_" in artifact.purchase_head.teacher
    assert artifact.purchase_payout_head is not None
    assert artifact.purchase_payout_head.teacher.startswith(
        "all_choice_log_closing_odds_residual"
    )
    assert np.isfinite(prediction.purchase_scores).all()


def test_oof_residual_scale_learns_signal_and_rejects_noise() -> None:
    market = [np.asarray([0.5, 0.5])] * 6
    signal = [np.asarray([1.0, -1.0])] * 6
    scale, market_loss, scaled_loss = v22._fit_multinomial_residual_scale(
        market, signal, [0] * 6, alpha=0.001
    )

    assert scale > 0.0
    assert scaled_loss < market_loss

    balanced_winners = [0, 1, 0, 1, 0, 1]
    noise_scale, noise_market_loss, noise_scaled_loss = (
        v22._fit_multinomial_residual_scale(
            market, signal, balanced_winners, alpha=0.001
        )
    )
    assert noise_scale == pytest.approx(0.0, abs=1e-7)
    assert noise_scaled_loss == pytest.approx(noise_market_loss)


def test_nested_oof_scaled_market_artifact_freezes_learned_scale() -> None:
    artifact = fit_four_head_nested_v22(
        labeled_races(start_day=1, days=8, races_per_day=3),
        minimum_inner_training_dates=2,
        minimum_purchase_training_dates=2,
        alpha=0.01,
        purchase_loss=(
            "multinomial_market_offset_oof_scaled_all_choice_closing"
        ),
    )
    race = labeled_races(start_day=9, days=1)[0]
    hit_probability = v22.predict_purchase_hit_probabilities(
        artifact, race.decision
    )

    assert 0.0 <= artifact.purchase_residual_scale <= 2.0
    assert artifact.purchase_oof_market_log_loss is not None
    assert artifact.purchase_oof_scaled_log_loss is not None
    assert (
        artifact.purchase_oof_scaled_log_loss
        <= artifact.purchase_oof_market_log_loss + 1e-10
    )
    assert hit_probability is not None
    assert hit_probability.sum() == pytest.approx(1.0)
    assert np.isfinite(predict_race(artifact, race.decision).purchase_scores).all()


def test_hurdle_purchase_heads_learn_hit_probability_and_conditional_payout() -> None:
    matrix = np.asarray(
        [[-2.0], [-1.0], [0.0], [1.0], [2.0]], dtype=np.float64
    )
    returns = np.asarray([-1.0, -1.0, -1.0, 2.0, 8.0])
    hit_head, payout_head = v22._fit_purchase_heads(
        [matrix],
        [returns],
        alpha=0.01,
        purchase_loss="hurdle_logistic_lognormal",
    )
    scores = v22._purchase_net_scores(hit_head, payout_head, matrix)

    assert payout_head is not None
    assert hit_head.teacher.startswith("logistic_hit_probability")
    assert payout_head.teacher.startswith("log_capped_gross_return")
    assert np.isfinite(scores).all()
    assert scores[-1] > scores[0]


def test_nested_hurdle_artifact_predicts_finite_net_returns() -> None:
    artifact = fit_four_head_nested_v22(
        labeled_races(start_day=1, days=8, races_per_day=3),
        minimum_inner_training_dates=2,
        minimum_purchase_training_dates=2,
        alpha=0.01,
        purchase_loss="hurdle_logistic_lognormal",
    )
    prediction = predict_race(
        artifact, labeled_races(start_day=9, days=1)[0].decision
    )

    assert artifact.purchase_payout_head is not None
    assert np.isfinite(prediction.purchase_scores).all()
    assert all(-1.0 < value <= 50.0 for value in prediction.purchase_scores)


def test_nested_calibrated_hurdle_artifact_uses_oof_return_calibration() -> None:
    artifact = fit_four_head_nested_v22(
        labeled_races(start_day=1, days=8, races_per_day=3),
        minimum_inner_training_dates=2,
        minimum_purchase_training_dates=2,
        alpha=0.01,
        purchase_loss="hurdle_logistic_lognormal_calibrated",
    )
    prediction = predict_race(
        artifact, labeled_races(start_day=9, days=1)[0].decision
    )

    assert artifact.purchase_payout_head is not None
    assert artifact.purchase_calibration_head is not None
    assert artifact.purchase_calibration_head.teacher.startswith(
        "poisson_calibration_of_strict_purchase_head_oof"
    )
    assert np.isfinite(prediction.purchase_scores).all()
    assert all(-1.0 < value <= 50.0 for value in prediction.purchase_scores)


def test_contextual_purchase_map_retains_decision_features() -> None:
    race = labeled_races(start_day=1, days=1)[0]
    choices = len(race.decision.current_odds)
    probability = np.full(choices, 1.0 / choices)
    ranking = np.linspace(1.0, 0.0, choices)
    closing = np.asarray(race.outcome.closing_odds)

    base = v22._purchase_matrix(
        race.decision, probability, ranking, closing
    )
    contextual = v22._purchase_matrix(
        race.decision,
        probability,
        ranking,
        closing,
        feature_map="decision_context_v2",
    )

    assert base.shape == (choices, 6)
    assert contextual.shape == (
        choices,
        6 + len(race.decision.features[0]),
    )
    assert np.allclose(contextual[:, 6:], race.decision.features)


def test_nested_contextual_hurdle_uses_decision_feature_map() -> None:
    artifact = fit_four_head_nested_v22(
        labeled_races(start_day=1, days=8, races_per_day=3),
        minimum_inner_training_dates=2,
        minimum_purchase_training_dates=2,
        alpha=0.01,
        purchase_loss="hurdle_contextual_lognormal",
    )
    prediction = predict_race(
        artifact, labeled_races(start_day=9, days=1)[0].decision
    )

    assert artifact.purchase_feature_map == "decision_context_v2"
    assert artifact.purchase_payout_head is not None
    assert len(artifact.purchase_head.coefficients) > 6
    assert np.isfinite(prediction.purchase_scores).all()


def test_contextual_interaction_map_learns_all_pairwise_terms() -> None:
    race = labeled_races(start_day=1, days=1)[0]
    choices = len(race.decision.current_odds)
    probability = np.full(choices, 1.0 / choices)
    ranking = np.linspace(1.0, 0.0, choices)
    closing = np.asarray(race.outcome.closing_odds)

    contextual = v22._purchase_matrix(
        race.decision,
        probability,
        ranking,
        closing,
        feature_map="decision_context_v2",
    )
    interactions = v22._purchase_matrix(
        race.decision,
        probability,
        ranking,
        closing,
        feature_map="decision_context_interactions_v3",
    )
    width = contextual.shape[1]

    assert interactions.shape == (
        choices,
        width + width * (width + 1) // 2,
    )
    assert np.allclose(interactions[:, :width], contextual)
    assert np.isfinite(interactions).all()


def test_nested_contextual_interaction_hurdle_uses_learned_terms() -> None:
    artifact = fit_four_head_nested_v22(
        labeled_races(start_day=1, days=8, races_per_day=3),
        minimum_inner_training_dates=2,
        minimum_purchase_training_dates=2,
        alpha=0.01,
        purchase_loss="hurdle_contextual_interactions_lognormal",
    )
    prediction = predict_race(
        artifact, labeled_races(start_day=9, days=1)[0].decision
    )

    assert artifact.purchase_feature_map == "decision_context_interactions_v3"
    assert artifact.purchase_payout_head is not None
    contextual_width = 6 + len(
        labeled_races(start_day=9, days=1)[0].decision.features[0]
    )
    expected_width = contextual_width + contextual_width * (
        contextual_width + 1
    ) // 2
    assert len(artifact.purchase_head.coefficients) == expected_width
    assert np.isfinite(prediction.purchase_scores).all()


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
