from __future__ import annotations

import numpy as np

from boatrace_ai.listwise.payout_estimator import (
    FEATURE_COUNT,
    ConditionalPayoutRegressor,
    ConditionalPayoutStatistics,
    ConditionalPayoutTailCalibrator,
    payout_tail_bin_indices,
    fit_conditional_payout,
    fit_conditional_payout_statistics,
    payout_features,
    predict_conditional_odds,
)


def test_prediction_uses_lognormal_mean_correction() -> None:
    model = ConditionalPayoutRegressor(
        weights=np.zeros(FEATURE_COUNT),
        residual_variance=2.0,
        ridge=10.0,
        training_samples=100,
    )

    predicted = predict_conditional_odds(
        model,
        [0.1],
        ["1-2-3"],
        [("r1", "2026-07-01", "01", 1)],
    )

    median = predict_conditional_odds(
        model,
        [0.1],
        ["1-2-3"],
        [("r1", "2026-07-01", "01", 1)],
        lognormal_mean_correction=False,
    )

    halfway = predict_conditional_odds(
        model,
        [0.1],
        ["1-2-3"],
        [("r1", "2026-07-01", "01", 1)],
        mean_correction_factor=0.5,
    )

    np.testing.assert_allclose(predicted, [np.e])
    np.testing.assert_allclose(halfway, [np.sqrt(np.e)])
    np.testing.assert_allclose(median, [1.1])


def test_payout_features_encode_probability_order_venue_and_race_number() -> None:
    matrix = payout_features(
        [0.2, 0.05],
        ["1-2-3", "6-5-4"],
        [
            ("r1", "2026-07-01", "01", 2),
            ("r2", "2026-07-01", "24", 11),
        ],
    )

    assert matrix.shape == (2, FEATURE_COUNT)
    assert matrix[0, 3] == 1.0
    assert matrix[1, 8] == 1.0
    assert matrix[0, 21] == 1.0
    assert matrix[1, 44] == 1.0
    assert matrix[0, 45] == 1.0
    assert matrix[1, 47] == 1.0
    assert matrix[1, 53] > matrix[0, 48]
    assert np.count_nonzero(matrix[0]) == 9
    assert np.count_nonzero(matrix[1]) == 9


def test_conditional_payout_regression_learns_probability_payout_relation() -> None:
    rng = np.random.default_rng(20260722)
    probabilities = np.linspace(0.01, 0.35, 500)
    combinations = ["1-2-3" if index % 2 else "6-5-4" for index in range(500)]
    race_keys = [
        (f"r{index}", "2026-07-01", f"{index % 24 + 1:02d}", index % 12 + 1)
        for index in range(500)
    ]
    true_odds = np.clip(0.78 / probabilities, 1.1, 2_000.0)
    payouts = true_odds * 100.0 * np.exp(rng.normal(0.0, 0.03, len(probabilities)))

    model = fit_conditional_payout(
        probabilities,
        combinations,
        race_keys,
        payouts,
    )
    statistics = ConditionalPayoutStatistics.empty()
    statistics.update(
        probabilities[:250],
        combinations[:250],
        race_keys[:250],
        payouts[:250],
    )
    statistics.update(
        probabilities[250:],
        combinations[250:],
        race_keys[250:],
        payouts[250:],
    )
    incremental_model = fit_conditional_payout_statistics(statistics)
    predicted = predict_conditional_odds(
        model,
        [0.02, 0.20],
        ["1-2-3", "1-2-3"],
        [
            ("a", "2026-07-02", "01", 1),
            ("b", "2026-07-02", "01", 1),
        ],
    )

    assert model.training_samples == 500
    np.testing.assert_allclose(incremental_model.weights, model.weights, atol=1e-12)
    assert np.isclose(
        incremental_model.residual_variance,
        model.residual_variance,
        atol=1e-12,
    )
    assert predicted[0] > predicted[1] * 5.0
    np.testing.assert_allclose(predicted, [39.0, 3.9], rtol=0.25)


def test_tail_calibrator_uses_fixed_bins_and_winsorized_sufficient_statistics() -> None:
    probabilities = np.asarray([0.02, 0.005, 0.001, 0.000999, 0.0])
    raw_odds = np.full(5, 10.0)
    actual_odds = np.asarray([0.1, 2.0, 20.0, 100.0, 40.0])
    calibrator = ConditionalPayoutTailCalibrator.empty()

    np.testing.assert_array_equal(
        payout_tail_bin_indices(probabilities),
        [0, 1, 2, 3, 3],
    )
    calibrator.update(probabilities, raw_odds, actual_odds)

    np.testing.assert_array_equal(calibrator.statistics.counts, [1, 1, 1, 2])
    np.testing.assert_allclose(calibrator.statistics.ratio_sums, [0.1, 0.2, 2.0, 8.0])
    np.testing.assert_allclose(
        calibrator.statistics.ratio_square_sums,
        [0.01, 0.04, 4.0, 32.0],
    )
    diagnostics = calibrator.diagnostics()
    assert diagnostics["samples"] == 5
    assert diagnostics["minimum_global_samples"] == 20
    np.testing.assert_array_equal(calibrator.eligible_mask(probabilities), [False] * 5)
    assert diagnostics["ratio_winsor_limits"] == [0.1, 4.0]


def test_tail_calibrator_shrinks_bins_to_global_statistics() -> None:
    probabilities = np.asarray([0.03] * 2 + [0.007] * 10)
    raw_odds = np.full(12, 10.0)
    actual_odds = raw_odds * np.asarray([0.2] * 2 + [0.8] * 10)
    calibrator = ConditionalPayoutTailCalibrator.empty(
        prior_samples=20.0,
        minimum_bin_samples=2,
        confidence_z=0.0,
    )

    calibrator.update(probabilities, raw_odds, actual_odds)
    factors = calibrator.factors()

    global_mean = (2 * 0.2 + 10 * 0.8) / 12
    expected_first_bin = (2 * 0.2 + 20 * global_mean) / 22
    expected_second_bin = (10 * 0.8 + 20 * global_mean) / 30
    np.testing.assert_allclose(
        factors[:2],
        [expected_first_bin, expected_second_bin],
    )
    np.testing.assert_allclose(factors[2:], [0.5, 0.5])


def test_tail_calibrator_uses_one_sided_lower_confidence_factor() -> None:
    probabilities = np.full(80, 0.003)
    raw_odds = np.full(80, 20.0)
    ratios = np.tile([0.4, 1.2], 40)
    conservative = ConditionalPayoutTailCalibrator.empty()
    posterior_mean = ConditionalPayoutTailCalibrator.empty(confidence_z=0.0)

    conservative.update(probabilities, raw_odds, raw_odds * ratios)
    posterior_mean.update(probabilities, raw_odds, raw_odds * ratios)

    conservative_factors = conservative.factors()
    mean_factors = posterior_mean.factors()
    assert np.all(np.isfinite(conservative_factors))
    assert np.all((conservative_factors > 0.0) & (conservative_factors <= 1.0))
    assert conservative_factors[2] < mean_factors[2]


def test_tail_calibrator_empty_and_small_samples_use_conservative_fallback() -> None:
    calibrator = ConditionalPayoutTailCalibrator.empty()
    probabilities = np.asarray([0.03, 0.007, 0.003, 0.0003])
    raw_odds = np.asarray([2.0, 10.0, 100.0, 1_000.0])

    empty_calibrated = calibrator.calibrate(probabilities, raw_odds)
    np.testing.assert_allclose(empty_calibrated, raw_odds * 0.5)

    calibrator.update([0.03], [10.0], [40.0])
    factors = calibrator.factors()
    assert factors[0] <= calibrator.fallback_factor
    assert np.all(np.isfinite(factors))
    assert np.all((factors > 0.0) & (factors <= 1.0))


def test_tail_calibrator_never_increases_raw_odds() -> None:
    calibrator = ConditionalPayoutTailCalibrator.empty(minimum_bin_samples=1)
    probabilities = np.asarray([0.03, 0.007, 0.003, 0.0003])
    raw_odds = np.asarray([2.0, 10.0, 100.0, 1_000.0])
    calibrator.update(
        np.repeat(probabilities, 30),
        np.tile(raw_odds, 30),
        np.tile(raw_odds * 10.0, 30),
    )

    calibrated = calibrator.calibrate(probabilities, raw_odds)

    assert np.all(np.isfinite(calibrated))
    assert np.all(calibrated > 0.0)
    assert np.all(calibrated <= raw_odds)


def test_tail_calibrator_batch_updates_match_single_update() -> None:
    rng = np.random.default_rng(20260723)
    probabilities = 10.0 ** rng.uniform(-4.0, -1.0, 500)
    raw_odds = rng.uniform(2.0, 500.0, 500)
    actual_odds = raw_odds * rng.uniform(0.01, 5.0, 500)
    single = ConditionalPayoutTailCalibrator.empty()
    batched = ConditionalPayoutTailCalibrator.empty()

    single.update(probabilities, raw_odds, actual_odds)
    for start, stop in ((0, 17), (17, 123), (123, 301), (301, 500)):
        batched.update(
            probabilities[start:stop],
            raw_odds[start:stop],
            actual_odds[start:stop],
        )

    np.testing.assert_array_equal(
        batched.statistics.counts,
        single.statistics.counts,
    )
    np.testing.assert_array_equal(
        batched.statistics.ratio_sums,
        single.statistics.ratio_sums,
    )
    np.testing.assert_array_equal(
        batched.statistics.ratio_square_sums,
        single.statistics.ratio_square_sums,
    )
    np.testing.assert_array_equal(batched.factors(), single.factors())
    assert batched.diagnostics() == single.diagnostics()


def test_tail_calibrator_marks_unsupported_bins_ineligible_without_raw_fallback() -> None:
    calibrator = ConditionalPayoutTailCalibrator.empty(
        minimum_bin_samples=4,
        minimum_global_samples=8,
        fallback_factor=0.4,
    )
    probabilities = np.asarray([0.02, 0.005, 0.001, 0.000999])
    raw_odds = np.asarray([10.0, 20.0, 100.0, 1_000.0])
    calibrator.update(probabilities, raw_odds, raw_odds * 0.8)

    calibrated, eligible = calibrator.calibrate_with_eligibility(
        probabilities,
        raw_odds,
    )

    np.testing.assert_array_equal(eligible, [False, False, False, False])
    np.testing.assert_allclose(calibrated, raw_odds * 0.4)
    assert np.all(calibrated < raw_odds)


def test_tail_calibrator_does_not_use_global_support_for_sparse_bins() -> None:
    calibrator = ConditionalPayoutTailCalibrator.empty(
        minimum_bin_samples=20,
        minimum_global_samples=8,
        fallback_factor=0.9,
        confidence_z=0.0,
    )
    probabilities = np.full(8, 0.03)
    raw_odds = np.full(8, 10.0)
    calibrator.update(probabilities, raw_odds, raw_odds * 0.7)

    calibrated, eligible = calibrator.calibrate_with_eligibility(
        [0.03, 0.007, 0.003, 0.0003],
        [10.0, 20.0, 100.0, 1_000.0],
    )

    np.testing.assert_array_equal(eligible, [False, False, False, False])
    np.testing.assert_allclose(calibrated, [7.0, 14.0, 70.0, 700.0])


def test_tail_calibrator_requires_support_in_each_probability_bin() -> None:
    calibrator = ConditionalPayoutTailCalibrator.empty(
        minimum_bin_samples=4,
        minimum_global_samples=8,
        confidence_z=0.0,
    )
    probabilities = np.asarray([0.03] * 4 + [0.007] * 4)
    raw_odds = np.full(8, 10.0)
    calibrator.update(probabilities, raw_odds, raw_odds * 0.7)

    eligible = calibrator.eligible_mask([0.03, 0.007, 0.003, 0.0003])

    np.testing.assert_array_equal(eligible, [True, True, False, False])


def test_tail_calibrator_bin_support_boundary_is_inclusive() -> None:
    calibrator = ConditionalPayoutTailCalibrator.empty()
    calibrator.update(
        np.full(19, 0.003),
        np.full(19, 100.0),
        np.full(19, 80.0),
    )
    np.testing.assert_array_equal(calibrator.eligible_mask([0.003]), [False])

    calibrator.update([0.003], [100.0], [80.0])

    np.testing.assert_array_equal(calibrator.eligible_mask([0.003]), [True])
    np.testing.assert_array_equal(calibrator.eligible_mask([0.0003]), [False])


def test_tail_calibrator_daily_batch_update_is_order_invariant() -> None:
    rng = np.random.default_rng(20260724)
    probabilities = 10.0 ** rng.uniform(-4.5, -1.0, 800)
    raw_odds = rng.uniform(2.0, 1_000.0, 800)
    actual_odds = raw_odds * rng.lognormal(-0.2, 0.9, 800)
    shuffled_indices = rng.permutation(len(probabilities))
    chronological = ConditionalPayoutTailCalibrator.empty()
    shuffled = ConditionalPayoutTailCalibrator.empty()

    chronological.update(probabilities, raw_odds, actual_odds)
    shuffled.update(
        probabilities[shuffled_indices],
        raw_odds[shuffled_indices],
        actual_odds[shuffled_indices],
    )

    np.testing.assert_array_equal(
        chronological.statistics.counts,
        shuffled.statistics.counts,
    )
    np.testing.assert_allclose(
        chronological.statistics.ratio_sums,
        shuffled.statistics.ratio_sums,
    )
    np.testing.assert_allclose(
        chronological.statistics.ratio_square_sums,
        shuffled.statistics.ratio_square_sums,
    )
    np.testing.assert_allclose(chronological.factors(), shuffled.factors())
    np.testing.assert_array_equal(
        chronological.eligible_mask(probabilities),
        shuffled.eligible_mask(probabilities),
    )


def test_prediction_can_apply_tail_calibrator_after_ridge() -> None:
    model = ConditionalPayoutRegressor(
        weights=np.zeros(FEATURE_COUNT),
        residual_variance=2.0,
        ridge=10.0,
        training_samples=100,
    )
    calibrator = ConditionalPayoutTailCalibrator.empty()

    raw = predict_conditional_odds(
        model,
        [0.1],
        ["1-2-3"],
        [("r1", "2026-07-01", "01", 1)],
    )
    calibrated = predict_conditional_odds(
        model,
        [0.1],
        ["1-2-3"],
        [("r1", "2026-07-01", "01", 1)],
        tail_calibrator=calibrator,
    )

    np.testing.assert_allclose(calibrated, raw * 0.5)
