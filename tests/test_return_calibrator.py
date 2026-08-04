from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest

from boatrace_ai.listwise.direct_bankroll import (
    COMBINATION_LABELS as ALL_COMBINATIONS,
    standard_direct_policy,
)
from boatrace_ai.listwise.return_bankroll import (
    _select_return_regularization,
    predict_expected_returns_from_state,
    simulate_expected_return_calibrated_bankroll,
    validate_expected_return_inference_state,
)

from boatrace_ai.listwise.return_calibrator import (
    FEATURE_COUNT,
    MIN_EXPECTED_RETURN,
    calibrate_combination_returns,
    expected_return_features,
    expected_return_poisson_loss,
    fit_combination_return_calibrator,
    fit_expected_return_calibrator,
    predict_expected_returns,
)


COMBINATIONS = ("1-2-3", "2-1-3")
COMBINATION_INDEX = {value: index for index, value in enumerate(COMBINATIONS)}
COMBINATION_LANES = np.asarray(((0, 1, 2), (1, 0, 2)), dtype=np.int64)


def test_expected_return_features_encode_candidate_market_edge() -> None:
    race_keys = [("r1", "2026-07-01", "01", 1)]
    candidate = np.asarray([[0.8, 0.2]])
    market = np.asarray([[0.5, 0.5]])

    matrix = expected_return_features(
        candidate,
        market,
        race_keys,
        COMBINATION_LANES,
    )

    assert matrix.shape == (2, FEATURE_COUNT)
    assert matrix[0, 0] == 1.0
    assert matrix[0, 3] == 1.0
    assert matrix[1, 4] == 1.0
    assert matrix[0, 21] == 1.0
    assert matrix[0, 45] == 1.0
    assert matrix[0, 56] > 0.0
    assert matrix[1, 56] < 0.0


def test_combination_calibration_uses_conservative_daily_lower_bound() -> None:
    race_keys = [
        (f"r-{day}", f"2026-06-{day:02d}", "01", 1)
        for day in range(1, 21)
    ]
    predicted = np.full((20, 2), 0.5, dtype=np.float64)
    payouts = {
        race_key[0]: {"combination": "1-2-3", "payout_yen": 200}
        for race_key in race_keys
    }

    calibrator = fit_combination_return_calibrator(
        predicted,
        race_keys,
        payouts,
        COMBINATION_INDEX,
        samples=500,
        seed=7,
    )
    adjusted = calibrate_combination_returns(calibrator, predicted[:1])

    assert calibrator.training_races == 20
    assert calibrator.training_days == 20
    assert calibrator.bootstrap_samples == 500
    assert calibrator.factors.tolist() == [2.0, 0.0]
    assert adjusted.tolist() == [[1.0, MIN_EXPECTED_RETURN]]


def test_combination_calibration_is_deterministic_and_ignores_future_rows() -> None:
    race_keys = [
        (f"r-{day}", f"2026-06-{day:02d}", "01", 1)
        for day in range(1, 11)
    ]
    predicted = np.full((10, 2), 0.8, dtype=np.float64)
    payouts = {
        race_key[0]: {
            "combination": COMBINATIONS[day % 2],
            "payout_yen": 300,
        }
        for day, race_key in enumerate(race_keys)
    }
    first = fit_combination_return_calibrator(
        predicted, race_keys, payouts, COMBINATION_INDEX, samples=500, seed=11
    )
    payouts["future"] = {"combination": "1-2-3", "payout_yen": 999_900}
    second = fit_combination_return_calibrator(
        predicted, race_keys, payouts, COMBINATION_INDEX, samples=500, seed=11
    )

    np.testing.assert_array_equal(first.factors, second.factors)
    np.testing.assert_array_equal(first.lower_bounds, second.lower_bounds)


def test_poisson_validation_loss_prefers_calibrated_returns() -> None:
    race_keys = [("r1", "2026-07-01", "01", 1)]
    payouts = {"r1": {"combination": "1-2-3", "payout_yen": 200}}
    calibrated = expected_return_poisson_loss(
        np.asarray([[2.0, 0.1]]), race_keys, payouts, COMBINATION_INDEX
    )
    reversed_values = expected_return_poisson_loss(
        np.asarray([[0.1, 2.0]]), race_keys, payouts, COMBINATION_INDEX
    )

    assert calibrated < reversed_values


def test_newton_return_calibrator_learns_relative_value() -> None:
    race_count = 1_000
    race_keys = [
        (f"r{index}", "2026-07-01", f"{index % 24 + 1:02d}", index % 12 + 1)
        for index in range(race_count)
    ]
    candidate = np.empty((race_count, 2), dtype=np.float64)
    candidate[: race_count // 2] = (0.8, 0.2)
    candidate[race_count // 2 :] = (0.2, 0.8)
    market = np.full((race_count, 2), 0.5, dtype=np.float64)
    payouts = {}
    for index, race_key in enumerate(race_keys):
        preferred = 0 if index < race_count // 2 else 1
        winner = preferred if index % 10 < 8 else 1 - preferred
        payouts[race_key[0]] = {
            "combination": COMBINATIONS[winner],
            "payout_yen": 150,
        }

    model = fit_expected_return_calibrator(
        candidate,
        market,
        race_keys,
        payouts,
        COMBINATION_LANES,
        COMBINATION_INDEX,
        regularization=0.001,
        max_iterations=30,
        batch_races=100,
    )
    predicted = predict_expected_returns(
        model,
        candidate,
        market,
        race_keys,
        COMBINATION_LANES,
        batch_races=100,
    )

    assert model.training_samples == race_count * 2
    assert np.isfinite(model.objective)
    assert np.isfinite(model.gradient_norm)
    assert model.iterations <= 30
    assert predicted[:500, 0].mean() > predicted[:500, 1].mean()
    assert predicted[500:, 1].mean() > predicted[500:, 0].mean()
    assert 0.8 < predicted[:500, 0].mean() < 1.5


def test_expected_return_calibration_rejects_missing_payouts() -> None:
    race_keys = [
        ("complete", "2026-07-01", "01", 1),
        ("missing", "2026-07-01", "01", 2),
    ]
    probabilities = np.full((2, 2), 0.5)
    payouts = {
        "complete": {"combination": "1-2-3", "payout_yen": 200}
    }

    with pytest.raises(ValueError, match="missing=1"):
        fit_expected_return_calibrator(
            probabilities,
            probabilities,
            race_keys,
            payouts,
            COMBINATION_LANES,
            COMBINATION_INDEX,
        )
    with pytest.raises(ValueError, match="missing=1"):
        fit_combination_return_calibrator(
            probabilities,
            race_keys,
            payouts,
            COMBINATION_INDEX,
        )
    with pytest.raises(ValueError, match="missing=1"):
        expected_return_poisson_loss(
            probabilities,
            race_keys,
            payouts,
            COMBINATION_INDEX,
        )


def test_return_regularization_uses_pre_policy_temporal_validation() -> None:
    target_index = ALL_COMBINATIONS.index("1-2-3")
    race_keys = [
        (f"r-{day}-{race}", f"2026-06-{day:02d}", "01", race)
        for day in range(1, 5)
        for race in range(1, 6)
    ]
    candidate = np.full((20, 120), 0.4 / 119.0)
    candidate[:, target_index] = 0.6
    market = np.full((20, 120), 0.8 / 119.0)
    market[:, target_index] = 0.2
    payouts = {
        race_key[0]: {"combination": "1-2-3", "payout_yen": 200}
        for race_key in race_keys
    }

    selected, diagnostics, period = _select_return_regularization(
        candidate,
        market,
        race_keys,
        payouts,
        fit_stop=15,
        validation_days=1,
        candidates=(0.1, 1.0),
        fallback=0.01,
        max_iterations=5,
        batch_races=10,
    )

    assert selected in {0.01, 0.1, 1.0}
    assert len(diagnostics) == 3
    assert all(np.isfinite(row["poisson_loss"]) for row in diagnostics)
    assert period["fit_through"] == "2026-06-02"
    assert period["validation_from"] == "2026-06-03"


def test_expected_return_bankroll_uses_pre_evaluation_calibration(
    tmp_path: Path,
) -> None:
    target_index = ALL_COMBINATIONS.index("1-2-3")
    calibration_keys = [
        (
            f"cal-{index}",
            f"2026-06-{index // 50 + 1:02d}",
            f"{index % 24 + 1:02d}",
            index % 12 + 1,
        )
        for index in range(200)
    ]
    calibration_candidate = np.full((200, 120), 0.4 / 119.0)
    calibration_candidate[:, target_index] = 0.6
    calibration_market = np.full((200, 120), 0.8 / 119.0)
    calibration_market[:, target_index] = 0.2
    payouts = {
        race_key[0]: {
            "combination": "1-2-3" if index % 10 < 8 else "1-3-2",
            "payout_yen": 200,
        }
        for index, race_key in enumerate(calibration_keys)
    }
    race_keys = [
        ("eval-1", "2026-07-01", "01", 1),
        ("eval-2", "2026-07-02", "01", 2),
    ]
    candidate = np.full((2, 120), 0.4 / 119.0)
    candidate[:, target_index] = 0.6
    market = np.full((2, 120), 0.8 / 119.0)
    market[:, target_index] = 0.2
    payouts.update(
        {
            "eval-1": {"combination": "1-2-3", "payout_yen": 200},
            "eval-2": {"combination": "1-2-3", "payout_yen": 200},
        }
    )
    policy = {**standard_direct_policy(), "ev_threshold": 0.9}
    state = {}

    result = simulate_expected_return_calibrated_bankroll(
        candidate,
        race_keys=race_keys,
        payouts=payouts,
        market_reference_probabilities=market,
        calibration_probabilities=calibration_candidate,
        calibration_market_reference_probabilities=calibration_market,
        calibration_race_keys=calibration_keys,
        policy=policy,
        regularization=0.001,
        regularization_candidates=(0.001,),
        regularization_validation_days=1,
        max_iterations=30,
        batch_races=50,
        policy_selection_days=2,
        minimum_selection_tickets=10_000,
        state_output=state,
    )

    assert result["evaluated_races"] == 2
    assert result["selected_tickets"] >= 2
    assert result["hit_tickets"] >= 2
    assert result["policy"]["expected_return_training_samples"] == 24_000
    assert result["policy"]["ev_threshold"] == 0.9
    assert result["policy_selection"]["source"] == "fallback_fixed_threshold"
    assert result["return_calibrator"]["iterations"] <= 30
    assert np.isfinite(result["return_calibrator"]["gradient_norm"])
    combination = result["return_calibrator"]["combination_calibration"]
    assert combination["training_races"] == 200
    assert combination["training_days"] == 4
    assert set(combination["factors"]) == set(ALL_COMBINATIONS)
    assert set(combination["point_ratios"]) == set(ALL_COMBINATIONS)
    assert set(combination["lower_bounds"]) == set(ALL_COMBINATIONS)
    assert combination["zero_factor_combinations"] == sum(
        value == 0.0 for value in combination["factors"].values()
    )
    assert combination["below_legacy_floor_combinations"] == sum(
        value < 0.25 for value in combination["factors"].values()
    )
    selection = result["policy_selection"]["combination_calibration"]
    assert selection["training_races"] == 100
    assert selection["training_days"] == 2
    assert set(selection["factors"]) == set(ALL_COMBINATIONS)
    validate_expected_return_inference_state(state)
    assert state["trained_through"] == "2026-07-02"
    assert state["valid_for_dates_after"] == "2026-07-02"
    assert state["contains_evaluation_outcomes"] is True
    assert state["holdout_replay_state"] is False
    assert state["return_calibrator"].training_samples == 202 * 120

    state_path = tmp_path / "expected-return-state.joblib"
    joblib.dump(state, state_path)
    restored = joblib.load(state_path)
    next_candidate = candidate[:1]
    next_market = market[:1]
    next_keys = [("next-1", "2026-07-03", "01", 1)]
    before = predict_expected_returns_from_state(
        state, next_candidate, next_market, next_keys
    )
    after = predict_expected_returns_from_state(
        restored, next_candidate, next_market, next_keys
    )
    np.testing.assert_allclose(before, after)
    with pytest.raises(ValueError, match="cannot score its training period"):
        predict_expected_returns_from_state(
            restored,
            next_candidate,
            next_market,
            [("replay", "2026-07-02", "01", 1)],
        )
    with pytest.raises(ValueError, match="cannot score its training period"):
        predict_expected_returns_from_state(
            restored,
            np.repeat(next_candidate, 2, axis=0),
            np.repeat(next_market, 2, axis=0),
            [
                ("future", "2026-07-03", "01", 1),
                ("hidden-replay", "2026-07-02", "01", 2),
            ],
        )


def test_expected_return_bankroll_rejects_temporal_leakage() -> None:
    probabilities = np.full((2, 120), 1.0 / 120.0)
    calibration = np.full((2, 120), 1.0 / 120.0)
    evaluation_keys = [
        ("eval-1", "2026-07-02", "01", 1),
        ("eval-2", "2026-07-03", "01", 2),
    ]
    overlap_keys = [
        ("cal-1", "2026-07-01", "01", 1),
        ("cal-2", "2026-07-02", "01", 2),
    ]

    with pytest.raises(ValueError, match="strictly precede"):
        simulate_expected_return_calibrated_bankroll(
            probabilities,
            race_keys=evaluation_keys,
            payouts={},
            market_reference_probabilities=probabilities,
            calibration_probabilities=calibration,
            calibration_market_reference_probabilities=calibration,
            calibration_race_keys=overlap_keys,
        )

    with pytest.raises(ValueError, match="non-decreasing"):
        simulate_expected_return_calibrated_bankroll(
            probabilities,
            race_keys=list(reversed(evaluation_keys)),
            payouts={},
            market_reference_probabilities=probabilities,
            calibration_probabilities=calibration,
            calibration_market_reference_probabilities=calibration,
            calibration_race_keys=[
                ("cal-1", "2026-06-01", "01", 1),
                ("cal-2", "2026-06-02", "01", 2),
            ],
        )
