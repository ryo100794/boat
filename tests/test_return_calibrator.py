from __future__ import annotations

import numpy as np

from boatrace_ai.listwise.direct_bankroll import (
    COMBINATION_LABELS as ALL_COMBINATIONS,
    standard_direct_policy,
)
from boatrace_ai.listwise.return_bankroll import (
    simulate_expected_return_calibrated_bankroll,
)

from boatrace_ai.listwise.return_calibrator import (
    FEATURE_COUNT,
    expected_return_features,
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


def test_expected_return_bankroll_uses_pre_evaluation_calibration() -> None:
    target_index = ALL_COMBINATIONS.index("1-2-3")
    calibration_keys = [
        (f"cal-{index}", "2026-06-01", f"{index % 24 + 1:02d}", index % 12 + 1)
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
        max_iterations=30,
        batch_races=50,
    )

    assert result["evaluated_races"] == 2
    assert result["selected_tickets"] >= 2
    assert result["hit_tickets"] >= 2
    assert result["policy"]["expected_return_training_samples"] == 24_000
    assert result["policy"]["ev_threshold"] == 0.9
    assert result["policy_selection"]["source"] == "fallback_fixed_threshold"
    assert result["return_calibrator"]["iterations"] <= 30
    assert np.isfinite(result["return_calibrator"]["gradient_norm"])
