from __future__ import annotations

import numpy as np

from boatrace_ai.listwise.v25_oracle_band_classifier_v37 import (
    _classification_vector,
    _oracle_band_label,
    _select_threshold,
)


def _example(*, closing_ev: float, closing_odds: float = 20.0) -> dict:
    return {
        "features": np.asarray([1.0, 2.0]),
        "model_probability": 0.05,
        "current_ev": 0.9,
        "closing_ev": closing_ev,
        "closing_odds": closing_odds,
    }


def test_oracle_band_label_has_explicit_boundaries() -> None:
    assert _oracle_band_label(_example(closing_ev=0.95)) == 1
    assert _oracle_band_label(_example(closing_ev=1.00)) == 1
    assert _oracle_band_label(_example(closing_ev=0.949)) == 0
    assert _oracle_band_label(_example(closing_ev=1.001)) == 0
    assert _oracle_band_label(_example(closing_ev=0.97, closing_odds=81.0)) == 0


def test_classifier_vector_excludes_result_and_payout() -> None:
    first = _example(closing_ev=0.97)
    second = {**first, "actual_combination": "6-5-4", "actual_payout_yen": 999999}
    np.testing.assert_array_equal(
        _classification_vector(first), _classification_vector(second)
    )


def test_threshold_selection_prefers_precise_prior_oof_boundary() -> None:
    scores = np.linspace(0.01, 0.99, 100)
    labels = np.asarray([0] * 90 + [1] * 10, dtype=np.int8)
    selected = _select_threshold(scores, labels)
    assert selected is not None
    assert selected["precision"] == 1.0
    assert selected["recall"] >= 0.5
