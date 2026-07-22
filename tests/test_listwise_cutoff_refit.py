from __future__ import annotations

import numpy as np
import pytest
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import StandardScaler

from boatrace_ai.listwise.cutoff_refit import (
    cutoff_boundaries,
    rescale_weights_preserving_scores,
    validate_source_artifact,
)
from boatrace_ai.listwise.model import ListwiseLinearModel


def test_cutoff_boundaries_require_adjacent_full_days() -> None:
    keys = [
        ("r1", "2026-07-17", "01", 1),
        ("r2", "2026-07-17", "01", 2),
        ("r3", "2026-07-18", "01", 1),
        ("r4", "2026-07-19", "01", 1),
    ]
    assert cutoff_boundaries(
        keys,
        training_cutoff="2026-07-17",
        evaluation_from="2026-07-18",
        evaluation_through="2026-07-19",
    ) == (2, 2, 4)
    with pytest.raises(ValueError, match="adjacent"):
        cutoff_boundaries(
            keys,
            training_cutoff="2026-07-16",
            evaluation_from="2026-07-18",
            evaluation_through="2026-07-19",
        )


def test_scaler_transfer_preserves_linear_scores() -> None:
    weights = np.asarray([2.0, -3.0, 4.0])
    old_scale = np.asarray([2.0, 4.0, 8.0])
    new_scale = np.asarray([1.0, 8.0, 2.0])
    rows = np.asarray([[1.0, 2.0, 3.0], [4.0, 0.5, 6.0]])
    transferred = rescale_weights_preserving_scores(
        weights,
        old_scale=old_scale,
        new_scale=new_scale,
    )
    assert (rows / old_scale) @ weights == pytest.approx(
        (rows / new_scale) @ transferred
    )


def test_scaler_transfer_rejects_invalid_scale() -> None:
    with pytest.raises(ValueError, match="positive scales"):
        rescale_weights_preserving_scores(
            np.asarray([1.0]),
            old_scale=np.asarray([0.0]),
            new_scale=np.asarray([1.0]),
        )


def _source_artifact(*, n_features: int = 3) -> dict[str, object]:
    scaler = StandardScaler(with_mean=False)
    scaler.scale_ = np.ones(n_features)
    scaler.n_features_in_ = n_features
    model = ListwiseLinearModel(
        weights=np.ones(n_features),
        scaler=scaler,
        target="top3_pl",
        alpha=1e-5,
        learning_rate=0.02,
        epochs=3,
    )
    return {
        "model": model,
        "hasher": FeatureHasher(
            n_features=n_features,
            input_type="dict",
            alternate_sign=False,
        ),
        "n_features": n_features,
        "trained_through": ("race", "2026-05-09", "01", 1),
    }


def test_source_artifact_requires_identical_fixed_structure() -> None:
    artifact = _source_artifact()
    model = validate_source_artifact(
        artifact,
        training_cutoff="2026-07-17",
        evaluation_from="2026-07-18",
    )
    assert len(model.weights) == 3

    artifact["n_features"] = 4
    with pytest.raises(ValueError, match="dimensions differ"):
        validate_source_artifact(
            artifact,
            training_cutoff="2026-07-17",
            evaluation_from="2026-07-18",
        )


def test_source_artifact_rejects_signed_hasher() -> None:
    artifact = _source_artifact()
    artifact["hasher"] = FeatureHasher(
        n_features=3,
        input_type="dict",
        alternate_sign=True,
    )
    with pytest.raises(ValueError, match="hasher settings"):
        validate_source_artifact(
            artifact,
            training_cutoff="2026-07-17",
            evaluation_from="2026-07-18",
        )
