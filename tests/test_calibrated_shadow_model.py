from unittest.mock import patch

import pytest

from boatrace_ai.calibrated_shadow_model import (
    normalize_model_kind,
    predict_probabilities,
    train_bundle,
)


def synthetic_entries(*_args, **_kwargs):
    for race in range(8):
        for lane in range(1, 7):
            yield (
                {
                    "lane": str(lane),
                    "lane_num": lane,
                    "ability": 7.0 - lane + race * 0.01,
                    "origin": "A" if lane % 2 else "B",
                },
                1 if lane == 1 else 0,
                {"race_id": f"r{race}", "lane": lane},
            )


@pytest.mark.parametrize("model_kind", ["linear", "mlp"])
def test_calibrated_shadow_trains_in_batches(model_kind) -> None:
    with patch(
        "boatrace_ai.calibrated_shadow_model.iter_training_entries",
        side_effect=synthetic_entries,
    ):
        bundle = train_bundle(
            None,
            include_races={f"r{race}" for race in range(8)},
            model_kind=model_kind,
            n_features=256,
            batch_size=12,
            epochs=1,
        )

    probabilities = predict_probabilities(
        bundle,
        [
            {"lane": "1", "lane_num": 1, "ability": 6.0, "origin": "A"},
            {"lane": "6", "lane_num": 6, "ability": 1.0, "origin": "B"},
        ],
    )

    assert bundle["model_kind"] == model_kind
    assert len(probabilities) == 2
    assert all(0.0 <= value <= 1.0 for value in probabilities)


def test_unknown_model_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown model kind"):
        normalize_model_kind("tree")
