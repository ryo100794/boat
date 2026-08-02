from __future__ import annotations

from copy import deepcopy

import pytest

from boatrace_ai.joint_scenario_model import (
    fit_conditional_joint_scenario_model,
)
from boatrace_ai.terminal_probability_oof import (
    build_terminal_probability_oof_artifact,
    joint_observations_from_terminal_oof,
)


OUTCOMES = ("A", "B")


def _build(races: list[dict], **kwargs: object) -> dict:
    return build_terminal_probability_oof_artifact(
        races,
        expected_outcomes=OUTCOMES,
        **kwargs,
    )


def _races() -> list[dict]:
    rows = []
    for day in range(1, 7):
        actual = "A" if day % 2 else "B"
        rows.append({
            "race_id": f"202607{day:02d}0101",
            "race_date": f"2026-07-{day:02d}",
            "jcd": "01" if day < 5 else "02",
            "actual_combination": actual,
            "model_probabilities": {"A": 0.6, "B": 0.4},
            "market_probabilities": {"A": 0.55, "B": 0.45},
            "official_closing_odds": (
                {"A": 1.5, "B": 4.0}
                if actual == "A"
                else {"A": 3.5, "B": 1.6}
            ),
        })
    return rows


def test_terminal_teacher_is_soft_strict_oof_with_verified_provenance() -> None:
    artifact = _build(
        _races(), minimum_training_days=2, regularization=1.0
    )

    assert artifact["actual_one_hot_exported_as_probability"] is False
    assert artifact["deployment_eligible"] is False
    assert artifact["predicted_races"] == 4
    assert artifact["strict_oof_metrics"]["evaluated_races"] == 4
    for prediction in artifact["predictions"]:
        assert prediction["teacher_trained_through_date"] < prediction["race_date"]
        assert 0.0 < min(prediction["probabilities"].values())
        assert max(prediction["probabilities"].values()) < 1.0
        assert len(prediction["prediction_sha256"]) == 64


def test_current_date_outcome_cannot_change_its_own_oof_prediction() -> None:
    races = _races()
    baseline = _build(races)
    changed = deepcopy(races)
    changed[-2]["actual_combination"] = "B"
    mutated = _build(changed)

    baseline_predictions = {
        row["race_id"]: row["prediction_sha256"] for row in baseline["predictions"]
    }
    mutated_predictions = {
        row["race_id"]: row["prediction_sha256"] for row in mutated["predictions"]
    }
    changed_race_id = changed[-2]["race_id"]
    assert mutated_predictions[changed_race_id] == baseline_predictions[changed_race_id]


def test_terminal_artifact_adapts_to_joint_generator_observations() -> None:
    races = _races()
    artifact = _build(races)
    observations = joint_observations_from_terminal_oof(races, artifact)
    model = fit_conditional_joint_scenario_model(
        observations, expected_outcomes=("A", "B"), rank=2
    )

    assert len(observations) == 4
    assert model.training_races == 4
    assert model.teacher_artifact_sha256s


def test_terminal_artifact_contract_tampering_is_rejected() -> None:
    races = _races()
    artifact = _build(races)
    artifact["predictions"][0]["prediction_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="prediction hash mismatch"):
        joint_observations_from_terminal_oof(races, artifact)

    artifact = _build(races)
    artifact["predictions"][0]["probabilities"] = {"A": 0.8, "B": 0.2}
    with pytest.raises(ValueError, match="prediction hash mismatch"):
        joint_observations_from_terminal_oof(races, artifact)

    artifact = _build(races)
    artifact["minimum_training_days"] = 3
    with pytest.raises(ValueError, match="contract hash mismatch"):
        joint_observations_from_terminal_oof(races, artifact)


def test_terminal_teacher_rejects_invalid_configuration_and_outcome() -> None:
    with pytest.raises(ValueError, match="minimum_training_days"):
        _build(_races(), minimum_training_days=True)
    with pytest.raises(ValueError, match="regularization"):
        _build(_races(), regularization=float("nan"))

    races = _races()
    races[0]["actual_combination"] = "C"
    with pytest.raises(ValueError, match="outside the outcome schema"):
        _build(races)


def test_terminal_teacher_defaults_to_complete_trifecta_schema() -> None:
    with pytest.raises(ValueError, match="expected set"):
        build_terminal_probability_oof_artifact(_races())
