from __future__ import annotations

from dataclasses import replace

import pytest

from boatrace_ai.joint_parameter_uncertainty import (
    bootstrap_joint_parameter_models,
    generate_parameter_path_draws,
)
from boatrace_ai.joint_scenario_model import (
    JointScenarioObservation,
    TEACHER_KIND,
    outcome_schema_fingerprint,
    terminal_probability_prediction_fingerprint,
)


OUTCOMES = ("A", "B")


def _observation(index: int) -> JointScenarioObservation:
    day = index + 2
    race_id = f"202607{day:02d}0101"
    terminal = {
        "A": 0.7 if index % 2 else 0.3,
        "B": 0.3 if index % 2 else 0.7,
    }
    artifact = "a" * 64
    fold_id = f"fold-{index}"
    fold_manifest = f"{index + 1:064x}"
    return JointScenarioObservation(
        race_date=f"2026-07-{day:02d}",
        race_id=race_id,
        teacher_trained_through_date=f"2026-07-{day - 1:02d}",
        terminal_probability_teacher_kind=TEACHER_KIND,
        terminal_probability_teacher_source="strict-oof-v1",
        terminal_probability_artifact_sha256=artifact,
        terminal_probability_fold_id=fold_id,
        terminal_probability_fold_manifest_sha256=fold_manifest,
        terminal_probability_prediction_sha256=(
            terminal_probability_prediction_fingerprint(
                race_id=race_id,
                probabilities=terminal,
                artifact_sha256=artifact,
                fold_id=fold_id,
                fold_manifest_sha256=fold_manifest,
                feature_cutoff_seconds=0,
                outcomes=OUTCOMES,
            )
        ),
        terminal_probability_outcome_schema_sha256=(
            outcome_schema_fingerprint(OUTCOMES)
        ),
        terminal_probability_feature_cutoff_seconds=0,
        venue="01",
        decision_horizon_seconds=300,
        popularity_band="favorite",
        decision_probabilities={"A": 0.5, "B": 0.5},
        terminal_probability_teacher=terminal,
        decision_market_shares={"A": 0.5, "B": 0.5},
        final_market_shares={"A": 0.6, "B": 0.4},
    )


def test_day_bootstrap_refits_are_deterministic_and_strictly_prior() -> None:
    rows = [_observation(index) for index in range(6)]
    kwargs = {
        "decision_date": "2026-07-09",
        "draws": 4,
        "seed": 43,
        "expected_outcomes": OUTCOMES,
        "fit_options": {"rank": 2, "learn_residual_scales": False},
    }

    first = bootstrap_joint_parameter_models(rows, **kwargs)
    second = bootstrap_joint_parameter_models(rows, **kwargs)

    assert [draw.manifest_sha256 for draw in first] == [
        draw.manifest_sha256 for draw in second
    ]
    assert all(draw.model.training_through < "2026-07-09" for draw in first)
    assert any(len(set(draw.sampled_days)) < 6 for draw in first)
    assert all(draw.model.training_races == 6 for draw in first)


def test_bootstrap_rejects_current_or_future_teacher() -> None:
    rows = [_observation(index) for index in range(6)]
    future = replace(rows[-1], race_date="2026-07-09")

    with pytest.raises(ValueError, match="strictly prior"):
        bootstrap_joint_parameter_models(
            [*rows[:-1], future],
            decision_date="2026-07-09",
            draws=2,
            expected_outcomes=OUTCOMES,
        )


def test_refitted_models_generate_one_inner_path_set_per_outer_draw() -> None:
    draws = bootstrap_joint_parameter_models(
        [_observation(index) for index in range(6)],
        decision_date="2026-07-09",
        draws=4,
        seed=47,
        expected_outcomes=OUTCOMES,
        fit_options={"rank": 2, "learn_residual_scales": False},
    )

    paths = generate_parameter_path_draws(
        draws,
        decision_probabilities={"A": 0.55, "B": 0.45},
        decision_market_shares={"A": 0.6, "B": 0.4},
        venue="01",
        decision_horizon_seconds=300,
        popularity_band="favorite",
        scenarios_per_draw=8,
        seed=53,
    )

    assert len(paths) == 4
    assert all(len(draw) == 8 for draw in paths)
    assert all(sum(row.weight for row in draw) == pytest.approx(1.0) for draw in paths)
    assert all(
        row.market_state["parameter_draw_index"] == draw_index
        and row.market_state["parameter_draw_manifest_sha256"]
        == draws[draw_index].manifest_sha256
        for draw_index, path_draw in enumerate(paths)
        for row in path_draw
    )
