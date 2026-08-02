from __future__ import annotations

import numpy as np
import pytest

from boatrace_ai.joint_scenario_model import (
    JointScenarioObservation,
    TEACHER_KIND,
    evaluate_joint_scenario_walk_forward,
    fit_conditional_joint_scenario_model,
    generate_joint_market_scenarios,
    joint_scenario_model_diagnostics,
    outcome_schema_fingerprint,
    terminal_probability_prediction_fingerprint,
)


def _observation(
    index: int,
    *,
    terminal_a: float,
    final_market_a: float,
    venue: str = "01",
    band: str = "favorite",
) -> JointScenarioObservation:
    day = index + 2
    outcomes = ("A", "B")
    terminal = {"A": terminal_a, "B": 1.0 - terminal_a}
    artifact_sha = "a" * 64
    fold_id = f"fold-{index}"
    fold_manifest_sha = f"{index + 1:064x}"
    race_id = f"202607{day:02d}0101"
    return JointScenarioObservation(
        race_date=f"2026-07-{day:02d}",
        race_id=race_id,
        teacher_trained_through_date=f"2026-07-{day - 1:02d}",
        terminal_probability_teacher_kind=TEACHER_KIND,
        terminal_probability_teacher_source="oof-terminal-v1",
        terminal_probability_artifact_sha256=artifact_sha,
        terminal_probability_fold_id=fold_id,
        terminal_probability_fold_manifest_sha256=fold_manifest_sha,
        terminal_probability_prediction_sha256=(
            terminal_probability_prediction_fingerprint(
                race_id=race_id,
                probabilities=terminal,
                artifact_sha256=artifact_sha,
                fold_id=fold_id,
                fold_manifest_sha256=fold_manifest_sha,
                feature_cutoff_seconds=0,
                outcomes=outcomes,
            )
        ),
        terminal_probability_outcome_schema_sha256=(
            outcome_schema_fingerprint(outcomes)
        ),
        terminal_probability_feature_cutoff_seconds=0,
        venue=venue,
        decision_horizon_seconds=300,
        popularity_band=band,
        decision_probabilities={"A": 0.5, "B": 0.5},
        terminal_probability_teacher=terminal,
        decision_market_shares={"A": 0.5, "B": 0.5},
        final_market_shares={"A": final_market_a, "B": 1.0 - final_market_a},
    )


def _training_rows() -> list[JointScenarioObservation]:
    rows = []
    for index in range(12):
        terminal = (
            0.9 if index >= 10 else 0.7 if index % 2 else 0.3
        )
        final_market = (
            0.1 if index >= 10 else 0.3 if index % 2 else 0.7
        )
        rows.append(_observation(
            index,
            terminal_a=terminal,
            final_market_a=final_market,
            venue="01" if index < 10 else "02",
        ))
    return rows


def test_fit_rejects_one_hot_or_non_prior_probability_teacher() -> None:
    row = _observation(0, terminal_a=0.7, final_market_a=0.3)
    invalid_kind = JointScenarioObservation(
        **{**row.__dict__, "terminal_probability_teacher_kind": "actual_one_hot"}
    )
    with pytest.raises(ValueError, match="strict OOF"):
        fit_conditional_joint_scenario_model([invalid_kind, row, row])

    leaking = JointScenarioObservation(
        **{**row.__dict__, "teacher_trained_through_date": row.race_date}
    )
    with pytest.raises(ValueError, match="strictly prior"):
        fit_conditional_joint_scenario_model([leaking, row, row])

    one_hot = {"A": 1.0, "B": 0.0}
    mislabeled_one_hot = JointScenarioObservation(
        **{
            **row.__dict__,
            "terminal_probability_teacher": one_hot,
            "terminal_probability_prediction_sha256": (
                terminal_probability_prediction_fingerprint(
                    race_id=row.race_id,
                    probabilities=one_hot,
                    artifact_sha256=row.terminal_probability_artifact_sha256,
                    fold_id=row.terminal_probability_fold_id,
                    fold_manifest_sha256=(
                        row.terminal_probability_fold_manifest_sha256
                    ),
                    feature_cutoff_seconds=0,
                    outcomes=("A", "B"),
                )
            ),
        }
    )
    with pytest.raises(ValueError, match="one-hot"):
        fit_conditional_joint_scenario_model([mislabeled_one_hot, row, row])


def test_fit_rejects_unverifiable_terminal_teacher_provenance() -> None:
    row = _observation(0, terminal_a=0.7, final_market_a=0.3)
    bad_hash = JointScenarioObservation(
        **{**row.__dict__, "terminal_probability_prediction_sha256": "b" * 64}
    )
    with pytest.raises(ValueError, match="prediction hash mismatch"):
        fit_conditional_joint_scenario_model([bad_hash, row, row])

    bad_cutoff = JointScenarioObservation(
        **{**row.__dict__, "terminal_probability_feature_cutoff_seconds": 10}
    )
    with pytest.raises(ValueError, match="zero seconds"):
        fit_conditional_joint_scenario_model([bad_cutoff, row, row])


def test_generated_probability_and_market_vectors_remain_simplexes() -> None:
    model = fit_conditional_joint_scenario_model(
        _training_rows(), rank=2, pooling_strength=4.0
    )
    scenarios = generate_joint_market_scenarios(
        model,
        decision_probabilities={"A": 0.55, "B": 0.45},
        decision_market_shares={"A": 0.6, "B": 0.4},
        venue="01",
        decision_horizon_seconds=300,
        popularity_band="favorite",
        scenarios=200,
        seed=9,
    )

    assert len(scenarios) == 200
    assert sum(row.weight for row in scenarios) == pytest.approx(1.0)
    assert all(sum(row.probabilities.values()) == pytest.approx(1.0) for row in scenarios)
    assert all(
        sum(row.market_state["final_market_shares"].values()) == pytest.approx(1.0)
        for row in scenarios
    )


def test_shared_factor_preserves_learned_negative_probability_market_dependence() -> None:
    model = fit_conditional_joint_scenario_model(
        _training_rows(),
        rank=2,
        pooling_strength=4.0,
        diagonal_noise_fraction=0.0,
    )
    scenarios = generate_joint_market_scenarios(
        model,
        decision_probabilities={"A": 0.5, "B": 0.5},
        decision_market_shares={"A": 0.5, "B": 0.5},
        venue="01",
        decision_horizon_seconds=300,
        popularity_band="favorite",
        scenarios=2_000,
        seed=11,
    )
    probability = np.asarray([row.probabilities["A"] for row in scenarios])
    market = np.asarray([
        row.market_state["final_market_shares"]["A"] for row in scenarios
    ])

    assert np.corrcoef(probability, market)[0, 1] < -0.9


def test_group_mean_is_partially_pooled_and_unseen_context_falls_back() -> None:
    weak_pooling = fit_conditional_joint_scenario_model(
        _training_rows(), rank=2, pooling_strength=1.0
    )
    strong_pooling = fit_conditional_joint_scenario_model(
        _training_rows(), rank=2, pooling_strength=100.0
    )
    key = ("02", 300, "favorite")
    weak_distance = np.linalg.norm(
        weak_pooling.group_means[key] - weak_pooling.residual_mean
    )
    strong_distance = np.linalg.norm(
        strong_pooling.group_means[key] - strong_pooling.residual_mean
    )
    assert 0.0 < strong_distance < weak_distance
    scenarios = generate_joint_market_scenarios(
        strong_pooling,
        decision_probabilities={"A": 0.5, "B": 0.5},
        decision_market_shares={"A": 0.5, "B": 0.5},
        venue="24",
        decision_horizon_seconds=300,
        popularity_band="unseen",
        scenarios=5,
        seed=4,
    )
    assert all(row.market_state["context_fallback"] is True for row in scenarios)


def test_unseen_interaction_reuses_available_hierarchical_main_effects() -> None:
    model = fit_conditional_joint_scenario_model(
        _training_rows(), rank=2, pooling_strength=10.0
    )
    scenarios = generate_joint_market_scenarios(
        model,
        decision_probabilities={"A": 0.5, "B": 0.5},
        decision_market_shares={"A": 0.5, "B": 0.5},
        venue="02",
        decision_horizon_seconds=300,
        popularity_band="new-band",
        scenarios=500,
        seed=5,
    )
    mean_probability = np.mean([row.probabilities["A"] for row in scenarios])
    assert mean_probability > 0.5
    assert all(row.market_state["context_fallback"] is True for row in scenarios)


def test_generation_is_deterministic_and_reports_teacher_boundary() -> None:
    model = fit_conditional_joint_scenario_model(_training_rows(), rank=2)
    kwargs = {
        "decision_probabilities": {"A": 0.5, "B": 0.5},
        "decision_market_shares": {"A": 0.5, "B": 0.5},
        "venue": "01",
        "decision_horizon_seconds": 300,
        "popularity_band": "favorite",
        "scenarios": 10,
        "seed": 17,
    }
    first = generate_joint_market_scenarios(model, **kwargs)
    second = generate_joint_market_scenarios(model, **kwargs)
    assert first == second
    diagnostics = joint_scenario_model_diagnostics(model)
    assert diagnostics["actual_one_hot_used_as_terminal_probability_teacher"] is False
    assert diagnostics["role"].endswith("not_yet_policy_connected")


def test_walk_forward_uses_only_strictly_prior_joint_observations() -> None:
    rows = _training_rows()
    actual = {
        row.race_id: "A" if index % 2 else "B"
        for index, row in enumerate(rows)
    }
    result = evaluate_joint_scenario_walk_forward(
        rows,
        actual,
        minimum_training_days=5,
        scenarios_per_race=32,
        rank=2,
        seed=19,
    )

    assert result["evaluated_days"] == 7
    assert result["evaluated_races"] == 7
    assert all(
        row["trained_through_date"] < row["date"] for row in result["days"]
    )
    assert result["role"].endswith("not_policy_or_ga_fitness")


def test_walk_forward_current_teacher_cannot_change_own_prediction() -> None:
    rows = _training_rows()
    actual = {row.race_id: "A" for row in rows}
    baseline = evaluate_joint_scenario_walk_forward(
        rows[:7],
        actual,
        minimum_training_days=5,
        scenarios_per_race=32,
        rank=2,
        seed=23,
    )
    changed_terminal = {"A": 0.8, "B": 0.2}
    target = rows[5]
    changed = JointScenarioObservation(
        **{
            **target.__dict__,
            "terminal_probability_teacher": changed_terminal,
            "terminal_probability_prediction_sha256": (
                terminal_probability_prediction_fingerprint(
                    race_id=target.race_id,
                    probabilities=changed_terminal,
                    artifact_sha256=target.terminal_probability_artifact_sha256,
                    fold_id=target.terminal_probability_fold_id,
                    fold_manifest_sha256=(
                        target.terminal_probability_fold_manifest_sha256
                    ),
                    feature_cutoff_seconds=0,
                    outcomes=("A", "B"),
                )
            ),
        }
    )
    mutated_rows = [*rows[:5], changed, rows[6]]
    mutated = evaluate_joint_scenario_walk_forward(
        mutated_rows,
        actual,
        minimum_training_days=5,
        scenarios_per_race=32,
        rank=2,
        seed=23,
    )

    assert (
        mutated["days"][0]["metrics"]["generated_log_loss"]
        == pytest.approx(baseline["days"][0]["metrics"]["generated_log_loss"])
    )
