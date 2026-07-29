from __future__ import annotations
import copy
import itertools
import random
from datetime import date, timedelta

import numpy as np
import pytest

from boatrace_ai.listwise.closing_odds_multihorizon_v11 import (
    CHECKPOINT_LABELS,
    CHECKPOINT_OFFSETS_SECONDS,
    FEATURE_NAMES,
    _examples_from_race,
    _finite_sample_lower_rank,
    _fit_point_model,
    build_checkpoint_feature_vector,
    closing_odds_multihorizon_v11_metrics,
    fit_closing_odds_multihorizon_v11,
    forecast_closing_odds_multihorizon_v11,
    missing_checkpoint_labels,
    normalize_labeled_checkpoints,
    select_teacher_final_odds,
)


COMBINATIONS = ("1-2-3", "1-2-4", "1-3-2", "2-1-3", "3-1-2", "4-1-2")
ALL_COMBINATIONS = tuple(
    "-".join(str(value) for value in combination)
    for combination in itertools.permutations(range(1, 7), 3)
)


def _synthetic_race(
    race_date: str,
    race_index: int,
    *,
    include_final: bool = True,
) -> dict[str, object]:
    day_index = (date.fromisoformat(race_date) - date(2026, 1, 1)).days
    venue = 1 + (day_index + race_index) % 24
    final = {
        combination: (
            4.0
            + 2.25 * combination_index
            + 0.11 * venue
            + 0.17 * race_index
            + 0.03 * day_index
        )
        for combination_index, combination in enumerate(COMBINATIONS)
    }
    checkpoints: dict[str, dict[str, object]] = {}
    for horizon_index, horizon in enumerate(CHECKPOINT_OFFSETS_SECONDS):
        odds = {}
        for combination_index, combination in enumerate(COMBINATIONS):
            deterministic_residual = (
                ((day_index + combination_index + race_index) % 5) - 2
            ) * 0.012
            log_final_to_current = (
                -0.025
                - 0.00016 * horizon
                + 0.018 * combination_index
                + deterministic_residual
            )
            odds[combination] = final[combination] / np.exp(log_final_to_current)
        checkpoints[f"t{horizon}"] = {
            "target_offset_seconds": horizon,
            "checkpoint_attempt": 1,
            "captured_at": (
                f"{race_date}T{6 + race_index:02d}:"
                f"{10 + horizon_index:02d}:00+00:00"
            ),
            "source_update_staleness_seconds": float(
                3 + horizon_index * 4 + day_index % 3
            ),
            "captured_age_seconds": float(horizon + 2 + horizon_index),
            "odds": odds,
        }
    race: dict[str, object] = {
        "race_date": race_date,
        "race_id": f"{race_date}-{venue:02d}-{race_index:02d}",
        "jcd": f"{venue:02d}",
        "rno": race_index,
        "closing_odds_checkpoints": checkpoints,
    }
    if include_final:
        race["closing_odds"] = final
    return race


def _training_races() -> list[dict[str, object]]:
    start = date(2026, 1, 1)
    return [
        _synthetic_race((start + timedelta(days=day)).isoformat(), race_index)
        for day in range(7)
        for race_index in (1, 2)
    ]


def _official_120_race(race_date: str, race_index: int) -> dict[str, object]:
    race = _synthetic_race(race_date, race_index, include_final=False)
    official = {
        combination: 3.0 + 0.25 * index
        for index, combination in enumerate(ALL_COMBINATIONS)
    }
    race["official_closing_odds"] = official
    race["closing_odds"] = {
        combination: value * 9.0 for combination, value in official.items()
    }
    for horizon_index, horizon in enumerate(CHECKPOINT_OFFSETS_SECONDS):
        race["closing_odds_checkpoints"][f"t{horizon}"]["odds"] = {
            combination: value / np.exp(-0.02 - 0.0001 * horizon)
            for combination, value in official.items()
        }
        race["closing_odds_checkpoints"][f"t{horizon}"][
            "captured_age_seconds"
        ] = float(horizon + horizon_index)
    return race


def _fit_ready(
    races: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return fit_closing_odds_multihorizon_v11(
        races or _training_races(),
        prediction_date="2026-01-08",
        regularization=0.05,
        minimum_training_days=5,
        minimum_training_races=8,
        minimum_examples_per_horizon=40,
        calibration_warmup_days=2,
        minimum_calibration_days=3,
        minimum_relative_mae_improvement=0.01,
    )


def test_multihorizon_fit_separates_point_and_safe_lower_predictions() -> None:
    model = _fit_ready()
    assert model["ready"] is True
    assert model["teacher"] == (
        "winsorized_log(selected_closing_odds/current_odds)"
    )
    assert model["checkpoint_labels"] == list(CHECKPOINT_LABELS)
    assert set(model["point_models"]) == set(CHECKPOINT_LABELS)
    assert all(model["point_models"][label]["ready"] for label in CHECKPOINT_LABELS)
    assert all(
        {
            "strict_prior_baseline_current_mae",
            "strict_prior_model_mae",
            "strict_prior_selected_mae",
            "selection_reason",
        }
        <= set(model["point_models"][label])
        for label in CHECKPOINT_LABELS
    )
    assert model["lower_quantile_model"]["ready"] is True
    assert model["selection_conformal_required"] is True
    assert all(
        model["lower_quantile_model"]["by_horizon"][label][
            "finite_sample_unit"
        ]
        == "prior_day_cluster"
        for label in CHECKPOINT_LABELS
    )
    assert all(
        model["lower_quantile_model"]["by_horizon"][label][
            "effective_sample_days"
        ]
        == len(
            model["lower_quantile_model"]["by_horizon"][label][
                "daily_lower_residuals"
            ]
        )
        for label in CHECKPOINT_LABELS
    )

    required_features = {
        "log_horizon_seconds",
        "log_current_odds",
        "market_rank",
        "log_odds_slope_per_minute",
        "log_odds_curvature_per_minute2",
        "log1p_source_update_staleness_minutes",
        "log1p_checkpoint_age_before_target_seconds",
        "checkpoint_age_before_target_missing",
        "venue_01",
        "venue_log_odds_01",
        "venue_rank_01",
        "rno_01",
        "hour_sin",
    }
    assert required_features <= set(FEATURE_NAMES)

    race = _synthetic_race("2026-01-08", 1, include_final=False)
    forecast = forecast_closing_odds_multihorizon_v11(
        race, model, as_of_offset_seconds=10
    )
    assert forecast["ready"] is True
    assert forecast["boundary_audit_passed"] is True
    for label in CHECKPOINT_LABELS:
        row = forecast["predictions"][label]
        assert row["ready"] is True
        assert row["future_checkpoint_offsets_used"] == []
        assert set(row["point_final_odds"]) == set(COMBINATIONS)
        for combination in COMBINATIONS:
            assert (
                MIN_ODDS_FOR_TEST
                <= row["lower_final_odds"][combination]
                <= row["point_final_odds"][combination]
            )


MIN_ODDS_FOR_TEST = 1.0


def test_training_and_features_cannot_see_same_day_or_future_checkpoints() -> None:
    training = _training_races()
    same_day_left = _synthetic_race("2026-01-08", 1)
    same_day_right = copy.deepcopy(same_day_left)
    same_day_right["closing_odds"] = {
        key: value * 1000.0
        for key, value in same_day_right["closing_odds"].items()
    }
    left = _fit_ready(training + [same_day_left])
    right = _fit_ready(training + [same_day_right])
    assert left == right
    audit = left["boundary_audit"]
    assert audit["excluded_non_past_races"] == 1
    assert audit["strict_training_boundary"] is True
    assert audit["strict_calibration_boundaries"] is True
    assert all(
        fold["trained_through_date"] < fold["evaluation_date"]
        for fold in audit["calibration_folds"]
    )

    feature_race = _synthetic_race("2026-01-08", 2, include_final=False)
    baseline, trace = build_checkpoint_feature_vector(
        feature_race, checkpoint="t120", combination=COMBINATIONS[0]
    )
    changed = copy.deepcopy(feature_race)
    for label in ("t60", "t30", "t10"):
        changed["closing_odds_checkpoints"][label]["odds"] = {
            key: value * 50.0
            for key, value in changed["closing_odds_checkpoints"][label]["odds"].items()
        }
    after, changed_trace = build_checkpoint_feature_vector(
        changed, checkpoint="t120", combination=COMBINATIONS[0]
    )
    np.testing.assert_array_equal(baseline, after)
    assert trace["used_checkpoint_offsets"] == [300, 120]
    assert changed_trace["future_checkpoint_offsets_used"] == []


def test_missing_checkpoint_is_explicit_and_never_filled_from_the_future() -> None:
    model = _fit_ready()
    race = _synthetic_race("2026-01-08", 1, include_final=False)
    del race["closing_odds_checkpoints"]["t120"]
    assert missing_checkpoint_labels(race) == ["t120"]

    forecast = forecast_closing_odds_multihorizon_v11(
        race, model, as_of_offset_seconds=10
    )
    missing = forecast["predictions"]["t120"]
    assert forecast["missing_checkpoints"] == ["t120"]
    assert missing == {
        "ready": False,
        "reason": "missing_checkpoint",
        "target_offset_seconds": 120,
        "point_final_odds": {},
        "lower_final_odds": {},
        "used_checkpoint_offsets": [],
        "future_checkpoint_offsets_used": [],
    }
    assert forecast["predictions"]["t60"]["ready"] is True
    assert forecast["future_checkpoint_imputation"] is False


def test_insufficient_data_returns_auditable_not_ready_artifact() -> None:
    sparse = [_synthetic_race("2026-01-01", 1)]
    del sparse[0]["closing_odds_checkpoints"]["t10"]
    model = fit_closing_odds_multihorizon_v11(
        sparse,
        prediction_date="2026-01-02",
        minimum_training_days=2,
        minimum_training_races=2,
        minimum_examples_per_horizon=12,
        calibration_warmup_days=1,
        minimum_calibration_days=1,
        minimum_relative_mae_improvement=0.01,
    )
    assert model["ready"] is False
    assert {
        "insufficient_training_days",
        "insufficient_training_races",
        "insufficient_training_examples_by_horizon",
        "insufficient_strict_prior_calibration",
    } <= set(model["not_ready_reasons"])
    assert (
        model["training_summary"]["missing_checkpoint_races_by_horizon"]["t10"] == 1
    )
    assert model["lower_quantile_model"]["by_horizon"]["t10"]["ready"] is False


def test_fit_is_reproducible_under_input_order_changes() -> None:
    races = _training_races()
    shuffled = list(races)
    random.Random(8128).shuffle(shuffled)
    first = _fit_ready(races)
    second = _fit_ready(shuffled)
    assert first == second

    with pytest.raises(ValueError, match="precedes artifact boundary"):
        forecast_closing_odds_multihorizon_v11(
            _synthetic_race("2026-01-07", 1, include_final=False),
            first,
            as_of_offset_seconds=10,
        )


def test_official_120_teacher_precedes_fallback_in_artifact_and_metrics() -> None:
    official_race = _official_120_race("2026-01-07", 3)
    selected, source = select_teacher_final_odds(official_race)
    assert source == "official_closing_odds"
    assert selected == official_race["official_closing_odds"]

    incomplete = copy.deepcopy(official_race)
    incomplete["official_closing_odds"].pop(ALL_COMBINATIONS[-1])
    selected, source = select_teacher_final_odds(incomplete)
    assert source == "closing_odds_fallback"
    assert selected == incomplete["closing_odds"]

    model = _fit_ready(_training_races() + [official_race])
    provenance = model["teacher_provenance"]
    assert provenance["training_races_by_source"]["official_closing_odds"] == 1
    assert provenance["selected_races_by_source"]["official_closing_odds"] == 1
    assert provenance["robustization"]["cluster_unit"] == (
        "race_date_x_venue_x_horizon"
    )

    metrics = closing_odds_multihorizon_v11_metrics(
        [_official_120_race("2026-01-08", 3)],
        model,
        as_of_offset_seconds=60,
    )
    assert metrics["evaluation_tickets"] == 120
    assert metrics["teacher_provenance"]["races_by_source"] == {
        "official_closing_odds": 1
    }
    assert metrics["baseline_current_log_mae"] is not None
    assert metrics["selected_point_log_mae"] is not None
    assert metrics["selection_conformal_required"] is True


def test_t300_as_of_never_reads_later_checkpoints() -> None:
    model = _fit_ready()
    race = _synthetic_race("2026-01-08", 1, include_final=False)
    baseline = forecast_closing_odds_multihorizon_v11(
        race, model, as_of_offset_seconds=300
    )
    changed = copy.deepcopy(race)
    for label in ("t120", "t60", "t30", "t10"):
        changed["closing_odds_checkpoints"][label]["odds"] = {
            key: value * 1000.0
            for key, value in changed["closing_odds_checkpoints"][label]["odds"].items()
        }
    after = forecast_closing_odds_multihorizon_v11(
        changed, model, as_of_offset_seconds=300
    )
    assert baseline == after
    assert baseline["predictions"]["t300"]["used_checkpoint_offsets"] == [300]
    assert baseline["after_as_of_checkpoints"] == ["t120", "t60", "t30", "t10"]
    assert all(
        baseline["predictions"][label]["reason"] == "after_as_of_checkpoint"
        for label in ("t120", "t60", "t30", "t10")
    )
    assert baseline["checkpoint_access_audit"]["future_checkpoint_offsets_used"] == []


def test_checkpoint_age_uses_latest_pre_target_and_rejects_future() -> None:
    race = _synthetic_race("2026-01-08", 1, include_final=False)
    base = race["closing_odds_checkpoints"]["t300"]
    latest = copy.deepcopy(base)
    latest["captured_age_seconds"] = 305.0
    latest["odds"] = {key: value * 2.0 for key, value in base["odds"].items()}
    older = copy.deepcopy(base)
    older["captured_age_seconds"] = 350.0
    future = copy.deepcopy(base)
    future["captured_age_seconds"] = 299.0
    future["odds"] = {key: value * 3.0 for key, value in base["odds"].items()}
    race.pop("closing_odds_checkpoints")
    race["checkpoints"] = [older, future, latest]

    normalized = normalize_labeled_checkpoints(
        race, as_of_offset_seconds=300
    )
    assert normalized["t300"]["checkpoint_age_before_target_seconds"] == 5.0
    assert normalized["t300"]["odds"] == latest["odds"]
    vector, trace = build_checkpoint_feature_vector(
        race,
        checkpoint="t300",
        combination=COMBINATIONS[0],
        as_of_offset_seconds=300,
    )
    age_index = FEATURE_NAMES.index(
        "log1p_checkpoint_age_before_target_seconds"
    )
    assert vector[age_index] == pytest.approx(np.log1p(5.0))
    assert trace["checkpoint_age_before_target_seconds"] == 5.0

    future_only = copy.deepcopy(race)
    future_only["checkpoints"] = [future]
    assert normalize_labeled_checkpoints(
        future_only, as_of_offset_seconds=300
    ) == {}


def test_strict_prior_worse_horizon_falls_back_to_current_odds() -> None:
    races = _training_races()
    for race in races:
        sign = 0.05 if date.fromisoformat(str(race["race_date"])).day % 2 else -0.05
        race["closing_odds_checkpoints"]["t300"]["odds"] = {
            key: value / np.exp(sign)
            for key, value in race["closing_odds"].items()
        }
    model = _fit_ready(races)
    t300 = model["point_models"]["t300"]
    assert t300["selected_mode"] == "current_odds_baseline"
    assert t300["selection_reason"] == (
        "strict_prior_mae_not_better_than_current_baseline"
    )
    assert t300["strict_prior_model_mae"] > (
        t300["strict_prior_baseline_current_mae"]
    )
    assert t300["strict_prior_selected_mae"] == (
        t300["strict_prior_baseline_current_mae"]
    )

    race = _synthetic_race("2026-01-08", 1, include_final=False)
    forecast = forecast_closing_odds_multihorizon_v11(
        race, model, as_of_offset_seconds=300
    )
    row = forecast["predictions"]["t300"]
    current = race["closing_odds_checkpoints"]["t300"]["odds"]
    assert row["point_source"] == "current_odds_baseline"
    assert row["point_final_odds"] == current


def test_batched_examples_match_single_feature_contract_and_order() -> None:
    race = _official_120_race("2026-01-07", 3)

    rows, audit = _examples_from_race(race, "2026-01-07")

    assert all(audit[f"incomplete_{label}"] == 0 for label in CHECKPOINT_LABELS)
    expected_order = [
        (horizon, combination)
        for horizon in CHECKPOINT_OFFSETS_SECONDS
        for combination in sorted(ALL_COMBINATIONS)
    ]
    assert [
        (int(row["horizon"]), str(row["combination"])) for row in rows
    ] == expected_order

    for row in rows:
        expected_vector, expected_trace = build_checkpoint_feature_vector(
            race,
            checkpoint=row["label"],
            combination=row["combination"],
        )
        np.testing.assert_allclose(
            row["features"],
            expected_vector,
            rtol=0.0,
            atol=1e-12,
        )
        assert row["trace"] == expected_trace
        current = race["closing_odds_checkpoints"][row["label"]]["odds"][
            row["combination"]
        ]
        final = race["official_closing_odds"][row["combination"]]
        assert row["raw_target_log_ratio"] == pytest.approx(
            np.log(final / current),
            rel=0.0,
            abs=1e-12,
        )

    missing_trend_race = copy.deepcopy(race)
    missing_trend_combination = ALL_COMBINATIONS[17]
    del missing_trend_race["closing_odds_checkpoints"]["t300"]["odds"][
        missing_trend_combination
    ]
    missing_rows, missing_audit = _examples_from_race(
        missing_trend_race,
        "2026-01-07",
    )
    assert missing_audit["incomplete_t300"] == 1
    assert missing_audit["missing_t300"] == 0
    missing_trend = next(
        row
        for row in missing_rows
        if row["label"] == "t120"
        and row["combination"] == missing_trend_combination
    )
    assert missing_trend["trace"]["used_checkpoint_offsets"] == [120]
    assert missing_trend["trace"]["future_checkpoint_offsets_used"] == []


def test_log_difference_stays_finite_when_odds_ratio_would_overflow() -> None:
    race = _synthetic_race("2026-01-07", 1)
    combination = COMBINATIONS[0]
    race["closing_odds"][combination] = 1e308
    race["closing_odds_checkpoints"]["t300"]["odds"][combination] = 1e-308

    rows, _audit = _examples_from_race(race, "2026-01-07")
    target = next(
        row["raw_target_log_ratio"]
        for row in rows
        if row["label"] == "t300" and row["combination"] == combination
    )

    assert np.isfinite(target)
    assert target == pytest.approx(np.log(1e308) - np.log(1e-308))


def test_point_fit_rejects_nonfinite_inputs_and_records_condition() -> None:
    rows, _audit = _examples_from_race(
        _official_120_race("2026-01-07", 3),
        "2026-01-07",
    )
    model = _fit_point_model(rows, 0.05, architecture="base")
    assert model is not None
    diagnostics = model["numerical_diagnostics"]
    assert diagnostics["gram_symmetrized"] is True
    assert diagnostics["gram_pre_symmetry_max_abs_error"] >= 0.0
    assert diagnostics["regularized_system_condition_number_finite"] is True
    assert diagnostics["regularized_system_condition_number"] > 0.0

    invalid_target = [dict(rows[0], target_log_ratio=float("inf"))]
    with pytest.raises(ValueError, match="targets must be finite"):
        _fit_point_model(invalid_target, 0.05, architecture="base")

    invalid_features = [
        dict(rows[0], features=np.full(len(FEATURE_NAMES), np.nan))
    ]
    with pytest.raises(ValueError, match="features must be finite"):
        _fit_point_model(invalid_features, 0.05, architecture="base")

    with pytest.raises(ValueError, match="finite observations"):
        _finite_sample_lower_rank(
            [0.0, float("nan")],
            target_coverage=0.8,
        )
