from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np

from .closing_odds import MAX_ODDS, MIN_ODDS
from .closing_odds_multihorizon_v11 import (
    EXPECTED_COMBINATIONS,
    FEATURE_NAMES,
    _examples_from_race,
    _finite_sample_lower_rank,
    _iso_date,
    _race_identity,
    _snapshot_odds,
    _teacher_selection,
    _winsorize_teachers_by_day_venue,
    build_checkpoint_feature_vector,
    normalize_labeled_checkpoints,
)

MODEL_NAME = "closing_odds_t300_nonlinear_v12"
MODEL_VERSION = 12
CHECKPOINT_OFFSET_SECONDS = 300
CHECKPOINT_LABEL = "t300"
DEFAULT_LOWER_QUANTILE = 0.20
DEFAULT_MINIMUM_RELATIVE_MAE_IMPROVEMENT = 0.01
FORBIDDEN_FEATURE_TOKENS = (
    "result",
    "finish",
    "rank_actual",
    "winner",
    "payout",
    "payoff",
    "return",
    "refund",
    "settlement",
)


def _feature_indices() -> np.ndarray:
    return np.asarray(
        [index for index, name in enumerate(FEATURE_NAMES) if name != "intercept"],
        dtype=np.int64,
    )


def _assert_feature_contract(feature_names: Sequence[str]) -> None:
    offending = sorted(
        name
        for name in feature_names
        if any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    )
    if offending:
        raise ValueError(f"result/payout features are forbidden: {offending}")


def _preferred_engine() -> str:
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        return "sklearn_hist_gradient_boosting"
    return "lightgbm"


def _new_estimator(engine: str, random_state: int) -> object:
    if engine == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError:
            engine = "sklearn_hist_gradient_boosting"
        else:
            return LGBMRegressor(
                objective="regression_l1",
                n_estimators=160,
                learning_rate=0.035,
                num_leaves=15,
                max_depth=5,
                min_child_samples=80,
                subsample=0.85,
                colsample_bytree=0.80,
                reg_alpha=0.05,
                reg_lambda=1.0,
                random_state=random_state,
                n_jobs=1,
                verbosity=-1,
            )
    if engine == "sklearn_hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.05,
            max_iter=80,
            max_leaf_nodes=15,
            max_depth=5,
            min_samples_leaf=30,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=random_state,
        )
    raise ValueError(f"unsupported nonlinear engine: {engine}")


def _fit_estimator(
    examples: Sequence[Mapping[str, object]], engine: str, random_state: int
) -> dict[str, object] | None:
    if not examples:
        return None
    indices = _feature_indices()
    names = [FEATURE_NAMES[index] for index in indices]
    _assert_feature_contract(names)
    matrix = np.stack(
        [np.asarray(row["features"], dtype=np.float64)[indices] for row in examples]
    )
    target = np.asarray(
        [float(row["target_log_ratio"]) for row in examples], dtype=np.float64
    )
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(target)):
        raise ValueError("non-finite T300 training data")
    estimator = _new_estimator(engine, random_state)
    estimator.fit(matrix, target)
    predicted = np.asarray(estimator.predict(matrix), dtype=np.float64)
    return {
        "model_type": "nonlinear_t300_log_closing_to_current_ratio",
        "engine": (
            "lightgbm"
            if estimator.__class__.__module__.startswith("lightgbm")
            else "sklearn_hist_gradient_boosting"
        ),
        "estimator": estimator,
        "feature_indices": indices.tolist(),
        "feature_names": names,
        "training_examples": len(examples),
        "training_log_ratio_mae": float(np.mean(np.abs(target - predicted))),
        "random_state": int(random_state),
    }


def _predict_log_ratio(vector: np.ndarray, fitted: Mapping[str, object]) -> float:
    names = [str(value) for value in fitted.get("feature_names", ())]
    _assert_feature_contract(names)
    selected = np.asarray(vector, dtype=np.float64)[
        np.asarray(fitted["feature_indices"], dtype=np.int64)
    ]
    if selected.shape != (len(names),) or not np.all(np.isfinite(selected)):
        raise ValueError("v12 nonlinear feature contract mismatch")
    estimator = fitted.get("estimator")
    if estimator is None or not hasattr(estimator, "predict"):
        raise ValueError("v12 nonlinear estimator is missing")
    value = float(np.asarray(estimator.predict(selected.reshape(1, -1)))[0])
    if not math.isfinite(value):
        raise ValueError("v12 nonlinear estimator returned a non-finite value")
    return float(np.clip(value, -8.0, 8.0))


def _predict_log_ratios(
    vectors: Sequence[np.ndarray], fitted: Mapping[str, object]
) -> np.ndarray:
    names = [str(value) for value in fitted.get("feature_names", ())]
    _assert_feature_contract(names)
    indices = np.asarray(fitted["feature_indices"], dtype=np.int64)
    matrix = np.stack(
        [np.asarray(vector, dtype=np.float64)[indices] for vector in vectors]
    )
    if matrix.shape[1:] != (len(names),) or not np.all(np.isfinite(matrix)):
        raise ValueError("v12 nonlinear feature contract mismatch")
    estimator = fitted.get("estimator")
    if estimator is None or not hasattr(estimator, "predict"):
        raise ValueError("v12 nonlinear estimator is missing")
    return np.clip(np.asarray(estimator.predict(matrix), dtype=np.float64), -8.0, 8.0)


def _collect_examples(
    races: Sequence[Mapping[str, object]], target_date: str
) -> tuple[list[dict[str, object]], dict[str, object]]:
    examples: list[dict[str, object]] = []
    sources_races: dict[str, int] = defaultdict(int)
    sources_tickets: dict[str, int] = defaultdict(int)
    training_sources: dict[str, set[str]] = defaultdict(set)
    excluded_non_past = excluded_invalid_date = 0
    official_incomplete = missing_teacher = missing_t300 = incomplete_t300 = 0
    prior_races = 0
    for race in races:
        try:
            race_date = _iso_date(race.get("race_date"), "race_date")
        except ValueError:
            excluded_invalid_date += 1
            continue
        if race_date >= target_date:
            excluded_non_past += 1
            continue
        prior_races += 1
        rows, audit = _examples_from_race(race, race_date)
        source = audit["teacher_source"]
        official_incomplete += int(bool(audit["official_closing_odds_incomplete"]))
        missing_t300 += int(audit["missing_t300"])
        incomplete_t300 += int(audit["incomplete_t300"])
        if source is None:
            missing_teacher += 1
        else:
            sources_races[str(source)] += 1
            sources_tickets[str(source)] += int(audit["teacher_tickets"])
        selected = [row for row in rows if row["label"] == CHECKPOINT_LABEL]
        if selected:
            identity = f"{race_date}|{_race_identity(race, race_date)}"
            training_sources[str(source)].add(identity)
            examples.extend(selected)
    examples.sort(
        key=lambda row: (
            str(row["race_date"]),
            str(row["race_id"]),
            str(row["combination"]),
        )
    )
    examples, robustization = _winsorize_teachers_by_day_venue(examples)
    provenance = {
        "selection_policy": (
            "official_closing_odds_when_120_valid_else_closing_odds_fallback"
        ),
        "official_required_points": EXPECTED_COMBINATIONS,
        "selected_races_by_source": dict(sorted(sources_races.items())),
        "selected_tickets_by_source": dict(sorted(sources_tickets.items())),
        "training_races_by_source": {
            key: len(value) for key, value in sorted(training_sources.items())
        },
        "training_examples_by_source": {
            key: sum(row["teacher_source"] == key for row in examples)
            for key in ("official_closing_odds", "closing_odds_fallback")
        },
        "official_closing_odds_incomplete_races": official_incomplete,
        "missing_teacher_races": missing_teacher,
        "teacher": "winsorized_log(selected_closing_odds/current_t300_odds)",
        "robustization": robustization,
    }
    return examples, {
        "prior_races": prior_races,
        "excluded_non_past_races": excluded_non_past,
        "excluded_invalid_date_races": excluded_invalid_date,
        "missing_t300_races": missing_t300,
        "incomplete_t300_races": incomplete_t300,
        "teacher_provenance": provenance,
    }


def fit_closing_odds_t300_nonlinear_v12(
    races: Iterable[Mapping[str, object]],
    *,
    prediction_date: object,
    minimum_training_days: int = 5,
    minimum_training_races: int = 100,
    minimum_training_examples: int = 500,
    calibration_warmup_days: int = 2,
    minimum_calibration_clusters: int = 4,
    minimum_relative_mae_improvement: float = DEFAULT_MINIMUM_RELATIVE_MAE_IMPROVEMENT,
    lower_quantile: float = DEFAULT_LOWER_QUANTILE,
    random_state: int = 120300,
    engine: str | None = None,
) -> dict[str, object]:
    """Fit a nonlinear T300 challenger using strict-prior whole-day folds."""
    target_date = _iso_date(prediction_date, "prediction_date")
    if not 0.0 <= minimum_relative_mae_improvement < 1.0:
        raise ValueError("minimum_relative_mae_improvement must be in [0, 1)")
    if not 0.0 < lower_quantile < 0.5:
        raise ValueError("lower_quantile must be between zero and 0.5")
    for name, value in (
        ("minimum_training_days", minimum_training_days),
        ("minimum_training_races", minimum_training_races),
        ("minimum_training_examples", minimum_training_examples),
        ("calibration_warmup_days", calibration_warmup_days),
        ("minimum_calibration_clusters", minimum_calibration_clusters),
    ):
        if int(value) < 1:
            raise ValueError(f"{name} must be positive")
    requested_engine = engine or _preferred_engine()
    if requested_engine not in ("lightgbm", "sklearn_hist_gradient_boosting"):
        raise ValueError(f"unsupported nonlinear engine: {requested_engine}")

    source = list(races)
    examples, audit = _collect_examples(source, target_date)
    training_dates = sorted({str(row["race_date"]) for row in examples})
    by_day: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in examples:
        by_day[str(row["race_date"])].append(row)

    baseline_errors: list[float] = []
    challenger_errors: list[float] = []
    baseline_residuals_by_cluster: dict[str, list[float]] = defaultdict(list)
    challenger_residuals_by_cluster: dict[str, list[float]] = defaultdict(list)
    folds: list[dict[str, object]] = []
    actual_engine: str | None = None
    for index in range(calibration_warmup_days, len(training_dates)):
        evaluation_date = training_dates[index]
        prior_dates = training_dates[:index]
        fold_training = [row for day in prior_dates for row in by_day[day]]
        holdout = list(by_day[evaluation_date])
        fitted = _fit_estimator(
            fold_training, requested_engine, random_state + index
        )
        if fitted is None or not holdout:
            continue
        actual_engine = str(fitted["engine"])
        predictions = _predict_log_ratios(
            [np.asarray(row["features"]) for row in holdout],
            fitted,
        )
        raw_targets = np.asarray(
            [float(row["raw_target_log_ratio"]) for row in holdout]
        )
        baseline_errors.extend(np.abs(raw_targets).tolist())
        challenger_errors.extend(np.abs(raw_targets - predictions).tolist())
        for row, residual in zip(holdout, raw_targets - predictions):
            cluster = f"{evaluation_date}|{row['venue_group']}"
            baseline_residuals_by_cluster[cluster].append(
                float(row["raw_target_log_ratio"])
            )
            challenger_residuals_by_cluster[cluster].append(float(residual))
        folds.append(
            {
                "evaluation_date": evaluation_date,
                "trained_through_date": prior_dates[-1],
                "training_dates": list(prior_dates),
                "training_examples": len(fold_training),
                "evaluation_examples": len(holdout),
                "evaluation_group_unit": "race_date_x_venue",
                "evaluation_groups": len(
                    {str(row["venue_group"]) for row in holdout}
                ),
                "strict_prior_day": prior_dates[-1] < evaluation_date,
            }
        )

    baseline_mae = float(np.mean(baseline_errors)) if baseline_errors else None
    challenger_mae = (
        float(np.mean(challenger_errors)) if challenger_errors else None
    )
    improvement = (
        1.0 - challenger_mae / baseline_mae
        if challenger_mae is not None
        and baseline_mae is not None
        and baseline_mae > 0.0
        else None
    )
    training_races = len(
        {(str(row["race_date"]), str(row["race_id"])) for row in examples}
    )
    data_ready = bool(
        len(training_dates) >= minimum_training_days
        and training_races >= minimum_training_races
        and len(examples) >= minimum_training_examples
        and len(challenger_residuals_by_cluster) >= minimum_calibration_clusters
        and folds
    )
    adopted = bool(
        data_ready
        and improvement is not None
        and improvement >= minimum_relative_mae_improvement
    )
    if not data_ready:
        reason = "insufficient_strict_prior_data"
    elif adopted:
        reason = "strict_prior_mae_beats_current_odds_by_gate"
    else:
        reason = "strict_prior_mae_does_not_beat_current_odds_by_gate"
    final_model = (
        _fit_estimator(examples, requested_engine, random_state) if adopted else None
    )
    if final_model is not None:
        actual_engine = str(final_model["engine"])

    selected_mode = "nonlinear_model" if adopted else "current_odds_baseline"
    selected_residuals_by_cluster = (
        challenger_residuals_by_cluster
        if adopted
        else baseline_residuals_by_cluster
    )
    cluster_lower: dict[str, float] = {}
    cluster_counts: dict[str, int] = {}
    for cluster, values in sorted(selected_residuals_by_cluster.items()):
        cluster_counts[cluster] = len(values)
        cluster_lower[cluster] = _finite_sample_lower_rank(
            values, target_coverage=1.0 - lower_quantile
        )[0]
    outer = (
        _finite_sample_lower_rank(
            list(cluster_lower.values()), target_coverage=1.0 - lower_quantile
        )
        if cluster_lower
        else None
    )
    conformal_ready = len(cluster_lower) >= minimum_calibration_clusters
    adjustment = min(0.0, float(outer[0])) if outer is not None else None
    trained_through = training_dates[-1] if training_dates else None
    strict_boundary = trained_through is None or trained_through < target_date
    strict_folds = all(bool(fold["strict_prior_day"]) for fold in folds)
    names = [FEATURE_NAMES[index] for index in _feature_indices()]
    _assert_feature_contract(names)
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "ready": bool(data_ready and conformal_ready and strict_boundary and strict_folds),
        "prediction_date": target_date,
        "trained_through_date": trained_through,
        "checkpoint_label": CHECKPOINT_LABEL,
        "checkpoint_offset_seconds": CHECKPOINT_OFFSET_SECONDS,
        "selected_mode": selected_mode,
        "challenger_adopted": adopted,
        "selection_reason": reason,
        "minimum_relative_mae_improvement": float(minimum_relative_mae_improvement),
        "strict_prior_baseline_current_mae": baseline_mae,
        "strict_prior_challenger_mae": challenger_mae,
        "strict_prior_relative_mae_improvement": improvement,
        "requested_engine": requested_engine,
        "actual_engine": actual_engine or requested_engine,
        "point_model": final_model,
        "feature_names": names,
        "forbidden_feature_tokens": list(FORBIDDEN_FEATURE_TOKENS),
        "teacher_provenance": audit["teacher_provenance"],
        "lower_quantile_model": {
            "ready": conformal_ready,
            "model_type": "day_x_venue_cluster_finite_sample_residual_lower_bound",
            "calibrated_point_source": selected_mode,
            "quantile": float(lower_quantile),
            "cluster_unit": "race_date_x_venue",
            "effective_sample_clusters": len(cluster_lower),
            "cluster_lower_residuals": cluster_lower,
            "cluster_ticket_counts": cluster_counts,
            "residual_log_ratio_adjustment": adjustment,
            "finite_sample_rank": outer[1] if outer is not None else None,
            "finite_sample_coverage": outer[2] if outer is not None else None,
            "monotone_safe_side": True,
        },
        "training_summary": {
            "training_dates": training_dates,
            "training_days": len(training_dates),
            "training_races": training_races,
            "training_examples": len(examples),
            "missing_t300_races": audit["missing_t300_races"],
            "incomplete_t300_races": audit["incomplete_t300_races"],
        },
        "boundary_audit": {
            "input_races": len(source),
            "eligible_prior_races": audit["prior_races"],
            "excluded_non_past_races": audit["excluded_non_past_races"],
            "excluded_invalid_date_races": audit["excluded_invalid_date_races"],
            "prediction_date": target_date,
            "trained_through_date": trained_through,
            "strict_training_boundary": strict_boundary,
            "outer_day_folds": folds,
            "strict_outer_day_boundaries": strict_folds,
            "group_day_unit": "race_date_x_venue",
            "future_checkpoint_imputation": False,
            "result_or_payout_features": [],
        },
    }


def _empty_forecast(
    model: Mapping[str, object], race_date: str, reason: str
) -> dict[str, object]:
    return {
        "model_name": MODEL_NAME,
        "ready": False,
        "reason": reason,
        "prediction_date": race_date,
        "checkpoint_label": CHECKPOINT_LABEL,
        "point_source": None,
        "point_final_odds": {},
        "lower_final_odds": {},
        "used_checkpoint_offsets": [],
        "future_checkpoint_offsets_used": [],
        "teacher_provenance": model.get("teacher_provenance"),
        "future_checkpoint_imputation": False,
    }


def forecast_closing_odds_t300_nonlinear_v12(
    race: Mapping[str, object],
    model: Mapping[str, object],
    *,
    prediction_date: object | None = None,
) -> dict[str, object]:
    """Forecast closing odds from exactly T300; never impute a checkpoint."""
    if str(model.get("model_name")) != MODEL_NAME:
        raise ValueError("not a closing odds T300 nonlinear v12 artifact")
    race_date = _iso_date(
        prediction_date if prediction_date is not None else race.get("race_date"),
        "prediction_date",
    )
    artifact_date = _iso_date(model.get("prediction_date"), "artifact prediction_date")
    trained_through = model.get("trained_through_date")
    if race_date < artifact_date:
        raise ValueError("prediction_date precedes artifact boundary")
    if trained_through is not None and _iso_date(
        trained_through, "trained_through_date"
    ) >= race_date:
        raise ValueError("artifact is not strictly prior to prediction_date")
    checkpoints = normalize_labeled_checkpoints(
        race, as_of_offset_seconds=CHECKPOINT_OFFSET_SECONDS
    )
    snapshot = checkpoints.get(CHECKPOINT_LABEL)
    if snapshot is None:
        return _empty_forecast(model, race_date, "missing_t300_checkpoint")
    if not bool(model.get("ready")):
        return _empty_forecast(model, race_date, "model_not_ready")
    selected_mode = str(model.get("selected_mode"))
    fitted = model.get("point_model")
    if selected_mode == "nonlinear_model" and not isinstance(fitted, Mapping):
        raise ValueError("adopted v12 nonlinear model is missing")
    if selected_mode not in ("nonlinear_model", "current_odds_baseline"):
        raise ValueError("invalid v12 selected mode")
    calibration = model.get("lower_quantile_model")
    if not isinstance(calibration, Mapping) or not calibration.get("ready"):
        raise ValueError("v12 lower-bound calibration is not ready")
    adjustment = min(0.0, float(calibration["residual_log_ratio_adjustment"]))
    current = _snapshot_odds(snapshot)
    if not current:
        return _empty_forecast(model, race_date, "missing_t300_checkpoint")
    points: dict[str, float] = {}
    lowers: dict[str, float] = {}
    used: set[int] = set()
    future: set[int] = set()
    for combination in sorted(current):
        vector, trace = build_checkpoint_feature_vector(
            race,
            checkpoint=CHECKPOINT_LABEL,
            combination=combination,
            as_of_offset_seconds=CHECKPOINT_OFFSET_SECONDS,
        )
        used.update(int(value) for value in trace["used_checkpoint_offsets"])
        future.update(int(value) for value in trace["future_checkpoint_offsets_used"])
        log_ratio = (
            _predict_log_ratio(vector, fitted)
            if selected_mode == "nonlinear_model"
            else 0.0
        )
        point = min(
            MAX_ODDS,
            max(MIN_ODDS, current[combination] * math.exp(log_ratio)),
        )
        points[combination] = point
        lowers[combination] = min(
            point,
            min(MAX_ODDS, max(MIN_ODDS, point * math.exp(adjustment))),
        )
    if future:
        raise ValueError("future checkpoint observation reached v12 forecast")
    return {
        "model_name": MODEL_NAME,
        "ready": True,
        "reason": None,
        "prediction_date": race_date,
        "artifact_prediction_date": artifact_date,
        "trained_through_date": trained_through,
        "checkpoint_label": CHECKPOINT_LABEL,
        "checkpoint_offset_seconds": CHECKPOINT_OFFSET_SECONDS,
        "point_source": selected_mode,
        "point_final_odds": points,
        "lower_final_odds": lowers,
        "lower_residual_log_ratio_adjustment": adjustment,
        "used_checkpoint_offsets": sorted(used, reverse=True),
        "future_checkpoint_offsets_used": [],
        "teacher_provenance": model.get("teacher_provenance"),
        "future_checkpoint_imputation": False,
        "boundary_audit_passed": bool(
            trained_through is None
            or _iso_date(trained_through, "trained_through_date") < race_date
        ),
    }


def closing_odds_t300_nonlinear_v12_metrics(
    races: Iterable[Mapping[str, object]], model: Mapping[str, object]
) -> dict[str, object]:
    """Evaluate final-odds MAE and lower-bound coverage for T300."""
    baseline_errors: list[float] = []
    selected_errors: list[float] = []
    covered: list[bool] = []
    missing_prediction = missing_teacher = evaluated_races = 0
    for race in races:
        teacher, source, _incomplete = _teacher_selection(race)
        if source is None:
            missing_teacher += 1
            continue
        forecast = forecast_closing_odds_t300_nonlinear_v12(race, model)
        if not forecast["ready"]:
            missing_prediction += 1
            continue
        snapshot = normalize_labeled_checkpoints(
            race, as_of_offset_seconds=CHECKPOINT_OFFSET_SECONDS
        ).get(CHECKPOINT_LABEL)
        current = _snapshot_odds(snapshot) if isinstance(snapshot, Mapping) else {}
        point = forecast["point_final_odds"]
        lower = forecast["lower_final_odds"]
        if set(current) != set(teacher) or set(point) != set(teacher):
            missing_prediction += 1
            continue
        for combination in sorted(teacher):
            target_log = math.log(teacher[combination])
            baseline_errors.append(abs(target_log - math.log(current[combination])))
            selected_errors.append(abs(target_log - math.log(point[combination])))
            covered.append(teacher[combination] >= lower[combination])
        evaluated_races += 1
    baseline_mae = float(np.mean(baseline_errors)) if baseline_errors else None
    selected_mae = float(np.mean(selected_errors)) if selected_errors else None
    return {
        "model_name": MODEL_NAME,
        "checkpoint_label": CHECKPOINT_LABEL,
        "evaluation_races": evaluated_races,
        "evaluation_tickets": len(selected_errors),
        "baseline_current_log_mae": baseline_mae,
        "selected_point_log_mae": selected_mae,
        "selected_relative_mae_improvement": (
            1.0 - selected_mae / baseline_mae
            if selected_mae is not None and baseline_mae not in (None, 0.0)
            else None
        ),
        "lower_bound_coverage": float(np.mean(covered)) if covered else None,
        "point_source": model.get("selected_mode"),
        "missing_prediction_races": missing_prediction,
        "missing_teacher_races": missing_teacher,
        "teacher_provenance": model.get("teacher_provenance"),
    }
