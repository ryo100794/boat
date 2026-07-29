from __future__ import annotations

import math
from collections import defaultdict
from datetime import date, timedelta
from typing import Any, Callable, Iterable, Mapping

from .closing_odds_multihorizon_v11 import (
    DEFAULT_LOWER_QUANTILE as V11_DEFAULT_LOWER_QUANTILE,
    checkpoint_label,
    closing_odds_multihorizon_v11_metrics,
    fit_closing_odds_multihorizon_v11,
    forecast_closing_odds_multihorizon_v11,
    select_teacher_final_odds,
)
from .closing_odds_t300_nonlinear_v12 import (
    DEFAULT_LOWER_QUANTILE as V12_DEFAULT_LOWER_QUANTILE,
    MODEL_NAME as V12_CLOSING_MODEL_NAME,
    closing_odds_t300_nonlinear_v12_metrics,
    fit_closing_odds_t300_nonlinear_v12,
    forecast_closing_odds_t300_nonlinear_v12,
)
from .odds_path_conservative_v7 import (
    CLOSING_QUANTILE,
    _crossfit_probability_rows,
    _cumulative_daily,
    _prospective_summary,
    _summarize_bankroll,
    _weighted_probability_metrics,
    fit_probability_lcb,
    probability_metrics,
)
from .odds_path_probability_v8 import (
    attach_odds_path_probability_v8,
    fit_odds_path_probability_v8,
)
from .odds_path_selection_conformal_v10 import (
    DISCRETE_POLICY,
    _aggregate_selection_conformal,
    _selection_coverage_gate,
    _simulate_selection_conformal_policy,
)
from .selection_conformal import (
    fit_selection_conformal_haircut,
    selected_safe_ev_candidates,
)


MODEL_NAME = "odds_path_role_integrated_t300_nonlinear_v12"
STRATEGY_NAME = MODEL_NAME
REGISTERED_AFTER = "2026-07-29"
PROSPECTIVE_OUTPUT_KEY = "prospective_role_integrated_v12_walk_forward"
DECISION_OFFSET_SECONDS = 300
DECISION_CHECKPOINT = checkpoint_label(DECISION_OFFSET_SECONDS)
RESEARCH_PRECONFORMAL_OUTPUT_KEY = "research_preconformal_upper_bound"
CLOSING_FALLBACK_V11 = "v11"
CLOSING_FALLBACK_NO_BET = "no_bet"
CLOSING_FALLBACK_POLICIES = (CLOSING_FALLBACK_V11, CLOSING_FALLBACK_NO_BET)

RESEARCH_PRECONFORMAL_ARTIFACT: dict[str, Any] = {
    "ready": True,
    "haircut": 1.0,
    "method": "fixed_identity_pre_selection_conformal_upper_bound",
    "research_only_non_deployable": True,
}

DISCRETE_POLICY_V12: dict[str, Any] = {
    **DISCRETE_POLICY,
    "name": "v12_t300_nonlinear_selection_conformal_discrete_log",
    "closing_model": V12_CLOSING_MODEL_NAME,
    "decision_checkpoint": DECISION_CHECKPOINT,
    "decision_offset_seconds": DECISION_OFFSET_SECONDS,
    "future_checkpoint_imputation": False,
}


def _closing_v12_report_model(model: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a V12 artifact for JSON reporting without its live estimator."""
    report_model = dict(model)
    point_model = model.get("point_model")
    if isinstance(point_model, Mapping):
        report_model["point_model"] = {
            key: value for key, value in point_model.items() if key != "estimator"
        }
    return report_model


def _closing_contract(
    v12_model: Mapping[str, object],
    v11_model: Mapping[str, object] | None,
    *,
    fallback_policy: str,
) -> dict[str, Any]:
    if fallback_policy not in CLOSING_FALLBACK_POLICIES:
        raise ValueError(f"unsupported v12 closing fallback policy: {fallback_policy}")
    v12_ready = bool(v12_model.get("ready"))
    v12_adopted = bool(v12_model.get("challenger_adopted"))
    if v12_ready and v12_adopted:
        selected = V12_CLOSING_MODEL_NAME
        reason = "v12_ready_and_adopted"
    elif (
        fallback_policy == CLOSING_FALLBACK_V11
        and isinstance(v11_model, Mapping)
        and bool(v11_model.get("ready"))
    ):
        selected = "closing_odds_multihorizon_v11"
        reason = "v12_not_ready_or_not_adopted_v11_fallback"
    else:
        selected = "no_bet"
        reason = (
            "v12_not_ready_or_not_adopted_no_bet_contract"
            if fallback_policy == CLOSING_FALLBACK_NO_BET
            else "v12_not_ready_or_not_adopted_v11_fallback_not_ready"
        )
    return {
        "requested_model": V12_CLOSING_MODEL_NAME,
        "selected_model": selected,
        "fallback_policy": fallback_policy,
        "selection_reason": reason,
        "v12_ready": v12_ready,
        "v12_adopted": v12_adopted,
        "v12_selection_reason": v12_model.get("selection_reason"),
        "v11_fallback_ready": bool(
            isinstance(v11_model, Mapping) and v11_model.get("ready")
        ),
        "ready_for_purchase": selected != "no_bet",
        "decision_checkpoint": DECISION_CHECKPOINT,
        "decision_offset_seconds": DECISION_OFFSET_SECONDS,
        "future_checkpoint_imputation": False,
    }


def _strict_prior_forecasts(
    races: Iterable[dict[str, Any]],
    v12_model: Mapping[str, object],
    v11_model: Mapping[str, object] | None,
    *,
    evaluation_date: str,
    fallback_policy: str,
    forecast_field: str = "lower_final_odds",
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    if forecast_field not in {"point_final_odds", "lower_final_odds"}:
        raise ValueError("unsupported closing forecast field")
    forecasts: dict[str, dict[str, float]] = {}
    contract = _closing_contract(
        v12_model,
        v11_model,
        fallback_policy=fallback_policy,
    )
    audit: dict[str, Any] = {
        "races": 0,
        "ready_races": 0,
        "missing_t300_races": 0,
        "incomplete_t300_races": 0,
        "future_checkpoint_violations": 0,
        "forecast_field": forecast_field,
        "closing_model_identity": contract,
    }
    selected_model = contract["selected_model"]
    for race in races:
        audit["races"] += 1
        if selected_model == "no_bet":
            audit["missing_t300_races"] += 1
            continue
        if selected_model == V12_CLOSING_MODEL_NAME:
            forecast = forecast_closing_odds_t300_nonlinear_v12(
                race,
                v12_model,
                prediction_date=evaluation_date,
            )
            ready = bool(forecast.get("ready"))
            forecast_values = forecast.get(forecast_field) or {}
            future = list(forecast.get("future_checkpoint_offsets_used") or [])
        else:
            assert isinstance(v11_model, Mapping)
            forecast = forecast_closing_odds_multihorizon_v11(
                race,
                v11_model,
                as_of_offset_seconds=DECISION_OFFSET_SECONDS,
                prediction_date=evaluation_date,
            )
            access = forecast.get("checkpoint_access_audit") or {}
            row = (forecast.get("predictions") or {}).get(DECISION_CHECKPOINT) or {}
            ready = bool(row.get("ready"))
            forecast_values = row.get(forecast_field) or {}
            future = list(access.get("future_checkpoint_offsets_used") or [])
            future.extend(row.get("future_checkpoint_offsets_used") or [])
        if future:
            audit["future_checkpoint_violations"] += 1
            raise ValueError("v12 role stack T300 forecast used a future checkpoint")
        if not ready:
            audit["missing_t300_races"] += 1
            continue
        selected_values = {
            str(combination): float(odds)
            for combination, odds in forecast_values.items()
            if math.isfinite(float(odds)) and float(odds) > 0.0
        }
        if len(selected_values) != 120:
            audit["incomplete_t300_races"] += 1
            continue
        forecasts[str(race["race_id"])] = selected_values
        audit["ready_races"] += 1
    return forecasts, audit


def _append_selection_observations(
    observations: list[dict[str, Any]],
    races: list[dict[str, Any]],
    *,
    closing_forecasts: dict[str, dict[str, float]],
    probability_lcb: dict[str, Any],
    evaluation_date: str,
) -> int:
    """Observe final odds only after the day's purchase selection is frozen."""
    selected = selected_safe_ev_candidates(
        races,
        closing_forecasts=closing_forecasts,
        probability_lcb=probability_lcb,
    )
    race_by_id = {str(race["race_id"]): race for race in races}
    appended = 0
    for candidate in selected:
        race = race_by_id[str(candidate["race_id"])]
        final_odds, _source = select_teacher_final_odds(race)
        actual = final_odds.get(str(candidate["combination"]))
        predicted = float(candidate["predicted_closing"])
        if actual is None or predicted <= 0.0:
            continue
        ratio = float(actual) / predicted
        if not math.isfinite(ratio) or ratio <= 0.0:
            continue
        observations.append({
            "race_date": evaluation_date,
            "race_id": str(candidate["race_id"]),
            "combination": str(candidate["combination"]),
            "closing_ratio": ratio,
        })
        appended += 1
    return appended


def _closing_metric_adapter(metrics: Mapping[str, object]) -> dict[str, Any]:
    return {
        "closing_q20_evaluation_races": int(metrics["evaluation_races"]),
        "closing_q20_evaluation_tickets": int(metrics["evaluation_tickets"]),
        "closing_q20_pinball_loss": None,
        "closing_q20_lower_coverage": metrics.get("lower_bound_coverage"),
        "closing_q20_target_coverage": 1.0 - CLOSING_QUANTILE,
    }


def _closing_metrics(
    races: list[dict[str, Any]],
    *,
    identity: Mapping[str, object],
    v12_model: Mapping[str, object],
    v11_model: Mapping[str, object] | None,
) -> dict[str, Any]:
    selected = str(identity["selected_model"])
    if selected == V12_CLOSING_MODEL_NAME:
        return dict(closing_odds_t300_nonlinear_v12_metrics(races, v12_model))
    if selected == "closing_odds_multihorizon_v11":
        assert isinstance(v11_model, Mapping)
        return dict(
            closing_odds_multihorizon_v11_metrics(
                races,
                v11_model,
                as_of_offset_seconds=DECISION_OFFSET_SECONDS,
            )
        )
    return {
        "model_name": "no_bet",
        "checkpoint_label": DECISION_CHECKPOINT,
        "evaluation_races": 0,
        "evaluation_tickets": 0,
        "baseline_current_log_mae": None,
        "selected_point_log_mae": None,
        "selected_relative_mae_improvement": None,
        "lower_bound_coverage": None,
        "point_source": None,
        "missing_prediction_races": len(races),
        "missing_teacher_races": 0,
    }


def _aggregate_closing_identity(folds: list[dict[str, Any]]) -> dict[str, Any]:
    identities = [
        fold.get("closing_model_identity") or {}
        for fold in folds
        if isinstance(fold.get("closing_model_identity"), Mapping)
    ]
    selected_counts: dict[str, int] = defaultdict(int)
    for identity in identities:
        selected_counts[str(identity.get("selected_model") or "unknown")] += 1
    return {
        "requested_model": V12_CLOSING_MODEL_NAME,
        "fallback_policy": (
            identities[-1].get("fallback_policy") if identities else None
        ),
        "selected_model_latest": (
            identities[-1].get("selected_model") if identities else None
        ),
        "selected_model_fold_counts": dict(sorted(selected_counts.items())),
        "evaluation_folds": len(identities),
        "v12_ready_folds": sum(bool(row.get("v12_ready")) for row in identities),
        "v12_adopted_folds": sum(bool(row.get("v12_adopted")) for row in identities),
        "v11_fallback_folds": selected_counts.get(
            "closing_odds_multihorizon_v11", 0
        ),
        "no_bet_folds": selected_counts.get("no_bet", 0),
        "decision_checkpoint": DECISION_CHECKPOINT,
        "decision_offset_seconds": DECISION_OFFSET_SECONDS,
        "future_checkpoint_imputation": False,
    }


def _next_date(dates: list[str]) -> str:
    if not dates:
        return date.today().isoformat()
    return (date.fromisoformat(dates[-1]) + timedelta(days=1)).isoformat()


def _research_preconformal_summary(
    daily: list[dict[str, Any]],
    *,
    evaluated_races: int,
    eligible_dates: list[str],
    skipped_dates: list[str],
    diagnostics_by_date: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    cumulative_daily = _cumulative_daily(daily)
    bankroll = _summarize_bankroll(
        cumulative_daily,
        evaluated_races=evaluated_races,
        purchase_diagnostic_accumulators=diagnostics_by_date.values(),
    )
    largest_hit = max(
        (
            int(row.get("largest_hit_return_yen") or 0)
            for row in cumulative_daily
        ),
        default=0,
    )
    return {
        "status": "research_only_non_deployable",
        "research_only_non_deployable": True,
        "deployable": False,
        "included_in_promotion_gate": False,
        "included_in_deployment_selected_policy": False,
        "included_in_operational_decision": False,
        "interpretation": (
            "Optimistic bankroll diagnostic before selection-conformal haircut; "
            "it is not an operational or promotion estimate"
        ),
        "fixed_selection_conformal": dict(RESEARCH_PRECONFORMAL_ARTIFACT),
        "eligible_dates": list(eligible_dates),
        "skipped_dates": list(skipped_dates),
        "largest_hit_return_yen": largest_hit,
        **bankroll,
    }


def walk_forward_evaluate_v12(
    races: list[dict[str, Any]],
    *,
    daily_budget_yen: int,
    min_calibration_days: int,
    evaluation_dates: Iterable[str] | None = None,
    closing_fallback_policy: str = CLOSING_FALLBACK_V11,
    closing_forecast_field: str = "lower_final_odds",
    probability_lcb_fit: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None,
    probability_lcb_metrics: Callable[..., dict[str, Any]] | None = None,
    probability_lcb_metrics_use_preallocation_population: bool = False,
    selection_conformal_fit: Callable[..., dict[str, Any]] | None = None,
    selection_observation_append: Callable[..., int] | None = None,
) -> dict[str, Any]:
    """Evaluate V12 closing decisions in the strict-prior role-separated stack."""
    lcb_fit = probability_lcb_fit or fit_probability_lcb
    conformal_fit = selection_conformal_fit or fit_selection_conformal_haircut
    observation_append = (
        selection_observation_append or _append_selection_observations
    )
    if closing_forecast_field not in {"point_final_odds", "lower_final_odds"}:
        raise ValueError("unsupported closing forecast field")
    if closing_fallback_policy not in CLOSING_FALLBACK_POLICIES:
        raise ValueError(
            f"unsupported v12 closing fallback policy: {closing_fallback_policy}"
        )
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for race in races:
        by_day[str(race["race_date"])].append(race)
    dates = sorted(by_day)
    requested = (
        set(dates)
        if evaluation_dates is None
        else {str(value) for value in evaluation_dates if str(value) in by_day}
    )
    eligible_dates = [
        value
        for index, value in enumerate(dates)
        if index >= min_calibration_days
    ]
    output_dates = {value for value in eligible_dates if value in requested}
    policy = {
        **DISCRETE_POLICY_V12,
        "daily_budget_yen": daily_budget_yen,
        "closing_fallback_policy": closing_fallback_policy,
        "closing_forecast_field": closing_forecast_field,
    }

    observations: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    daily: list[dict[str, Any]] = []
    diagnostics_by_date: dict[str, dict[str, Any]] = {}
    artifacts_by_date: dict[str, dict[str, Any]] = {}
    research_daily: list[dict[str, Any]] = []
    research_diagnostics_by_date: dict[str, dict[str, Any]] = {}
    research_evaluated_races = 0
    research_eligible_dates: list[str] = []
    research_skipped_dates: list[str] = []

    for evaluation_date in eligible_dates:
        prior_dates = [value for value in dates if value < evaluation_date]
        training = [race for value in prior_dates for race in by_day[value]]
        holdout = by_day[evaluation_date]

        probability_model = fit_odds_path_probability_v8(training)
        transformed = attach_odds_path_probability_v8(holdout, probability_model)
        crossfit_rows = _crossfit_probability_rows(
            training,
            probability_fit=fit_odds_path_probability_v8,
            probability_attach=attach_odds_path_probability_v8,
        )
        probability_lcb = lcb_fit(crossfit_rows)
        closing_v12_model = fit_closing_odds_t300_nonlinear_v12(
            training,
            prediction_date=evaluation_date,
            lower_quantile=V12_DEFAULT_LOWER_QUANTILE,
        )
        closing_v11_model = (
            fit_closing_odds_multihorizon_v11(
                training,
                prediction_date=evaluation_date,
                lower_quantile=V11_DEFAULT_LOWER_QUANTILE,
            )
            if closing_fallback_policy == CLOSING_FALLBACK_V11
            else None
        )
        conformal = conformal_fit(
            observations,
            evaluation_date=evaluation_date,
        )
        artifacts_by_date[evaluation_date] = conformal

        closing_forecasts, access_audit = _strict_prior_forecasts(
            transformed,
            closing_v12_model,
            closing_v11_model,
            evaluation_date=evaluation_date,
            fallback_policy=closing_fallback_policy,
            forecast_field=closing_forecast_field,
        )
        closing_identity = dict(access_audit["closing_model_identity"])
        bankroll, purchase_diagnostic = _simulate_selection_conformal_policy(
            transformed,
            closing_forecasts=closing_forecasts,
            probability_lcb=probability_lcb,
            daily_budget_yen=daily_budget_yen,
            selection_conformal=conformal,
            capture_preallocation_candidates=(
                probability_lcb_metrics_use_preallocation_population
            ),
        )
        preallocation_population = purchase_diagnostic.pop(
            "_preallocation_candidates", []
        )
        research_bankroll: dict[str, Any] | None = None
        research_diagnostic: dict[str, Any] | None = None
        research_ready = bool(closing_identity["ready_for_purchase"]) and bool(
            probability_lcb.get("ready")
        )
        if research_ready:
            research_bankroll, research_diagnostic = (
                _simulate_selection_conformal_policy(
                    transformed,
                    closing_forecasts=closing_forecasts,
                    probability_lcb=probability_lcb,
                    daily_budget_yen=daily_budget_yen,
                    selection_conformal=dict(RESEARCH_PRECONFORMAL_ARTIFACT),
                )
            )

        # Purchase/allocation has completed. Final odds can now become a teacher
        # for later days, while payout data remains inside allocator settlement.
        appended = observation_append(
            observations,
            transformed,
            closing_forecasts=closing_forecasts,
            probability_lcb=probability_lcb,
            evaluation_date=evaluation_date,
        )
        if evaluation_date not in output_dates:
            continue

        if probability_lcb_metrics is None:
            lcb_evaluation = None
        elif probability_lcb_metrics_use_preallocation_population:
            lcb_evaluation = probability_lcb_metrics(
                transformed,
                closing_forecasts=closing_forecasts,
                probability_lcb=probability_lcb,
                selected_candidates=preallocation_population,
            )
        else:
            lcb_evaluation = probability_lcb_metrics(
                transformed,
                closing_forecasts=closing_forecasts,
                probability_lcb=probability_lcb,
            )

        if research_bankroll is not None and research_diagnostic is not None:
            research_daily.extend(research_bankroll["daily"])
            research_diagnostics_by_date[evaluation_date] = research_diagnostic
            research_evaluated_races += len(holdout)
            research_eligible_dates.append(evaluation_date)
        else:
            research_skipped_dates.append(evaluation_date)

        probability_result = probability_metrics(transformed)
        closing_result = _closing_metrics(
            transformed,
            identity=closing_identity,
            v12_model=closing_v12_model,
            v11_model=closing_v11_model,
        )
        closing_v12_report_model = _closing_v12_report_model(closing_v12_model)
        closing_compat = _closing_metric_adapter(closing_result)
        selection_artifact = dict(bankroll.get("selection_conformal") or conformal)
        trained_boundaries = {
            "probability": probability_model.get("trained_through_date"),
            "probability_lcb": probability_lcb.get("trained_through_date"),
            "closing_v12": closing_v12_model.get("trained_through_date"),
            "closing_v11_fallback": (
                closing_v11_model.get("trained_through_date")
                if isinstance(closing_v11_model, Mapping)
                else None
            ),
            "selection_conformal": selection_artifact.get("trained_through_date"),
        }
        leakage_pass = all(
            value is None or str(value) < evaluation_date
            for value in trained_boundaries.values()
        ) and access_audit["future_checkpoint_violations"] == 0
        diagnostics_by_date[evaluation_date] = purchase_diagnostic
        daily.extend(bankroll["daily"])
        folds.append({
            "fold": len(folds) + 1,
            "calibration_dates": prior_dates,
            "evaluation_date": evaluation_date,
            "calibration_races": len(training),
            "evaluation_races": len(holdout),
            "operational_model": probability_model,
            "probability_lcb": probability_lcb,
            **(
                {"probability_lcb_metrics": lcb_evaluation}
                if lcb_evaluation is not None
                else {}
            ),
            "closing_model": closing_v12_report_model,
            "closing_v12_model": closing_v12_report_model,
            "closing_v11_fallback_model": closing_v11_model,
            "closing_model_identity": closing_identity,
            "closing_ready": bool(closing_identity["ready_for_purchase"]),
            "closing_t300_access_audit": access_audit,
            "selection_conformal": selection_artifact,
            "selection_observations_appended_after_decision": appended,
            "selected_policy": (
                dict(policy)
                if closing_identity["ready_for_purchase"]
                and probability_lcb.get("ready")
                and conformal.get("ready")
                else {"name": "no_bet", "no_bet": True}
            ),
            "probability_metrics": probability_result,
            "closing_model_metrics": closing_result,
            "closing_t300_v12_metrics": (
                closing_result
                if closing_identity["selected_model"] == V12_CLOSING_MODEL_NAME
                else None
            ),
            "closing_multihorizon_v11_metrics": (
                closing_result
                if closing_identity["selected_model"]
                == "closing_odds_multihorizon_v11"
                else None
            ),
            "closing_q20_metrics": closing_compat,
            "bankroll": {
                key: value for key, value in bankroll.items() if key != "daily"
            },
            "leakage_guard": {
                "outer_date": evaluation_date,
                "probability_trained_through": trained_boundaries["probability"],
                "probability_lcb_trained_through": trained_boundaries[
                    "probability_lcb"
                ],
                "closing_v12_trained_through": trained_boundaries["closing_v12"],
                "closing_v11_fallback_trained_through": trained_boundaries[
                    "closing_v11_fallback"
                ],
                "selection_conformal_trained_through": trained_boundaries[
                    "selection_conformal"
                ],
                "as_of_offset_seconds": DECISION_OFFSET_SECONDS,
                "future_checkpoint_imputation": False,
                "settlement_after_purchase_decision": True,
                "pass": leakage_pass,
            },
        })

    if not folds:
        return {
            "model": MODEL_NAME,
            "calibrator_strategy": STRATEGY_NAME,
            "status": "waiting_for_clean_evaluation_day",
            "comparison_role": "role_separated_t300_v12_shadow",
            "registered_after": REGISTERED_AFTER,
            "available_races": len(races),
            "available_days": len(dates),
            "evaluation_days": 0,
            "evaluation_races": 0,
            "evaluated_races": 0,
            "folds": [],
            "daily": [],
            "closing_model_identity": {
                "requested_model": V12_CLOSING_MODEL_NAME,
                "fallback_policy": closing_fallback_policy,
                "selected_model_latest": None,
                "selected_model_fold_counts": {},
                "evaluation_folds": 0,
                "v12_ready_folds": 0,
                "v12_adopted_folds": 0,
                "v11_fallback_folds": 0,
                "no_bet_folds": 0,
                "decision_checkpoint": DECISION_CHECKPOINT,
                "decision_offset_seconds": DECISION_OFFSET_SECONDS,
                "future_checkpoint_imputation": False,
            },
            RESEARCH_PRECONFORMAL_OUTPUT_KEY: _research_preconformal_summary(
                [],
                evaluated_races=0,
                eligible_dates=[],
                skipped_dates=[],
                diagnostics_by_date={},
            ),
            "promotion_gate": {},
            "promotion_eligible": False,
        }

    daily = _cumulative_daily(daily)
    probability = _weighted_probability_metrics(folds)
    bankroll = _summarize_bankroll(
        daily,
        evaluated_races=int(probability["evaluated_races"]),
        purchase_diagnostic_accumulators=diagnostics_by_date.values(),
    )
    selection_summary = _aggregate_selection_conformal(folds)
    closing_identity_summary = _aggregate_closing_identity(folds)
    prospective_folds = [
        fold for fold in folds if str(fold["evaluation_date"]) > REGISTERED_AFTER
    ]
    prospective_dates = {
        str(fold["evaluation_date"]) for fold in prospective_folds
    }
    prospective = _prospective_summary(
        prospective_folds,
        [row for row in daily if str(row["race_date"]) in prospective_dates],
        purchase_diagnostic_accumulators=(
            diagnostics_by_date[value] for value in prospective_dates
        ),
        registered_after=REGISTERED_AFTER,
        comparison_role="pre_registered_strict_outer_day_v12_shadow",
    )
    prospective_selection = _aggregate_selection_conformal(prospective_folds)
    prospective["selection_conformal"] = prospective_selection
    prospective["closing_model_identity"] = _aggregate_closing_identity(
        prospective_folds
    )
    gate = prospective.get("promotion_gate") or {}
    gate.update(_selection_coverage_gate(prospective_selection))
    checks = [value for key, value in gate.items() if key.endswith("_pass")]
    prospective["promotion_gate"] = gate
    prospective["promotion_eligible"] = bool(checks) and all(checks)

    deployment_date = _next_date(dates)
    deployment_closing_v12 = fit_closing_odds_t300_nonlinear_v12(
        races,
        prediction_date=deployment_date,
        lower_quantile=V12_DEFAULT_LOWER_QUANTILE,
    )
    deployment_closing_v11 = (
        fit_closing_odds_multihorizon_v11(
            races,
            prediction_date=deployment_date,
            lower_quantile=V11_DEFAULT_LOWER_QUANTILE,
        )
        if closing_fallback_policy == CLOSING_FALLBACK_V11
        else None
    )
    deployment_closing_identity = _closing_contract(
        deployment_closing_v12,
        deployment_closing_v11,
        fallback_policy=closing_fallback_policy,
    )
    deployment_probability = fit_odds_path_probability_v8(races)
    deployment_lcb = lcb_fit(
        _crossfit_probability_rows(
            races,
            probability_fit=fit_odds_path_probability_v8,
            probability_attach=attach_odds_path_probability_v8,
        )
    )
    deployment_conformal = conformal_fit(
        observations,
        evaluation_date=deployment_date,
    )
    deployment_ready = all((
        bool(deployment_closing_identity["ready_for_purchase"]),
        bool(deployment_lcb.get("ready")),
        bool(deployment_conformal.get("ready")),
    ))
    deployment = {
        "role": "next_day_refit_not_evaluation",
        "prediction_date": deployment_date,
        "calibrator_strategy": STRATEGY_NAME,
        "operational_model": deployment_probability,
        "probability_lcb": deployment_lcb,
        "closing_t300_v12_model": _closing_v12_report_model(
            deployment_closing_v12
        ),
        "closing_v11_fallback_model": deployment_closing_v11,
        "closing_model_identity": deployment_closing_identity,
        "selection_conformal": deployment_conformal,
        "daily_budget_yen": daily_budget_yen,
        "candidate_policy": dict(policy),
        "selected_policy": {"name": "no_bet", "no_bet": True},
        "operational_status": (
            "shadow_only_until_v12_promotion_gate"
            if deployment_ready
            else "v12_components_or_closing_fallback_not_ready"
        ),
    }
    research_preconformal = _research_preconformal_summary(
        research_daily,
        evaluated_races=research_evaluated_races,
        eligible_dates=research_eligible_dates,
        skipped_dates=research_skipped_dates,
        diagnostics_by_date=research_diagnostics_by_date,
    )

    return {
        "model": MODEL_NAME,
        "calibrator_strategy": STRATEGY_NAME,
        "comparison_role": "role_separated_t300_v12_shadow",
        "validation_design": (
            "Each outer day fits probability, probability LCB, V12 nonlinear closing "
            "odds and optional V11 fallback on strict prior days; T300 selection conformal and "
            "discrete allocation precede final-odds observation and settlement"
        ),
        "registered_after": REGISTERED_AFTER,
        "daily_budget_yen": daily_budget_yen,
        "fixed_policy": dict(policy),
        "available_races": len(races),
        "available_days": len(dates),
        "evaluation_days": len(folds),
        "evaluation_races": int(probability["evaluated_races"]),
        "evaluated_races": int(probability["evaluated_races"]),
        "probability_metrics": probability,
        **probability,
        "trifecta_log_loss": probability["calibrated_trifecta_log_loss"],
        "trifecta_top5_hit_rate": probability[
            "calibrated_trifecta_top5_hit_rate"
        ],
        **{key: value for key, value in bankroll.items() if key != "daily"},
        "selection_conformal": selection_summary,
        "closing_model_identity": closing_identity_summary,
        "selection_conformal_artifacts_by_date": artifacts_by_date,
        "folds": folds,
        "daily": daily,
        RESEARCH_PRECONFORMAL_OUTPUT_KEY: research_preconformal,
        PROSPECTIVE_OUTPUT_KEY: prospective,
        "promotion_gate": gate,
        "promotion_eligible": prospective["promotion_eligible"],
        "deployment_configuration": deployment,
    }
