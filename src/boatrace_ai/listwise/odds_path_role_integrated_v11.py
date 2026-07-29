from __future__ import annotations

import math
from collections import defaultdict
from datetime import date, timedelta
from typing import Any, Iterable, Mapping

from .closing_odds_multihorizon_v11 import (
    DEFAULT_LOWER_QUANTILE,
    checkpoint_label,
    closing_odds_multihorizon_v11_metrics,
    fit_closing_odds_multihorizon_v11,
    forecast_closing_odds_multihorizon_v11,
    select_teacher_final_odds,
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


MODEL_NAME = "odds_path_role_integrated_multihorizon_v11"
STRATEGY_NAME = MODEL_NAME
REGISTERED_AFTER = "2026-07-29"
PROSPECTIVE_OUTPUT_KEY = "prospective_role_integrated_v11_walk_forward"
DECISION_OFFSET_SECONDS = 300
DECISION_CHECKPOINT = checkpoint_label(DECISION_OFFSET_SECONDS)

DISCRETE_POLICY_V11: dict[str, Any] = {
    **DISCRETE_POLICY,
    "name": "v11_t300_multihorizon_selection_conformal_discrete_log",
    "closing_model": "closing_odds_multihorizon_v11",
    "decision_checkpoint": DECISION_CHECKPOINT,
    "decision_offset_seconds": DECISION_OFFSET_SECONDS,
    "future_checkpoint_imputation": False,
}


def _strict_prior_forecasts(
    races: Iterable[dict[str, Any]],
    model: Mapping[str, object],
    *,
    evaluation_date: str,
) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    forecasts: dict[str, dict[str, float]] = {}
    audit = {
        "races": 0,
        "ready_races": 0,
        "missing_t300_races": 0,
        "incomplete_t300_races": 0,
        "future_checkpoint_violations": 0,
    }
    for race in races:
        audit["races"] += 1
        forecast = forecast_closing_odds_multihorizon_v11(
            race,
            model,
            as_of_offset_seconds=DECISION_OFFSET_SECONDS,
            prediction_date=evaluation_date,
        )
        access = forecast.get("checkpoint_access_audit") or {}
        future = list(access.get("future_checkpoint_offsets_used") or [])
        row = (forecast.get("predictions") or {}).get(DECISION_CHECKPOINT) or {}
        row_future = list(row.get("future_checkpoint_offsets_used") or [])
        if future or row_future:
            audit["future_checkpoint_violations"] += 1
            raise ValueError("v11 T300 forecast used a future checkpoint")
        if not row.get("ready"):
            audit["missing_t300_races"] += 1
            continue
        lower = {
            str(combination): float(odds)
            for combination, odds in (row.get("lower_final_odds") or {}).items()
            if math.isfinite(float(odds)) and float(odds) > 0.0
        }
        if len(lower) != 120:
            audit["incomplete_t300_races"] += 1
            continue
        forecasts[str(race["race_id"])] = lower
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


def _next_date(dates: list[str]) -> str:
    if not dates:
        return date.today().isoformat()
    return (date.fromisoformat(dates[-1]) + timedelta(days=1)).isoformat()


def walk_forward_evaluate_v11(
    races: list[dict[str, Any]],
    *,
    daily_budget_yen: int,
    min_calibration_days: int,
    evaluation_dates: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Evaluate T300 decisions with a strict prior-day role-separated stack."""
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
    policy = {**DISCRETE_POLICY_V11, "daily_budget_yen": daily_budget_yen}

    observations: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    daily: list[dict[str, Any]] = []
    diagnostics_by_date: dict[str, dict[str, Any]] = {}
    artifacts_by_date: dict[str, dict[str, Any]] = {}

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
        probability_lcb = fit_probability_lcb(crossfit_rows)
        closing_model = fit_closing_odds_multihorizon_v11(
            training,
            prediction_date=evaluation_date,
            lower_quantile=DEFAULT_LOWER_QUANTILE,
        )
        conformal = fit_selection_conformal_haircut(
            observations,
            evaluation_date=evaluation_date,
        )
        artifacts_by_date[evaluation_date] = conformal

        closing_forecasts, access_audit = _strict_prior_forecasts(
            transformed,
            closing_model,
            evaluation_date=evaluation_date,
        )
        bankroll, purchase_diagnostic = _simulate_selection_conformal_policy(
            transformed,
            closing_forecasts=closing_forecasts,
            probability_lcb=probability_lcb,
            daily_budget_yen=daily_budget_yen,
            selection_conformal=conformal,
        )

        # Purchase/allocation has completed. Final odds can now become a teacher
        # for later days, while payout data remains inside allocator settlement.
        appended = _append_selection_observations(
            observations,
            transformed,
            closing_forecasts=closing_forecasts,
            probability_lcb=probability_lcb,
            evaluation_date=evaluation_date,
        )
        if evaluation_date not in output_dates:
            continue

        probability_result = probability_metrics(transformed)
        closing_result = closing_odds_multihorizon_v11_metrics(
            transformed,
            closing_model,
            as_of_offset_seconds=DECISION_OFFSET_SECONDS,
        )
        closing_compat = _closing_metric_adapter(closing_result)
        selection_artifact = dict(bankroll.get("selection_conformal") or conformal)
        trained_boundaries = (
            probability_model.get("trained_through_date"),
            probability_lcb.get("trained_through_date"),
            closing_model.get("trained_through_date"),
            selection_artifact.get("trained_through_date"),
        )
        leakage_pass = all(
            value is None or str(value) < evaluation_date
            for value in trained_boundaries
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
            "closing_model": closing_model,
            "closing_ready": bool(closing_model.get("ready")),
            "closing_t300_access_audit": access_audit,
            "selection_conformal": selection_artifact,
            "selection_observations_appended_after_decision": appended,
            "selected_policy": (
                dict(policy)
                if closing_model.get("ready")
                and probability_lcb.get("ready")
                and conformal.get("ready")
                else {"name": "no_bet", "no_bet": True}
            ),
            "probability_metrics": probability_result,
            "closing_multihorizon_v11_metrics": closing_result,
            "closing_q20_metrics": closing_compat,
            "bankroll": {
                key: value for key, value in bankroll.items() if key != "daily"
            },
            "leakage_guard": {
                "outer_date": evaluation_date,
                "probability_trained_through": trained_boundaries[0],
                "probability_lcb_trained_through": trained_boundaries[1],
                "closing_trained_through": trained_boundaries[2],
                "selection_conformal_trained_through": trained_boundaries[3],
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
            "comparison_role": "role_separated_t300_v11_shadow",
            "registered_after": REGISTERED_AFTER,
            "available_races": len(races),
            "available_days": len(dates),
            "evaluation_days": 0,
            "evaluation_races": 0,
            "evaluated_races": 0,
            "folds": [],
            "daily": [],
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
        comparison_role="pre_registered_strict_outer_day_v11_shadow",
    )
    prospective_selection = _aggregate_selection_conformal(prospective_folds)
    prospective["selection_conformal"] = prospective_selection
    gate = prospective.get("promotion_gate") or {}
    gate.update(_selection_coverage_gate(prospective_selection))
    checks = [value for key, value in gate.items() if key.endswith("_pass")]
    prospective["promotion_gate"] = gate
    prospective["promotion_eligible"] = bool(checks) and all(checks)

    deployment_date = _next_date(dates)
    deployment_closing = fit_closing_odds_multihorizon_v11(
        races,
        prediction_date=deployment_date,
        lower_quantile=DEFAULT_LOWER_QUANTILE,
    )
    deployment_probability = fit_odds_path_probability_v8(races)
    deployment_lcb = fit_probability_lcb(
        _crossfit_probability_rows(
            races,
            probability_fit=fit_odds_path_probability_v8,
            probability_attach=attach_odds_path_probability_v8,
        )
    )
    deployment_conformal = fit_selection_conformal_haircut(
        observations,
        evaluation_date=deployment_date,
    )
    deployment_ready = all((
        bool(deployment_closing.get("ready")),
        bool(deployment_lcb.get("ready")),
        bool(deployment_conformal.get("ready")),
    ))
    deployment = {
        "role": "next_day_refit_not_evaluation",
        "prediction_date": deployment_date,
        "calibrator_strategy": STRATEGY_NAME,
        "operational_model": deployment_probability,
        "probability_lcb": deployment_lcb,
        "closing_multihorizon_v11_model": deployment_closing,
        "selection_conformal": deployment_conformal,
        "daily_budget_yen": daily_budget_yen,
        "candidate_policy": dict(policy),
        "selected_policy": {"name": "no_bet", "no_bet": True},
        "operational_status": (
            "shadow_only_until_v11_promotion_gate"
            if deployment_ready
            else "v11_components_not_ready"
        ),
    }

    return {
        "model": MODEL_NAME,
        "calibrator_strategy": STRATEGY_NAME,
        "comparison_role": "role_separated_t300_v11_shadow",
        "validation_design": (
            "Each outer day fits probability, probability LCB, and multihorizon "
            "closing odds on strict prior days; T300 selection conformal and "
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
        "selection_conformal_artifacts_by_date": artifacts_by_date,
        "folds": folds,
        "daily": daily,
        PROSPECTIVE_OUTPUT_KEY: prospective,
        "promotion_gate": gate,
        "promotion_eligible": prospective["promotion_eligible"],
        "deployment_configuration": deployment,
    }
