from __future__ import annotations

from typing import Any, Mapping


RESIDUAL_MODELS = (
    (
        "pruned_direct_context_market_residual_v27",
        "pruned_direct_context_market_residual_v27",
        "metrics",
        "artifact",
    ),
    (
        "direct_context_empirical_lcb_v26",
        "direct_context_empirical_lcb_v26",
        "probability_metrics",
        "final_probability_artifact",
    ),
    (
        "direct_context_market_residual_v25",
        "direct_context_market_residual_v25",
        "metrics",
        "artifact",
    ),
    (
        "contextual_market_residual_v24",
        "contextual_market_residual_v24",
        "metrics",
        "artifact",
    ),
)


def apply_archive_residual_summary(
    payload: Mapping[str, Any],
    summary: dict[str, Any],
) -> None:
    temporal = payload.get("temporal_residual_diagnostic")
    if not isinstance(temporal, Mapping):
        return
    selected_name = selected = selected_metric_key = selected_artifact_key = None
    for result_key, model_name, metric_key, artifact_key in RESIDUAL_MODELS:
        candidate = temporal.get(result_key)
        if isinstance(candidate, Mapping):
            selected_name = model_name
            selected = candidate
            selected_metric_key = metric_key
            selected_artifact_key = artifact_key
            break
    if None in (selected, selected_name, selected_metric_key, selected_artifact_key):
        return
    metrics = selected.get(str(selected_metric_key))
    if not isinstance(metrics, Mapping):
        return

    summary.update({
        "model": selected_name,
        "comparison_role": "unavailable_at_decision_closing_oracle_research_only",
        "promotion_eligible": False,
        "evaluated_races": metrics.get("evaluated_races"),
        "trifecta_log_loss": metrics.get("trifecta_log_loss"),
        "calibrated_trifecta_log_loss": metrics.get("trifecta_log_loss"),
        "market_trifecta_log_loss": metrics.get("market_trifecta_log_loss"),
        "model_trifecta_log_loss": metrics.get("raw_model_trifecta_log_loss"),
        "trifecta_top5_hit_rate": metrics.get("trifecta_top5_hit_rate"),
        "calibrated_trifecta_top5_hit_rate": metrics.get(
            "trifecta_top5_hit_rate"
        ),
        "market_trifecta_top5_hit_rate": metrics.get(
            "market_trifecta_top5_hit_rate"
        ),
        "residual_calibration_from": temporal.get("calibration_from"),
        "residual_calibration_through": temporal.get("calibration_through"),
        "residual_evaluation_from": temporal.get("evaluation_from"),
        "residual_evaluation_through": temporal.get("evaluation_through"),
    })
    artifact = selected.get(str(selected_artifact_key))
    if isinstance(artifact, Mapping):
        for key in (
            "feature_dimension",
            "feature_variant",
            "active_context_feature_count",
            "regularization",
            "objective",
            "gradient_norm",
            "iterations",
            "converged",
            "training_races",
        ):
            if key in artifact:
                summary[f"residual_{key}"] = artifact[key]

    selection = selected.get("selected_candidate")
    if isinstance(selection, Mapping):
        summary["residual_selection"] = {
            key: selection.get(key)
            for key in (
                "variant",
                "active_context_feature_count",
                "feature_dimension",
                "regularization",
                "converged",
            )
        }

    compact_policies = []
    for row in selected.get("purchase_diagnostics") or ():
        if not isinstance(row, Mapping):
            continue
        policy = row.get("policy")
        simulation = row.get("simulation")
        bootstrap = row.get("bootstrap")
        if not isinstance(policy, Mapping) or not isinstance(simulation, Mapping):
            continue
        compact_policies.append({
            "name": policy.get("name"),
            "tickets": simulation.get("tickets"),
            "races_bet": simulation.get("races_bet"),
            "hit_tickets": simulation.get("hit_tickets"),
            "stake_yen": simulation.get("stake_yen"),
            "return_yen": simulation.get("return_yen"),
            "profit_yen": simulation.get("profit_yen"),
            "roi": simulation.get("roi"),
            "roi_ci95_lower": (
                bootstrap.get("roi_ci95_lower")
                if isinstance(bootstrap, Mapping)
                else None
            ),
            "probability_roi_above_one": (
                bootstrap.get("probability_roi_above_one")
                if isinstance(bootstrap, Mapping)
                else None
            ),
        })
    bankroll = selected.get("bankroll")
    if isinstance(bankroll, Mapping):
        for key in (
            "evaluated_races",
            "evaluation_days",
            "tickets",
            "hit_tickets",
            "stake_yen",
            "return_yen",
            "profit_yen",
            "roi",
            "max_drawdown_yen",
        ):
            if bankroll.get(key) is not None:
                summary[key] = bankroll[key]
        summary["promotion_eligible"] = bool(selected.get("promotion_eligible"))
        compact_policies = [{
            "name": "empirical_ev_lcb95_adaptive_kelly",
            "tickets": bankroll.get("tickets"),
            "hit_tickets": bankroll.get("hit_tickets"),
            "stake_yen": bankroll.get("stake_yen"),
            "return_yen": bankroll.get("return_yen"),
            "profit_yen": bankroll.get("profit_yen"),
            "roi": bankroll.get("roi"),
            "status": bankroll.get("status"),
        }]
        calibration = selected.get("empirical_ev_calibration")
        if isinstance(calibration, Mapping):
            summary["residual_empirical_ev_calibration"] = {
                key: calibration.get(key)
                for key in (
                    "ready",
                    "ready_reasons",
                    "trained_through_date",
                    "training_days",
                    "tickets",
                    "candidate_days",
                    "context_ready_cells",
                    "context_cells",
                    "excluded_non_past_records",
                )
                if calibration.get(key) is not None
            }
    summary["residual_purchase_policies"] = compact_policies
