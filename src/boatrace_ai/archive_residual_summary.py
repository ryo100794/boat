from __future__ import annotations

from typing import Any, Mapping


RESIDUAL_MODELS = (
    ("direct_context_market_residual_v25", "direct_context_market_residual_v25"),
    ("contextual_market_residual_v24", "contextual_market_residual_v24"),
)


def apply_archive_residual_summary(
    payload: Mapping[str, Any],
    summary: dict[str, Any],
) -> None:
    temporal = payload.get("temporal_residual_diagnostic")
    if not isinstance(temporal, Mapping):
        return
    selected_name = selected = None
    for result_key, model_name in RESIDUAL_MODELS:
        candidate = temporal.get(result_key)
        if isinstance(candidate, Mapping):
            selected_name = model_name
            selected = candidate
            break
    if selected is None or selected_name is None:
        return
    metrics = selected.get("metrics")
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
    artifact = selected.get("artifact")
    if isinstance(artifact, Mapping):
        for key in (
            "feature_dimension",
            "regularization",
            "objective",
            "gradient_norm",
            "iterations",
            "converged",
            "training_races",
        ):
            if key in artifact:
                summary[f"residual_{key}"] = artifact[key]

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
    summary["residual_purchase_policies"] = compact_policies
