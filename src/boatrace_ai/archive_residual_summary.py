from __future__ import annotations

from typing import Any, Mapping


RESIDUAL_MODELS = (
    (
        "nonlinear_market_offset_context_search_v41",
        "nonlinear_market_offset_context_search_v41",
        "metrics",
        "artifact",
    ),
    (
        "nonlinear_market_offset_residual_v38",
        "nonlinear_market_offset_residual_v38",
        "metrics",
        "artifact",
    ),
    (
        "ticket_utility_meta_ranking_v31",
        "ticket_utility_meta_ranking_v31",
        "probability_metrics",
        "probability_artifact",
    ),
    (
        "conditional_ticket_residual_v30",
        "conditional_ticket_residual_v30",
        "metrics",
        "artifact",
    ),
    (
        "payout_weighted_role_model_v29",
        "payout_weighted_role_model_v29",
        "probability_metrics",
        "probability_artifact",
    ),
    (
        "course_interaction_market_residual_v28",
        "course_interaction_market_residual_v28",
        "metrics",
        "artifact",
    ),
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
        "evaluation_from": temporal.get("evaluation_from"),
        "evaluation_through": temporal.get("evaluation_through"),
    })
    for key in (
        "evaluated_days",
        "log_loss_delta_vs_market",
        "days_better_than_market",
    ):
        if metrics.get(key) is not None:
            summary[f"residual_{key}"] = metrics.get(key)
    for key in (
        "market_is_exact_nested_null",
        "outer_period_used_for_selection",
        "selected_tree_preset",
        "selected_shrinkage",
        "selected_context_variant",
        "selected_context_features",
        "inner_fit_through",
        "inner_validation_from",
    ):
        if selected.get(key) is not None:
            summary[f"residual_{key}"] = selected.get(key)
    artifact = selected.get(str(selected_artifact_key))
    if isinstance(artifact, Mapping):
        for key in (
            "feature_dimension",
            "context_features",
            "feature_variant",
            "architecture",
            "structure_variant",
            "active_context_feature_count",
            "active_ticket_feature_count",
            "regularization",
            "objective",
            "gradient_norm",
            "iterations",
            "converged",
            "training_races",
            "tree_preset",
            "booster_sha256",
        ):
            if key in artifact:
                summary[f"residual_{key}"] = artifact[key]

    selection = selected.get("selected_candidate")
    if isinstance(selection, Mapping):
        summary["residual_selection"] = {
            key: selection.get(key)
            for key in (
                "variant",
                "structure_variant",
                "architecture",
                "feature_variant",
                "active_context_feature_count",
                "active_ticket_feature_count",
                "feature_dimension",
                "regularization",
                "payout_weight_exponent",
                "label_scheme",
                "tree_preset",
                "top_k",
                "converged",
            )
        }
    ranking_metrics = selected.get("ranking_metrics")
    if isinstance(ranking_metrics, Mapping):
        summary["residual_ranking_metrics"] = {
            key: ranking_metrics.get(key)
            for key in (
                "evaluated_races",
                "trifecta_log_loss",
                "trifecta_top5_hit_rate",
                "top5_flat_stake_yen",
                "top5_flat_return_yen",
                "top5_flat_profit_yen",
                "top5_flat_roi",
            )
        }
        selected_top_k = ranking_metrics.get("selected_top_k_metrics")
        if isinstance(selected_top_k, Mapping):
            summary["residual_ranking_metrics"] = {
                "selected_top_k": ranking_metrics.get("selected_top_k"),
                **{
                    key: selected_top_k.get(key)
                    for key in (
                        "evaluated_races",
                        "hit_races",
                        "hit_rate",
                        "stake_yen",
                        "return_yen",
                        "profit_yen",
                        "roi",
                        "roi_ci95_lower",
                        "probability_roi_above_one",
                    )
                },
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
        compact_policy = {
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
        }
        if row.get("role") is not None:
            compact_policy["role"] = row.get("role")
        if row.get("shrinkage") is not None:
            compact_policy["shrinkage"] = row.get("shrinkage")
        compact_policies.append(compact_policy)
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
            "roi_ci95_lower",
            "probability_roi_above_one",
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
            "roi_ci95_lower": bankroll.get("roi_ci95_lower"),
            "probability_roi_above_one": bankroll.get("probability_roi_above_one"),
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
    nested = temporal.get("nested_nonlinear_value_calibration_v40")
    if isinstance(nested, Mapping):
        nested_probability = nested.get("evaluation_probability_metrics")
        nested_probability = (
            nested_probability
            if isinstance(nested_probability, Mapping)
            else {}
        )
        nested_calibration = nested.get("empirical_ev_calibration")
        nested_calibration = (
            nested_calibration
            if isinstance(nested_calibration, Mapping)
            else {}
        )
        nested_bankroll = nested.get("bankroll")
        nested_bankroll = (
            nested_bankroll if isinstance(nested_bankroll, Mapping) else {}
        )
        summary.update({
            "nested_value_model": nested.get("model"),
            "nested_value_status": nested.get("status"),
            "nested_value_model_training_from": nested.get(
                "model_training_from"
            ),
            "nested_value_model_training_through": nested.get(
                "model_training_through"
            ),
            "nested_value_model_training_days": nested.get(
                "model_training_days"
            ),
            "nested_value_calibration_from": nested.get(
                "value_calibration_from"
            ),
            "nested_value_calibration_through": nested.get(
                "value_calibration_through"
            ),
            "nested_value_calibration_days": nested.get(
                "value_calibration_days"
            ),
            "nested_value_evaluation_from": nested.get("evaluation_from"),
            "nested_value_evaluation_through": nested.get(
                "evaluation_through"
            ),
            "nested_value_evaluated_races": nested_probability.get(
                "evaluated_races"
            ),
            "nested_value_trifecta_log_loss": nested_probability.get(
                "trifecta_log_loss"
            ),
            "nested_value_market_trifecta_log_loss": nested_probability.get(
                "market_trifecta_log_loss"
            ),
            "nested_value_log_loss_delta_vs_market": nested_probability.get(
                "log_loss_delta_vs_market"
            ),
            "nested_value_calibration_ready": nested_calibration.get("ready"),
            "nested_value_calibration_ready_reasons": nested_calibration.get(
                "ready_reasons"
            ),
            "nested_value_calibration_candidates": nested.get(
                "calibration_ledger_candidates"
            ),
            "nested_value_evaluation_candidates": nested.get(
                "evaluation_ledger_candidates"
            ),
            "nested_value_decile_audit": nested.get("value_decile_audit"),
            "nested_value_tickets": nested_bankroll.get("tickets"),
            "nested_value_stake_yen": nested_bankroll.get("stake_yen"),
            "nested_value_return_yen": nested_bankroll.get("return_yen"),
            "nested_value_profit_yen": nested_bankroll.get("profit_yen"),
            "nested_value_roi": nested_bankroll.get("roi"),
            "nested_value_roi_display": nested_bankroll.get("roi_display"),
            "nested_value_roi_ci95_lower": nested_bankroll.get(
                "roi_ci95_lower"
            ),
            "nested_value_probability_roi_above_one": nested_bankroll.get(
                "probability_roi_above_one"
            ),
            "nested_value_promotion_eligible": bool(
                nested.get("promotion_eligible")
            ),
        })
