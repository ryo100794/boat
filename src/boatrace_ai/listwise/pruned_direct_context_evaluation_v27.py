from __future__ import annotations

from typing import Any, Iterable, Mapping

from ..bankroll_bootstrap import bootstrap_daily_roi
from .flat_policy import simulate_chronological_flat_policy
from .market_calibration import blend_probabilities
from .pruned_direct_context_v27 import (
    fit_temporal_pruned_residual,
    pruned_probabilities,
)


def evaluate_temporal_pruned_residual(
    calibration: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
    *,
    policies: Iterable[Mapping[str, Any]],
    daily_budget_yen: int,
) -> dict[str, Any]:
    result = fit_temporal_pruned_residual(calibration, evaluation)
    scored = [
        {
            **race,
            "model_probabilities": pruned_probabilities(
                race,
                result["artifact"],
            ),
        }
        for race in evaluation
    ]
    purchase_diagnostics = []
    for policy in policies:
        simulation = simulate_chronological_flat_policy(
            scored,
            calibrator={"model_weight": 1.0, "temperature": 1.0},
            policy=dict(policy),
            probability_blender=blend_probabilities,
            initial_bankroll_yen=daily_budget_yen,
        )
        bootstrap = (
            bootstrap_daily_roi(simulation["daily"])
            if simulation["daily"]
            else {
                "days": 0,
                "roi": None,
                "roi_ci95_lower": None,
                "probability_roi_above_one": None,
            }
        )
        purchase_diagnostics.append({
            "policy": dict(policy),
            "simulation": simulation,
            "bootstrap": bootstrap,
        })
    result["purchase_diagnostics"] = purchase_diagnostics
    return result
