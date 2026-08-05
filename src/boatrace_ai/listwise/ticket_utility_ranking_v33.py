from __future__ import annotations

from typing import Any, Iterable, Mapping

from .ticket_utility_ranking_v31 import (
    LABEL_SCHEMES,
    POLICY_CALIBRATION_DAYS,
    TREE_PRESETS,
    evaluate_temporal_ticket_utility_roles,
)


MODEL_NAME = "ticket_utility_calibration_aligned_v33"
FAMILYWISE_SELECTION_ALPHA = 0.05


def evaluate_calibration_aligned_ticket_utility(
    calibration: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
    *,
    daily_budget_yen: int,
    policy_calibration_days: int = POLICY_CALIBRATION_DAYS,
    label_schemes: Iterable[str] = LABEL_SCHEMES,
    tree_presets: Iterable[Mapping[str, Any]] = TREE_PRESETS,
    bootstrap_samples: int = 2_000,
) -> dict[str, Any]:
    """Evaluate a pre-policy-frozen scorer with multiplicity-aware selection.

    Ranking and probability generators are trained before the policy-calibration
    window, then reused unchanged for both the calibration ledger and untouched
    evaluation. The empirical ROI LCB therefore remains attached to the score
    generators that produced its training population.
    """

    result = evaluate_temporal_ticket_utility_roles(
        calibration,
        evaluation,
        daily_budget_yen=daily_budget_yen,
        policy_calibration_days=policy_calibration_days,
        label_schemes=label_schemes,
        tree_presets=tree_presets,
        probability_artifact=None,
        bootstrap_samples=bootstrap_samples,
        result_model_name=MODEL_NAME,
        freeze_calibration_generators=True,
        familywise_selection_alpha=FAMILYWISE_SELECTION_ALPHA,
    )
    if result.get("status") == "completed":
        result["validation_design"] = (
            "Prediction generators are frozen before the policy-calibration "
            "window and reused byte-identically for calibration and untouched "
            "evaluation. Candidate selection uses a Bonferroni family-wise "
            "daily-bootstrap lower bound plus largest-hit and chronological-"
            "block stress gates."
        )
    return result


__all__ = ["evaluate_calibration_aligned_ticket_utility"]
