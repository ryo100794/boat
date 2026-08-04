from __future__ import annotations

from typing import Any, Iterable, Mapping

from .direct_context_market_residual_v25 import CONTEXT_FEATURES
from .nonlinear_market_residual_v38 import (
    SHRINKAGES,
    TREE_PRESETS,
    fit_nonlinear_market_residual,
    nonlinear_residual_metrics,
)
from .ticket_utility_ranking_v31 import ACTIVE_CONTEXT_FEATURES


MODEL_NAME = "nonlinear_market_offset_context_search_v41"
CONTEXT_VARIANTS: dict[str, tuple[str, ...]] = {
    "independent_core_10": ACTIVE_CONTEXT_FEATURES,
    "full_context_20": CONTEXT_FEATURES,
}


def fit_temporal_nonlinear_context_search(
    calibration: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
    *,
    context_variants: Mapping[str, tuple[str, ...]] = CONTEXT_VARIANTS,
    tree_presets: Iterable[Mapping[str, Any]] = TREE_PRESETS,
    shrinkages: Iterable[float] = SHRINKAGES,
    num_threads: int = 4,
) -> dict[str, Any]:
    """Select feature breadth without consulting the outer evaluation period."""
    dates = sorted({str(race["race_date"]) for race in calibration})
    if len(dates) < 5:
        raise ValueError("at least five V41 calibration days are required")
    split_index = max(1, min(len(dates) - 1, int(len(dates) * 0.8)))
    fit_dates = set(dates[:split_index])
    validation_dates = set(dates[split_index:])
    inner_fit = [race for race in calibration if str(race["race_date"]) in fit_dates]
    inner_validation = [
        race for race in calibration if str(race["race_date"]) in validation_dates
    ]
    variant_values = {
        str(name): tuple(str(feature) for feature in features)
        for name, features in context_variants.items()
    }
    if not variant_values:
        raise ValueError("V41 requires context variants")
    if any(not features or len(set(features)) != len(features) for features in variant_values.values()):
        raise ValueError("V41 context variants must be non-empty and unique")
    preset_values = tuple(dict(value) for value in tree_presets)
    shrinkage_values = tuple(float(value) for value in shrinkages)
    if not preset_values or not shrinkage_values:
        raise ValueError("V41 requires tree and shrinkage candidates")

    candidates: list[dict[str, Any]] = []
    for variant_name, context_features in variant_values.items():
        for preset in preset_values:
            inner_artifact = fit_nonlinear_market_residual(
                inner_fit,
                tree_preset=preset,
                context_features=context_features,
                num_threads=num_threads,
            )
            for shrinkage in shrinkage_values:
                candidates.append({
                    "context_variant": variant_name,
                    "context_feature_count": len(context_features),
                    "tree_preset": str(preset["name"]),
                    "shrinkage": shrinkage,
                    "metrics": nonlinear_residual_metrics(
                        inner_validation,
                        inner_artifact,
                        shrinkage=shrinkage,
                    ),
                })
    selected = min(
        candidates,
        key=lambda row: (
            float(row["metrics"]["trifecta_log_loss"]),
            int(row["context_feature_count"]),
            float(row["shrinkage"]),
            str(row["tree_preset"]),
            str(row["context_variant"]),
        ),
    )
    selected_variant = str(selected["context_variant"])
    selected_preset = next(
        value
        for value in preset_values
        if str(value["name"]) == str(selected["tree_preset"])
    )
    artifact = fit_nonlinear_market_residual(
        calibration,
        tree_preset=selected_preset,
        context_features=variant_values[selected_variant],
        num_threads=num_threads,
    )
    selected_shrinkage = float(selected["shrinkage"])
    return {
        "model": MODEL_NAME,
        "validation_design": (
            "Context breadth, tree complexity, and residual shrinkage including "
            "the exact-market null are selected on the latest inner prior-day "
            "block only; the winner is refit on all prior days before outer scoring"
        ),
        "outer_period_used_for_selection": False,
        "market_is_exact_nested_null": True,
        "inner_fit_through": dates[split_index - 1],
        "inner_validation_from": dates[split_index],
        "context_variants": {
            name: list(features) for name, features in variant_values.items()
        },
        "candidates": candidates,
        "selected_context_variant": selected_variant,
        "selected_context_features": list(variant_values[selected_variant]),
        "selected_tree_preset": str(selected["tree_preset"]),
        "selected_shrinkage": selected_shrinkage,
        "artifact": artifact,
        "metrics": nonlinear_residual_metrics(
            evaluation, artifact, shrinkage=selected_shrinkage
        ),
    }
