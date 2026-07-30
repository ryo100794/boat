from __future__ import annotations

from collections.abc import Mapping
from typing import Any


CHRONOLOGICAL_BANKROLL_KEYS = (
    "race_days",
    "evaluated_races",
    "selected_races",
    "tickets",
    "hit_tickets",
    "stake_yen",
    "return_yen",
    "profit_yen",
    "roi",
    "max_drawdown_yen",
    "winning_days",
    "profitable_day_fraction",
    "roi_without_largest_hit",
    "largest_hit_return_share",
    "effective_hit_count",
    "daily_cluster_bootstrap_roi_lower_95",
    "bootstrap_probability_roi_above_one",
    "normalized_drawdown",
    "daily_stake_limit_fraction",
)

_CHRONOLOGICAL_MARKERS = {
    "chronological",
    "chronological_bankroll",
    "time_ordered",
}

CHRONOLOGICAL_PROMOTION_ALIASES = {
    "largest_hit_excluded_roi": "roi_without_largest_hit",
    "roi_ci95_lower": "daily_cluster_bootstrap_roi_lower_95",
    "probability_roi_above_one": "bootstrap_probability_roi_above_one",
}


def canonicalize_primary_bankroll(
    values: Mapping[str, Any],
    *,
    chronological_bankroll: Mapping[str, Any] | None = None,
    primary_bankroll: Any = None,
) -> dict[str, Any]:
    """Use chronological metrics as headlines when they are promotion-primary."""
    result = dict(values)
    chronological = (
        chronological_bankroll
        if chronological_bankroll is not None
        else values.get("chronological_bankroll")
    )
    nested = dict(chronological) if isinstance(chronological, Mapping) else {}
    for key in CHRONOLOGICAL_BANKROLL_KEYS:
        flattened = values.get(f"chronological_{key}")
        if key not in nested and flattened is not None:
            nested[key] = flattened

    promotion_gate = values.get("promotion_gate")
    gate_primary = (
        promotion_gate.get("primary_bankroll")
        if isinstance(promotion_gate, Mapping)
        else None
    )
    primary = (
        primary_bankroll
        if primary_bankroll is not None
        else values.get("primary_bankroll")
    )
    is_chronological = (
        str(primary or "").strip().lower() in _CHRONOLOGICAL_MARKERS
        or str(gate_primary or "").strip().lower() in _CHRONOLOGICAL_MARKERS
        or nested.get("primary_promotion_bankroll") is True
    )
    if not is_chronological or not nested:
        return result

    stored_legacy = values.get("legacy_batch_bankroll")
    legacy = (
        dict(stored_legacy)
        if isinstance(stored_legacy, Mapping)
        else {
            key: values[key]
            for key in (
                *CHRONOLOGICAL_BANKROLL_KEYS,
                *CHRONOLOGICAL_PROMOTION_ALIASES,
            )
            if values.get(key) is not None
        }
    )
    if legacy:
        result["legacy_batch_bankroll"] = legacy
        for key, value in legacy.items():
            result[f"legacy_batch_{key}"] = value

    for key in CHRONOLOGICAL_BANKROLL_KEYS:
        value = nested.get(key)
        if value is not None:
            result[key] = value
            result[f"chronological_{key}"] = value

    for headline_key, chronological_key in CHRONOLOGICAL_PROMOTION_ALIASES.items():
        value = nested.get(chronological_key)
        if value is not None:
            result[headline_key] = value
            result[f"chronological_{headline_key}"] = value

    result["primary_bankroll"] = "chronological"
    return result
