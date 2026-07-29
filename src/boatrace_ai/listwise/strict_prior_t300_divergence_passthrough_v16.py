from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

from .edge_conditional_probability_lcb_v13 import PROBABILITY_BANDS, RANK_GROUPS
from .edge_conditional_probability_lcb_v14 import (
    METHOD as V14_DISPATCH_METHOD,
    REGISTERED_DIVERGENCE_LABEL,
    REGISTERED_DIVERGENCE_LOWER,
    REGISTERED_DIVERGENCE_UPPER,
)

MODEL_NAME = "strict_prior_t300_divergence_passthrough_v16"
METHOD = "fixed_t300_divergence_raw_probability_passthrough_v16"
REGISTERED_AFTER = "2026-07-29"


def _population_fingerprint(rows: Iterable[Mapping[str, Any]]) -> str:
    population = sorted(
        (str(row.get("race_date") or ""), str(row.get("race_id") or ""))
        for row in rows
    )
    payload = json.dumps(population, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fit_strict_prior_t300_divergence_passthrough_v16(
    crossfit_races: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a result-free fixed-band artifact for the V12 callback."""
    dates = sorted({str(row.get("race_date") or "") for row in crossfit_races})
    dates = [date for date in dates if date]
    rank_nodes = {
        rank: {"factor": 1.0, "fixed": True, "uses_result": False,
               "resolution": "raw_probability_passthrough"}
        for rank in RANK_GROUPS
    }
    conditional_cells = {
        "|".join((rank, probability_band, REGISTERED_DIVERGENCE_LABEL)): {
            "factor": 1.0, "parent_factor": 1.0, "usable": True,
            "fixed": True, "uses_result": False,
            "resolution": "raw_probability_passthrough",
        }
        for rank in RANK_GROUPS
        for _upper, probability_band in PROBABILITY_BANDS
    }
    return {
        "model_name": MODEL_NAME,
        "artifact_method": METHOD,
        # Stable dispatcher protocol; V16 does not fit V14's conditional LCB.
        "method": V14_DISPATCH_METHOD,
        "ready": True,
        "reason": None,
        "registered_after": REGISTERED_AFTER,
        "decision_checkpoint": "t300",
        "fixed_filter": True,
        "strict_prior": True,
        "raw_probability_passthrough": True,
        "uses_result": False,
        "uses_payout": False,
        "fit_parameters_from_outcomes": False,
        "registered_divergence_definition": (
            "log(model_probability / normalized_T300_market_probability)"
        ),
        "registered_divergence_lower_inclusive": REGISTERED_DIVERGENCE_LOWER,
        "registered_divergence_upper_exclusive": REGISTERED_DIVERGENCE_UPPER,
        "registered_divergence_label": REGISTERED_DIVERGENCE_LABEL,
        "training_days": len(dates),
        "training_races": len(crossfit_races),
        "training_dates": dates,
        "trained_through_date": dates[-1] if dates else None,
        "input_population_fingerprint": _population_fingerprint(crossfit_races),
        "rank_nodes": rank_nodes,
        "conditional_cells": conditional_cells,
    }
