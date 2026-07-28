from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from ..packed_bankroll import PackedCandidates
from .contextual_empirical_ev_calibration import (
    ContextualEmpiricalEVCalibrationArtifact,
    fit_contextual_empirical_ev_calibration,
)


CONTEXTUAL_EV_BIN_EDGES = (
    float("-inf"),
    1.0,
    1.2,
    1.5,
    2.0,
    2.5,
    3.5,
    5.0,
    10.0,
    float("inf"),
)


def probability_ranks(packed: PackedCandidates) -> np.ndarray:
    """Rank retained candidates within each race without using outcomes."""
    if not packed.tickets:
        return np.empty(0, dtype=np.int16)
    indices = np.arange(packed.tickets, dtype=np.int64)
    order = np.lexsort((indices, -packed.probability, packed.race_codes))
    sorted_races = packed.race_codes[order]
    starts = np.empty(packed.tickets, dtype=np.bool_)
    starts[0] = True
    starts[1:] = sorted_races[1:] != sorted_races[:-1]
    group_starts = np.maximum.accumulate(np.where(starts, indices, 0))
    sorted_ranks = indices - group_starts + 1
    if int(sorted_ranks.max()) > np.iinfo(np.int16).max:
        raise ValueError("too many candidates in one race")
    ranks = np.empty(packed.tickets, dtype=np.int16)
    ranks[order] = sorted_ranks.astype(np.int16)
    return ranks


def packed_empirical_records(
    packed: PackedCandidates,
) -> list[dict[str, Any]]:
    ranks = probability_ranks(packed)
    records: list[dict[str, Any]] = []
    for day_index, race_date in enumerate(packed.dates):
        start, stop = map(int, packed.offsets[day_index : day_index + 2])
        for index in range(start, stop):
            records.append({
                "race_date": race_date,
                "raw_estimated_ev": float(packed.estimated_ev[index]),
                "gross_return_per_yen": (
                    float(packed.actual_payout_yen[index]) / 100.0
                    if bool(packed.hit[index])
                    else 0.0
                ),
                "probability_rank": int(ranks[index]),
                "forecast_odds": float(packed.estimated_odds[index]),
            })
    return records


def fit_packed_contextual_ev(
    packed: PackedCandidates,
    *,
    prediction_date: str,
    bootstrap_samples: int,
    seed: int,
) -> ContextualEmpiricalEVCalibrationArtifact:
    return fit_contextual_empirical_ev_calibration(
        packed_empirical_records(packed),
        prediction_date=prediction_date,
        bin_edges=CONTEXTUAL_EV_BIN_EDGES,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )


def apply_contextual_ev(
    packed: PackedCandidates,
    artifact: ContextualEmpiricalEVCalibrationArtifact,
    *,
    estimate: str,
) -> PackedCandidates:
    if estimate not in {"point", "lcb95"}:
        raise ValueError("estimate must be point or lcb95")
    if not artifact.ready:
        raise ValueError("EV calibration artifact is not ready")
    if packed.dates and (
        artifact.trained_through_date is None
        or artifact.trained_through_date >= packed.dates[0]
    ):
        raise ValueError("EV calibration must be trained before evaluation dates")
    ranks = probability_ranks(packed)
    calibrated = np.zeros(packed.tickets, dtype=np.float32)
    field = "empirical_ev" if estimate == "point" else "empirical_ev_lcb95"
    for index in range(packed.tickets):
        prediction = artifact.predict(
            float(packed.estimated_ev[index]),
            int(ranks[index]),
            float(packed.estimated_odds[index]),
        )
        value = prediction.get(field)
        calibrated[index] = max(0.0, float(value)) if value is not None else 0.0
    return replace(packed, estimated_ev=calibrated)
