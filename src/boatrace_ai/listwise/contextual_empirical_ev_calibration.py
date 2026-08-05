from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from math import isfinite
from typing import Iterable, Mapping, Sequence

import numpy as np

from boatrace_ai.listwise.empirical_ev_calibration import (
    DEFAULT_BIN_EDGES,
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_SEED,
    EmpiricalEVCalibrationArtifact,
    fit_empirical_ev_calibration,
)


RANK_GROUPS = ("top5", "6-20", "21+")
ODDS_BANDS = ("<20", "20-50", "50-101", ">=101")
CONTEXTUAL_CALIBRATION_VERSION = 3


@dataclass(frozen=True)
class ContextualEVBin:
    index: int
    lower: float
    upper: float
    empirical_ev: float | None
    empirical_ev_lcb95: float | None
    calibration_level: str
    cell_support: int
    cell_support_days: int
    rank_support: int
    rank_support_days: int
    shrinkage_weight: float
    positive_return_days: int
    return_hhi: float | None

    def as_dict(self) -> dict[str, int | float | str | None]:
        return {
            "bin_index": self.index,
            "lower": self.lower,
            "upper": self.upper,
            "empirical_ev": self.empirical_ev,
            "empirical_ev_lcb95": self.empirical_ev_lcb95,
            "calibration_level": self.calibration_level,
            "cell_support": self.cell_support,
            "cell_support_days": self.cell_support_days,
            "rank_support": self.rank_support,
            "rank_support_days": self.rank_support_days,
            "shrinkage_weight": self.shrinkage_weight,
            "positive_return_days": self.positive_return_days,
            "return_hhi": self.return_hhi,
        }


@dataclass(frozen=True)
class ContextualEVCell:
    rank_group: str
    odds_band: str
    ready: bool
    ready_reasons: tuple[str, ...]
    support: int
    support_days: int
    bins: tuple[ContextualEVBin, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "rank_group": self.rank_group,
            "odds_band": self.odds_band,
            "ready": self.ready,
            "ready_reasons": list(self.ready_reasons),
            "support": self.support,
            "support_days": self.support_days,
            "bins": [bin_.as_dict() for bin_ in self.bins],
        }


@dataclass(frozen=True)
class ContextualEmpiricalEVCalibrationArtifact:
    cells: tuple[ContextualEVCell, ...]
    global_calibration: EmpiricalEVCalibrationArtifact
    ready: bool
    ready_reasons: tuple[str, ...]
    prediction_date: str
    trained_through_date: str | None
    training_days: int
    tickets: int
    excluded_non_past_records: int
    context_ready_cells: int
    min_rank_days: int
    min_rank_tickets: int
    min_cell_days: int
    min_cell_tickets: int
    rank_prior_tickets: float
    cell_prior_tickets: float
    bootstrap_samples: int
    seed: int
    shape_constraint: str = "isotonic"
    calibration_version: int = CONTEXTUAL_CALIBRATION_VERSION

    @property
    def candidate_days(self) -> int:
        return self.global_calibration.candidate_days

    def predict(
        self,
        raw_ev: float,
        probability_rank: int,
        forecast_odds: float,
    ) -> dict[str, object]:
        value = _finite_float(raw_ev, "raw_ev")
        rank_group = _rank_group(probability_rank)
        odds_band = _odds_band(forecast_odds)
        cell = next(
            cell
            for cell in self.cells
            if cell.rank_group == rank_group and cell.odds_band == odds_band
        )
        bin_index = _bin_index(
            value,
            tuple(bin_.upper for bin_ in cell.bins),
        )
        result: dict[str, object] = cell.bins[bin_index].as_dict()
        global_prediction = self.global_calibration.predict(value)
        for key in (
            "training_raw_ev_min",
            "training_raw_ev_max",
            "input_in_training_range",
            "input_in_local_block_range",
            "local_block_candidates",
            "local_block_candidate_days",
            "local_block_ess",
            "local_block_exposure_weight",
            "local_block_raw_ev_min",
            "local_block_raw_ev_max",
            "local_support_ready",
            "local_support_reasons",
        ):
            result[key] = global_prediction.get(key)
        result["purchase_lcb95_available"] = bool(
            global_prediction.get("purchase_lcb95_available")
            and cell.ready
            and result.get("empirical_ev_lcb95") is not None
        )
        result.update(
            {
                "rank_group": rank_group,
                "odds_band": odds_band,
                "cell_ready": cell.ready,
                "artifact_ready": self.ready,
                "trained_through_date": self.trained_through_date,
            }
        )
        return result

    def as_dict(self) -> dict[str, object]:
        return {
            "calibration_version": self.calibration_version,
            "ready": self.ready,
            "ready_reasons": list(self.ready_reasons),
            "prediction_date": self.prediction_date,
            "trained_through_date": self.trained_through_date,
            "training_days": self.training_days,
            "tickets": self.tickets,
            "candidate_days": self.candidate_days,
            "excluded_non_past_records": self.excluded_non_past_records,
            "context_ready_cells": self.context_ready_cells,
            "context_cells": len(self.cells),
            "min_rank_days": self.min_rank_days,
            "min_rank_tickets": self.min_rank_tickets,
            "min_cell_days": self.min_cell_days,
            "min_cell_tickets": self.min_cell_tickets,
            "rank_prior_tickets": self.rank_prior_tickets,
            "cell_prior_tickets": self.cell_prior_tickets,
            "bootstrap_samples": self.bootstrap_samples,
            "seed": self.seed,
            "shape_constraint": self.shape_constraint,
            "global_calibration": self.global_calibration.as_dict(),
            "cells": [cell.as_dict() for cell in self.cells],
        }


@dataclass(frozen=True)
class _Record:
    race_date: str
    raw_ev: float
    gross_return: float
    rank_index: int
    odds_index: int


def _finite_float(value: object, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _normalize_date(value: object, name: str) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError as exc:
        raise ValueError(f"{name} must start with an ISO date") from exc


def _rank_group(value: object) -> str:
    try:
        numeric = float(value)
        rank = int(numeric)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("probability_rank must be a positive integer") from exc
    if not isfinite(numeric) or numeric != rank or rank < 1:
        raise ValueError("probability_rank must be a positive integer")
    if rank <= 5:
        return RANK_GROUPS[0]
    if rank <= 20:
        return RANK_GROUPS[1]
    return RANK_GROUPS[2]


def _rank_index(value: object) -> int:
    return RANK_GROUPS.index(_rank_group(value))


def _odds_band(value: object) -> str:
    odds = _finite_float(value, "forecast_odds")
    if odds < 0.0:
        raise ValueError("forecast_odds must not be negative")
    if odds < 20.0:
        return ODDS_BANDS[0]
    if odds < 50.0:
        return ODDS_BANDS[1]
    if odds < 101.0:
        return ODDS_BANDS[2]
    return ODDS_BANDS[3]


def _odds_index(value: object) -> int:
    return ODDS_BANDS.index(_odds_band(value))


def _normalize_records(
    records: Iterable[Mapping[str, object]],
) -> list[_Record]:
    normalized: list[_Record] = []
    required = (
        "race_date",
        "raw_estimated_ev",
        "gross_return_per_yen",
        "probability_rank",
        "forecast_odds",
    )
    for record in records:
        missing = next((name for name in required if name not in record), None)
        if missing is not None:
            raise ValueError(f"missing required field: {missing}")
        raw_ev = _finite_float(record["raw_estimated_ev"], "raw_estimated_ev")
        gross_return = _finite_float(
            record["gross_return_per_yen"],
            "gross_return_per_yen",
        )
        if raw_ev < 0.0:
            raise ValueError("raw_estimated_ev must not be negative")
        if gross_return < 0.0:
            raise ValueError("gross_return_per_yen must not be negative")
        normalized.append(
            _Record(
                race_date=_normalize_date(record["race_date"], "race_date"),
                raw_ev=raw_ev,
                gross_return=gross_return,
                rank_index=_rank_index(record["probability_rank"]),
                odds_index=_odds_index(record["forecast_odds"]),
            )
        )
    return sorted(
        normalized,
        key=lambda row: (
            row.race_date,
            row.rank_index,
            row.odds_index,
            row.raw_ev,
            row.gross_return,
        ),
    )


def _validate_edges(bin_edges: Sequence[float]) -> tuple[float, ...]:
    edges = tuple(float(value) for value in bin_edges)
    if len(edges) < 2:
        raise ValueError("bin_edges must contain at least two values")
    if any(np.isnan(value) for value in edges):
        raise ValueError("bin_edges must not contain NaN")
    if any(left >= right for left, right in zip(edges, edges[1:])):
        raise ValueError("bin_edges must be strictly increasing")
    if edges[0] != float("-inf") or edges[-1] != float("inf"):
        raise ValueError("bin_edges must span negative to positive infinity")
    return edges


def _bin_index(value: float, upper_edges: Sequence[float]) -> int:
    return int(np.searchsorted(upper_edges, value, side="right"))


def _weighted_pava(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    block_values: list[float] = []
    block_weights: list[float] = []
    block_starts: list[int] = []
    for index, (value, weight) in enumerate(zip(values, weights, strict=True)):
        block_values.append(float(value))
        block_weights.append(float(weight))
        block_starts.append(index)
        while len(block_values) >= 2 and block_values[-2] > block_values[-1]:
            merged_weight = block_weights[-2] + block_weights[-1]
            left_share = block_weights[-2] / merged_weight
            right_share = block_weights[-1] / merged_weight
            merged_value = (
                block_values[-2] * left_share
                + block_values[-1] * right_share
            )
            block_values[-2:] = [merged_value]
            block_weights[-2:] = [merged_weight]
            block_starts.pop()
    result = np.empty(len(values), dtype=np.float64)
    for block, value in enumerate(block_values):
        start = block_starts[block]
        stop = (
            block_starts[block + 1]
            if block + 1 < len(block_starts)
            else len(values)
        )
        result[start:stop] = value
    return result


def _isotonic_bins(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    occupied = np.flatnonzero(counts > 0)
    result = np.full(len(counts), np.nan, dtype=np.float64)
    if not len(occupied):
        return result
    fitted = _weighted_pava(sums[occupied] / counts[occupied], counts[occupied])
    result[occupied] = fitted
    first = int(occupied[0])
    result[:first] = fitted[0]
    for left_position, right_position in zip(occupied, occupied[1:]):
        left = int(left_position)
        right = int(right_position)
        result[left + 1 : right] = result[left]
    result[int(occupied[-1]) + 1 :] = fitted[-1]
    return result


def _bandwise_bins(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    occupied = np.flatnonzero(counts > 0)
    result = np.full(len(counts), np.nan, dtype=np.float64)
    if not len(occupied):
        return result
    result[occupied] = sums[occupied] / counts[occupied]
    first = int(occupied[0])
    result[:first] = result[first]
    for left_position, right_position in zip(occupied, occupied[1:]):
        left = int(left_position)
        right = int(right_position)
        result[left + 1 : right] = result[left]
    result[int(occupied[-1]) + 1 :] = result[int(occupied[-1])]
    return result


def _fit_bins(
    sums: np.ndarray,
    counts: np.ndarray,
    *,
    shape_constraint: str,
) -> np.ndarray:
    if shape_constraint == "isotonic":
        return _isotonic_bins(sums, counts)
    if shape_constraint == "bandwise":
        return _bandwise_bins(sums, counts)
    raise ValueError("shape_constraint must be isotonic or bandwise")


def _context_reasons(
    support: int,
    support_days: int,
    *,
    min_tickets: int,
    min_days: int,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if support_days < min_days:
        reasons.append("insufficient_support_days")
    if support < min_tickets:
        reasons.append("insufficient_support")
    return tuple(reasons)


def _add_finite(array: np.ndarray, index: object, value: float) -> None:
    updated = float(array[index]) + value
    if not isfinite(updated):
        raise ValueError("gross return aggregates exceed float64 range")
    array[index] = updated


def _shrunken_bins(
    sums: np.ndarray,
    counts: np.ndarray,
    parent: np.ndarray,
    prior_tickets: float,
    shape_constraint: str,
) -> np.ndarray:
    """Fit one child curve with the parent as finite pseudo-support."""
    if not np.all(np.isfinite(parent)):
        return np.array(parent, dtype=np.float64, copy=True)
    local = np.divide(
        sums,
        counts,
        out=np.zeros_like(sums, dtype=np.float64),
        where=counts > 0,
    )
    local_weight = counts / (counts + prior_tickets)
    targets = local_weight * local + (1.0 - local_weight) * parent
    weights = counts.astype(np.float64) + prior_tickets
    if shape_constraint == "isotonic":
        return _weighted_pava(targets, weights)
    if shape_constraint == "bandwise":
        return targets
    raise ValueError("shape_constraint must be isotonic or bandwise")


def fit_contextual_empirical_ev_calibration(
    records: Iterable[Mapping[str, object]],
    *,
    prediction_date: object,
    bin_edges: Sequence[float] = DEFAULT_BIN_EDGES,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_SEED,
    min_days: int = 30,
    min_tickets: int = 300,
    min_candidate_days: int = 20,
    min_local_candidates: int = 50,
    min_local_candidate_days: int = 20,
    min_local_ess: float = 10.0,
    candidate_min_raw_ev: float = 0.0,
    min_rank_days: int = 15,
    min_rank_tickets: int = 150,
    min_cell_days: int = 10,
    min_cell_tickets: int = 50,
    rank_prior_tickets: float = 100.0,
    cell_prior_tickets: float = 50.0,
    shape_constraint: str = "isotonic",
) -> ContextualEmpiricalEVCalibrationArtifact:
    """Fit leakage-safe rank/odds contextual empirical EV calibration.

    Records on or after ``prediction_date`` are excluded before every aggregate,
    bootstrap sample, and readiness calculation. Context estimates shrink through
    rank group estimates to the configured global raw-EV shape constraint.
    """
    edges = _validate_edges(bin_edges)
    if shape_constraint not in {"isotonic", "bandwise"}:
        raise ValueError("shape_constraint must be isotonic or bandwise")
    boundary = _normalize_date(prediction_date, "prediction_date")
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    thresholds = (
        min_days,
        min_tickets,
        min_candidate_days,
        min_rank_days,
        min_rank_tickets,
        min_cell_days,
        min_cell_tickets,
    )
    if any(value < 1 for value in thresholds):
        raise ValueError("readiness thresholds must be positive")
    rank_prior_tickets = _finite_float(rank_prior_tickets, "rank_prior_tickets")
    cell_prior_tickets = _finite_float(cell_prior_tickets, "cell_prior_tickets")
    if rank_prior_tickets <= 0.0 or cell_prior_tickets <= 0.0:
        raise ValueError("prior ticket strengths must be positive")

    normalized = _normalize_records(records)
    rows = [row for row in normalized if row.race_date < boundary]
    excluded = len(normalized) - len(rows)
    global_records = [
        {
            "race_date": row.race_date,
            "raw_estimated_ev": row.raw_ev,
            "gross_return_per_yen": row.gross_return,
        }
        for row in rows
    ]
    global_artifact = fit_empirical_ev_calibration(
        global_records,
        bin_edges=edges,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        min_days=min_days,
        min_tickets=min_tickets,
        min_candidate_days=min_candidate_days,
        min_local_candidates=min_local_candidates,
        min_local_candidate_days=min_local_candidate_days,
        min_local_ess=min_local_ess,
        candidate_min_raw_ev=candidate_min_raw_ev,
        shape_constraint=shape_constraint,
    )

    dates = sorted({row.race_date for row in rows})
    date_indices = {value: index for index, value in enumerate(dates)}
    bin_count = len(edges) - 1
    shape = (len(dates), len(RANK_GROUPS), len(ODDS_BANDS), bin_count)
    day_sums = np.zeros(shape, dtype=np.float64)
    day_counts = np.zeros(shape, dtype=np.int64)
    upper_edges = edges[1:]
    for row in rows:
        index = _bin_index(row.raw_ev, upper_edges)
        coordinates = (
            date_indices[row.race_date],
            row.rank_index,
            row.odds_index,
            index,
        )
        _add_finite(day_sums, coordinates, row.gross_return)
        day_counts[coordinates] += 1

    if not np.all(np.isfinite(day_sums)):
        raise ValueError("gross return aggregates exceed float64 range")

    sums = day_sums.sum(axis=0)
    counts = day_counts.sum(axis=0)
    cell_support = counts.sum(axis=2)
    cell_days = np.count_nonzero(day_counts.sum(axis=3), axis=0)
    rank_counts = counts.sum(axis=1)
    rank_sums = sums.sum(axis=1)
    rank_support = rank_counts.sum(axis=1)
    rank_days = np.count_nonzero(day_counts.sum(axis=(2, 3)), axis=0)
    global_point = np.array(
        [
            np.nan if bin_.empirical_ev is None else bin_.empirical_ev
            for bin_ in global_artifact.bins
        ],
        dtype=np.float64,
    )

    rank_ready = np.zeros(len(RANK_GROUPS), dtype=bool)
    rank_weights = np.zeros((len(RANK_GROUPS), bin_count), dtype=np.float64)
    rank_point = np.tile(global_point, (len(RANK_GROUPS), 1))
    for rank_index in range(len(RANK_GROUPS)):
        reasons = _context_reasons(
            int(rank_support[rank_index]),
            int(rank_days[rank_index]),
            min_tickets=min_rank_tickets,
            min_days=min_rank_days,
        )
        rank_ready[rank_index] = not reasons
        if rank_ready[rank_index]:
            rank_point[rank_index] = _shrunken_bins(
                rank_sums[rank_index],
                rank_counts[rank_index],
                global_point,
                rank_prior_tickets,
                shape_constraint,
            )
            rank_weights[rank_index] = rank_counts[rank_index] / (
                rank_counts[rank_index] + rank_prior_tickets
            )

    cell_ready = np.zeros((len(RANK_GROUPS), len(ODDS_BANDS)), dtype=bool)
    cell_weights = np.zeros(
        (len(RANK_GROUPS), len(ODDS_BANDS), bin_count),
        dtype=np.float64,
    )
    cell_point = np.repeat(rank_point[:, None, :], len(ODDS_BANDS), axis=1)
    for rank_index in range(len(RANK_GROUPS)):
        for odds_index in range(len(ODDS_BANDS)):
            reasons = _context_reasons(
                int(cell_support[rank_index, odds_index]),
                int(cell_days[rank_index, odds_index]),
                min_tickets=min_cell_tickets,
                min_days=min_cell_days,
            )
            cell_ready[rank_index, odds_index] = not reasons
            if cell_ready[rank_index, odds_index]:
                cell_point[rank_index, odds_index] = _shrunken_bins(
                    sums[rank_index, odds_index],
                    counts[rank_index, odds_index],
                    rank_point[rank_index],
                    cell_prior_tickets,
                    shape_constraint,
                )
                cell_weights[rank_index, odds_index] = counts[
                    rank_index, odds_index
                ] / (counts[rank_index, odds_index] + cell_prior_tickets)

    cell_samples = np.empty(
        (bootstrap_samples, len(RANK_GROUPS), len(ODDS_BANDS), bin_count),
        dtype=np.float64,
    )
    if dates:
        rng = np.random.default_rng(seed)
        for sample in range(bootstrap_samples):
            selected = rng.integers(0, len(dates), size=len(dates))
            sampled_sums = day_sums[selected].sum(axis=0)
            sampled_counts = day_counts[selected].sum(axis=0)
            if not np.all(np.isfinite(sampled_sums)):
                raise ValueError("bootstrap aggregates exceed float64 range")
            sampled_global_sums = sampled_sums.sum(axis=(0, 1))
            sampled_global_counts = sampled_counts.sum(axis=(0, 1))
            sampled_global = _fit_bins(
                sampled_global_sums,
                sampled_global_counts,
                shape_constraint=shape_constraint,
            )
            if shape_constraint == "bandwise":
                sampled_global[sampled_global_counts == 0] = 0.0
            sampled_rank = np.tile(sampled_global, (len(RANK_GROUPS), 1))
            for rank_index in range(len(RANK_GROUPS)):
                if rank_ready[rank_index]:
                    sampled_rank[rank_index] = _shrunken_bins(
                        sampled_sums[rank_index].sum(axis=0),
                        sampled_counts[rank_index].sum(axis=0),
                        sampled_global,
                        rank_prior_tickets,
                        shape_constraint,
                    )
            sample_cells = np.repeat(
                sampled_rank[:, None, :],
                len(ODDS_BANDS),
                axis=1,
            )
            for rank_index in range(len(RANK_GROUPS)):
                for odds_index in range(len(ODDS_BANDS)):
                    if cell_ready[rank_index, odds_index]:
                        sample_cells[rank_index, odds_index] = _shrunken_bins(
                            sampled_sums[rank_index, odds_index],
                            sampled_counts[rank_index, odds_index],
                            sampled_rank[rank_index],
                            cell_prior_tickets,
                            shape_constraint,
                        )
            cell_samples[sample] = sample_cells
        cell_lcb = np.quantile(cell_samples, 0.05, axis=0)
        cell_lcb = np.minimum(cell_point, cell_lcb)
    else:
        cell_lcb = np.full_like(cell_point, np.nan)

    cells: list[ContextualEVCell] = []
    for rank_index, rank_group in enumerate(RANK_GROUPS):
        for odds_index, odds_band in enumerate(ODDS_BANDS):
            reasons = _context_reasons(
                int(cell_support[rank_index, odds_index]),
                int(cell_days[rank_index, odds_index]),
                min_tickets=min_cell_tickets,
                min_days=min_cell_days,
            )
            bins: list[ContextualEVBin] = []
            for bin_index in range(bin_count):
                cell_weight = cell_weights[rank_index, odds_index, bin_index]
                rank_weight = rank_weights[rank_index, bin_index]
                if cell_ready[rank_index, odds_index]:
                    level = "rank_odds_cell"
                    weight = cell_weight
                    lcb_value = cell_lcb[rank_index, odds_index, bin_index]
                    stability_sums = day_sums[:, rank_index, odds_index, bin_index]
                elif rank_ready[rank_index]:
                    level = "rank_group"
                    weight = rank_weight
                    lcb_value = cell_lcb[rank_index, odds_index, bin_index]
                    stability_sums = day_sums[:, rank_index, :, bin_index].sum(
                        axis=1
                    )
                else:
                    level = "global"
                    weight = 0.0
                    global_bin = global_artifact.bins[bin_index]
                    global_lcb = global_bin.empirical_ev_lcb95
                    lcb_value = np.nan if global_lcb is None else global_lcb
                    stability_sums = day_sums[:, :, :, bin_index].sum(
                        axis=(1, 2)
                    )
                stability_total = float(stability_sums.sum())
                positive_return_days = int(
                    np.count_nonzero(stability_sums > 0.0)
                )
                return_hhi = (
                    float(np.square(stability_sums).sum() / stability_total**2)
                    if stability_total > 0.0
                    else None
                )
                point_value = cell_point[rank_index, odds_index, bin_index]
                bins.append(
                    ContextualEVBin(
                        index=bin_index,
                        lower=edges[bin_index],
                        upper=edges[bin_index + 1],
                        empirical_ev=(
                            None if np.isnan(point_value) else float(point_value)
                        ),
                        empirical_ev_lcb95=(
                            None if np.isnan(lcb_value) else float(lcb_value)
                        ),
                        calibration_level=level,
                        cell_support=int(counts[rank_index, odds_index, bin_index]),
                        cell_support_days=int(
                            np.count_nonzero(
                                day_counts[:, rank_index, odds_index, bin_index]
                            )
                        ),
                        rank_support=int(rank_counts[rank_index, bin_index]),
                        rank_support_days=int(
                            np.count_nonzero(
                                day_counts[:, rank_index, :, bin_index].sum(axis=1)
                            )
                        ),
                        shrinkage_weight=weight,
                        positive_return_days=positive_return_days,
                        return_hhi=return_hhi,
                    )
                )
            cells.append(
                ContextualEVCell(
                    rank_group=rank_group,
                    odds_band=odds_band,
                    ready=not reasons,
                    ready_reasons=reasons,
                    support=int(cell_support[rank_index, odds_index]),
                    support_days=int(cell_days[rank_index, odds_index]),
                    bins=tuple(bins),
                )
            )

    return ContextualEmpiricalEVCalibrationArtifact(
        cells=tuple(cells),
        global_calibration=global_artifact,
        ready=global_artifact.ready,
        ready_reasons=global_artifact.ready_reasons,
        prediction_date=boundary,
        trained_through_date=global_artifact.trained_through_date,
        training_days=global_artifact.training_days,
        tickets=global_artifact.tickets,
        excluded_non_past_records=excluded,
        context_ready_cells=int(np.count_nonzero(cell_ready)),
        min_rank_days=min_rank_days,
        min_rank_tickets=min_rank_tickets,
        min_cell_days=min_cell_days,
        min_cell_tickets=min_cell_tickets,
        rank_prior_tickets=float(rank_prior_tickets),
        cell_prior_tickets=float(cell_prior_tickets),
        bootstrap_samples=bootstrap_samples,
        seed=int(seed),
        shape_constraint=shape_constraint,
    )
