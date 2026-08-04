from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from math import isfinite
from typing import Iterable, Mapping, Sequence

import numpy as np


DEFAULT_BIN_EDGES = (
    float("-inf"),
    1.0,
    1.05,
    1.10,
    1.20,
    1.50,
    float("inf"),
)
DEFAULT_BOOTSTRAP_SAMPLES = 5_000
DEFAULT_SEED = 20260728
LCB_TAIL_PROBABILITY = 0.05
LCB_CONFIDENCE_LEVEL = 1.0 - LCB_TAIL_PROBABILITY


@dataclass(frozen=True)
class EmpiricalEVBin:
    index: int
    lower: float
    upper: float
    empirical_ev: float | None
    empirical_ev_lcb95: float | None
    support: int
    exposure_weight: float
    support_days: int
    positive_return_days: int
    return_hhi: float | None
    isotonic_block_id: int | None
    local_block_candidates: int
    local_block_candidate_days: int
    local_block_ess: float
    local_block_exposure_weight: float
    local_block_raw_ev_min: float | None
    local_block_raw_ev_max: float | None
    local_support_ready: bool
    local_support_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "bin_index": self.index,
            "lower": self.lower,
            "upper": self.upper,
            "empirical_ev": self.empirical_ev,
            "empirical_ev_lcb95": self.empirical_ev_lcb95,
            "support": self.support,
            "exposure_weight": self.exposure_weight,
            "support_days": self.support_days,
            "positive_return_days": self.positive_return_days,
            "return_hhi": self.return_hhi,
            "isotonic_block_id": self.isotonic_block_id,
            "local_block_candidates": self.local_block_candidates,
            "local_block_candidate_days": self.local_block_candidate_days,
            "local_block_ess": self.local_block_ess,
            "local_block_exposure_weight": (
                self.local_block_exposure_weight
            ),
            "local_block_raw_ev_min": self.local_block_raw_ev_min,
            "local_block_raw_ev_max": self.local_block_raw_ev_max,
            "local_support_ready": self.local_support_ready,
            "local_support_reasons": list(self.local_support_reasons),
        }


@dataclass(frozen=True)
class EmpiricalEVCalibrationArtifact:
    bins: tuple[EmpiricalEVBin, ...]
    ready: bool
    ready_reasons: tuple[str, ...]
    trained_through_date: str | None
    training_days: int
    tickets: int
    total_exposure_weight: float
    training_raw_ev_min: float | None
    training_raw_ev_max: float | None
    candidate_days: int
    candidate_min_raw_ev: float
    min_days: int
    min_tickets: int
    min_candidate_days: int
    min_local_candidates: int
    min_local_candidate_days: int
    min_local_ess: float
    isotonic_block_count: int
    bootstrap_samples: int
    seed: int
    shape_constraint: str = "isotonic"
    quantile_method: str = "linear"

    def predict(
        self,
        raw_ev: float,
        probability_rank: int | None = None,
        forecast_odds: float | None = None,
    ) -> dict[str, object]:
        # Context is accepted so global and contextual artifacts share one policy API.
        del probability_rank, forecast_odds
        value = _finite_float(raw_ev, "raw_ev")
        index = _bin_index(value, tuple(bin_.upper for bin_ in self.bins))
        bin_ = self.bins[index]
        input_in_local_block_range = bool(
            bin_.local_block_raw_ev_min is not None
            and bin_.local_block_raw_ev_max is not None
            and bin_.local_block_raw_ev_min <= value
            <= bin_.local_block_raw_ev_max
        )
        prediction: dict[str, object] = {
            **bin_.as_dict(),
            "training_raw_ev_min": self.training_raw_ev_min,
            "training_raw_ev_max": self.training_raw_ev_max,
            "input_in_training_range": bool(
                self.training_raw_ev_min is not None
                and self.training_raw_ev_max is not None
                and self.training_raw_ev_min <= value
                <= self.training_raw_ev_max
            ),
            "input_in_local_block_range": input_in_local_block_range,
            "purchase_lcb95_available": bool(
                bin_.local_support_ready
                and input_in_local_block_range
                and bin_.empirical_ev_lcb95 is not None
            ),
        }
        return prediction

    def as_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "ready_reasons": list(self.ready_reasons),
            "trained_through_date": self.trained_through_date,
            "training_days": self.training_days,
            "tickets": self.tickets,
            "total_exposure_weight": self.total_exposure_weight,
            "training_raw_ev_min": self.training_raw_ev_min,
            "training_raw_ev_max": self.training_raw_ev_max,
            "weighting": "optional_sample_weight_default_1",
            "candidate_days": self.candidate_days,
            "candidate_min_raw_ev": self.candidate_min_raw_ev,
            "min_days": self.min_days,
            "min_tickets": self.min_tickets,
            "min_candidate_days": self.min_candidate_days,
            "min_local_candidates": self.min_local_candidates,
            "min_local_candidate_days": self.min_local_candidate_days,
            "min_local_ess": self.min_local_ess,
            "isotonic_block_count": self.isotonic_block_count,
            "bootstrap_samples": self.bootstrap_samples,
            "seed": self.seed,
            "lcb_tail_probability": LCB_TAIL_PROBABILITY,
            "lcb_confidence_level": LCB_CONFIDENCE_LEVEL,
            "lcb_sidedness": "one_sided_lower",
            "lcb_estimator": (
                "nonparametric_race_date_cluster_percentile_bootstrap"
            ),
            "bootstrap_cluster_unit": "race_date",
            "bootstrap_resample_cluster_count": self.training_days,
            "within_day_candidates_resampled_together": True,
            "ticket_level_independence_assumed": False,
            "bootstrap_observation": (
                "complete_race_date_vector_of_bin_returns_and_exposures"
            ),
            "lcb_capped_at_point_estimate": True,
            "shape_constraint": self.shape_constraint,
            "quantile_method": self.quantile_method,
            "bins": [bin_.as_dict() for bin_ in self.bins],
        }


@dataclass(frozen=True)
class _Record:
    race_date: str
    raw_ev: float
    gross_return: float
    sample_weight: float


def _finite_float(value: object, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _normalize_date(value: object) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if not text:
        raise ValueError("race_date must not be empty")
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError as exc:
        raise ValueError("race_date must start with an ISO date") from exc


def _normalize_records(
    records: Iterable[Mapping[str, object]],
) -> list[_Record]:
    normalized: list[_Record] = []
    for record in records:
        try:
            raw_ev_value = record["raw_estimated_ev"]
            gross_return_value = record["gross_return_per_yen"]
        except KeyError as exc:
            raise ValueError(f"missing required field: {exc.args[0]}") from exc
        raw_ev = _finite_float(raw_ev_value, "raw_estimated_ev")
        gross_return = _finite_float(
            gross_return_value,
            "gross_return_per_yen",
        )
        sample_weight = _finite_float(
            record.get("sample_weight", 1.0),
            "sample_weight",
        )
        if raw_ev < 0.0:
            raise ValueError("raw_estimated_ev must not be negative")
        if gross_return < 0.0:
            raise ValueError("gross_return_per_yen must not be negative")
        if sample_weight <= 0.0:
            raise ValueError("sample_weight must be positive")
        normalized.append(
            _Record(
                race_date=_normalize_date(record.get("race_date", "")),
                raw_ev=raw_ev,
                gross_return=gross_return,
                sample_weight=sample_weight,
            )
        )
    return normalized


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


def _add_finite(array: np.ndarray, index: object, value: float) -> None:
    updated = float(array[index]) + value
    if not isfinite(updated):
        raise ValueError("gross return aggregates exceed float64 range")
    array[index] = updated


def _weighted_pava(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if len(values) != len(weights):
        raise ValueError("values and weights must have equal length")
    if not len(values):
        return np.empty(0, dtype=np.float64)

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
        stop = block_starts[block + 1] if block + 1 < len(block_starts) else len(values)
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
    """Estimate each EV band independently while filling unsupported bands."""
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


def _contiguous_calibration_block_ids(
    fitted: np.ndarray,
) -> tuple[np.ndarray, int]:
    block_ids = np.full(len(fitted), -1, dtype=np.int64)
    block_id = -1
    previous: float | None = None
    for index, raw_value in enumerate(fitted):
        value = float(raw_value)
        if not isfinite(value):
            previous = None
            continue
        if previous is None or value != previous:
            block_id += 1
        block_ids[index] = block_id
        previous = value
    return block_ids, block_id + 1


def _bootstrap_lcb(
    day_sums: np.ndarray,
    day_counts: np.ndarray,
    *,
    samples: int,
    seed: int,
    shape_constraint: str,
    quantile_method: str,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    days, bin_count = day_sums.shape
    predictions = np.empty((samples, bin_count), dtype=np.float64)
    for sample in range(samples):
        selected = rng.integers(0, days, size=days)
        sampled_sums = day_sums[selected].sum(axis=0)
        sampled_counts = day_counts[selected].sum(axis=0)
        predictions[sample] = _fit_bins(
            sampled_sums,
            sampled_counts,
            shape_constraint=shape_constraint,
        )
        if shape_constraint == "bandwise":
            predictions[sample, sampled_counts == 0] = 0.0
        if not np.all(np.isfinite(predictions[sample])):
            raise ValueError("bootstrap aggregates exceed float64 range")
    return np.quantile(
        predictions,
        LCB_TAIL_PROBABILITY,
        axis=0,
        method=quantile_method,
    )


def fit_empirical_ev_calibration(
    records: Iterable[Mapping[str, object]],
    *,
    bin_edges: Sequence[float] = DEFAULT_BIN_EDGES,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_SEED,
    min_days: int = 30,
    min_tickets: int = 300,
    min_candidate_days: int = 20,
    min_local_candidates: int = 50,
    min_local_candidate_days: int = 20,
    min_local_ess: float = 10.0,
    candidate_min_raw_ev: float = 1.0,
    shape_constraint: str = "isotonic",
    quantile_method: str = "linear",
) -> EmpiricalEVCalibrationArtifact:
    """Fit a date-clustered empirical return calibration artifact.

    Callers must supply records available strictly before prediction time.
    The artifact clusters by race date and records the latest included date.
    """
    edges = _validate_edges(bin_edges)
    if shape_constraint not in {"isotonic", "bandwise"}:
        raise ValueError("shape_constraint must be isotonic or bandwise")
    if quantile_method not in {"linear", "inverted_cdf"}:
        raise ValueError(
            "quantile_method must be linear or inverted_cdf"
        )
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    if min_days < 1 or min_tickets < 1 or min_candidate_days < 1:
        raise ValueError("ready gate thresholds must be positive")
    if min_local_candidates < 1 or min_local_candidate_days < 1:
        raise ValueError("local support count thresholds must be positive")
    local_ess_threshold = _finite_float(min_local_ess, "min_local_ess")
    if local_ess_threshold <= 0.0:
        raise ValueError("min_local_ess must be positive")
    candidate_threshold = _finite_float(candidate_min_raw_ev, "candidate_min_raw_ev")
    if candidate_threshold < 0.0:
        raise ValueError("candidate_min_raw_ev must not be negative")

    rows = _normalize_records(records)
    dates = sorted({row.race_date for row in rows})
    date_index = {race_date: index for index, race_date in enumerate(dates)}
    bin_count = len(edges) - 1
    sums = np.zeros(bin_count, dtype=np.float64)
    exposure_weights = np.zeros(bin_count, dtype=np.float64)
    counts = np.zeros(bin_count, dtype=np.int64)
    day_sums = np.zeros((len(dates), bin_count), dtype=np.float64)
    day_exposure_weights = np.zeros(
        (len(dates), bin_count), dtype=np.float64
    )
    day_counts = np.zeros((len(dates), bin_count), dtype=np.int64)
    candidate_dates: set[str] = set()
    row_bin_indices: list[int] = []

    upper_edges = edges[1:]
    for row in rows:
        bin_index = _bin_index(row.raw_ev, upper_edges)
        row_bin_indices.append(bin_index)
        day_index = date_index[row.race_date]
        weighted_return = row.gross_return * row.sample_weight
        _add_finite(sums, bin_index, weighted_return)
        _add_finite(exposure_weights, bin_index, row.sample_weight)
        counts[bin_index] += 1
        _add_finite(day_sums, (day_index, bin_index), weighted_return)
        _add_finite(
            day_exposure_weights,
            (day_index, bin_index),
            row.sample_weight,
        )
        day_counts[day_index, bin_index] += 1
        if row.raw_ev >= candidate_threshold:
            candidate_dates.add(row.race_date)

    if not np.all(np.isfinite(day_sums)):
        raise ValueError("gross return aggregates exceed float64 range")

    point = _fit_bins(
        sums, exposure_weights, shape_constraint=shape_constraint
    )
    lcb = (
        _bootstrap_lcb(
            day_sums,
            day_exposure_weights,
            samples=bootstrap_samples,
            seed=seed,
            shape_constraint=shape_constraint,
            quantile_method=quantile_method,
        )
        if rows
        else np.full(bin_count, np.nan, dtype=np.float64)
    )
    lcb = np.minimum(point, lcb)
    block_ids, isotonic_block_count = _contiguous_calibration_block_ids(
        point
    )
    block_metadata: dict[int, dict[str, object]] = {}
    for block_id in range(isotonic_block_count):
        block_rows = [
            row
            for row, bin_index in zip(rows, row_bin_indices, strict=True)
            if int(block_ids[bin_index]) == block_id
        ]
        day_exposures: dict[str, float] = {}
        for row in block_rows:
            day_exposures[row.race_date] = (
                day_exposures.get(row.race_date, 0.0)
                + row.sample_weight
            )
        total_local_exposure = sum(day_exposures.values())
        squared_day_exposure = sum(
            value * value for value in day_exposures.values()
        )
        local_ess = (
            total_local_exposure * total_local_exposure
            / squared_day_exposure
            if squared_day_exposure > 0.0 else 0.0
        )
        local_reasons: list[str] = []
        if len(block_rows) < min_local_candidates:
            local_reasons.append("insufficient_local_candidates")
        if len(day_exposures) < min_local_candidate_days:
            local_reasons.append("insufficient_local_candidate_days")
        if local_ess < local_ess_threshold:
            local_reasons.append("insufficient_local_day_cluster_ess")
        block_metadata[block_id] = {
            "candidates": len(block_rows),
            "candidate_days": len(day_exposures),
            "ess": float(local_ess),
            "exposure_weight": float(total_local_exposure),
            "raw_ev_min": (
                min(row.raw_ev for row in block_rows)
                if block_rows else None
            ),
            "raw_ev_max": (
                max(row.raw_ev for row in block_rows)
                if block_rows else None
            ),
            "support_ready": not local_reasons,
            "support_reasons": tuple(local_reasons),
        }

    reasons: list[str] = []
    if len(dates) < min_days:
        reasons.append("insufficient_training_days")
    if len(rows) < min_tickets:
        reasons.append("insufficient_tickets")
    if len(candidate_dates) < min_candidate_days:
        reasons.append("insufficient_candidate_days")

    bins: list[EmpiricalEVBin] = []
    for index in range(bin_count):
        block_id = int(block_ids[index])
        local = block_metadata.get(block_id, {
            "candidates": 0,
            "candidate_days": 0,
            "ess": 0.0,
            "exposure_weight": 0.0,
            "raw_ev_min": None,
            "raw_ev_max": None,
            "support_ready": False,
            "support_reasons": (
                "insufficient_local_candidates",
                "insufficient_local_candidate_days",
                "insufficient_local_day_cluster_ess",
            ),
        })
        bins.append(
            EmpiricalEVBin(
                index=index,
                lower=edges[index],
                upper=edges[index + 1],
                empirical_ev=None if np.isnan(point[index]) else float(point[index]),
                empirical_ev_lcb95=None if np.isnan(lcb[index]) else float(lcb[index]),
                support=int(counts[index]),
                exposure_weight=float(exposure_weights[index]),
                support_days=int(np.count_nonzero(day_counts[:, index])),
                positive_return_days=int(
                    np.count_nonzero(day_sums[:, index] > 0.0)
                ),
                return_hhi=(
                    float(np.square(day_sums[:, index]).sum() / sums[index] ** 2)
                    if sums[index] > 0.0
                    else None
                ),
                isotonic_block_id=block_id if block_id >= 0 else None,
                local_block_candidates=int(local["candidates"]),
                local_block_candidate_days=int(local["candidate_days"]),
                local_block_ess=float(local["ess"]),
                local_block_exposure_weight=float(
                    local["exposure_weight"]
                ),
                local_block_raw_ev_min=local["raw_ev_min"],
                local_block_raw_ev_max=local["raw_ev_max"],
                local_support_ready=bool(local["support_ready"]),
                local_support_reasons=tuple(local["support_reasons"]),
            )
        )

    return EmpiricalEVCalibrationArtifact(
        bins=tuple(bins),
        ready=not reasons,
        ready_reasons=tuple(reasons),
        trained_through_date=dates[-1] if dates else None,
        training_days=len(dates),
        tickets=len(rows),
        total_exposure_weight=float(exposure_weights.sum()),
        training_raw_ev_min=(
            min(row.raw_ev for row in rows) if rows else None
        ),
        training_raw_ev_max=(
            max(row.raw_ev for row in rows) if rows else None
        ),
        candidate_days=len(candidate_dates),
        candidate_min_raw_ev=candidate_threshold,
        min_days=min_days,
        min_tickets=min_tickets,
        min_candidate_days=min_candidate_days,
        min_local_candidates=min_local_candidates,
        min_local_candidate_days=min_local_candidate_days,
        min_local_ess=local_ess_threshold,
        isotonic_block_count=isotonic_block_count,
        bootstrap_samples=bootstrap_samples,
        seed=int(seed),
        shape_constraint=shape_constraint,
        quantile_method=quantile_method,
    )
