from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from math import fsum, isfinite
from typing import Any

import numpy as np


DEFAULT_BOOTSTRAP_SAMPLES = 20_000
DEFAULT_SEED = 20260728
DEFAULT_CHUNK_SIZE = 2_000
MAX_EXACT_YEN = float(2**53 - 1)


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _normalized_date(value: object) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if value is None:
        raise ValueError("race_date must not be empty")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("race_date must not be empty")
    return normalized


def _yen_value(row: Mapping[str, object], key: str) -> float:
    try:
        raw = row[key]
        if isinstance(raw, bool):
            raise TypeError
        value = float(raw)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{key} must be a finite non-negative number") from exc
    if not isfinite(value) or value < 0.0:
        raise ValueError(f"{key} must be a finite non-negative number")
    if value > MAX_EXACT_YEN:
        raise ValueError(f"{key} exceeds the exact supported yen range")
    return value


def _aggregate_days(
    daily_rows: Iterable[Mapping[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    daily: dict[str, tuple[list[float], list[float]]] = defaultdict(
        lambda: ([], [])
    )
    try:
        iterator = iter(daily_rows)
    except TypeError as exc:
        raise ValueError("daily_rows must be an iterable of mappings") from exc

    for row in iterator:
        if not isinstance(row, Mapping):
            raise ValueError("each daily row must be a mapping")
        day = _normalized_date(row.get("race_date"))
        stake = _yen_value(row, "stake_yen")
        returned = _yen_value(row, "return_yen")
        daily[day][0].append(stake)
        daily[day][1].append(returned)

    if not daily:
        raise ValueError("daily_rows must not be empty")

    ordered_days = sorted(daily)
    aggregates = [
        (fsum(sorted(daily[day][0])), fsum(sorted(daily[day][1])))
        for day in ordered_days
    ]
    if any(
        not isfinite(stake)
        or not isfinite(returned)
        or stake > MAX_EXACT_YEN
        or returned > MAX_EXACT_YEN
        for stake, returned in aggregates
    ):
        raise ValueError("daily aggregate exceeds the exact supported yen range")
    stakes = np.asarray([row[0] for row in aggregates], dtype=np.float64)
    returns = np.asarray([row[1] for row in aggregates], dtype=np.float64)
    if stakes.sum() > MAX_EXACT_YEN or returns.sum() > MAX_EXACT_YEN:
        raise ValueError("observed aggregate exceeds the exact supported yen range")
    return stakes, returns


def bootstrap_daily_roi(
    daily_rows: Iterable[Mapping[str, object]],
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_SEED,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, Any]:
    """Estimate bankroll ROI uncertainty by resampling complete days.

    Multiple records for one race date are aggregated before sampling. Zero-
    stake days remain in the cluster population. Draws with zero total stake
    have undefined ROI and are excluded from the percentile and probability.
    """

    sample_count = _positive_integer(samples, "samples")
    step = _positive_integer(chunk_size, "chunk_size")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    normalized_seed = int(seed)
    if normalized_seed < 0:
        raise ValueError("seed must be a non-negative integer")

    stakes, returns = _aggregate_days(daily_rows)
    observed_stake = float(stakes.sum())
    observed_return = float(returns.sum())
    observed_roi = (
        float(observed_return / observed_stake) if observed_stake > 0.0 else None
    )

    rng = np.random.default_rng(normalized_seed)
    bootstrap_roi = np.empty(sample_count, dtype=np.float64)
    valid_count = 0
    day_count = len(stakes)
    for start in range(0, sample_count, step):
        current_size = min(step, sample_count - start)
        sampled_days = rng.integers(
            0,
            day_count,
            size=(current_size, day_count),
        )
        sampled_stakes = stakes[sampled_days].sum(axis=1)
        sampled_returns = returns[sampled_days].sum(axis=1)
        if not np.all(np.isfinite(sampled_stakes)) or not np.all(
            np.isfinite(sampled_returns)
        ):
            raise ValueError("bootstrap aggregate exceeds float64 range")
        valid = sampled_stakes > 0.0
        valid_in_chunk = int(np.count_nonzero(valid))
        if valid_in_chunk:
            end = valid_count + valid_in_chunk
            bootstrap_roi[valid_count:end] = (
                sampled_returns[valid] / sampled_stakes[valid]
            )
            valid_count = end

    if valid_count:
        valid_roi = bootstrap_roi[:valid_count]
        roi_lower = float(np.quantile(valid_roi, 0.05))
        probability_above_one = float(np.mean(valid_roi > 1.0))
    else:
        roi_lower = None
        probability_above_one = None

    return {
        "days": int(day_count),
        "samples": sample_count,
        "valid_samples": valid_count,
        "stake_yen": observed_stake,
        "return_yen": observed_return,
        "profit_yen": float(observed_return - observed_stake),
        "roi": observed_roi,
        "roi_ci95_lower": roi_lower,
        "probability_roi_above_one": probability_above_one,
    }
