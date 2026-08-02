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


def _normalized_venue(value: object) -> str:
    if value is None:
        raise ValueError("jcd must not be empty")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("jcd must not be empty")
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


def moving_block_bootstrap_roi(
    daily_rows: Iterable[Mapping[str, object]],
    *,
    block_days: int,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_SEED,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, Any]:
    """Estimate ROI sensitivity by resampling moving blocks of complete days.

    Rows are aggregated by date before overlapping blocks are formed. Blocks
    follow the sorted observed-day sequence and never wrap from the final day
    to the first. The final sampled block is truncated to preserve the observed
    number of days in every bootstrap draw.
    """

    sample_count = _positive_integer(samples, "samples")
    step = _positive_integer(chunk_size, "chunk_size")
    block_length = _positive_integer(block_days, "block_days")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    normalized_seed = int(seed)
    if normalized_seed < 0:
        raise ValueError("seed must be a non-negative integer")

    stakes, returns = _aggregate_days(daily_rows)
    day_count = len(stakes)
    if block_length > day_count:
        raise ValueError("block_days must not exceed the number of observed days")

    observed_stake = float(stakes.sum())
    observed_return = float(returns.sum())
    observed_roi = (
        float(observed_return / observed_stake) if observed_stake > 0.0 else None
    )
    block_count = (day_count + block_length - 1) // block_length
    start_count = day_count - block_length + 1
    rng = np.random.default_rng(normalized_seed)
    bootstrap_roi = np.empty(sample_count, dtype=np.float64)
    valid_count = 0

    for start in range(0, sample_count, step):
        current_size = min(step, sample_count - start)
        block_starts = rng.integers(
            0,
            start_count,
            size=(current_size, block_count),
        )
        sampled_stakes = np.zeros(current_size, dtype=np.float64)
        sampled_returns = np.zeros(current_size, dtype=np.float64)
        remaining_days = day_count
        for block_index in range(block_count):
            take_days = min(block_length, remaining_days)
            starts = block_starts[:, block_index]
            for offset in range(take_days):
                indices = starts + offset
                sampled_stakes += stakes[indices]
                sampled_returns += returns[indices]
            remaining_days -= take_days
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
        "block_days": block_length,
        "blocks_per_sample": block_count,
        "samples": sample_count,
        "valid_samples": valid_count,
        "stake_yen": observed_stake,
        "return_yen": observed_return,
        "profit_yen": float(observed_return - observed_stake),
        "roi": observed_roi,
        "roi_ci95_lower": roi_lower,
        "probability_roi_above_one": probability_above_one,
    }


def leave_one_venue_out_roi(
    rows: Iterable[Mapping[str, object]],
) -> dict[str, Any]:
    """Report observed ROI after removing each venue as a complete cluster."""

    try:
        iterator = iter(rows)
    except TypeError as exc:
        raise ValueError("rows must be an iterable of mappings") from exc

    by_venue: dict[str, tuple[list[float], list[float]]] = defaultdict(
        lambda: ([], [])
    )
    for row in iterator:
        if not isinstance(row, Mapping):
            raise ValueError("each row must be a mapping")
        _normalized_date(row.get("race_date"))
        venue = _normalized_venue(row.get("jcd"))
        by_venue[venue][0].append(_yen_value(row, "stake_yen"))
        by_venue[venue][1].append(_yen_value(row, "return_yen"))

    if not by_venue:
        raise ValueError("rows must not be empty")

    aggregates = {
        venue: (fsum(sorted(values[0])), fsum(sorted(values[1])))
        for venue, values in by_venue.items()
    }
    if any(
        not isfinite(stake)
        or not isfinite(returned)
        or stake > MAX_EXACT_YEN
        or returned > MAX_EXACT_YEN
        for stake, returned in aggregates.values()
    ):
        raise ValueError("venue aggregate exceeds the exact supported yen range")
    total_stake = fsum(sorted(stake for stake, _ in aggregates.values()))
    total_return = fsum(sorted(returned for _, returned in aggregates.values()))
    if total_stake > MAX_EXACT_YEN or total_return > MAX_EXACT_YEN:
        raise ValueError("observed aggregate exceeds the exact supported yen range")

    diagnostics = []
    for venue in sorted(aggregates):
        omitted_stake, omitted_return = aggregates[venue]
        remaining_stake = total_stake - omitted_stake
        remaining_return = total_return - omitted_return
        diagnostics.append(
            {
                "jcd": venue,
                "omitted_stake_yen": omitted_stake,
                "omitted_return_yen": omitted_return,
                "omitted_profit_yen": omitted_return - omitted_stake,
                "remaining_stake_yen": remaining_stake,
                "remaining_return_yen": remaining_return,
                "remaining_profit_yen": remaining_return - remaining_stake,
                "remaining_roi": (
                    float(remaining_return / remaining_stake)
                    if remaining_stake > 0.0
                    else None
                ),
            }
        )

    return {
        "venues": len(aggregates),
        "stake_yen": total_stake,
        "return_yen": total_return,
        "profit_yen": total_return - total_stake,
        "roi": float(total_return / total_stake) if total_stake > 0.0 else None,
        "leave_one_venue_out": diagnostics,
    }
