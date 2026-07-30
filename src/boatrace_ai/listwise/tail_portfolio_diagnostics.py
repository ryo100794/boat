from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from math import isfinite
from typing import Any, Iterable, Mapping

import numpy as np


DEFAULT_TAIL_ODDS = 101.0
DEFAULT_BOOTSTRAP_SAMPLES = 20_000
DEFAULT_SEED = 20260727


@dataclass(frozen=True)
class _Ticket:
    race_date: str
    race_id: str
    odds: float
    stake: float
    returned: float


def _normalized_date(value: object) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if not text:
        raise ValueError("date must not be empty")
    return text


def _finite_number(row: Mapping[str, object], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a finite number") from exc
    if not isfinite(value):
        raise ValueError(f"{key} must be a finite number")
    return value


def _normalize_rows(rows: Iterable[Mapping[str, object]]) -> list[_Ticket]:
    normalized: list[_Ticket] = []
    for row in rows:
        race_id = str(row.get("race_id", "")).strip()
        if not race_id:
            raise ValueError("race_id must not be empty")
        odds = _finite_number(row, "odds")
        stake = _finite_number(row, "stake")
        returned = _finite_number(row, "return")
        if odds <= 0.0:
            raise ValueError("odds must be greater than zero")
        if stake < 0.0:
            raise ValueError("stake must not be negative")
        if returned < 0.0:
            raise ValueError("return must not be negative")
        if stake == 0.0 and returned != 0.0:
            raise ValueError("return must be zero when stake is zero")
        normalized.append(
            _Ticket(
                race_date=_normalized_date(row.get("date", "")),
                race_id=race_id,
                odds=odds,
                stake=stake,
                returned=returned,
            )
        )
    return normalized


def _cluster_bootstrap_roi_lower_bound(
    tickets: list[_Ticket],
    *,
    samples: int,
    seed: int,
    chunk_size: int = 2_000,
) -> float | None:
    purchased = [ticket for ticket in tickets if ticket.stake > 0.0]
    if not purchased:
        return None

    daily: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for ticket in purchased:
        daily[ticket.race_date][0] += ticket.stake
        daily[ticket.race_date][1] += ticket.returned
    ordered_days = sorted(daily)
    stakes = np.asarray([daily[day][0] for day in ordered_days], dtype=np.float64)
    returns = np.asarray([daily[day][1] for day in ordered_days], dtype=np.float64)

    rng = np.random.default_rng(seed)
    bootstrap_roi = np.empty(samples, dtype=np.float64)
    cluster_count = len(ordered_days)
    for start in range(0, samples, chunk_size):
        stop = min(samples, start + chunk_size)
        sampled = rng.integers(
            0,
            cluster_count,
            size=(stop - start, cluster_count),
        )
        sampled_stake = stakes[sampled].sum(axis=1)
        sampled_return = returns[sampled].sum(axis=1)
        bootstrap_roi[start:stop] = sampled_return / sampled_stake
    return float(np.quantile(bootstrap_roi, 0.05))


def _segment_metrics(
    tickets: list[_Ticket],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    purchased = [ticket for ticket in tickets if ticket.stake > 0.0]
    stake = sum(ticket.stake for ticket in purchased)
    returned = sum(ticket.returned for ticket in purchased)
    hits = [ticket for ticket in purchased if ticket.returned > 0.0]
    roi = returned / stake if stake else None

    largest_hit = max(hits, key=lambda ticket: ticket.returned, default=None)
    return_without_largest_hit = (
        returned - largest_hit.returned if largest_hit is not None else returned
    )
    return {
        "status": "purchased" if purchased else "no_purchases",
        "tickets": len(purchased),
        "hits": len(hits),
        "hit_days": len({ticket.race_date for ticket in hits}),
        "stake": stake,
        "return": returned,
        "profit": returned - stake,
        "roi": roi,
        "roi_excluding_largest_hit": (
            return_without_largest_hit / stake if stake else None
        ),
        "largest_hit_return": (
            largest_hit.returned if largest_hit is not None else None
        ),
        "largest_hit_date": (
            largest_hit.race_date if largest_hit is not None else None
        ),
        "largest_hit_race_id": (
            largest_hit.race_id if largest_hit is not None else None
        ),
        "daily_cluster_bootstrap_roi_lower_95": (
            _cluster_bootstrap_roi_lower_bound(
                purchased,
                samples=bootstrap_samples,
                seed=seed,
            )
        ),
    }


def diagnose_tail_portfolio(
    rows: Iterable[Mapping[str, object]],
    *,
    tail_odds: float = DEFAULT_TAIL_ODDS,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Compare ordinary and long-shot purchased tickets without mutating input.

    The bootstrap samples active purchase dates as clusters.  The largest-hit
    diagnostic keeps every stake and removes only the largest payout, which is
    deliberately conservative when measuring dependence on a single windfall.
    """
    if not isfinite(float(tail_odds)) or float(tail_odds) <= 0.0:
        raise ValueError("tail_odds must be a finite positive number")
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")

    tickets = _normalize_rows(rows)
    purchased_count = sum(ticket.stake > 0.0 for ticket in tickets)
    if not tickets:
        status = "empty"
    elif not purchased_count:
        status = "no_purchases"
    else:
        status = "purchased"

    threshold = float(tail_odds)
    ordinary = [ticket for ticket in tickets if ticket.odds < threshold]
    tail = [ticket for ticket in tickets if ticket.odds >= threshold]
    return {
        "status": status,
        "input_rows": len(tickets),
        "purchased_tickets": purchased_count,
        "tail_odds_threshold": threshold,
        "bootstrap_samples": int(bootstrap_samples),
        "seed": int(seed),
        "normal": _segment_metrics(
            ordinary,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        "tail": _segment_metrics(
            tail,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
    }
