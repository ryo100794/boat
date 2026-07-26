from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .adaptive_allocation import validate_policy


@dataclass(frozen=True)
class PackedCandidates:
    """Compact, reusable ticket candidates for policy-only simulations."""

    dates: tuple[str, ...]
    offsets: np.ndarray
    evaluated_races: np.ndarray
    race_codes: np.ndarray
    estimated_odds: np.ndarray
    estimated_ev: np.ndarray
    probability: np.ndarray
    actual_payout_yen: np.ndarray
    hit: np.ndarray

    @property
    def tickets(self) -> int:
        return int(self.offsets[-1]) if len(self.offsets) else 0


def pack_candidates(
    candidates_by_date: Mapping[str, Sequence[Mapping[str, Any]]],
    evaluated_races_by_date: Mapping[str, int],
) -> PackedCandidates:
    dates = tuple(sorted(set(candidates_by_date) | set(evaluated_races_by_date)))
    offsets = [0]
    race_ids: list[str] = []
    odds: list[float] = []
    ev: list[float] = []
    probability: list[float] = []
    payout: list[int] = []
    hit: list[bool] = []
    for race_date in dates:
        rows = candidates_by_date.get(race_date, ())
        for row in rows:
            race_ids.append(str(row["race_id"]))
            odds.append(float(row["estimated_odds"]))
            ev.append(float(row["estimated_ev"]))
            probability.append(float(row["probability"]))
            payout.append(int(row["actual_payout_yen"]))
            hit.append(bool(row["hit"]))
        offsets.append(len(race_ids))
    unique_races = {race_id: index for index, race_id in enumerate(dict.fromkeys(race_ids))}
    return PackedCandidates(
        dates=dates,
        offsets=np.asarray(offsets, dtype=np.int64),
        evaluated_races=np.asarray(
            [evaluated_races_by_date.get(date, 0) for date in dates], dtype=np.int32
        ),
        race_codes=np.asarray([unique_races[value] for value in race_ids], dtype=np.int32),
        estimated_odds=np.asarray(odds, dtype=np.float32),
        estimated_ev=np.asarray(ev, dtype=np.float32),
        probability=np.asarray(probability, dtype=np.float32),
        actual_payout_yen=np.asarray(payout, dtype=np.int32),
        hit=np.asarray(hit, dtype=np.bool_),
    )


def evaluate_packed_policy(
    packed: PackedCandidates,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate one allocation policy without rebuilding ticket dictionaries."""
    daily_budget = int(policy["daily_budget_yen"])
    fractional_kelly = float(policy["fractional_kelly"])
    maximum_exposure = float(policy["max_daily_exposure_fraction"])
    minimum_exposure = float(policy["min_daily_exposure_fraction"])
    race_cap = float(policy["race_cap_fraction"])
    ticket_cap = float(policy["ticket_cap_fraction"])
    max_tickets_value = policy.get("max_daily_tickets")
    max_tickets = int(max_tickets_value) if max_tickets_value else None
    allocation_mode = str(policy["allocation_mode"])
    granularity = int(policy["stake_granularity_yen"])
    min_stake = int(policy["min_stake_yen"])
    validate_policy(
        daily_budget_yen=daily_budget,
        fractional_kelly=fractional_kelly,
        max_daily_exposure_fraction=maximum_exposure,
        min_daily_exposure_fraction=minimum_exposure,
        race_cap_fraction=race_cap,
        ticket_cap_fraction=ticket_cap,
        max_daily_tickets=max_tickets,
        allocation_mode=allocation_mode,
        stake_granularity_yen=granularity,
        min_stake_yen=min_stake,
    )

    totals = {
        "candidate_tickets": packed.tickets,
        "positive_edge_tickets": 0,
        "allocation_candidate_tickets": 0,
        "evaluated_races": int(packed.evaluated_races.sum()),
        "selected_races": 0,
        "tickets": 0,
        "hit_tickets": 0,
        "hit_races": 0,
        "stake_yen": 0,
        "return_yen": 0,
        "days_with_bets": 0,
        "winning_days": 0,
        "losing_days": 0,
        "breakeven_days": 0,
    }
    cumulative_profit = peak_profit = max_drawdown = 0
    daily: list[dict[str, Any]] = []
    for day_index, race_date in enumerate(packed.dates):
        start, stop = map(int, packed.offsets[day_index : day_index + 2])
        result = _evaluate_day(packed, start, stop, policy)
        result.update(
            race_date=race_date,
            evaluated_races=int(packed.evaluated_races[day_index]),
        )
        profit = int(result["return_yen"]) - int(result["stake_yen"])
        result["profit_yen"] = profit
        cumulative_profit += profit
        peak_profit = max(peak_profit, cumulative_profit)
        max_drawdown = max(max_drawdown, peak_profit - cumulative_profit)
        result["cumulative_profit_yen"] = cumulative_profit
        daily.append(result)
        for key in (
            "positive_edge_tickets", "allocation_candidate_tickets", "tickets",
            "selected_races", "hit_tickets", "hit_races", "stake_yen", "return_yen",
        ):
            totals[key] += int(result[key])
        if result["stake_yen"]:
            totals["days_with_bets"] += 1
            totals["winning_days" if profit > 0 else "losing_days" if profit < 0 else "breakeven_days"] += 1

    stake = int(totals["stake_yen"])
    returned = int(totals["return_yen"])
    totals.update(
        profit_yen=returned - stake,
        roi=returned / stake if stake else 0.0,
        ticket_hit_rate=(totals["hit_tickets"] / totals["tickets"] if totals["tickets"] else 0.0),
        race_hit_rate=(totals["hit_races"] / totals["selected_races"] if totals["selected_races"] else 0.0),
        max_drawdown_yen=max_drawdown,
        daily=daily,
    )
    return totals


def _evaluate_day(
    packed: PackedCandidates,
    start: int,
    stop: int,
    policy: Mapping[str, Any],
) -> dict[str, int]:
    odds = packed.estimated_odds[start:stop].astype(np.float64, copy=False)
    ev = packed.estimated_ev[start:stop].astype(np.float64, copy=False)
    probability = packed.probability[start:stop].astype(np.float64, copy=False)
    races = packed.race_codes[start:stop]
    hit = packed.hit[start:stop]
    payout = packed.actual_payout_yen[start:stop]
    edge = ev - 1.0
    valid = (odds > 1.0) & (edge > 0.0) & np.isfinite(odds) & np.isfinite(edge)
    indices = np.flatnonzero(valid)
    positive_count = int(indices.size)
    if not positive_count:
        return _empty_day(stop - start)

    kelly = edge[indices] / (odds[indices] - 1.0)
    valid_kelly = (kelly > 0.0) & np.isfinite(kelly)
    indices = indices[valid_kelly]
    kelly = kelly[valid_kelly]
    fractions = np.minimum(
        float(policy["ticket_cap_fraction"]),
        float(policy["fractional_kelly"]) * kelly,
    )
    max_tickets = int(policy.get("max_daily_tickets") or 0)
    if max_tickets and len(indices) > max_tickets:
        order = np.lexsort((-fractions, -probability[indices], -ev[indices]))
        indices, fractions = indices[order[:max_tickets]], fractions[order[:max_tickets]]

    maximum_exposure = float(policy["max_daily_exposure_fraction"])
    minimum_exposure = float(policy["min_daily_exposure_fraction"])
    ticket_cap = float(policy["ticket_cap_fraction"])
    if policy["allocation_mode"] == "normalized_kelly":
        total = float(fractions.sum())
        if 0.0 < total < minimum_exposure:
            fractions = np.minimum(ticket_cap, fractions * (min(maximum_exposure, minimum_exposure) / total))

    selected_races = races[indices]
    for race_code in np.unique(selected_races):
        mask = selected_races == race_code
        total = float(fractions[mask].sum())
        if total > float(policy["race_cap_fraction"]):
            fractions[mask] *= float(policy["race_cap_fraction"]) / total
    total = float(fractions.sum())
    if total > maximum_exposure:
        fractions *= maximum_exposure / total

    budget = int(policy["daily_budget_yen"])
    granularity = int(policy["stake_granularity_yen"])
    stakes = np.floor((budget * fractions) / granularity).astype(np.int64) * granularity
    if policy["allocation_mode"] == "normalized_kelly":
        _fill_normalized_stakes(
            stakes, fractions, selected_races, ev[indices], probability[indices], policy
        )
    accepted = stakes >= int(policy["min_stake_yen"])
    indices, selected_races, stakes = indices[accepted], selected_races[accepted], stakes[accepted]
    returns = np.where(hit[indices], stakes * payout[indices] // 100, 0)
    hit_mask = hit[indices]
    return {
        "candidate_tickets": stop - start,
        "positive_edge_tickets": positive_count,
        "allocation_candidate_tickets": int(len(fractions)),
        "tickets": int(len(indices)),
        "selected_races": int(np.unique(selected_races).size),
        "hit_tickets": int(hit_mask.sum()),
        "hit_races": int(np.unique(selected_races[hit_mask]).size),
        "stake_yen": int(stakes.sum()),
        "return_yen": int(returns.sum()),
    }


def _fill_normalized_stakes(
    stakes: np.ndarray,
    fractions: np.ndarray,
    races: np.ndarray,
    ev: np.ndarray,
    probability: np.ndarray,
    policy: Mapping[str, Any],
) -> None:
    granularity = int(policy["stake_granularity_yen"])
    budget = int(policy["daily_budget_yen"])
    daily_cap = int(np.floor(budget * float(policy["max_daily_exposure_fraction"]) / granularity) * granularity)
    target = min(daily_cap, int(np.floor(budget * float(policy["min_daily_exposure_fraction"]) / granularity) * granularity))
    ticket_cap = int(np.floor(budget * float(policy["ticket_cap_fraction"]) / granularity) * granularity)
    race_cap = int(np.floor(budget * float(policy["race_cap_fraction"]) / granularity) * granularity)
    unique, inverse = np.unique(races, return_inverse=True)
    del unique
    race_stakes = np.bincount(inverse, weights=stakes, minlength=int(inverse.max()) + 1).astype(np.int64)
    while int(stakes.sum()) < target:
        eligible = (
            (stakes + granularity <= ticket_cap)
            & (race_stakes[inverse] + granularity <= race_cap)
            & (int(stakes.sum()) + granularity <= daily_cap)
        )
        candidates = np.flatnonzero(eligible)
        if not len(candidates):
            break
        residual = budget * fractions[candidates] - stakes[candidates]
        order = np.lexsort((-probability[candidates], -ev[candidates], -residual))
        chosen = int(candidates[order[0]])
        stakes[chosen] += granularity
        race_stakes[inverse[chosen]] += granularity


def _empty_day(candidate_count: int) -> dict[str, int]:
    return {
        "candidate_tickets": candidate_count,
        "positive_edge_tickets": 0,
        "allocation_candidate_tickets": 0,
        "tickets": 0,
        "selected_races": 0,
        "hit_tickets": 0,
        "hit_races": 0,
        "stake_yen": 0,
        "return_yen": 0,
    }
