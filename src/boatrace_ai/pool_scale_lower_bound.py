from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from .joint_market_value import JointMarketScenario
from .parimutuel_settlement import ParimutuelSettlementRules


POOL_SCALE_METHOD = "displayed_odds_integer_feasibility_lower_bound_v1"
UNPRICED_MARKERS = frozenset({"", "-", "--", "---"})


@dataclass(frozen=True)
class PoolScaleLowerBound:
    """Smallest integer wager pool consistent with displayed decimal odds."""

    method: str
    total_face_units: int
    total_sales_yen: int
    distributable_pool_yen: int
    ticket_stakes_yen: Mapping[str, int]
    priced_outcomes: tuple[str, ...]
    unpriced_outcomes: tuple[str, ...]
    allocation_sha256: str


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _outcomes(
    displayed_odds: Mapping[str, object],
    ordinary_outcomes: Sequence[str] | None,
) -> tuple[str, ...]:
    if not isinstance(displayed_odds, Mapping) or not displayed_odds:
        raise ValueError("displayed_odds must be a non-empty mapping")
    if any(not isinstance(key, str) or not key for key in displayed_odds):
        raise ValueError("odds keys must be non-empty strings")
    outcomes = (
        tuple(displayed_odds)
        if ordinary_outcomes is None
        else tuple(ordinary_outcomes)
    )
    if not outcomes or len(set(outcomes)) != len(outcomes) or any(
        not isinstance(outcome, str) or not outcome for outcome in outcomes
    ):
        raise ValueError("ordinary_outcomes must be unique non-empty strings")
    if set(displayed_odds) != set(outcomes):
        raise ValueError("displayed odds must match ordinary outcomes")
    return outcomes


def _payout_per_face_unit(
    value: object,
    *,
    face_unit_yen: int,
    allow_unpriced: bool,
) -> int | None:
    if value is None or (
        isinstance(value, str) and value.strip() in UNPRICED_MARKERS
    ):
        if allow_unpriced:
            return None
        raise ValueError("unpriced odds require allow_unpriced=True")
    if isinstance(value, bool):
        raise ValueError("odds must be finite decimal multipliers")
    try:
        odds = Decimal(str(value).strip())
    except (InvalidOperation, ValueError, AttributeError) as exc:
        raise ValueError("odds must be finite decimal multipliers") from exc
    if not odds.is_finite() or odds < Decimal(1):
        raise ValueError("odds must be finite multipliers of at least 1.0")
    payout = odds * face_unit_yen
    integral = payout.to_integral_value()
    if payout != integral:
        raise ValueError("odds do not resolve to an integer face-unit payout")
    return int(integral)


def _allocation_hash(
    outcomes: Sequence[str],
    stakes_yen: Mapping[str, int],
) -> str:
    encoded = json.dumps(
        [[outcome, stakes_yen[outcome]] for outcome in outcomes],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def infer_minimum_pool_scale(
    displayed_odds: Mapping[str, object],
    *,
    ordinary_outcomes: Sequence[str] | None = None,
    rules: ParimutuelSettlementRules | None = None,
    allow_unpriced: bool = False,
    max_total_sales_yen: int = 200_000_000,
    batch_size: int = 8192,
) -> PoolScaleLowerBound:
    """Infer a conservative absolute scale without claiming the actual pool.

    The search uses integer 10-yen face units and the same statutory payout
    arithmetic as the settlement adapter. Unpriced outcomes are rejected by
    default so collector omissions cannot silently collapse the pool estimate.
    """
    if not isinstance(allow_unpriced, bool):
        raise ValueError("allow_unpriced must be boolean")
    settlement_rules = rules or ParimutuelSettlementRules()
    settlement_rules.validate()
    outcomes = _outcomes(displayed_odds, ordinary_outcomes)
    maximum_sales = _positive_integer(
        max_total_sales_yen, "max_total_sales_yen"
    )
    search_batch = _positive_integer(batch_size, "batch_size")
    face = settlement_rules.face_unit_yen
    if maximum_sales % face:
        raise ValueError("max_total_sales_yen must use face-unit increments")

    payout_by_outcome = {
        outcome: _payout_per_face_unit(
            displayed_odds[outcome],
            face_unit_yen=face,
            allow_unpriced=allow_unpriced,
        )
        for outcome in outcomes
    }
    priced = tuple(
        outcome for outcome in outcomes if payout_by_outcome[outcome] is not None
    )
    unpriced = tuple(outcome for outcome in outcomes if outcome not in priced)
    if not priced:
        raise ValueError("at least one outcome must have displayed odds")
    payout_values = [int(payout_by_outcome[outcome]) for outcome in priced]
    if any(value > np.iinfo(np.int64).max - 1 for value in payout_values):
        raise ValueError("displayed odds exceed the integer search range")
    payouts = np.asarray(payout_values, dtype=np.int64)
    numerator = settlement_rules.payout_rate_numerator
    denominator = settlement_rules.payout_rate_denominator
    maximum_units = maximum_sales // face
    if maximum_units > np.iinfo(np.int64).max // (face * numerator):
        raise ValueError("max_total_sales_yen exceeds the integer search range")
    first_units = max(
        len(priced),
        max(
            (int(payout) * denominator + face * numerator - 1)
            // (face * numerator)
            for payout in payouts
        ),
    )
    if first_units > maximum_units:
        raise ValueError("no odds-consistent pool found within max_total_sales_yen")

    selected_units: int | None = None
    selected_pool: int | None = None
    selected_lower: np.ndarray | None = None
    selected_upper: np.ndarray | None = None
    cursor = first_units
    while cursor <= maximum_units:
        stop = min(cursor + search_batch, maximum_units + 1)
        totals = np.arange(cursor, stop, dtype=np.int64)
        pools = totals * face * numerator // denominator
        lower = pools[:, None] // (payouts[None, :] + 1) + 1
        upper = pools[:, None] // payouts[None, :]
        minimum_return = payouts == face
        if np.any(minimum_return):
            upper[:, minimum_return] = totals[:, None]
        feasible = (
            np.all(lower <= upper, axis=1)
            & (np.sum(lower, axis=1) <= totals)
            & (np.sum(upper, axis=1) >= totals)
        )
        matches = np.flatnonzero(feasible)
        if matches.size:
            row = int(matches[0])
            selected_units = int(totals[row])
            selected_pool = int(pools[row])
            selected_lower = lower[row].copy()
            selected_upper = upper[row].copy()
            break
        cursor = stop
    if selected_units is None or selected_pool is None:
        raise ValueError("no odds-consistent pool found within max_total_sales_yen")
    assert selected_lower is not None and selected_upper is not None

    allocated = selected_lower
    remaining = selected_units - int(np.sum(allocated))
    for index in range(len(allocated)):
        addition = min(remaining, int(selected_upper[index] - allocated[index]))
        allocated[index] += addition
        remaining -= addition
        if remaining == 0:
            break
    if remaining:
        raise RuntimeError("feasible pool allocation could not be constructed")

    unit_allocation = {outcome: 0 for outcome in outcomes}
    unit_allocation.update({
        outcome: int(allocated[index]) for index, outcome in enumerate(priced)
    })
    for outcome in priced:
        units = unit_allocation[outcome]
        reproduced = max(face, selected_pool // units)
        if reproduced != payout_by_outcome[outcome]:
            raise RuntimeError("constructed allocation does not reproduce odds")
    stakes = {
        outcome: unit_allocation[outcome] * face for outcome in outcomes
    }
    return PoolScaleLowerBound(
        method=POOL_SCALE_METHOD,
        total_face_units=selected_units,
        total_sales_yen=selected_units * face,
        distributable_pool_yen=selected_pool,
        ticket_stakes_yen=stakes,
        priced_outcomes=priced,
        unpriced_outcomes=unpriced,
        allocation_sha256=_allocation_hash(outcomes, stakes),
    )


def attach_pool_scale_lower_bound(
    parameter_draws: Sequence[Sequence[JointMarketScenario]],
    *,
    displayed_odds: Mapping[str, object],
    ordinary_outcomes: Sequence[str] | None = None,
    odds_asof: str,
    rules: ParimutuelSettlementRules | None = None,
    allow_unpriced: bool = False,
    max_total_sales_yen: int = 200_000_000,
    batch_size: int = 8192,
) -> tuple[tuple[tuple[JointMarketScenario, ...], ...], PoolScaleLowerBound]:
    """Attach the conservative scale to generated paths before settlement."""
    if not parameter_draws or any(not draw for draw in parameter_draws):
        raise ValueError("parameter_draws must contain non-empty draws")
    if not isinstance(odds_asof, str) or not odds_asof.strip():
        raise ValueError("odds_asof must be a non-empty timestamp or label")
    forbidden = {"external_ticket_stakes_yen", "external_total_sales_yen"}
    for draw in parameter_draws:
        for scenario in draw:
            if forbidden & set(scenario.market_state):
                raise ValueError("absolute pool scale is already attached")
    lower_bound = infer_minimum_pool_scale(
        displayed_odds,
        ordinary_outcomes=ordinary_outcomes,
        rules=rules,
        allow_unpriced=allow_unpriced,
        max_total_sales_yen=max_total_sales_yen,
        batch_size=batch_size,
    )
    attached = tuple(
        tuple(
            JointMarketScenario(
                probabilities=scenario.probabilities,
                market_state={
                    **scenario.market_state,
                    "external_total_sales_yen": lower_bound.total_sales_yen,
                    "external_pool_scale_method": lower_bound.method,
                    "external_pool_scale_asof": odds_asof.strip(),
                    "external_pool_scale_allocation_sha256": (
                        lower_bound.allocation_sha256
                    ),
                },
                weight=scenario.weight,
            )
            for scenario in draw
        )
        for draw in parameter_draws
    )
    return attached, lower_bound


__all__ = [
    "POOL_SCALE_METHOD",
    "PoolScaleLowerBound",
    "attach_pool_scale_lower_bound",
    "infer_minimum_pool_scale",
]
