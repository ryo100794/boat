from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Iterable


DEFAULT_STAKE_UNIT_YEN = 100
DEFAULT_MAX_SEARCH_UNITS = 1_000


@dataclass(frozen=True, slots=True)
class MultinomialKellyCandidate:
    """One mutually exclusive race outcome offered at decimal odds."""

    selection: str
    probability: float
    final_odds: float


@dataclass(frozen=True, slots=True)
class MultinomialKellyAllocation:
    selection: str
    probability: float
    final_odds: float
    units: int
    stake_yen: int


@dataclass(frozen=True, slots=True)
class MultinomialKellyResult:
    bankroll_yen: int
    stake_unit_yen: int
    effective_cap_yen: int
    total_units: int
    total_stake_yen: int
    cash_yen: int
    expected_log_wealth: float
    expected_log_growth: float
    allocations: tuple[MultinomialKellyAllocation, ...]

    @property
    def purchased(self) -> tuple[MultinomialKellyAllocation, ...]:
        return tuple(allocation for allocation in self.allocations if allocation.units)

    def wealth_if(self, selection: str) -> float:
        """Return post-settlement wealth if ``selection`` wins."""

        for allocation in self.allocations:
            if allocation.selection == selection:
                return self.cash_yen + allocation.stake_yen * allocation.final_odds
        raise KeyError(selection)


def allocate_multinomial_kelly(
    bankroll_yen: int,
    candidates: Iterable[MultinomialKellyCandidate],
    *,
    stake_unit_yen: int = DEFAULT_STAKE_UNIT_YEN,
    race_cap_yen: int | None = None,
    daily_cap_yen: int | None = None,
    daily_staked_yen: int = 0,
    probability_tolerance: float = 1e-9,
    max_search_units: int = DEFAULT_MAX_SEARCH_UNITS,
) -> MultinomialKellyResult:
    """Maximize expected log wealth on the integer stake grid.

    The candidates must describe every mutually exclusive outcome of one race,
    so their probabilities must sum to one. Decimal odds include returned stake.
    Cash is retained whenever zero or partial investment is optimal.

    For a fixed total number of units, remaining cash is common to every
    outcome. The objective then becomes a separable concave resource allocation
    problem whose descending marginal-gain allocation is exact. Comparing all
    feasible totals therefore gives the global integer optimum without
    enumerating every stake vector.
    """

    _validate_positive_int("bankroll_yen", bankroll_yen)
    _validate_positive_int("stake_unit_yen", stake_unit_yen)
    _validate_nonnegative_int("daily_staked_yen", daily_staked_yen)
    _validate_positive_int("max_search_units", max_search_units)
    if race_cap_yen is not None:
        _validate_nonnegative_int("race_cap_yen", race_cap_yen)
    if daily_cap_yen is not None:
        _validate_nonnegative_int("daily_cap_yen", daily_cap_yen)
        if daily_staked_yen > daily_cap_yen:
            raise ValueError("daily_staked_yen must not exceed daily_cap_yen")
    if not math.isfinite(probability_tolerance) or probability_tolerance < 0.0:
        raise ValueError("probability_tolerance must be finite and nonnegative")

    prepared = _prepare_candidates(candidates, probability_tolerance)
    cap_yen = _effective_cap_yen(
        bankroll_yen,
        stake_unit_yen,
        race_cap_yen=race_cap_yen,
        daily_cap_yen=daily_cap_yen,
        daily_staked_yen=daily_staked_yen,
    )
    cap_units = cap_yen // stake_unit_yen
    if cap_units > max_search_units:
        raise ValueError(
            f"effective cap requires {cap_units} units; "
            f"increase max_search_units from {max_search_units} explicitly"
        )

    best_units = [0] * len(prepared)
    best_total_units = 0
    probability_sum = math.fsum(candidate.probability for candidate in prepared)
    baseline_objective = probability_sum * math.log(bankroll_yen)
    best_objective = baseline_objective

    for total_units in range(1, cap_units + 1):
        cash_yen = bankroll_yen - total_units * stake_unit_yen
        units, objective = _best_allocation_for_total(
            prepared,
            total_units=total_units,
            cash_yen=cash_yen,
            stake_unit_yen=stake_unit_yen,
        )
        # Ascending total stake and canonical candidate ordering define ties.
        if objective > best_objective:
            best_objective = objective
            best_total_units = total_units
            best_units = units

    total_stake_yen = best_total_units * stake_unit_yen
    allocations = tuple(
        MultinomialKellyAllocation(
            selection=candidate.selection,
            probability=candidate.probability,
            final_odds=candidate.final_odds,
            units=units,
            stake_yen=units * stake_unit_yen,
        )
        for candidate, units in zip(prepared, best_units, strict=True)
    )
    return MultinomialKellyResult(
        bankroll_yen=bankroll_yen,
        stake_unit_yen=stake_unit_yen,
        effective_cap_yen=cap_yen,
        total_units=best_total_units,
        total_stake_yen=total_stake_yen,
        cash_yen=bankroll_yen - total_stake_yen,
        expected_log_wealth=best_objective,
        expected_log_growth=best_objective - baseline_objective,
        allocations=allocations,
    )


def _prepare_candidates(
    candidates: Iterable[MultinomialKellyCandidate],
    probability_tolerance: float,
) -> tuple[MultinomialKellyCandidate, ...]:
    prepared = tuple(candidates)
    if not prepared:
        raise ValueError("candidates must not be empty")
    if any(not isinstance(candidate, MultinomialKellyCandidate) for candidate in prepared):
        raise TypeError("every candidate must be a MultinomialKellyCandidate")

    selections: set[str] = set()
    for candidate in prepared:
        if not isinstance(candidate.selection, str) or not candidate.selection:
            raise ValueError("candidate selection must be a nonempty string")
        if candidate.selection in selections:
            raise ValueError(f"duplicate selection: {candidate.selection}")
        selections.add(candidate.selection)
        if not math.isfinite(candidate.probability) or candidate.probability < 0.0:
            raise ValueError("candidate probabilities must be finite and nonnegative")
        if not math.isfinite(candidate.final_odds) or candidate.final_odds <= 0.0:
            raise ValueError("candidate final_odds must be finite and positive")

    probability_sum = math.fsum(candidate.probability for candidate in prepared)
    if not math.isclose(
        probability_sum,
        1.0,
        rel_tol=0.0,
        abs_tol=probability_tolerance,
    ):
        raise ValueError(
            f"candidate probabilities must sum to 1; got {probability_sum:.17g}"
        )
    return tuple(sorted(prepared, key=lambda candidate: candidate.selection))


def _effective_cap_yen(
    bankroll_yen: int,
    stake_unit_yen: int,
    *,
    race_cap_yen: int | None,
    daily_cap_yen: int | None,
    daily_staked_yen: int,
) -> int:
    # Keeping at least one yen means every outcome has strictly positive wealth,
    # including outcomes receiving no stake.
    limits = [bankroll_yen - 1]
    if race_cap_yen is not None:
        limits.append(race_cap_yen)
    if daily_cap_yen is not None:
        limits.append(daily_cap_yen - daily_staked_yen)
    return max(0, min(limits) // stake_unit_yen * stake_unit_yen)


def _best_allocation_for_total(
    candidates: tuple[MultinomialKellyCandidate, ...],
    *,
    total_units: int,
    cash_yen: int,
    stake_unit_yen: int,
) -> tuple[list[int], float]:
    units = [0] * len(candidates)
    heap: list[tuple[float, str, int]] = []
    for index, candidate in enumerate(candidates):
        gain = _marginal_gain(
            candidate,
            current_units=0,
            cash_yen=cash_yen,
            stake_unit_yen=stake_unit_yen,
        )
        heap.append((-gain, candidate.selection, index))
    heapq.heapify(heap)

    objective = (
        math.fsum(candidate.probability for candidate in candidates)
        * math.log(cash_yen)
    )
    for _ in range(total_units):
        negative_gain, _selection, index = heapq.heappop(heap)
        objective -= negative_gain
        units[index] += 1
        candidate = candidates[index]
        next_gain = _marginal_gain(
            candidate,
            current_units=units[index],
            cash_yen=cash_yen,
            stake_unit_yen=stake_unit_yen,
        )
        heapq.heappush(heap, (-next_gain, candidate.selection, index))
    return units, objective


def _marginal_gain(
    candidate: MultinomialKellyCandidate,
    *,
    current_units: int,
    cash_yen: int,
    stake_unit_yen: int,
) -> float:
    current_wealth = (
        cash_yen + current_units * stake_unit_yen * candidate.final_odds
    )
    next_wealth = current_wealth + stake_unit_yen * candidate.final_odds
    return candidate.probability * math.log(next_wealth / current_wealth)


def _validate_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_nonnegative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
