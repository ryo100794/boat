from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from math import fsum, isfinite, log
from typing import Any, Callable, Mapping, Sequence

import numpy as np


SIMPLEX_TOLERANCE = 1e-8
MAX_EXACT_YEN = 2**53 - 1
TRIFECTA_OUTCOMES = tuple(
    "-".join(str(lane) for lane in order)
    for order in permutations(range(1, 7), 3)
)


@dataclass(frozen=True)
class JointMarketScenario:
    """One already-generated joint path for probabilities and market state."""

    probabilities: Mapping[str, float]
    market_state: Mapping[str, Any] = field(default_factory=dict)
    weight: float = 1.0


GrossPayoffModel = Callable[
    [JointMarketScenario, Mapping[str, int]],
    Mapping[str, Mapping[str, int]],
]


def _finite_non_negative(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and non-negative")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and non-negative") from exc
    if not isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return parsed


def _non_negative_yen(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a non-negative integer yen amount")
    parsed = int(value)
    if parsed < 0 or parsed > MAX_EXACT_YEN:
        raise ValueError(f"{name} must be a non-negative integer yen amount")
    return parsed


def validate_probability_simplex(
    probabilities: Mapping[str, float],
    *,
    expected_outcomes: Sequence[str] | None = None,
    tolerance: float = SIMPLEX_TOLERANCE,
) -> dict[str, float]:
    if not probabilities:
        raise ValueError("probabilities must not be empty")
    if any(not isinstance(key, str) or not key for key in probabilities):
        raise ValueError("probability keys must be non-empty strings")
    parsed = {
        key: _finite_non_negative(value, "probability")
        for key, value in probabilities.items()
    }
    if expected_outcomes is not None:
        if any(not isinstance(value, str) or not value for value in expected_outcomes):
            raise ValueError("expected_outcomes must be non-empty strings")
        expected = set(expected_outcomes)
        if len(expected) != len(expected_outcomes):
            raise ValueError("expected_outcomes must be unique")
        if set(parsed) != expected:
            raise ValueError("probability outcomes do not match the expected set")
    if abs(fsum(parsed.values()) - 1.0) > tolerance:
        raise ValueError("probabilities must sum to one")
    return parsed


def validate_trifecta_probability_simplex(
    probabilities: Mapping[str, float],
) -> dict[str, float]:
    return validate_probability_simplex(
        probabilities, expected_outcomes=TRIFECTA_OUTCOMES
    )


def _normalized_weights(scenarios: Sequence[JointMarketScenario]) -> np.ndarray:
    if not scenarios:
        raise ValueError("each parameter draw must contain at least one scenario")
    weights = np.asarray(
        [_finite_non_negative(row.weight, "scenario weight") for row in scenarios],
        dtype=np.float64,
    )
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("scenario weights must have positive total mass")
    return weights / total


def _weighted_lower_tail_mean(
    values: np.ndarray,
    weights: np.ndarray,
    fraction: float,
) -> float:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("inner_tail_fraction must be in (0, 1]")
    order = np.argsort(values, kind="stable")
    remaining = float(fraction)
    total = 0.0
    for index in order:
        mass = min(remaining, float(weights[index]))
        total += mass * float(values[index])
        remaining -= mass
        if remaining <= 1e-15:
            break
    return total / fraction


def _aggregate_path(
    values: np.ndarray,
    weights: np.ndarray,
    inner_tail_fraction: float | None,
) -> float:
    return (
        float(np.dot(weights, values))
        if inner_tail_fraction is None
        else _weighted_lower_tail_mean(values, weights, inner_tail_fraction)
    )


def _outer_summary(values: Sequence[float], alpha: float) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "lower_quantile": float(
            np.quantile(array, alpha, method="inverted_cdf")
        ),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "quantile_method": "inverted_cdf",
    }


def _validate_bets(bets_yen: Mapping[str, int]) -> dict[str, int]:
    parsed = {}
    for combination, amount in bets_yen.items():
        if not isinstance(combination, str) or not combination:
            raise ValueError("bet keys must be non-empty strings")
        value = _non_negative_yen(amount, "bet")
        if value > 0:
            parsed[combination] = value
    if not parsed:
        raise ValueError("at least one positive bet is required")
    return parsed


def _scenario_paths(
    scenarios: Sequence[JointMarketScenario],
    *,
    bets: Mapping[str, int],
    gross_payoff_model: GrossPayoffModel,
    costs: Mapping[str, int],
    expected_outcomes: Sequence[str] | None,
    include_ticket_diagnostics: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    weights = _normalized_weights(scenarios)
    ticket_edges = (
        {ticket: np.empty(len(scenarios), dtype=np.float64) for ticket in bets}
        if include_ticket_diagnostics
        else {}
    )
    portfolio_edges = np.empty(len(scenarios), dtype=np.float64)
    probability_paths = (
        {ticket: np.empty(len(scenarios), dtype=np.float64) for ticket in bets}
        if include_ticket_diagnostics
        else {}
    )
    winning_multiplier_paths = (
        {ticket: np.empty(len(scenarios), dtype=np.float64) for ticket in bets}
        if include_ticket_diagnostics
        else {}
    )
    other_receipt_paths = (
        {ticket: np.empty(len(scenarios), dtype=np.float64) for ticket in bets}
        if include_ticket_diagnostics
        else {}
    )
    total_stake = sum(bets.values())
    total_cost = sum(costs.get(ticket, 0) for ticket in bets)

    for scenario_index, scenario in enumerate(scenarios):
        probabilities = validate_probability_simplex(
            scenario.probabilities, expected_outcomes=expected_outcomes
        )
        missing_probabilities = set(bets) - set(probabilities)
        if missing_probabilities:
            raise ValueError(
                "scenario is missing bet outcomes: "
                + ", ".join(sorted(missing_probabilities))
            )
        payoff_table = gross_payoff_model(scenario, bets)
        if set(payoff_table) != set(bets):
            raise ValueError(
                "payoff model ticket keys must match the complete bet vector"
            )
        portfolio_gross = 0.0
        for ticket, stake in bets.items():
            receipts = payoff_table[ticket]
            if not isinstance(receipts, Mapping):
                raise ValueError("payoff table values must map outcomes to yen")
            if any(not isinstance(state, str) or not state for state in receipts):
                raise ValueError("payoff terminal states must be non-empty strings")
            unknown = set(receipts) - set(probabilities)
            if unknown:
                raise ValueError(
                    "payoff model returned unknown terminal states: "
                    + ", ".join(sorted(unknown))
                )
            parsed_receipts = {
                state: _non_negative_yen(amount, "gross receipt")
                for state, amount in receipts.items()
            }
            gross = fsum(
                probabilities[state] * receipt
                for state, receipt in parsed_receipts.items()
            )
            portfolio_gross += gross
            if include_ticket_diagnostics:
                ticket_edges[ticket][scenario_index] = (
                    gross - stake - costs.get(ticket, 0)
                ) / stake
                hit_receipt = parsed_receipts.get(ticket, 0)
                hit_probability = probabilities[ticket]
                probability_paths[ticket][scenario_index] = hit_probability
                winning_multiplier_paths[ticket][scenario_index] = (
                    hit_receipt / stake
                )
                other_receipt_paths[ticket][scenario_index] = (
                    gross - hit_probability * hit_receipt
                ) / stake
        portfolio_edges[scenario_index] = (
            portfolio_gross - total_stake - total_cost
        ) / total_stake
    diagnostics = {
        "probabilities": probability_paths,
        "winning_multipliers": winning_multiplier_paths,
        "other_receipts_per_yen": other_receipt_paths,
    }
    return weights, ticket_edges, portfolio_edges, diagnostics


def evaluate_joint_market_value(
    parameter_draws: Sequence[Sequence[JointMarketScenario]],
    *,
    bets_yen: Mapping[str, int],
    gross_payoff_model: GrossPayoffModel,
    operational_costs_yen: Mapping[str, int] | None = None,
    expected_outcomes: Sequence[str] | None = None,
    outer_alpha: float = 0.05,
    inner_tail_fraction: float | None = None,
    buy_margin: float = 0.0,
    minimum_outer_draws: int = 20,
    minimum_inner_tail_effective_samples: float = 5.0,
    include_marginal_contributions: bool = True,
    include_ticket_diagnostics: bool = True,
) -> dict[str, Any]:
    """Evaluate already-generated joint scenarios; this does not generate them."""
    if not parameter_draws:
        raise ValueError("parameter_draws must not be empty")
    if not 0.0 < outer_alpha < 1.0:
        raise ValueError("outer_alpha must be in (0, 1)")
    if isinstance(minimum_outer_draws, bool) or not isinstance(
        minimum_outer_draws, int
    ) or minimum_outer_draws < 1:
        raise ValueError("minimum_outer_draws must be positive")
    if not isinstance(include_marginal_contributions, bool):
        raise ValueError("include_marginal_contributions must be boolean")
    if not isinstance(include_ticket_diagnostics, bool):
        raise ValueError("include_ticket_diagnostics must be boolean")
    minimum_tail_ess = _finite_non_negative(
        minimum_inner_tail_effective_samples,
        "minimum_inner_tail_effective_samples",
    )
    threshold = _finite_non_negative(buy_margin, "buy_margin")
    bets = _validate_bets(bets_yen)
    costs = {
        ticket: _non_negative_yen(value, "operational cost")
        for ticket, value in (operational_costs_yen or {}).items()
    }
    unknown_costs = set(costs) - set(bets)
    if unknown_costs:
        raise ValueError("operational costs contain unknown tickets")
    outer_ticket_values = (
        {ticket: [] for ticket in bets} if include_ticket_diagnostics else {}
    )
    outer_portfolio_values: list[float] = []
    moments_by_draw = []
    inner_effective_samples = []

    for draw_index, draw in enumerate(parameter_draws):
        scenarios = list(draw)
        weights, ticket_paths, portfolio_path, diagnostics = _scenario_paths(
            scenarios,
            bets=bets,
            gross_payoff_model=gross_payoff_model,
            costs=costs,
            expected_outcomes=expected_outcomes,
            include_ticket_diagnostics=include_ticket_diagnostics,
        )
        ess = 1.0 / float(np.dot(weights, weights))
        inner_effective_samples.append(ess)
        draw_moments = {"draw": draw_index, "tickets": {}}
        for ticket in bets if include_ticket_diagnostics else ():
            outer_ticket_values[ticket].append(
                _aggregate_path(ticket_paths[ticket], weights, inner_tail_fraction)
            )
            probabilities = diagnostics["probabilities"][ticket]
            multipliers = diagnostics["winning_multipliers"][ticket]
            other_receipts = diagnostics["other_receipts_per_yen"][ticket]
            mean_probability = float(np.dot(weights, probabilities))
            mean_multiplier = float(np.dot(weights, multipliers))
            mean_product = float(np.dot(weights, probabilities * multipliers))
            mean_other = float(np.dot(weights, other_receipts))
            draw_moments["tickets"][ticket] = {
                "mean_probability": mean_probability,
                "mean_winning_multiplier": mean_multiplier,
                "mean_hit_probability_times_multiplier": mean_product,
                "probability_multiplier_covariance": (
                    mean_product - mean_probability * mean_multiplier
                ),
                "mean_other_terminal_receipts_per_yen": mean_other,
                "joint_expected_edge": float(np.dot(weights, ticket_paths[ticket])),
                "ordinary_hit_independence_approximation_edge": (
                    mean_probability * mean_multiplier
                    + mean_other
                    - 1.0
                    - costs.get(ticket, 0) / bets[ticket]
                ),
            }
        outer_portfolio_values.append(
            _aggregate_path(portfolio_path, weights, inner_tail_fraction)
        )
        if include_ticket_diagnostics:
            moments_by_draw.append(draw_moments)

    tickets = {
        ticket: {
            **_outer_summary(values, outer_alpha),
            "role": "diagnostic_only_not_an_independent_purchase_command",
        }
        for ticket, values in outer_ticket_values.items()
    }
    portfolio = _outer_summary(outer_portfolio_values, outer_alpha)
    tail_effective_samples = (
        min(inner_effective_samples) * inner_tail_fraction
        if inner_tail_fraction is not None
        else None
    )
    gate_reasons = []
    if len(parameter_draws) < minimum_outer_draws:
        gate_reasons.append("insufficient_outer_parameter_draws")
    if tail_effective_samples is not None and tail_effective_samples < minimum_tail_ess:
        gate_reasons.append("insufficient_inner_tail_effective_samples")
    gate_evaluable = not gate_reasons
    portfolio["purchase_gate_evaluable"] = gate_evaluable
    portfolio["passes_purchase_gate"] = bool(
        gate_evaluable and portfolio["lower_quantile"] > threshold
    )
    portfolio["purchase_gate_reasons"] = gate_reasons

    marginal_contributions = {}
    for removed_ticket in bets if include_marginal_contributions else ():
        reduced_bets = {
            ticket: amount for ticket, amount in bets.items() if ticket != removed_ticket
        }
        if not reduced_bets:
            marginal_contributions[removed_ticket] = {
                "available": False,
                "reason": "removing_the_only_ticket_leaves_no_portfolio",
            }
            continue
        reduced_values = []
        reduced_costs = {
            ticket: amount for ticket, amount in costs.items() if ticket in reduced_bets
        }
        for draw in parameter_draws:
            weights, _tickets, portfolio_path, _diagnostics = _scenario_paths(
                list(draw),
                bets=reduced_bets,
                gross_payoff_model=gross_payoff_model,
                costs=reduced_costs,
                expected_outcomes=expected_outcomes,
                include_ticket_diagnostics=False,
            )
            reduced_values.append(
                _aggregate_path(portfolio_path, weights, inner_tail_fraction)
            )
        differences = [
            full - reduced
            for full, reduced in zip(outer_portfolio_values, reduced_values)
        ]
        marginal_contributions[removed_ticket] = {
            "available": True,
            "definition": "V(full_bet_vector)-V(vector_without_ticket)",
            **_outer_summary(differences, outer_alpha),
        }

    return {
        "version": "joint_market_value_evaluator_v0",
        "scope": "evaluates_pre_generated_joint_scenarios_only",
        "definition": "E[gross_receipt(b)|F_t]/stake - 1 - costs/stake",
        "parameter_draws": len(parameter_draws),
        "minimum_outer_draws": minimum_outer_draws,
        "inner_aggregation": (
            "weighted_mean"
            if inner_tail_fraction is None
            else "portfolio_path_weighted_lower_tail_mean"
        ),
        "inner_tail_fraction": inner_tail_fraction,
        "inner_effective_samples_min": min(inner_effective_samples),
        "inner_tail_effective_samples_min": tail_effective_samples,
        "minimum_inner_tail_effective_samples": minimum_tail_ess,
        "outer_alpha": outer_alpha,
        "outer_quantile_method": "inverted_cdf",
        "buy_margin": threshold,
        "tickets": tickets,
        "ticket_diagnostics_computed": include_ticket_diagnostics,
        "portfolio": portfolio,
        "marginal_contributions_computed": include_marginal_contributions,
        "marginal_contributions": marginal_contributions,
        "moments_by_draw": moments_by_draw,
    }


def evaluate_joint_bankroll_growth(
    parameter_draws: Sequence[Sequence[JointMarketScenario]],
    *,
    bets_yen: Mapping[str, int],
    gross_payoff_model: GrossPayoffModel,
    available_bankroll_yen: int,
    operational_costs_yen: Mapping[str, int] | None = None,
    expected_outcomes: Sequence[str] | None = None,
    outer_alpha: float = 0.05,
    inner_tail_fraction: float | None = None,
    minimum_outer_draws: int = 20,
    minimum_inner_tail_effective_samples: float = 5.0,
    minimum_terminal_wealth_yen: int = 1,
) -> dict[str, Any]:
    """Evaluate conservative expected log growth after integer settlement."""
    if not parameter_draws:
        raise ValueError("parameter_draws must not be empty")
    if not 0.0 < outer_alpha < 1.0:
        raise ValueError("outer_alpha must be in (0, 1)")
    if isinstance(minimum_outer_draws, bool) or not isinstance(
        minimum_outer_draws, int
    ) or minimum_outer_draws < 1:
        raise ValueError("minimum_outer_draws must be positive")
    bankroll = _non_negative_yen(available_bankroll_yen, "available bankroll")
    wealth_floor = _non_negative_yen(
        minimum_terminal_wealth_yen, "minimum terminal wealth"
    )
    if bankroll < 1 or wealth_floor < 1 or wealth_floor > bankroll:
        raise ValueError(
            "bankroll and minimum terminal wealth must be positive, "
            "with the floor no larger than bankroll"
        )
    minimum_tail_ess = _finite_non_negative(
        minimum_inner_tail_effective_samples,
        "minimum_inner_tail_effective_samples",
    )
    bets = _validate_bets(bets_yen)
    costs = {
        ticket: _non_negative_yen(value, "operational cost")
        for ticket, value in (operational_costs_yen or {}).items()
    }
    if set(costs) - set(bets):
        raise ValueError("operational costs contain unknown tickets")
    total_outlay = sum(bets.values()) + sum(costs.values())
    if total_outlay > bankroll:
        raise ValueError("bet stakes and costs exceed available bankroll")

    outer_values: list[float] = []
    inner_effective_samples: list[float] = []
    ruin_probability_upper = 0.0
    for draw in parameter_draws:
        scenarios = list(draw)
        weights = _normalized_weights(scenarios)
        inner_effective_samples.append(1.0 / float(np.dot(weights, weights)))
        scenario_growth = np.empty(len(scenarios), dtype=np.float64)
        draw_ruin_probability = 0.0
        for scenario_index, scenario in enumerate(scenarios):
            probabilities = validate_probability_simplex(
                scenario.probabilities, expected_outcomes=expected_outcomes
            )
            payoff_table = gross_payoff_model(scenario, bets)
            if set(payoff_table) != set(bets):
                raise ValueError(
                    "payoff model ticket keys must match the complete bet vector"
                )
            gross_by_outcome: dict[str, int] = {}
            for ticket, receipts in payoff_table.items():
                if not isinstance(receipts, Mapping):
                    raise ValueError("payoff table values must map outcomes to yen")
                unknown = set(receipts) - set(probabilities)
                if unknown:
                    raise ValueError(
                        "payoff model returned unknown terminal states: "
                        + ", ".join(sorted(unknown))
                    )
                for outcome, amount in receipts.items():
                    gross_by_outcome[outcome] = (
                        gross_by_outcome.get(outcome, 0)
                        + _non_negative_yen(amount, "gross receipt")
                    )
            cash_after_outlay = bankroll - total_outlay
            base_wealth = max(wealth_floor, cash_after_outlay)
            base_growth = log(base_wealth / bankroll)
            expected_growth = base_growth
            positive_receipt_probability = 0.0
            for outcome, gross in gross_by_outcome.items():
                if gross <= 0:
                    continue
                probability = probabilities[outcome]
                positive_receipt_probability += probability
                expected_growth += probability * (
                    log(
                        max(wealth_floor, cash_after_outlay + gross)
                        / bankroll
                    )
                    - base_growth
                )
            ruin_probability = (
                max(0.0, 1.0 - positive_receipt_probability)
                if cash_after_outlay <= 0
                else 0.0
            )
            scenario_growth[scenario_index] = expected_growth
            draw_ruin_probability += float(weights[scenario_index]) * ruin_probability
        ruin_probability_upper = max(
            ruin_probability_upper, draw_ruin_probability
        )
        outer_values.append(
            _aggregate_path(scenario_growth, weights, inner_tail_fraction)
        )

    tail_effective_samples = (
        min(inner_effective_samples) * inner_tail_fraction
        if inner_tail_fraction is not None
        else None
    )
    gate_reasons = []
    if len(parameter_draws) < minimum_outer_draws:
        gate_reasons.append("insufficient_outer_parameter_draws")
    if tail_effective_samples is not None and tail_effective_samples < minimum_tail_ess:
        gate_reasons.append("insufficient_inner_tail_effective_samples")
    summary = _outer_summary(outer_values, outer_alpha)
    summary.update({
        "purchase_gate_evaluable": not gate_reasons,
        "passes_growth_gate": bool(
            not gate_reasons and summary["lower_quantile"] > 0.0
        ),
        "purchase_gate_reasons": gate_reasons,
        "maximum_conditional_ruin_probability": ruin_probability_upper,
    })
    return {
        "version": "joint_bankroll_growth_evaluator_v1",
        "definition": (
            "lower parameter quantile of scenario-tail expected "
            "log(terminal_wealth/available_bankroll)"
        ),
        "available_bankroll_yen": bankroll,
        "total_outlay_yen": total_outlay,
        "minimum_terminal_wealth_yen": wealth_floor,
        "inner_aggregation": (
            "weighted_mean"
            if inner_tail_fraction is None
            else "weighted_lower_tail_mean"
        ),
        "inner_tail_fraction": inner_tail_fraction,
        "inner_effective_samples_min": min(inner_effective_samples),
        "inner_tail_effective_samples_min": tail_effective_samples,
        "outer_alpha": outer_alpha,
        "outer_quantile_method": "inverted_cdf",
        "growth": summary,
    }


__all__ = [
    "GrossPayoffModel",
    "JointMarketScenario",
    "TRIFECTA_OUTCOMES",
    "evaluate_joint_bankroll_growth",
    "evaluate_joint_market_value",
    "validate_probability_simplex",
    "validate_trifecta_probability_simplex",
]
