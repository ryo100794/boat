from __future__ import annotations

from dataclasses import dataclass, field
from math import fsum, isfinite
from typing import Any, Callable, Mapping, Sequence

import numpy as np


SIMPLEX_TOLERANCE = 1e-8


@dataclass(frozen=True)
class JointMarketScenario:
    """One shared future path for outcome probabilities and market state."""

    probabilities: Mapping[str, float]
    market_state: Mapping[str, Any] = field(default_factory=dict)
    weight: float = 1.0


PayoutModel = Callable[
    [JointMarketScenario, Mapping[str, int]], Mapping[str, float]
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


def validate_probability_simplex(
    probabilities: Mapping[str, float],
    *,
    tolerance: float = SIMPLEX_TOLERANCE,
) -> dict[str, float]:
    if not probabilities:
        raise ValueError("probabilities must not be empty")
    parsed = {
        str(combination): _finite_non_negative(value, "probability")
        for combination, value in probabilities.items()
    }
    total = fsum(parsed.values())
    if abs(total - 1.0) > tolerance:
        raise ValueError("probabilities must sum to one")
    return parsed


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


def _outer_summary(values: Sequence[float], alpha: float) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "lower_quantile": float(np.quantile(array, alpha)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def evaluate_joint_market_value(
    parameter_draws: Sequence[Sequence[JointMarketScenario]],
    *,
    bets_yen: Mapping[str, int],
    payout_model: PayoutModel,
    operational_margins: Mapping[str, float] | None = None,
    outer_alpha: float = 0.05,
    inner_tail_fraction: float | None = None,
    buy_margin: float = 0.0,
) -> dict[str, Any]:
    """Evaluate ticket value without assuming probability/price independence.

    The payout model receives the complete bet vector for every scenario. It is
    responsible for self-impact, refunds, rounding and special payouts. The
    lower tail, when requested, is taken over conditional expected edge paths;
    realized Bernoulli returns are intentionally not used as a purchase gate.
    """

    if not parameter_draws:
        raise ValueError("parameter_draws must not be empty")
    if not 0.0 < outer_alpha < 1.0:
        raise ValueError("outer_alpha must be in (0, 1)")
    threshold = _finite_non_negative(buy_margin, "buy_margin")
    parsed_bets: dict[str, int] = {}
    for combination, raw_value in bets_yen.items():
        if isinstance(raw_value, bool) or not isinstance(raw_value, int):
            raise ValueError("bets_yen values must be non-negative integers")
        if raw_value < 0:
            raise ValueError("bets_yen values must be non-negative integers")
        if raw_value > 0:
            parsed_bets[str(combination)] = raw_value
    if not parsed_bets:
        raise ValueError("at least one positive bet is required")
    margins = {
        combination: _finite_non_negative(value, "operational margin")
        for combination, value in (operational_margins or {}).items()
    }
    total_stake = float(sum(parsed_bets.values()))
    outer_ticket_values = {combination: [] for combination in parsed_bets}
    outer_portfolio_values: list[float] = []
    moments_by_draw: list[dict[str, Any]] = []

    for draw_index, draw in enumerate(parameter_draws):
        scenarios = list(draw)
        weights = _normalized_weights(scenarios)
        edge_paths = {
            combination: np.empty(len(scenarios), dtype=np.float64)
            for combination in parsed_bets
        }
        probability_paths = {
            combination: np.empty(len(scenarios), dtype=np.float64)
            for combination in parsed_bets
        }
        payout_paths = {
            combination: np.empty(len(scenarios), dtype=np.float64)
            for combination in parsed_bets
        }
        for scenario_index, scenario in enumerate(scenarios):
            probabilities = validate_probability_simplex(scenario.probabilities)
            missing = set(parsed_bets) - set(probabilities)
            if missing:
                raise ValueError(
                    "scenario is missing bet probabilities: "
                    + ", ".join(sorted(missing))
                )
            payouts = payout_model(scenario, parsed_bets)
            missing = set(parsed_bets) - set(payouts)
            if missing:
                raise ValueError(
                    "payout model is missing bet multipliers: "
                    + ", ".join(sorted(missing))
                )
            for combination in parsed_bets:
                probability = probabilities[combination]
                payout = _finite_non_negative(
                    payouts[combination], "payout multiplier"
                )
                probability_paths[combination][scenario_index] = probability
                payout_paths[combination][scenario_index] = payout
                edge_paths[combination][scenario_index] = (
                    probability * payout - 1.0 - margins.get(combination, 0.0)
                )

        draw_ticket_values: dict[str, float] = {}
        draw_moments: dict[str, Any] = {"draw": draw_index, "tickets": {}}
        for combination in parsed_bets:
            edges = edge_paths[combination]
            value = (
                float(np.dot(weights, edges))
                if inner_tail_fraction is None
                else _weighted_lower_tail_mean(
                    edges, weights, float(inner_tail_fraction)
                )
            )
            draw_ticket_values[combination] = value
            outer_ticket_values[combination].append(value)
            mean_probability = float(
                np.dot(weights, probability_paths[combination])
            )
            mean_payout = float(np.dot(weights, payout_paths[combination]))
            mean_product = float(
                np.dot(
                    weights,
                    probability_paths[combination] * payout_paths[combination],
                )
            )
            draw_moments["tickets"][combination] = {
                "mean_probability": mean_probability,
                "mean_payout_multiplier": mean_payout,
                "mean_probability_times_payout": mean_product,
                "probability_payout_covariance": (
                    mean_product - mean_probability * mean_payout
                ),
                "joint_expected_edge": (
                    mean_product - 1.0 - margins.get(combination, 0.0)
                ),
                "independence_approximation_edge": (
                    mean_probability * mean_payout
                    - 1.0
                    - margins.get(combination, 0.0)
                ),
            }
        outer_portfolio_values.append(
            fsum(
                parsed_bets[combination] * draw_ticket_values[combination]
                for combination in parsed_bets
            )
            / total_stake
        )
        moments_by_draw.append(draw_moments)

    tickets = {}
    for combination, values in outer_ticket_values.items():
        summary = _outer_summary(values, outer_alpha)
        summary["passes_purchase_gate"] = summary["lower_quantile"] > threshold
        tickets[combination] = summary
    portfolio = _outer_summary(outer_portfolio_values, outer_alpha)
    portfolio["passes_purchase_gate"] = portfolio["lower_quantile"] > threshold
    return {
        "definition": "E[pi_T * D_T(b) | F_t] - 1 - operational_margin",
        "parameter_draws": len(parameter_draws),
        "inner_aggregation": (
            "weighted_mean"
            if inner_tail_fraction is None
            else "weighted_lower_tail_mean"
        ),
        "inner_tail_fraction": inner_tail_fraction,
        "outer_alpha": outer_alpha,
        "buy_margin": threshold,
        "tickets": tickets,
        "portfolio": portfolio,
        "moments_by_draw": moments_by_draw,
    }


__all__ = [
    "JointMarketScenario",
    "PayoutModel",
    "evaluate_joint_market_value",
    "validate_probability_simplex",
]
