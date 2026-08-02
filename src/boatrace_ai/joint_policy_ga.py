from __future__ import annotations

from dataclasses import dataclass
import json
import math
import random
from typing import Any, Mapping, Sequence

from .genetic_search import GeneticSearchSettings, evolve_population
from .joint_market_value import (
    GrossPayoffModel,
    JointMarketScenario,
    evaluate_joint_bankroll_growth,
    evaluate_joint_market_value,
)


@dataclass(frozen=True)
class JointPortfolioGenome:
    stake_units: tuple[int, ...]


@dataclass(frozen=True)
class JointPolicySearchConfig:
    purchase_unit_yen: int = 100
    maximum_portfolio_stake_yen: int = 10_000
    maximum_ticket_stake_yen: int = 5_000
    maximum_selected_tickets: int = 12
    outer_alpha: float = 0.05
    inner_tail_fraction: float | None = 0.10
    buy_margin: float = 0.0
    minimum_outer_draws: int = 20
    minimum_inner_tail_effective_samples: float = 5.0

    def validate(self, *, available_bankroll_yen: int) -> None:
        integer_fields = {
            "purchase_unit_yen": self.purchase_unit_yen,
            "maximum_portfolio_stake_yen": self.maximum_portfolio_stake_yen,
            "maximum_ticket_stake_yen": self.maximum_ticket_stake_yen,
            "maximum_selected_tickets": self.maximum_selected_tickets,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in integer_fields.values()
        ):
            raise ValueError("joint policy integer limits must be positive")
        if isinstance(available_bankroll_yen, bool) or not isinstance(
            available_bankroll_yen, int
        ) or available_bankroll_yen < self.purchase_unit_yen:
            raise ValueError("available bankroll must cover one purchase unit")
        if self.maximum_portfolio_stake_yen % self.purchase_unit_yen:
            raise ValueError("maximum portfolio stake must use purchase units")
        if self.maximum_ticket_stake_yen % self.purchase_unit_yen:
            raise ValueError("maximum ticket stake must use purchase units")
        if not 0.0 < self.outer_alpha < 1.0:
            raise ValueError("outer_alpha must be in (0, 1)")
        if self.inner_tail_fraction is not None and not (
            0.0 < self.inner_tail_fraction <= 1.0
        ):
            raise ValueError("inner_tail_fraction must be in (0, 1]")
        if not math.isfinite(self.buy_margin) or self.buy_margin < 0.0:
            raise ValueError("buy_margin must be finite and non-negative")
        if isinstance(self.minimum_outer_draws, bool) or not isinstance(
            self.minimum_outer_draws, int
        ) or self.minimum_outer_draws < 1:
            raise ValueError("minimum_outer_draws must be positive")
        if not math.isfinite(self.minimum_inner_tail_effective_samples) or (
            self.minimum_inner_tail_effective_samples < 0.0
        ):
            raise ValueError(
                "minimum inner tail effective samples must be non-negative"
            )


def _repair_units(
    units: Sequence[int],
    *,
    budget_units: int,
    ticket_limit_units: int,
    maximum_selected_tickets: int,
) -> JointPortfolioGenome:
    repaired = [max(0, min(ticket_limit_units, int(value))) for value in units]
    selected = sorted(
        (index for index, value in enumerate(repaired) if value > 0),
        key=lambda index: (-repaired[index], index),
    )
    for index in selected[maximum_selected_tickets:]:
        repaired[index] = 0
    excess = sum(repaired) - budget_units
    while excess > 0:
        index = max(range(len(repaired)), key=lambda item: (repaired[item], -item))
        decrement = min(excess, repaired[index])
        repaired[index] -= decrement
        excess -= decrement
    return JointPortfolioGenome(tuple(repaired))


def optimize_joint_portfolio(
    parameter_draws: Sequence[Sequence[JointMarketScenario]],
    *,
    candidate_tickets: Sequence[str],
    gross_payoff_model: GrossPayoffModel,
    available_bankroll_yen: int,
    expected_outcomes: Sequence[str] | None = None,
    operational_costs_yen: Mapping[str, int] | None = None,
    config: JointPolicySearchConfig | None = None,
    genetic_settings: GeneticSearchSettings | None = None,
) -> dict[str, Any]:
    """Search complete stake vectors; this function never submits a wager."""
    tickets = tuple(candidate_tickets)
    if not tickets or len(set(tickets)) != len(tickets) or any(
        not isinstance(ticket, str) or not ticket for ticket in tickets
    ):
        raise ValueError("candidate_tickets must be unique non-empty strings")
    if not parameter_draws:
        raise ValueError("parameter_draws must not be empty")
    policy = config or JointPolicySearchConfig()
    policy.validate(available_bankroll_yen=available_bankroll_yen)
    settings = genetic_settings or GeneticSearchSettings(
        population_size=24,
        generations=12,
        elite_count=4,
        mutation_rate=0.35,
        random_injections=2,
        max_workers=4,
        seed=33038,
    )
    budget_yen = min(
        available_bankroll_yen,
        policy.maximum_portfolio_stake_yen,
    )
    budget_units = budget_yen // policy.purchase_unit_yen
    ticket_limit_units = min(
        budget_units,
        policy.maximum_ticket_stake_yen // policy.purchase_unit_yen,
    )
    selected_limit = min(policy.maximum_selected_tickets, len(tickets))

    def bets(genome: JointPortfolioGenome) -> dict[str, int]:
        return {
            ticket: units * policy.purchase_unit_yen
            for ticket, units in zip(tickets, genome.stake_units)
            if units > 0
        }

    def candidate_key(genome: JointPortfolioGenome) -> str:
        return ",".join(str(value) for value in genome.stake_units)

    def serialize(genome: JointPortfolioGenome) -> Mapping[str, Any]:
        vector = bets(genome)
        return {
            "bets_yen": vector,
            "total_stake_yen": sum(vector.values()),
        }

    def random_candidate(rng: random.Random) -> JointPortfolioGenome:
        if rng.random() < 0.10:
            return JointPortfolioGenome((0,) * len(tickets))
        count = rng.randint(1, selected_limit)
        indexes = rng.sample(range(len(tickets)), count)
        units = [0] * len(tickets)
        remaining = budget_units
        for position, index in enumerate(indexes):
            required_after = len(indexes) - position - 1
            ceiling = min(ticket_limit_units, remaining - required_after)
            if ceiling < 1:
                break
            units[index] = rng.randint(1, ceiling)
            remaining -= units[index]
        return _repair_units(
            units,
            budget_units=budget_units,
            ticket_limit_units=ticket_limit_units,
            maximum_selected_tickets=selected_limit,
        )

    def crossover(
        left: JointPortfolioGenome,
        right: JointPortfolioGenome,
        rng: random.Random,
    ) -> JointPortfolioGenome:
        units = [
            left_value if rng.random() < 0.5 else right_value
            for left_value, right_value in zip(
                left.stake_units, right.stake_units
            )
        ]
        return _repair_units(
            units,
            budget_units=budget_units,
            ticket_limit_units=ticket_limit_units,
            maximum_selected_tickets=selected_limit,
        )

    def mutate(
        genome: JointPortfolioGenome,
        rng: random.Random,
        mutation_rate: float,
    ) -> JointPortfolioGenome:
        units = list(genome.stake_units)
        step_limit = max(1, ticket_limit_units // 4)
        for index in range(len(units)):
            if rng.random() < mutation_rate:
                units[index] += rng.randint(-step_limit, step_limit)
        return _repair_units(
            units,
            budget_units=budget_units,
            ticket_limit_units=ticket_limit_units,
            maximum_selected_tickets=selected_limit,
        )

    def evaluate(genome: JointPortfolioGenome) -> Mapping[str, Any]:
        vector = bets(genome)
        total_stake = sum(vector.values())
        if not vector:
            return {
                "total_stake_yen": 0,
                "conservative_expected_profit_yen": 0.0,
                "search_fitness": 0.0,
                "search_feasible": False,
                "constraint_violation": None,
                "portfolio": {
                    "lower_quantile": 0.0,
                    "purchase_gate_evaluable": True,
                    "passes_purchase_gate": False,
                    "purchase_gate_reasons": ["no_bet_vector"],
                },
            }
        result = evaluate_joint_market_value(
            parameter_draws,
            bets_yen=vector,
            gross_payoff_model=gross_payoff_model,
            operational_costs_yen={
                ticket: (operational_costs_yen or {}).get(ticket, 0)
                for ticket in vector
            },
            expected_outcomes=expected_outcomes,
            outer_alpha=policy.outer_alpha,
            inner_tail_fraction=policy.inner_tail_fraction,
            buy_margin=policy.buy_margin,
            minimum_outer_draws=policy.minimum_outer_draws,
            minimum_inner_tail_effective_samples=(
                policy.minimum_inner_tail_effective_samples
            ),
            include_marginal_contributions=False,
            include_ticket_diagnostics=False,
        )
        growth = evaluate_joint_bankroll_growth(
            parameter_draws,
            bets_yen=vector,
            gross_payoff_model=gross_payoff_model,
            available_bankroll_yen=available_bankroll_yen,
            operational_costs_yen={
                ticket: (operational_costs_yen or {}).get(ticket, 0)
                for ticket in vector
            },
            expected_outcomes=expected_outcomes,
            outer_alpha=policy.outer_alpha,
            inner_tail_fraction=policy.inner_tail_fraction,
            minimum_outer_draws=policy.minimum_outer_draws,
            minimum_inner_tail_effective_samples=(
                policy.minimum_inner_tail_effective_samples
            ),
        )
        portfolio = dict(result["portfolio"])
        growth_summary = dict(growth["growth"])
        conservative_profit = (
            (float(portfolio["lower_quantile"]) - policy.buy_margin)
            * total_stake
            if portfolio["purchase_gate_evaluable"]
            else None
        )
        search_evaluable = bool(
            portfolio["purchase_gate_evaluable"]
            and growth_summary["purchase_gate_evaluable"]
        )
        edge_excess = float(portfolio["lower_quantile"]) - policy.buy_margin
        growth_excess = float(growth_summary["lower_quantile"])
        search_feasible = bool(
            search_evaluable
            and portfolio["passes_purchase_gate"]
            and growth_summary["passes_growth_gate"]
        )
        constraint_violation = (
            max(0.0, -edge_excess) + max(0.0, -growth_excess)
            if search_evaluable
            else None
        )
        # Preserve the hard production gate while giving evolution a gradient
        # toward feasibility when the initial population has no valid vector.
        search_fitness = (
            1.0 + growth_excess
            if search_feasible
            else (
                1.0 / (1.0 + float(constraint_violation))
                if constraint_violation is not None
                else -1.0
            )
        )
        return {
            "total_stake_yen": total_stake,
            "conservative_expected_profit_yen": conservative_profit,
            "search_fitness": search_fitness,
            "search_feasible": search_feasible,
            "constraint_violation": constraint_violation,
            "edge_excess": edge_excess,
            "growth_excess": growth_excess,
            "portfolio": portfolio,
            "bankroll_growth": growth,
        }

    def fitness(
        metrics: Mapping[str, Any], _genome: JointPortfolioGenome
    ) -> float:
        return float(metrics["search_fitness"])

    zero = JointPortfolioGenome((0,) * len(tickets))
    first_full = _repair_units(
        [ticket_limit_units, *([0] * (len(tickets) - 1))],
        budget_units=budget_units,
        ticket_limit_units=ticket_limit_units,
        maximum_selected_tickets=selected_limit,
    )
    equal = _repair_units(
        [max(1, budget_units // selected_limit)] * selected_limit
        + [0] * (len(tickets) - selected_limit),
        budget_units=budget_units,
        ticket_limit_units=ticket_limit_units,
        maximum_selected_tickets=selected_limit,
    )
    ranked, history = evolve_population(
        settings=settings,
        evaluator=evaluate,
        fitness=fitness,
        random_candidate=random_candidate,
        crossover=crossover,
        mutate=mutate,
        candidate_key=candidate_key,
        serialize=serialize,
        immigrants=(zero, first_full, equal),
    )
    feasible = [
        row for row in ranked if bool(row.metrics.get("search_feasible"))
    ]
    if feasible:
        selected = feasible[0]
    else:
        selected = next(
            row
            for row in ranked
            if not any(row.candidate.stake_units)
        )
    best_search_candidate = ranked[0]
    selected_bets = bets(selected.candidate)
    detailed_value = None
    if selected_bets:
        detailed_value = evaluate_joint_market_value(
            parameter_draws,
            bets_yen=selected_bets,
            gross_payoff_model=gross_payoff_model,
            operational_costs_yen={
                ticket: (operational_costs_yen or {}).get(ticket, 0)
                for ticket in selected_bets
            },
            expected_outcomes=expected_outcomes,
            outer_alpha=policy.outer_alpha,
            inner_tail_fraction=policy.inner_tail_fraction,
            buy_margin=policy.buy_margin,
            minimum_outer_draws=policy.minimum_outer_draws,
            minimum_inner_tail_effective_samples=(
                policy.minimum_inner_tail_effective_samples
            ),
            include_marginal_contributions=True,
        )
        detailed_growth = evaluate_joint_bankroll_growth(
            parameter_draws,
            bets_yen=selected_bets,
            gross_payoff_model=gross_payoff_model,
            available_bankroll_yen=available_bankroll_yen,
            operational_costs_yen={
                ticket: (operational_costs_yen or {}).get(ticket, 0)
                for ticket in selected_bets
            },
            expected_outcomes=expected_outcomes,
            outer_alpha=policy.outer_alpha,
            inner_tail_fraction=policy.inner_tail_fraction,
            minimum_outer_draws=policy.minimum_outer_draws,
            minimum_inner_tail_effective_samples=(
                policy.minimum_inner_tail_effective_samples
            ),
        )
    else:
        detailed_growth = None
    purchase_authorized = bool(
        selected_bets
        and detailed_value
        and detailed_growth
        and detailed_value["portfolio"]["passes_purchase_gate"]
        and detailed_growth["growth"]["passes_growth_gate"]
        and selected.fitness > 0.0
    )
    return {
        "version": "joint_portfolio_policy_ga_v2",
        "role": "diagnostic_only_never_submits_wagers",
        "parameter_draws": len(parameter_draws),
        "candidate_tickets": list(tickets),
        "available_bankroll_yen": available_bankroll_yen,
        "search_budget_yen": budget_yen,
        "purchase_authorized": purchase_authorized,
        "feasible_candidates_found": len(feasible),
        "best_search_candidate": {
            **serialize(best_search_candidate.candidate),
            "fitness": best_search_candidate.fitness,
            "metrics": dict(best_search_candidate.metrics),
            "first_generation": best_search_candidate.first_generation,
        },
        "selected": {
            **serialize(selected.candidate),
            "fitness": selected.fitness,
            "metrics": dict(selected.metrics),
            "joint_value": detailed_value,
            "bankroll_growth": detailed_growth,
        },
        "ranked_candidates": [
            {
                **serialize(row.candidate),
                "fitness": row.fitness,
                "metrics": dict(row.metrics),
                "first_generation": row.first_generation,
            }
            for row in ranked[:20]
        ],
        "history": history,
        "search_settings": json.loads(json.dumps(settings.__dict__)),
        "policy_config": json.loads(json.dumps(policy.__dict__)),
    }


__all__ = [
    "JointPolicySearchConfig",
    "JointPortfolioGenome",
    "optimize_joint_portfolio",
]
