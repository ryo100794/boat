from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from numbers import Integral
from typing import Any, Mapping, Sequence

from .joint_market_value import GrossPayoffModel, JointMarketScenario


@dataclass(frozen=True)
class ParimutuelSettlementRules:
    """Integer-yen settlement rules for one wager pool."""

    payout_rate_numerator: int = 75
    payout_rate_denominator: int = 100
    face_unit_yen: int = 10
    purchase_unit_yen: int = 100
    refund_terminal_states: tuple[str, ...] = ("cancelled", "not_held")

    def validate(self) -> None:
        if isinstance(self.payout_rate_numerator, bool) or not isinstance(
            self.payout_rate_numerator, int
        ):
            raise ValueError("payout_rate_numerator must be an integer")
        if isinstance(self.payout_rate_denominator, bool) or not isinstance(
            self.payout_rate_denominator, int
        ):
            raise ValueError("payout_rate_denominator must be an integer")
        if not 0 < self.payout_rate_numerator <= self.payout_rate_denominator:
            raise ValueError("payout rate must be in (0, 1]")
        if isinstance(self.face_unit_yen, bool) or not isinstance(
            self.face_unit_yen, int
        ) or self.face_unit_yen < 1:
            raise ValueError("face_unit_yen must be positive")
        if isinstance(self.purchase_unit_yen, bool) or not isinstance(
            self.purchase_unit_yen, int
        ) or self.purchase_unit_yen < self.face_unit_yen:
            raise ValueError("purchase_unit_yen must be at least face_unit_yen")
        if self.purchase_unit_yen % self.face_unit_yen:
            raise ValueError("purchase_unit_yen must be divisible by face_unit_yen")
        if len(set(self.refund_terminal_states)) != len(
            self.refund_terminal_states
        ) or any(
            not isinstance(state, str) or not state
            for state in self.refund_terminal_states
        ):
            raise ValueError("refund terminal states must be unique strings")


def _yen(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer yen amount")
    return int(value)


def _external_stakes(
    market_state: Mapping[str, Any],
    *,
    ordinary_outcomes: Sequence[str],
    face_unit_yen: int,
) -> dict[str, int]:
    raw = market_state.get("external_ticket_stakes_yen")
    if raw is None:
        total_sales = market_state.get("external_total_sales_yen")
        shares = market_state.get("final_market_shares")
        if total_sales is None or not isinstance(shares, Mapping):
            raise ValueError(
                "market state requires absolute external_ticket_stakes_yen or "
                "external_total_sales_yen plus final_market_shares; normalized "
                "market shares alone cannot identify self impact"
            )
        return apportion_external_stakes(
            total_sales_yen=total_sales,
            market_shares=shares,
            ordinary_outcomes=ordinary_outcomes,
            face_unit_yen=face_unit_yen,
        )
    if not isinstance(raw, Mapping):
        raise ValueError(
            "external_ticket_stakes_yen must be a mapping"
        )
    if set(raw) != set(ordinary_outcomes):
        raise ValueError("external ticket stakes must match ordinary outcomes")
    stakes = {
        outcome: _yen(raw[outcome], "external ticket stake")
        for outcome in ordinary_outcomes
    }
    if any(value % face_unit_yen for value in stakes.values()):
        raise ValueError("external ticket stakes must use face-unit increments")
    if sum(stakes.values()) <= 0:
        raise ValueError("external ticket stakes must have positive total sales")
    return stakes


def apportion_external_stakes(
    *,
    total_sales_yen: object,
    market_shares: Mapping[str, float],
    ordinary_outcomes: Sequence[str],
    face_unit_yen: int = 10,
) -> dict[str, int]:
    """Allocate known absolute pool scale by largest remainder in face units."""
    if isinstance(face_unit_yen, bool) or not isinstance(
        face_unit_yen, int
    ) or face_unit_yen < 1:
        raise ValueError("face_unit_yen must be positive")
    total = _yen(total_sales_yen, "external total sales")
    if total <= 0 or total % face_unit_yen:
        raise ValueError("external total sales must use positive face units")
    outcomes = tuple(ordinary_outcomes)
    if not outcomes or len(set(outcomes)) != len(outcomes):
        raise ValueError("ordinary outcomes must be unique and non-empty")
    if set(market_shares) != set(outcomes):
        raise ValueError("market shares must match ordinary outcomes")
    parsed = {}
    for outcome in outcomes:
        value = market_shares[outcome]
        if isinstance(value, bool):
            raise ValueError("market shares must be finite and non-negative")
        try:
            parsed[outcome] = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "market shares must be finite and non-negative"
            ) from exc
    if any(not math.isfinite(value) or value < 0.0 for value in parsed.values()):
        raise ValueError("market shares must be finite and non-negative")
    share_total = sum(parsed.values())
    if abs(share_total - 1.0) > 1e-8:
        raise ValueError("market shares must sum to one")
    parsed = {
        outcome: value / share_total for outcome, value in parsed.items()
    }
    total_units = total // face_unit_yen
    raw_units = {
        outcome: parsed[outcome] * total_units for outcome in outcomes
    }
    allocated = {outcome: math.floor(raw_units[outcome]) for outcome in outcomes}
    remaining = total_units - sum(allocated.values())
    order = sorted(
        outcomes,
        key=lambda outcome: (
            -(raw_units[outcome] - allocated[outcome]),
            outcomes.index(outcome),
        ),
    )
    for outcome in order[:remaining]:
        allocated[outcome] += 1
    return {
        outcome: allocated[outcome] * face_unit_yen for outcome in outcomes
    }


def build_parimutuel_gross_payoff_model(
    *,
    ordinary_outcomes: Sequence[str],
    rules: ParimutuelSettlementRules | None = None,
) -> GrossPayoffModel:
    """Build a strict payout callback; no pool-size estimate is invented."""
    outcomes = tuple(ordinary_outcomes)
    if not outcomes or len(set(outcomes)) != len(outcomes) or any(
        not isinstance(outcome, str) or not outcome for outcome in outcomes
    ):
        raise ValueError("ordinary_outcomes must be unique non-empty strings")
    settlement_rules = rules or ParimutuelSettlementRules()
    settlement_rules.validate()
    refund_states = settlement_rules.refund_terminal_states
    if set(outcomes) & set(refund_states):
        raise ValueError("ordinary outcomes and refund states must be disjoint")
    payout_rate = Fraction(
        settlement_rules.payout_rate_numerator,
        settlement_rules.payout_rate_denominator,
    )

    def settle(
        scenario: JointMarketScenario,
        bets_yen: Mapping[str, int],
    ) -> Mapping[str, Mapping[str, int]]:
        if not set(bets_yen) <= set(outcomes):
            raise ValueError("bet vector contains a non-ordinary outcome")
        bets = {
            ticket: _yen(amount, "bet") for ticket, amount in bets_yen.items()
        }
        if any(
            amount <= 0 or amount % settlement_rules.purchase_unit_yen
            for amount in bets.values()
        ):
            raise ValueError("bets must use positive purchase-unit increments")
        external = _external_stakes(
            scenario.market_state,
            ordinary_outcomes=outcomes,
            face_unit_yen=settlement_rules.face_unit_yen,
        )
        special_addition = _yen(
            scenario.market_state.get("special_addition_yen", 0),
            "special payout addition",
        )
        active_full_refunds = tuple(
            state for state in refund_states if state in scenario.probabilities
        )
        raw_partial_refunds = scenario.market_state.get(
            "partial_refund_tickets_by_state", {}
        )
        if not isinstance(raw_partial_refunds, Mapping):
            raise ValueError("partial refund states must map to ticket sequences")
        partial_refunds: dict[str, set[str]] = {}
        for state, refunded_tickets in raw_partial_refunds.items():
            if not isinstance(state, str) or not state:
                raise ValueError("partial refund states must be non-empty strings")
            if state not in scenario.probabilities:
                raise ValueError("partial refund state is absent from probabilities")
            if state in outcomes or state in refund_states:
                raise ValueError("partial refund state must be a separate terminal state")
            if isinstance(refunded_tickets, (str, bytes)) or not isinstance(
                refunded_tickets, Sequence
            ):
                raise ValueError("partial refund tickets must be a sequence")
            if any(
                not isinstance(ticket, str) or not ticket
                for ticket in refunded_tickets
            ):
                raise ValueError("partial refund tickets must be strings")
            parsed_tickets = set(refunded_tickets)
            if not parsed_tickets <= set(outcomes):
                raise ValueError("partial refund contains an unknown ticket")
            partial_refunds[state] = parsed_tickets
        total_sales = sum(external.values()) + sum(bets.values())
        distributable = (
            total_sales * payout_rate.numerator // payout_rate.denominator
        ) + special_addition
        payoff: dict[str, dict[str, int]] = {}
        for ticket, own_stake in bets.items():
            winning_stake = external[ticket] + own_stake
            winning_face_units = winning_stake // settlement_rules.face_unit_yen
            payout_per_face_unit = max(
                settlement_rules.face_unit_yen,
                distributable // winning_face_units,
            )
            own_face_units = own_stake // settlement_rules.face_unit_yen
            receipts = {ticket: own_face_units * payout_per_face_unit}
            receipts.update({state: own_stake for state in active_full_refunds})
            receipts.update({
                state: own_stake
                for state, refunded_tickets in partial_refunds.items()
                if ticket in refunded_tickets
            })
            payoff[ticket] = receipts
        return payoff

    return settle


__all__ = [
    "ParimutuelSettlementRules",
    "apportion_external_stakes",
    "build_parimutuel_gross_payoff_model",
]
