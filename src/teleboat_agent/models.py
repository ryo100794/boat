from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


class ValidationError(ValueError):
    pass


@dataclass(frozen=True)
class BettingNumber:
    value: str

    @classmethod
    def parse(cls, value: Any) -> "BettingNumber":
        text = str(value)
        if len(text) != 3 or any(char not in "123456" for char in text):
            raise ValidationError("betting number must contain exactly three lanes from 1 to 6")
        if len(set(text)) != 3:
            raise ValidationError("betting number lanes must be unique")
        return cls(text)


@dataclass(frozen=True)
class Stadium:
    tel_code: int

    @classmethod
    def parse(cls, value: Any) -> "Stadium":
        try:
            tel_code = int(value)
        except (TypeError, ValueError) as exc:
            raise ValidationError("stadium_tel_code must be an integer") from exc
        if not 1 <= tel_code <= 24:
            raise ValidationError("stadium_tel_code must be between 1 and 24")
        return cls(tel_code)

    @property
    def formal_tel_code(self) -> str:
        return f"{self.tel_code:02d}"


@dataclass(frozen=True)
class Ticket:
    betting_number: BettingNumber
    quantity: int

    @classmethod
    def parse(cls, value: Any) -> "Ticket":
        if not isinstance(value, dict):
            raise ValidationError("each odds entry must be an object")
        try:
            quantity = int(value.get("quantity"))
        except (TypeError, ValueError) as exc:
            raise ValidationError("quantity must be an integer") from exc
        if not 1 <= quantity <= 999:
            raise ValidationError("quantity must be between 1 and 999")
        return cls(
            betting_number=BettingNumber.parse(value.get("number")),
            quantity=quantity,
        )

    @property
    def stake_yen(self) -> int:
        return self.quantity * 100

    def simple_betting_code(self, race_number: int) -> str:
        if not 1 <= race_number <= 12:
            raise ValidationError("race number must be between 1 and 12")
        return f"{race_number:02d}31{self.betting_number.value}{self.quantity:03d}"


@dataclass(frozen=True)
class VoteRequest:
    stadium: Stadium
    race_number: int
    tickets: tuple[Ticket, ...]

    @classmethod
    def parse(
        cls,
        payload: Any,
        *,
        max_tickets: int,
        max_total_stake_yen: int,
    ) -> "VoteRequest":
        if not isinstance(payload, dict):
            raise ValidationError("request body must be an object")
        race = payload.get("race")
        if not isinstance(race, dict):
            raise ValidationError("race must be an object")
        try:
            race_number = int(race.get("number"))
        except (TypeError, ValueError) as exc:
            raise ValidationError("race number must be an integer") from exc
        if not 1 <= race_number <= 12:
            raise ValidationError("race number must be between 1 and 12")
        raw_tickets = payload.get("odds")
        if not isinstance(raw_tickets, list) or not raw_tickets:
            raise ValidationError("odds must be a non-empty list")
        if len(raw_tickets) > max_tickets:
            raise ValidationError(f"ticket count exceeds configured limit {max_tickets}")
        tickets = tuple(Ticket.parse(item) for item in raw_tickets)
        numbers = [ticket.betting_number.value for ticket in tickets]
        if len(numbers) != len(set(numbers)):
            raise ValidationError("duplicate betting numbers are not allowed")
        total_stake = sum(ticket.stake_yen for ticket in tickets)
        if total_stake > max_total_stake_yen:
            raise ValidationError(
                f"total stake {total_stake} exceeds configured limit {max_total_stake_yen}"
            )
        return cls(
            stadium=Stadium.parse(race.get("stadium_tel_code")),
            race_number=race_number,
            tickets=tickets,
        )

    @property
    def total_stake_yen(self) -> int:
        return sum(ticket.stake_yen for ticket in self.tickets)

    def batches(self, batch_size: int) -> Iterable[tuple[Ticket, ...]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for start in range(0, len(self.tickets), batch_size):
            yield self.tickets[start : start + batch_size]
