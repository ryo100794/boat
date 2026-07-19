from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations, permutations, product
from typing import Any, ClassVar, Iterable


class ValidationError(ValueError):
    pass


class BetType(str, Enum):
    WIN = "win"
    PLACE = "place"
    EXACTA = "exacta"
    QUINELLA = "quinella"
    QUINELLA_PLACE = "quinella_place"
    TRIFECTA = "trifecta"
    TRIO = "trio"

    @classmethod
    def parse(cls, value: Any) -> "BetType":
        text = str(value or "trifecta").strip().lower()
        aliases = {
            "win": cls.WIN,
            "単勝": cls.WIN,
            "place": cls.PLACE,
            "複勝": cls.PLACE,
            "exacta": cls.EXACTA,
            "2連単": cls.EXACTA,
            "二連単": cls.EXACTA,
            "quinella": cls.QUINELLA,
            "2連複": cls.QUINELLA,
            "二連複": cls.QUINELLA,
            "quinella_place": cls.QUINELLA_PLACE,
            "wide": cls.QUINELLA_PLACE,
            "拡連複": cls.QUINELLA_PLACE,
            "trifecta": cls.TRIFECTA,
            "3連単": cls.TRIFECTA,
            "三連単": cls.TRIFECTA,
            "trio": cls.TRIO,
            "3連複": cls.TRIO,
            "三連複": cls.TRIO,
        }
        try:
            return aliases[text]
        except KeyError as exc:
            raise ValidationError(f"unsupported bet_type: {value}") from exc

    @property
    def label(self) -> str:
        return {
            self.WIN: "単勝",
            self.PLACE: "複勝",
            self.EXACTA: "2連単",
            self.QUINELLA: "2連複",
            self.QUINELLA_PLACE: "拡連複",
            self.TRIFECTA: "3連単",
            self.TRIO: "3連複",
        }[self]

    @property
    def official_value(self) -> str:
        return {
            self.WIN: "1",
            self.PLACE: "2",
            self.EXACTA: "3",
            self.QUINELLA: "4",
            self.QUINELLA_PLACE: "5",
            self.TRIFECTA: "6",
            self.TRIO: "7",
        }[self]

    @property
    def lane_count(self) -> int:
        return 1 if self in {self.WIN, self.PLACE} else 2 if self in {
            self.EXACTA,
            self.QUINELLA,
            self.QUINELLA_PLACE,
        } else 3

    @property
    def ordered(self) -> bool:
        return self in {self.EXACTA, self.TRIFECTA}


class BetMethod(str, Enum):
    REGULAR = "regular"
    BOX = "box"
    FORMATION = "formation"

    @classmethod
    def parse(cls, value: Any) -> "BetMethod":
        text = str(value or "regular").strip().lower()
        aliases = {
            "regular": cls.REGULAR,
            "normal": cls.REGULAR,
            "通常": cls.REGULAR,
            "box": cls.BOX,
            "ボックス": cls.BOX,
            "formation": cls.FORMATION,
            "フォーメーション": cls.FORMATION,
        }
        try:
            return aliases[text]
        except KeyError as exc:
            raise ValidationError(f"unsupported method: {value}") from exc

    @property
    def official_value(self) -> str:
        return {
            self.REGULAR: "1",
            self.BOX: "3",
            self.FORMATION: "4",
        }[self]


@dataclass(frozen=True)
class BettingNumber:
    value: str

    @classmethod
    def parse(cls, value: Any, *, expected_lanes: int = 3) -> "BettingNumber":
        text = str(value)
        if len(text) != expected_lanes or any(char not in "123456" for char in text):
            raise ValidationError(
                f"betting number must contain exactly {expected_lanes} lanes from 1 to 6"
            )
        if len(set(text)) != expected_lanes:
            raise ValidationError("betting number lanes must be unique")
        return cls(text)

    def normalized(self, bet_type: BetType) -> "BettingNumber":
        if bet_type.ordered or bet_type.lane_count == 1:
            return self
        return BettingNumber("".join(sorted(self.value)))

    def display(self, bet_type: BetType) -> str:
        separator = "-" if bet_type.ordered else "="
        return separator.join(self.value)


@dataclass(frozen=True)
class Stadium:
    tel_code: int

    NAMES: ClassVar[tuple[str, ...]] = (
        "桐生", "戸田", "江戸川", "平和島", "多摩川", "浜名湖",
        "蒲郡", "常滑", "津", "三国", "びわこ", "住之江",
        "尼崎", "鳴門", "丸亀", "児島", "宮島", "徳山",
        "下関", "若松", "芦屋", "福岡", "唐津", "大村",
    )

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

    @property
    def name(self) -> str:
        return self.NAMES[self.tel_code - 1]


@dataclass(frozen=True)
class Ticket:
    betting_number: BettingNumber
    quantity: int

    @classmethod
    def parse(
        cls,
        value: Any,
        *,
        bet_type: BetType = BetType.TRIFECTA,
    ) -> "Ticket":
        if not isinstance(value, dict):
            raise ValidationError("each ticket must be an object")
        try:
            quantity = int(value.get("quantity"))
        except (TypeError, ValueError) as exc:
            raise ValidationError("quantity must be an integer in 100-yen units") from exc
        if not 1 <= quantity <= 999:
            raise ValidationError("quantity must be between 1 and 999")
        number = BettingNumber.parse(
            value.get("number"),
            expected_lanes=bet_type.lane_count,
        ).normalized(bet_type)
        return cls(betting_number=number, quantity=quantity)

    @property
    def stake_yen(self) -> int:
        return self.quantity * 100

    def simple_betting_code(self, race_number: int) -> str:
        if len(self.betting_number.value) != 3:
            raise ValidationError("legacy simple betting code only supports trifecta")
        if not 1 <= race_number <= 12:
            raise ValidationError("race number must be between 1 and 12")
        return f"{race_number:02d}31{self.betting_number.value}{self.quantity:03d}"


@dataclass(frozen=True)
class VoteRequest:
    stadium: Stadium
    race_number: int
    bet_type: BetType
    method: BetMethod
    tickets: tuple[Ticket, ...]
    source_positions: tuple[tuple[int, ...], ...] = ()
    quantity: int | None = None

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

        bet_type = BetType.parse(payload.get("bet_type"))
        method = BetMethod.parse(payload.get("method"))
        if method is BetMethod.REGULAR:
            tickets = cls._parse_regular_tickets(payload, bet_type)
            positions: tuple[tuple[int, ...], ...] = ()
            quantity = None
        else:
            if bet_type.lane_count == 1:
                raise ValidationError("box and formation are not available for win/place")
            quantity = cls._parse_quantity(payload.get("quantity"))
            positions = cls._parse_positions(payload, method, bet_type)
            numbers = cls._expand_positions(positions, method, bet_type)
            tickets = tuple(
                Ticket(BettingNumber(number), quantity)
                for number in numbers
            )

        cls._validate_expansion(
            tickets,
            max_tickets=max_tickets,
            max_total_stake_yen=max_total_stake_yen,
        )
        return cls(
            stadium=Stadium.parse(race.get("stadium_tel_code")),
            race_number=race_number,
            bet_type=bet_type,
            method=method,
            tickets=tickets,
            source_positions=positions,
            quantity=quantity,
        )

    @staticmethod
    def _parse_regular_tickets(
        payload: dict[str, Any],
        bet_type: BetType,
    ) -> tuple[Ticket, ...]:
        raw_tickets = payload.get("tickets", payload.get("odds"))
        if not isinstance(raw_tickets, list) or not raw_tickets:
            raise ValidationError("tickets must be a non-empty list")
        tickets = tuple(Ticket.parse(item, bet_type=bet_type) for item in raw_tickets)
        numbers = [ticket.betting_number.value for ticket in tickets]
        if len(numbers) != len(set(numbers)):
            raise ValidationError("duplicate betting numbers are not allowed")
        return tickets

    @classmethod
    def _parse_positions(
        cls,
        payload: dict[str, Any],
        method: BetMethod,
        bet_type: BetType,
    ) -> tuple[tuple[int, ...], ...]:
        raw = payload.get("selections") if method is BetMethod.BOX else payload.get("formation")
        if method is BetMethod.BOX:
            lanes = cls._parse_lane_set(raw, "selections")
            if len(lanes) < bet_type.lane_count:
                raise ValidationError(
                    f"box requires at least {bet_type.lane_count} distinct lanes"
                )
            return (lanes,)
        if not isinstance(raw, list) or len(raw) != bet_type.lane_count:
            raise ValidationError(
                f"formation requires exactly {bet_type.lane_count} position lists"
            )
        return tuple(
            cls._parse_lane_set(position, f"formation[{index}]")
            for index, position in enumerate(raw)
        )

    @staticmethod
    def _parse_lane_set(value: Any, field: str) -> tuple[int, ...]:
        if not isinstance(value, list) or not value:
            raise ValidationError(f"{field} must be a non-empty lane list")
        try:
            lanes = tuple(int(item) for item in value)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{field} lanes must be integers") from exc
        if any(lane < 1 or lane > 6 for lane in lanes):
            raise ValidationError(f"{field} lanes must be between 1 and 6")
        if len(lanes) != len(set(lanes)):
            raise ValidationError(f"{field} contains duplicate lanes")
        return lanes

    @staticmethod
    def _parse_quantity(value: Any) -> int:
        try:
            quantity = int(value)
        except (TypeError, ValueError) as exc:
            raise ValidationError("quantity must be an integer in 100-yen units") from exc
        if not 1 <= quantity <= 999:
            raise ValidationError("quantity must be between 1 and 999")
        return quantity

    @staticmethod
    def _expand_positions(
        positions: tuple[tuple[int, ...], ...],
        method: BetMethod,
        bet_type: BetType,
    ) -> tuple[str, ...]:
        if method is BetMethod.BOX:
            lanes = positions[0]
            rows = (
                permutations(lanes, bet_type.lane_count)
                if bet_type.ordered
                else combinations(sorted(lanes), bet_type.lane_count)
            )
        else:
            rows = product(*positions)
        unique: dict[str, None] = {}
        for row in rows:
            if len(set(row)) != bet_type.lane_count:
                continue
            normalized = row if bet_type.ordered else tuple(sorted(row))
            unique.setdefault("".join(str(lane) for lane in normalized), None)
        if not unique:
            raise ValidationError("selection expands to zero valid tickets")
        return tuple(unique)

    @staticmethod
    def _validate_expansion(
        tickets: tuple[Ticket, ...],
        *,
        max_tickets: int,
        max_total_stake_yen: int,
    ) -> None:
        if len(tickets) > max_tickets:
            raise ValidationError(
                f"expanded ticket count {len(tickets)} exceeds configured limit {max_tickets}"
            )
        total_stake = sum(ticket.stake_yen for ticket in tickets)
        if total_stake > max_total_stake_yen:
            raise ValidationError(
                f"total stake {total_stake} exceeds configured limit {max_total_stake_yen}"
            )

    @property
    def total_stake_yen(self) -> int:
        return sum(ticket.stake_yen for ticket in self.tickets)

    @property
    def expanded_ticket_count(self) -> int:
        return len(self.tickets)

    def batches(self, batch_size: int) -> Iterable[tuple[Ticket, ...]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for start in range(0, len(self.tickets), batch_size):
            yield self.tickets[start : start + batch_size]
