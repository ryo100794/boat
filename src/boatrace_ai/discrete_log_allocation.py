from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any


PAYOUT_UNIT_YEN = 100
MIN_STAKE_UNIT_YEN = 100
_SCORE_TOLERANCE = 1e-15
SETTLEMENT_FIELDS = frozenset(
    {
        "actual_combination",
        "actual_payout_yen",
        "hit",
    }
)
SettlementKey = tuple[str, str]
SettlementMap = dict[SettlementKey, int]


class SettlementCandidate(dict[str, Any]):
    """Decision-only mapping with settlement outside its key space."""

    __slots__ = ("settlement_rows",)

    def __init__(
        self,
        decision: Mapping[str, Any],
        settlement_rows: Iterable[Mapping[str, Any]],
    ) -> None:
        super().__init__(
            (key, value)
            for key, value in decision.items()
            if key not in SETTLEMENT_FIELDS
        )
        self.settlement_rows = tuple(dict(row) for row in settlement_rows)


def candidate_with_settlements(
    decision: Mapping[str, Any],
    settlement_rows: Iterable[Mapping[str, Any]],
) -> SettlementCandidate:
    """Attach settlement data without exposing it as decision features."""

    return SettlementCandidate(decision, settlement_rows)


def split_decision_candidates_and_settlements(
    candidates: Iterable[Mapping[str, Any]],
    *,
    settlements: Mapping[SettlementKey, int] | None = None,
) -> tuple[list[dict[str, Any]], SettlementMap]:
    """Normalize legacy/new candidates into disjoint decision and result data."""

    decision_candidates: list[dict[str, Any]] = []
    settlement_map: SettlementMap = {}
    if settlements:
        for (race_id, combination), payout_yen in settlements.items():
            _add_settlement(
                settlement_map,
                race_id=str(race_id),
                combination=str(combination),
                payout_yen=payout_yen,
            )

    for source in candidates:
        decision_candidates.append({
            key: value
            for key, value in source.items()
            if key not in SETTLEMENT_FIELDS
        })
        for row in getattr(source, "settlement_rows", ()):
            _add_settlement_row(settlement_map, row)

        # Migrate legacy candidates at the API boundary without exposing their
        # result fields to portfolio enumeration or ranking.
        actual_combination = source.get("actual_combination")
        actual_payout_yen = source.get("actual_payout_yen")
        if actual_combination is not None and actual_payout_yen is not None:
            _add_settlement(
                settlement_map,
                race_id=str(source["race_id"]),
                combination=str(actual_combination),
                payout_yen=actual_payout_yen,
            )
    return decision_candidates, settlement_map


def settle_decision_ticket(
    decision: Mapping[str, Any],
    *,
    stake_yen: int,
    settlements: Mapping[SettlementKey, int],
) -> dict[str, Any]:
    """Join an already-selected ticket to official settlement data."""

    race_id = str(decision["race_id"])
    combination = str(decision["combination"])
    payout_yen = settlements.get((race_id, combination))
    hit = payout_yen is not None
    return_yen = (
        int(round(stake_yen * int(payout_yen) / PAYOUT_UNIT_YEN))
        if hit
        else 0
    )
    return {
        **decision,
        "actual_payout_yen": int(payout_yen) if hit else 0,
        "hit": hit,
        "stake_yen": int(stake_yen),
        "return_yen": return_yen,
        "profit_yen": return_yen - int(stake_yen),
    }


def _add_settlement_row(
    settlement_map: SettlementMap,
    row: Mapping[str, Any],
) -> None:
    _add_settlement(
        settlement_map,
        race_id=str(row["race_id"]),
        combination=str(row["combination"]),
        payout_yen=row["payout_yen"],
    )


def _add_settlement(
    settlement_map: SettlementMap,
    *,
    race_id: str,
    combination: str,
    payout_yen: Any,
) -> None:
    payout = int(payout_yen)
    if payout < 0:
        raise ValueError("payout_yen must be non-negative")
    key = race_id, combination
    existing = settlement_map.get(key)
    if existing is not None and existing != payout:
        raise ValueError(
            "conflicting settlement for "
            f"{race_id} {combination}: {existing} != {payout}"
        )
    settlement_map[key] = payout


@dataclass(frozen=True)
class _Candidate:
    race_id: str
    combination: str
    probability: float
    estimated_odds: float
    source: dict[str, Any] = field(compare=False, repr=False)

    @property
    def key(self) -> tuple[str, str]:
        return self.race_id, self.combination


@dataclass(frozen=True)
class _RaceOption:
    race_id: str
    selections: tuple[tuple[_Candidate, int], ...]
    stake_yen: int
    expected_log_growth: float

    @property
    def ticket_count(self) -> int:
        return len(self.selections)

    @property
    def tie_key(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (candidate.combination, stake_yen)
            for candidate, stake_yen in self.selections
        )


@dataclass(frozen=True)
class _DayState:
    expected_log_growth: float
    options: tuple[_RaceOption, ...]

    @property
    def tie_key(self) -> tuple[tuple[str, tuple[tuple[str, int], ...]], ...]:
        return tuple((option.race_id, option.tie_key) for option in self.options)


def allocate_discrete_log_day(
    race_date: str,
    candidates: list[dict[str, Any]],
    evaluated_races: set[str],
    *,
    daily_budget_yen: int,
    max_daily_exposure_fraction: float,
    race_cap_fraction: float,
    ticket_cap_fraction: float,
    max_daily_tickets: int | None,
    stake_granularity_yen: int = MIN_STAKE_UNIT_YEN,
    min_stake_yen: int = MIN_STAKE_UNIT_YEN,
    max_tickets_per_race: int = 2,
    settlements: Mapping[SettlementKey, int] | None = None,
) -> dict[str, Any]:
    """Allocate a day by maximizing conservative discrete log growth.

    Probability is already conservative. Legacy result fields are stripped at
    the API boundary. Enumeration and ranking receive decision data only;
    settlement is joined afterward.
    """

    _validate_policy(
        daily_budget_yen=daily_budget_yen,
        max_daily_exposure_fraction=max_daily_exposure_fraction,
        race_cap_fraction=race_cap_fraction,
        ticket_cap_fraction=ticket_cap_fraction,
        max_daily_tickets=max_daily_tickets,
        stake_granularity_yen=stake_granularity_yen,
        min_stake_yen=min_stake_yen,
        max_tickets_per_race=max_tickets_per_race,
    )
    decision_candidates, settlement_map = (
        split_decision_candidates_and_settlements(
            candidates,
            settlements=settlements,
        )
    )
    prepared = _prepare_candidates(decision_candidates)

    daily_cap_yen = _floor_to_granularity(
        daily_budget_yen * max_daily_exposure_fraction,
        stake_granularity_yen,
    )
    race_cap_yen = _floor_to_granularity(
        daily_budget_yen * race_cap_fraction,
        stake_granularity_yen,
    )
    ticket_cap_yen = _floor_to_granularity(
        daily_budget_yen * ticket_cap_fraction,
        stake_granularity_yen,
    )

    by_race: dict[str, list[_Candidate]] = defaultdict(list)
    for candidate in prepared:
        by_race[candidate.race_id].append(candidate)

    options_by_race: dict[str, list[_RaceOption]] = {}
    allocation_candidate_keys: set[tuple[str, str]] = set()
    for race_id in sorted(by_race):
        options, positive_option_keys = _enumerate_race_options(
            race_id,
            by_race[race_id],
            bankroll_yen=daily_budget_yen,
            daily_cap_yen=daily_cap_yen,
            race_cap_yen=race_cap_yen,
            ticket_cap_yen=ticket_cap_yen,
            stake_granularity_yen=stake_granularity_yen,
            min_stake_yen=min_stake_yen,
            max_tickets_per_race=max_tickets_per_race,
        )
        options_by_race[race_id] = options
        allocation_candidate_keys.update(positive_option_keys)

    selected_state = _select_day_portfolio(
        options_by_race,
        daily_cap_yen=daily_cap_yen,
        stake_granularity_yen=stake_granularity_yen,
        max_daily_tickets=max_daily_tickets,
    )
    selected = _settle_selected_portfolio(
        selected_state,
        race_date=race_date,
        daily_budget_yen=daily_budget_yen,
        settlements=settlement_map,
    )

    stake_yen = sum(int(item["stake_yen"]) for item in selected)
    return_yen = sum(int(item["return_yen"]) for item in selected)
    hit_tickets = sum(bool(item["hit"]) for item in selected)
    selected_races = {str(item["race_id"]) for item in selected}
    hit_races = {str(item["race_id"]) for item in selected if item["hit"]}
    hit_returns = [int(item["return_yen"]) for item in selected if item["hit"]]
    profit_yen = return_yen - stake_yen

    return {
        "race_date": race_date,
        "evaluated_races": len(evaluated_races),
        "candidate_tickets": len(decision_candidates),
        "positive_edge_tickets": sum(
            candidate.probability * candidate.estimated_odds > 1.0
            for candidate in prepared
        ),
        "allocation_candidate_tickets": len(allocation_candidate_keys),
        "tickets": len(selected),
        "races_bet": len(selected_races),
        "hit_tickets": hit_tickets,
        "hit_races": len(hit_races),
        "largest_hit_return_yen": max(hit_returns, default=0),
        "hit_return_square_sum_yen2": sum(value * value for value in hit_returns),
        "stake_yen": stake_yen,
        "return_yen": return_yen,
        "profit_yen": profit_yen,
        "roi": return_yen / stake_yen if stake_yen else None,
        "budget_used_fraction": (
            stake_yen / daily_budget_yen if daily_budget_yen else 0.0
        ),
        "avg_stake_yen": stake_yen / len(selected) if selected else 0.0,
        "max_stake_yen": max(
            (int(item["stake_yen"]) for item in selected),
            default=0,
        ),
        "expected_log_growth": selected_state.expected_log_growth,
        "allocation_method": "discrete_conservative_expected_log",
        "race_portfolios": [
            {
                "race_id": option.race_id,
                "tickets": option.ticket_count,
                "stake_yen": option.stake_yen,
                "expected_log_growth": option.expected_log_growth,
            }
            for option in selected_state.options
        ],
        "_tail_portfolio_rows": [
            {
                "date": race_date,
                "race_id": str(item["race_id"]),
                "odds": float(item["estimated_odds"]),
                "stake": int(item["stake_yen"]),
                "return": int(item["return_yen"]),
            }
            for item in selected
        ],
        "selected_sample": _selection_sample(selected),
    }


def _prepare_candidates(candidates: list[dict[str, Any]]) -> list[_Candidate]:
    prepared = []
    seen: set[tuple[str, str]] = set()
    for item in candidates:
        race_id = str(item["race_id"])
        combination = str(item["combination"])
        probability = float(item["probability"])
        estimated_odds = float(item["estimated_odds"])
        if not race_id or not combination:
            raise ValueError("race_id and combination must be non-empty")
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be finite and between 0 and 1")
        if not math.isfinite(estimated_odds) or estimated_odds <= 1.0:
            raise ValueError("estimated_odds must be finite and greater than 1")
        key = (race_id, combination)
        if key in seen:
            raise ValueError(f"duplicate candidate: {race_id} {combination}")
        seen.add(key)
        prepared.append(
            _Candidate(
                race_id=race_id,
                combination=combination,
                probability=probability,
                estimated_odds=estimated_odds,
                source=item,
            )
        )
    return sorted(prepared, key=lambda candidate: candidate.key)


def _enumerate_race_options(
    race_id: str,
    candidates: list[_Candidate],
    *,
    bankroll_yen: int,
    daily_cap_yen: int,
    race_cap_yen: int,
    ticket_cap_yen: int,
    stake_granularity_yen: int,
    min_stake_yen: int,
    max_tickets_per_race: int,
) -> tuple[list[_RaceOption], set[tuple[str, str]]]:
    no_bet = _RaceOption(race_id, (), 0, 0.0)
    maximum_ticket_stake = min(ticket_cap_yen, race_cap_yen, daily_cap_yen)
    if maximum_ticket_stake < min_stake_yen:
        return [no_bet], set()

    stake_values = range(
        min_stake_yen,
        maximum_ticket_stake + 1,
        stake_granularity_yen,
    )
    best_by_size: dict[tuple[int, int], _RaceOption] = {}
    positive_option_keys: set[tuple[str, str]] = set()

    for candidate in candidates:
        for stake_yen in stake_values:
            option = _build_race_option(
                race_id,
                ((candidate, stake_yen),),
                bankroll_yen=bankroll_yen,
            )
            if option is not None:
                positive_option_keys.add(candidate.key)
                _retain_best_option(best_by_size, option)

    if max_tickets_per_race >= 2:
        for left_index, left in enumerate(candidates):
            for right in candidates[left_index + 1 :]:
                if left.probability + right.probability > 1.0 + _SCORE_TOLERANCE:
                    continue
                for left_stake_yen in stake_values:
                    for right_stake_yen in stake_values:
                        if left_stake_yen + right_stake_yen > race_cap_yen:
                            continue
                        option = _build_race_option(
                            race_id,
                            (
                                (left, left_stake_yen),
                                (right, right_stake_yen),
                            ),
                            bankroll_yen=bankroll_yen,
                        )
                        if option is not None:
                            positive_option_keys.update((left.key, right.key))
                            _retain_best_option(best_by_size, option)

    options = [no_bet, *best_by_size.values()]
    return sorted(
        options,
        key=lambda option: (
            option.stake_yen,
            option.ticket_count,
            option.tie_key,
        ),
    ), positive_option_keys


def _build_race_option(
    race_id: str,
    selections: tuple[tuple[_Candidate, int], ...],
    *,
    bankroll_yen: int,
) -> _RaceOption | None:
    stake_yen = sum(stake for _candidate, stake in selections)
    expected_log_growth = _expected_log_growth(
        bankroll_yen,
        selections,
    )
    if not math.isfinite(expected_log_growth) or expected_log_growth <= 0.0:
        return None
    return _RaceOption(
        race_id=race_id,
        selections=selections,
        stake_yen=stake_yen,
        expected_log_growth=expected_log_growth,
    )


def _expected_log_growth(
    bankroll_yen: int,
    selections: tuple[tuple[_Candidate, int], ...],
) -> float:
    total_stake_yen = sum(stake for _candidate, stake in selections)
    if total_stake_yen <= 0:
        return 0.0
    losing_wealth_yen = bankroll_yen - total_stake_yen
    if losing_wealth_yen <= 0:
        return -math.inf

    selected_probability = sum(
        candidate.probability for candidate, _stake in selections
    )
    if selected_probability > 1.0 + _SCORE_TOLERANCE:
        return -math.inf
    selected_probability = min(1.0, selected_probability)
    log_growth = (1.0 - selected_probability) * math.log(
        losing_wealth_yen / bankroll_yen
    )
    for candidate, stake_yen in selections:
        winning_wealth_yen = (
            losing_wealth_yen + stake_yen * candidate.estimated_odds
        )
        if winning_wealth_yen <= 0:
            return -math.inf
        log_growth += candidate.probability * math.log(
            winning_wealth_yen / bankroll_yen
        )
    return log_growth


def _retain_best_option(
    best_by_size: dict[tuple[int, int], _RaceOption],
    option: _RaceOption,
) -> None:
    key = option.stake_yen, option.ticket_count
    current = best_by_size.get(key)
    if current is None or _option_is_better(option, current):
        best_by_size[key] = option


def _option_is_better(candidate: _RaceOption, current: _RaceOption) -> bool:
    difference = candidate.expected_log_growth - current.expected_log_growth
    if difference > _SCORE_TOLERANCE:
        return True
    if abs(difference) <= _SCORE_TOLERANCE:
        return candidate.tie_key < current.tie_key
    return False


def _select_day_portfolio(
    options_by_race: dict[str, list[_RaceOption]],
    *,
    daily_cap_yen: int,
    stake_granularity_yen: int,
    max_daily_tickets: int | None,
) -> _DayState:
    daily_cap_units = daily_cap_yen // stake_granularity_yen
    states: dict[tuple[int, int], _DayState] = {(0, 0): _DayState(0.0, ())}

    for race_id in sorted(options_by_race):
        next_states: dict[tuple[int, int], _DayState] = {}
        for (used_units, used_tickets), state in states.items():
            for option in options_by_race[race_id]:
                option_units = option.stake_yen // stake_granularity_yen
                next_units = used_units + option_units
                next_tickets = used_tickets + option.ticket_count
                if next_units > daily_cap_units:
                    continue
                if (
                    max_daily_tickets is not None
                    and next_tickets > max_daily_tickets
                ):
                    continue
                next_options = (
                    state.options + (option,)
                    if option.ticket_count
                    else state.options
                )
                candidate_state = _DayState(
                    state.expected_log_growth + option.expected_log_growth,
                    next_options,
                )
                state_key = next_units, next_tickets
                current = next_states.get(state_key)
                if current is None or _state_is_better_same_size(
                    candidate_state,
                    current,
                ):
                    next_states[state_key] = candidate_state
        states = next_states

    best = states.get((0, 0), _DayState(0.0, ()))
    best_size = (0, 0)
    for size, state in states.items():
        if state.expected_log_growth <= 0.0:
            continue
        if _state_is_better_final(state, size, best, best_size):
            best = state
            best_size = size
    return best


def _state_is_better_same_size(candidate: _DayState, current: _DayState) -> bool:
    difference = candidate.expected_log_growth - current.expected_log_growth
    if difference > _SCORE_TOLERANCE:
        return True
    if abs(difference) <= _SCORE_TOLERANCE:
        return candidate.tie_key < current.tie_key
    return False


def _state_is_better_final(
    candidate: _DayState,
    candidate_size: tuple[int, int],
    current: _DayState,
    current_size: tuple[int, int],
) -> bool:
    difference = candidate.expected_log_growth - current.expected_log_growth
    if difference > _SCORE_TOLERANCE:
        return True
    if abs(difference) > _SCORE_TOLERANCE:
        return False
    if candidate_size != current_size:
        return candidate_size < current_size
    return candidate.tie_key < current.tie_key


def _settle_selected_portfolio(
    state: _DayState,
    *,
    race_date: str,
    daily_budget_yen: int,
    settlements: Mapping[SettlementKey, int],
) -> list[dict[str, Any]]:
    selected = []
    for option in state.options:
        for candidate, stake_yen in option.selections:
            estimated_ev = candidate.probability * candidate.estimated_odds
            kelly_fraction = max(
                0.0,
                (estimated_ev - 1.0) / (candidate.estimated_odds - 1.0),
            )
            standalone_log_growth = _expected_log_growth(
                daily_budget_yen,
                ((candidate, stake_yen),),
            )
            selected.append(
                settle_decision_ticket(
                    {
                        **candidate.source,
                        "race_id": candidate.race_id,
                        "race_date": candidate.source.get(
                            "race_date", race_date
                        ),
                        "combination": candidate.combination,
                        "probability": candidate.probability,
                        "estimated_odds": candidate.estimated_odds,
                        "estimated_ev": estimated_ev,
                        "kelly_fraction": kelly_fraction,
                        "stake_fraction": stake_yen / daily_budget_yen,
                        "expected_log_growth": standalone_log_growth,
                        "race_portfolio_expected_log_growth": (
                            option.expected_log_growth
                        ),
                    },
                    stake_yen=stake_yen,
                    settlements=settlements,
                )
            )
    return selected


def _selection_sample(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = sorted(
        selected,
        key=lambda item: (
            -int(item["stake_yen"]),
            -float(item["estimated_ev"]),
            str(item["race_id"]),
            str(item["combination"]),
        ),
    )[:12]
    return [
        {
            "race_id": item["race_id"],
            "combination": item["combination"],
            "stake_yen": item["stake_yen"],
            "probability": round(float(item["probability"]), 9),
            "estimated_odds": round(float(item["estimated_odds"]), 6),
            "estimated_ev": round(float(item["estimated_ev"]), 6),
            "kelly_fraction": round(float(item["kelly_fraction"]), 9),
            "expected_log_growth": float(item["expected_log_growth"]),
            "hit": bool(item["hit"]),
            "return_yen": item["return_yen"],
        }
        for item in rows
    ]


def _validate_policy(
    *,
    daily_budget_yen: int,
    max_daily_exposure_fraction: float,
    race_cap_fraction: float,
    ticket_cap_fraction: float,
    max_daily_tickets: int | None,
    stake_granularity_yen: int,
    min_stake_yen: int,
    max_tickets_per_race: int,
) -> None:
    if daily_budget_yen <= 0:
        raise ValueError("daily_budget_yen must be positive")
    for name, value in (
        ("max_daily_exposure_fraction", max_daily_exposure_fraction),
        ("race_cap_fraction", race_cap_fraction),
        ("ticket_cap_fraction", ticket_cap_fraction),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and between 0 and 1")
    if race_cap_fraction > max_daily_exposure_fraction:
        raise ValueError(
            "race_cap_fraction must not exceed max_daily_exposure_fraction"
        )
    if ticket_cap_fraction > race_cap_fraction:
        raise ValueError("ticket_cap_fraction must not exceed race_cap_fraction")
    if max_daily_tickets is not None and max_daily_tickets < 1:
        raise ValueError("max_daily_tickets must be positive when set")
    if max_tickets_per_race not in {1, 2}:
        raise ValueError("max_tickets_per_race must be 1 or 2")
    if stake_granularity_yen < MIN_STAKE_UNIT_YEN:
        raise ValueError("stake_granularity_yen must be at least 100")
    if stake_granularity_yen % MIN_STAKE_UNIT_YEN:
        raise ValueError("stake_granularity_yen must be divisible by 100")
    if min_stake_yen < MIN_STAKE_UNIT_YEN:
        raise ValueError("min_stake_yen must be at least 100")
    if min_stake_yen % stake_granularity_yen:
        raise ValueError("min_stake_yen must be divisible by stake_granularity_yen")
    if daily_budget_yen < min_stake_yen:
        raise ValueError("daily_budget_yen must be at least min_stake_yen")
    if daily_budget_yen % stake_granularity_yen:
        raise ValueError(
            "daily_budget_yen must be divisible by stake_granularity_yen"
        )


def _floor_to_granularity(value: float, granularity: int) -> int:
    if not math.isfinite(value) or value <= 0.0:
        return 0
    return int(math.floor(value / granularity) * granularity)
