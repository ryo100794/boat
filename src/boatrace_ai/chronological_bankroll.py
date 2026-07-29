from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

from .discrete_log_allocation import allocate_discrete_log_day


STAKE_UNIT_YEN = 100
DEFAULT_SETTLEMENT_DELAY_MINUTES = 10
_RESULT_KEYS = frozenset({
    "actual_combination", "actual_payout_yen", "hit", "payout_yen",
    "profit_yen", "result_available_at", "return_yen", "settlement_at",
})
Allocator = Callable[..., dict[str, Any]]


def _timestamp(value: Any, *, field: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif value:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"invalid {field}: {value}") from exc
    else:
        raise ValueError(f"missing {field}")
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _decision_at(candidate: Mapping[str, Any]) -> datetime:
    for field in ("decision_at", "real_odds_deadline_at", "odds_deadline_at"):
        if candidate.get(field):
            return _timestamp(candidate[field], field=field)
    race_date = candidate.get("race_date")
    rno = candidate.get("rno")
    if race_date and rno is not None:
        base = _timestamp(f"{race_date}T00:00:00+00:00", field="race_date")
        return base + timedelta(minutes=int(rno))
    raise ValueError(f"candidate lacks decision_at: {candidate.get('race_id')}")


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _RESULT_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def decision_information_fingerprint(
    candidates: Iterable[Mapping[str, Any]],
) -> str:
    """Fingerprint only information permitted at purchase decision time."""
    rows = sorted(
        (_canonical(candidate) for candidate in candidates),
        key=lambda row: (
            str(row.get("race_id") or ""), str(row.get("combination") or "")
        ),
    )
    encoded = json.dumps(
        rows, ensure_ascii=True, allow_nan=False, separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def settlement_events_from_races(
    races: Iterable[Mapping[str, Any]],
    *,
    fallback_delay_minutes: int = DEFAULT_SETTLEMENT_DELAY_MINUTES,
) -> list[dict[str, Any]]:
    """Build settlement events while preserving an auditable time source."""
    events = []
    for race in races:
        race_id = str(race["race_id"])
        explicit = race.get("result_available_at")
        if explicit:
            available_at = _timestamp(explicit, field="result_available_at")
            time_source = "result_available_at"
        else:
            raw_decision = race.get("odds_deadline_at") or race.get("decision_at")
            if raw_decision:
                decision_at = _timestamp(raw_decision, field="odds_deadline_at")
                decision_source = "decision_at"
            else:
                decision_at = _decision_at(race)
                decision_source = "synthetic_rno_order"
            available_at = decision_at + timedelta(minutes=fallback_delay_minutes)
            time_source = (
                f"{decision_source}_plus_{fallback_delay_minutes}m_fallback"
            )
        settlements = race.get("settlements")
        if not settlements:
            combination = race.get("actual_combination")
            payout_yen = race.get("actual_payout_yen")
            settlements = (
                [{"combination": combination, "payout_yen": payout_yen}]
                if combination is not None and payout_yen is not None else []
            )
        events.append({
            "race_id": race_id,
            "result_available_at": available_at.isoformat(),
            "result_available_at_source": time_source,
            "payouts": {
                str(row["combination"]): int(row["payout_yen"])
                for row in settlements
            },
        })
    return events


def simulate_chronological_bankroll_day(
    race_date: str,
    candidates: Iterable[Mapping[str, Any]],
    evaluated_races: Iterable[str],
    *,
    settlement_events: Iterable[Mapping[str, Any]],
    initial_bankroll_yen: int = 10_000,
    max_decision_exposure_fraction: float = 0.20,
    race_cap_fraction: float = 0.03,
    ticket_cap_fraction: float = 0.01,
    max_tickets_per_race: int = 2,
    stake_granularity_yen: int = STAKE_UNIT_YEN,
    allocate_day: Allocator = allocate_discrete_log_day,
    allocator_kwargs: Mapping[str, Any] | None = None,
    allocation_method: str | None = None,
) -> dict[str, Any]:
    """Run decisions in time order and release returns only at settlement."""
    if initial_bankroll_yen < STAKE_UNIT_YEN:
        raise ValueError("initial_bankroll_yen must fund at least one 100-yen unit")
    if initial_bankroll_yen % STAKE_UNIT_YEN:
        raise ValueError("initial_bankroll_yen must be divisible by 100")

    decisions = [dict(candidate) for candidate in candidates]
    by_race: dict[str, list[dict[str, Any]]] = defaultdict(list)
    decision_times: dict[str, datetime] = {}
    for candidate in decisions:
        race_id = str(candidate["race_id"])
        decision_at = _decision_at(candidate)
        existing = decision_times.get(race_id)
        if existing is not None and existing != decision_at:
            raise ValueError(f"inconsistent decision_at for {race_id}")
        decision_times[race_id] = decision_at
        by_race[race_id].append(candidate)

    event_by_race: dict[str, dict[str, Any]] = {}
    for source in settlement_events:
        race_id = str(source["race_id"])
        if race_id in event_by_race:
            raise ValueError(f"duplicate settlement event: {race_id}")
        event_by_race[race_id] = {
            "available_at": _timestamp(
                source.get("result_available_at"), field="result_available_at"
            ),
            "source": str(source.get("result_available_at_source") or "explicit"),
            "payouts": {
                str(combination): int(payout)
                for combination, payout in dict(source.get("payouts") or {}).items()
            },
        }

    cash_yen = initial_bankroll_yen
    peak_equity_yen = initial_bankroll_yen
    max_drawdown_yen = 0
    pending: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    configured_allocator_kwargs = dict(allocator_kwargs or {})

    def settle_due(as_of: datetime) -> None:
        nonlocal cash_yen, peak_equity_yen, max_drawdown_yen
        ordered = sorted(
            pending, key=lambda row: (row["available_at"], row["race_id"])
        )
        due = [item for item in ordered if item["available_at"] <= as_of]
        remaining = [item for item in ordered if item["available_at"] > as_of]
        outstanding = sum(int(row["stake_yen"]) for row in remaining)
        for item in due:
            cash_yen += int(item["return_yen"])
            equity = cash_yen + outstanding
            peak_equity_yen = max(peak_equity_yen, equity)
            max_drawdown_yen = max(max_drawdown_yen, peak_equity_yen - equity)
            ledger.append({
                "event": "settlement", "race_id": item["race_id"],
                "at": item["available_at"].isoformat(),
                "stake_yen": item["stake_yen"], "return_yen": item["return_yen"],
                "cash_after_yen": cash_yen,
                "outstanding_stake_yen": outstanding,
            })
        pending[:] = remaining

    for race_id in sorted(by_race, key=lambda key: (decision_times[key], key)):
        decision_at = decision_times[race_id]
        settle_due(decision_at)
        event = event_by_race.get(race_id)
        if event is None:
            raise ValueError(f"missing settlement event: {race_id}")
        if event["available_at"] <= decision_at:
            raise ValueError(f"settlement is not after decision: {race_id}")
        before_cash = cash_yen
        race_candidates = by_race[race_id]
        fingerprint = decision_information_fingerprint(race_candidates)
        allocatable_bankroll_yen = (
            cash_yen // stake_granularity_yen * stake_granularity_yen
        )
        if allocatable_bankroll_yen < stake_granularity_yen:
            allocation = {"selected_sample": [], "allocation_candidate_tickets": 0}
        else:
            call_kwargs = {
                **configured_allocator_kwargs,
                "daily_budget_yen": allocatable_bankroll_yen,
                "max_daily_exposure_fraction": max_decision_exposure_fraction,
                "race_cap_fraction": race_cap_fraction,
                "ticket_cap_fraction": ticket_cap_fraction,
                "max_daily_tickets": None,
                "stake_granularity_yen": stake_granularity_yen,
                "min_stake_yen": stake_granularity_yen,
                "settlements": {
                    (race_id, combination): payout
                    for combination, payout in event["payouts"].items()
                },
            }
            if allocate_day is allocate_discrete_log_day:
                call_kwargs.setdefault(
                    "max_tickets_per_race", max_tickets_per_race
                )
            allocation = allocate_day(
                race_date,
                race_candidates,
                {race_id},
                **call_kwargs,
            )
        tickets = [dict(row) for row in allocation["selected_sample"]]
        for ticket in tickets:
            ticket_stake = int(ticket["stake_yen"])
            if ticket_stake < 0 or ticket_stake % stake_granularity_yen:
                raise ValueError(
                    f"allocator returned invalid stake for {race_id}: "
                    f"{ticket_stake}"
                )
        race_stake = sum(int(row["stake_yen"]) for row in tickets)
        if race_stake > allocatable_bankroll_yen:
            raise ValueError(
                f"allocator exceeded available cash for {race_id}: "
                f"{race_stake} > {allocatable_bankroll_yen}"
            )
        race_return = sum(int(row["return_yen"]) for row in tickets)
        cash_yen -= race_stake
        pending.append({
            "race_id": race_id, "available_at": event["available_at"],
            "stake_yen": race_stake, "return_yen": race_return,
        })
        selected.extend(tickets)
        outstanding = sum(int(row["stake_yen"]) for row in pending)
        equity = cash_yen + outstanding
        max_drawdown_yen = max(max_drawdown_yen, peak_equity_yen - equity)
        ledger.append({
            "event": "decision", "race_id": race_id,
            "at": decision_at.isoformat(),
            "result_available_at": event["available_at"].isoformat(),
            "result_available_at_source": event["source"],
            "decision_information_sha256": fingerprint,
            "candidate_tickets": len(race_candidates),
            "allocation_candidate_tickets": int(
                allocation["allocation_candidate_tickets"]
            ),
            "tickets": len(tickets), "stake_yen": race_stake,
            "cash_before_yen": before_cash, "cash_after_yen": cash_yen,
            "outstanding_stake_yen": outstanding,
            "selections": [
                {"combination": str(row["combination"]),
                 "stake_yen": int(row["stake_yen"])}
                for row in tickets
            ],
        })

    settle_due(datetime.max.replace(tzinfo=timezone.utc))
    stake_yen = sum(int(row["stake_yen"]) for row in selected)
    return_yen = sum(int(row["return_yen"]) for row in selected)
    return {
        "race_date": race_date, "initial_bankroll_yen": initial_bankroll_yen,
        "closing_bankroll_yen": cash_yen, "available_cash_yen": cash_yen,
        "outstanding_stake_yen": 0,
        "evaluated_races": len(set(str(value) for value in evaluated_races)),
        "candidate_tickets": len(decisions), "tickets": len(selected),
        "races_bet": len({str(row["race_id"]) for row in selected}),
        "hit_tickets": sum(bool(row["hit"]) for row in selected),
        "stake_yen": stake_yen, "return_yen": return_yen,
        "profit_yen": cash_yen - initial_bankroll_yen,
        "roi": return_yen / stake_yen if stake_yen else None,
        "max_drawdown_yen": max_drawdown_yen,
        "allocation_method": allocation_method or (
            "chronological_discrete_conservative_expected_log"
            if allocate_day is allocate_discrete_log_day
            else f"chronological_{getattr(allocate_day, '__name__', 'allocator')}"
        ),
        "profit_reinvestment": True,
        "stake_granularity_yen": stake_granularity_yen,
        "real_betting_enabled": False,
        "decision_information_sha256": decision_information_fingerprint(decisions),
        "information_boundary": {
            "decision_fields_exclude_result_and_payout": True,
            "forbidden_decision_fields": sorted(_RESULT_KEYS),
            "settlement_joined_after_allocation": True,
        },
        "ledger": ledger,
    }


def summarize_chronological_bankroll_days(
    daily_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate nested chronological results without changing legacy totals."""
    rows = [dict(row) for row in daily_rows]
    stake_yen = sum(int(row.get("stake_yen") or 0) for row in rows)
    return_yen = sum(int(row.get("return_yen") or 0) for row in rows)
    cumulative_profit_yen = 0
    peak_profit_yen = 0
    interday_drawdown_yen = 0
    for row in rows:
        cumulative_profit_yen += int(row.get("profit_yen") or 0)
        peak_profit_yen = max(peak_profit_yen, cumulative_profit_yen)
        interday_drawdown_yen = max(
            interday_drawdown_yen,
            peak_profit_yen - cumulative_profit_yen,
        )
    return {
        "race_days": len(rows),
        "evaluated_races": sum(
            int(row.get("evaluated_races") or 0) for row in rows
        ),
        "tickets": sum(int(row.get("tickets") or 0) for row in rows),
        "races_bet": sum(int(row.get("races_bet") or 0) for row in rows),
        "hit_tickets": sum(
            int(row.get("hit_tickets") or 0) for row in rows
        ),
        "stake_yen": stake_yen,
        "return_yen": return_yen,
        "profit_yen": return_yen - stake_yen,
        "roi": return_yen / stake_yen if stake_yen else 0.0,
        "winning_days": sum(
            int(int(row.get("profit_yen") or 0) > 0) for row in rows
        ),
        "max_drawdown_yen": max(
            interday_drawdown_yen,
            max((int(row.get("max_drawdown_yen") or 0) for row in rows), default=0),
        ),
        "profit_reinvestment": True,
        "stake_granularity_yen": STAKE_UNIT_YEN,
        "real_betting_enabled": False,
        "daily": rows,
    }
