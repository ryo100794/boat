from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from .discrete_log_allocation import allocate_discrete_log_day


STAKE_UNIT_YEN = 100
DEFAULT_SETTLEMENT_DELAY_MINUTES = 10
_RESULT_KEYS = frozenset({
    "actual_combination", "actual_payout_yen", "hit", "payout_yen",
    "profit_yen", "result_available_at", "return_yen", "settlement_at",
})
Allocator = Callable[..., dict[str, Any]]
ScheduleQuotaRounding = Literal["floor", "ceil"]


def _empirical_quantile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = math.ceil((len(ordered) - 1) * quantile)
    return float(ordered[index])


def _validate_schedule_quota_opportunity(
    policy: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if policy is None:
        return None
    configured = dict(policy)
    after_fraction = float(configured.get("after_fraction", 0.95))
    score_quantile = float(configured.get("score_quantile", 0.75))
    reserve_slots = int(configured.get("reserve_slots", 1))
    minimum_score = float(configured.get("minimum_score", 0.0))
    quota_mode = str(configured.get("quota_mode") or "late_tail_release")
    if not 0.0 <= after_fraction <= 1.0:
        raise ValueError("opportunity after_fraction must be between zero and one")
    if not 0.0 <= score_quantile <= 1.0:
        raise ValueError("opportunity score_quantile must be between zero and one")
    if reserve_slots <= 0:
        raise ValueError("opportunity reserve_slots must be positive")
    if not math.isfinite(minimum_score):
        raise ValueError("opportunity minimum_score must be finite")
    if quota_mode not in {"late_tail_release", "online_reserve"}:
        raise ValueError("unsupported opportunity quota mode")
    return {
        "method": "strictly_observed_intraday_score_quantile",
        "quota_mode": quota_mode,
        "after_fraction": after_fraction,
        "score_quantile": score_quantile,
        "reserve_slots": reserve_slots,
        "minimum_score": minimum_score,
        "score_field": "estimated_ev",
        "result_or_payout_fields_used": False,
    }


def cumulative_schedule_ticket_quota(
    *,
    limit: int,
    elapsed: int,
    total: int,
    rounding: ScheduleQuotaRounding = "floor",
) -> int:
    if limit < 0 or total <= 0 or not 0 <= elapsed <= total:
        raise ValueError("invalid schedule ticket quota inputs")
    if rounding not in {"floor", "ceil"}:
        raise ValueError("schedule quota rounding must be floor or ceil")
    numerator = limit * elapsed
    if rounding == "ceil" and numerator:
        return min(limit, (numerator + total - 1) // total)
    return min(limit, numerator // total)


def opportunity_adjusted_ticket_quota(
    *,
    limit: int,
    base_quota: int,
    used_tickets: int,
    elapsed: int,
    total: int,
    current_score: float | None,
    observed_scores: list[float],
    policy: Mapping[str, Any] | None,
) -> tuple[int, float | None, bool]:
    configured = _validate_schedule_quota_opportunity(policy)
    preserved_quota = max(base_quota, min(used_tickets, limit))
    if (
        configured is None
        or limit <= 0
        or total <= 0
        or current_score is None
    ):
        return preserved_quota, None, False
    if configured["quota_mode"] == "online_reserve":
        reserve_slots = min(limit, int(configured["reserve_slots"]))
        ordinary_limit = limit - reserve_slots
        ordinary_quota = cumulative_schedule_ticket_quota(
            limit=ordinary_limit,
            elapsed=elapsed,
            total=total,
            rounding="floor",
        )
        preserved_quota = max(ordinary_quota, min(used_tickets, limit))
    history_threshold = _empirical_quantile(
        observed_scores, float(configured["score_quantile"])
    )
    if history_threshold is None:
        return preserved_quota, None, False
    score_threshold = max(
        float(configured["minimum_score"]), history_threshold
    )
    if configured["quota_mode"] == "online_reserve":
        if (
            elapsed / total >= float(configured["after_fraction"])
            and preserved_quota < limit
            and current_score >= score_threshold
        ):
            return min(
                limit, max(preserved_quota, used_tickets + 1)
            ), score_threshold, True
        return preserved_quota, score_threshold, False
    reserve_floor = max(0, limit - int(configured["reserve_slots"]))
    if (
        elapsed / total >= float(configured["after_fraction"])
        and preserved_quota >= reserve_floor
        and preserved_quota < limit
        and current_score >= score_threshold
    ):
        return preserved_quota + 1, score_threshold, True
    return preserved_quota, score_threshold, False


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
    daily_stake_limit_fraction: float = 1.0,
    max_decision_exposure_fraction: float = 0.20,
    race_cap_fraction: float = 0.03,
    ticket_cap_fraction: float = 0.01,
    max_tickets_per_race: int = 2,
    max_daily_tickets: int | None = None,
    schedule: Iterable[Mapping[str, Any]] | None = None,
    schedule_quota_rounding: ScheduleQuotaRounding = "floor",
    schedule_quota_opportunity: Mapping[str, Any] | None = None,
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
    if max_daily_tickets is not None and max_daily_tickets < 0:
        raise ValueError("max_daily_tickets must be non-negative")
    if schedule_quota_rounding not in {"floor", "ceil"}:
        raise ValueError("schedule quota rounding must be floor or ceil")
    opportunity_policy = _validate_schedule_quota_opportunity(
        schedule_quota_opportunity
    )

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

    schedule_times: dict[str, datetime] = {}
    for item in schedule if schedule is not None else decisions:
        race_id = str(item["race_id"])
        decision_at = _decision_at(item)
        existing = schedule_times.get(race_id)
        if existing is not None and existing != decision_at:
            raise ValueError(f"inconsistent schedule time for {race_id}")
        schedule_times[race_id] = decision_at
    missing_schedule = set(by_race) - set(schedule_times)
    if missing_schedule:
        raise ValueError(
            "candidate races missing from schedule: "
            + ", ".join(sorted(missing_schedule))
        )
    ordered_schedule_times = sorted(schedule_times.values())

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

    if not 0.0 <= daily_stake_limit_fraction <= 1.0:
        raise ValueError(
            "daily_stake_limit_fraction must be between zero and one"
        )
    cash_yen = initial_bankroll_yen
    initial_gross_stake_allowance_yen = (
        int(initial_bankroll_yen * daily_stake_limit_fraction)
        // stake_granularity_yen
        * stake_granularity_yen
    )
    gross_stake_yen = 0
    realized_cumulative_profit_yen = 0
    peak_equity_yen = initial_bankroll_yen
    max_drawdown_yen = 0
    pending: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    configured_allocator_kwargs = dict(allocator_kwargs or {})
    observed_candidate_scores: list[float] = []

    def settle_due(as_of: datetime) -> None:
        nonlocal cash_yen, peak_equity_yen, max_drawdown_yen
        nonlocal realized_cumulative_profit_yen
        ordered = sorted(
            pending, key=lambda row: (row["available_at"], row["race_id"])
        )
        due = [item for item in ordered if item["available_at"] <= as_of]
        remaining = [item for item in ordered if item["available_at"] > as_of]
        outstanding = sum(int(row["stake_yen"]) for row in remaining)
        for item in due:
            cash_yen += int(item["return_yen"])
            realized_profit_yen = (
                int(item["return_yen"]) - int(item["stake_yen"])
            )
            realized_cumulative_profit_yen += realized_profit_yen
            equity = cash_yen + outstanding
            peak_equity_yen = max(peak_equity_yen, equity)
            max_drawdown_yen = max(max_drawdown_yen, peak_equity_yen - equity)
            ledger.append({
                "event": "settlement", "race_id": item["race_id"],
                "at": item["available_at"].isoformat(),
                "stake_yen": item["stake_yen"], "return_yen": item["return_yen"],
                "cash_after_yen": cash_yen,
                "outstanding_stake_yen": outstanding,
                "realized_cumulative_profit_yen": (
                    realized_cumulative_profit_yen
                ),
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
        gross_stake_allowance_yen = (
            initial_gross_stake_allowance_yen
            + max(0, realized_cumulative_profit_yen)
        )
        remaining_gross_stake_allowance_yen = max(
            0, gross_stake_allowance_yen - gross_stake_yen
        )
        allocatable_bankroll_yen = (
            min(cash_yen, remaining_gross_stake_allowance_yen)
            // stake_granularity_yen * stake_granularity_yen
        )
        schedule_races_elapsed = sum(
            value <= decision_at for value in ordered_schedule_times
        )
        schedule_races_total = len(ordered_schedule_times)
        cumulative_ticket_quota = (
            cumulative_schedule_ticket_quota(
                limit=max_daily_tickets,
                elapsed=schedule_races_elapsed,
                total=schedule_races_total,
                rounding=schedule_quota_rounding,
            )
            if max_daily_tickets is not None and schedule_races_total
            else max_daily_tickets
        )
        current_candidate_score = max(
            (
                float(candidate["estimated_ev"])
                for candidate in race_candidates
                if candidate.get("estimated_ev") is not None
                and math.isfinite(float(candidate["estimated_ev"]))
            ),
            default=None,
        )
        opportunity_score_threshold = None
        opportunity_quota_released = False
        if max_daily_tickets is not None:
            (
                cumulative_ticket_quota,
                opportunity_score_threshold,
                opportunity_quota_released,
            ) = opportunity_adjusted_ticket_quota(
                limit=max_daily_tickets,
                base_quota=int(cumulative_ticket_quota or 0),
                used_tickets=len(selected),
                elapsed=schedule_races_elapsed,
                total=schedule_races_total,
                current_score=current_candidate_score,
                observed_scores=observed_candidate_scores,
                policy=opportunity_policy,
            )
        remaining_ticket_quota = (
            max(0, cumulative_ticket_quota - len(selected))
            if cumulative_ticket_quota is not None else None
        )
        if (
            allocatable_bankroll_yen < stake_granularity_yen
            or remaining_ticket_quota == 0
        ):
            allocation = {"selected_sample": [], "allocation_candidate_tickets": 0}
        else:
            call_kwargs = {
                **configured_allocator_kwargs,
                "daily_budget_yen": allocatable_bankroll_yen,
                "max_daily_exposure_fraction": max_decision_exposure_fraction,
                "race_cap_fraction": race_cap_fraction,
                "ticket_cap_fraction": ticket_cap_fraction,
                "max_daily_tickets": remaining_ticket_quota,
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
        tickets = []
        cap_remaining_yen = allocatable_bankroll_yen
        for source in allocation["selected_sample"]:
            ticket = dict(source)
            original_stake = int(ticket["stake_yen"])
            if (
                remaining_ticket_quota is not None
                and len(tickets) >= remaining_ticket_quota
            ):
                break
            capped_stake = min(original_stake, cap_remaining_yen)
            capped_stake = (
                capped_stake // stake_granularity_yen * stake_granularity_yen
            )
            if capped_stake <= 0:
                continue
            original_return = int(ticket.get("return_yen") or 0)
            ticket["stake_yen"] = capped_stake
            ticket["return_yen"] = (
                original_return * capped_stake // original_stake
                if original_stake > 0 else 0
            )
            ticket["hit"] = bool(ticket["return_yen"] > 0)
            tickets.append(ticket)
            cap_remaining_yen -= capped_stake
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
        gross_stake_yen += race_stake
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
            "initial_gross_stake_allowance_yen": (
                initial_gross_stake_allowance_yen
            ),
            "gross_stake_allowance_yen": gross_stake_allowance_yen,
            "remaining_gross_stake_allowance_yen": max(
                0, gross_stake_allowance_yen - gross_stake_yen
            ),
            "gross_stake_yen": gross_stake_yen,
            "learned_daily_ticket_limit": max_daily_tickets,
            "schedule_races_elapsed": schedule_races_elapsed,
            "schedule_races_total": schedule_races_total,
            "cumulative_ticket_quota": cumulative_ticket_quota,
            "opportunity_candidate_score": current_candidate_score,
            "opportunity_score_threshold": (
                opportunity_score_threshold
                if opportunity_score_threshold is not None
                and math.isfinite(opportunity_score_threshold)
                else None
            ),
            "opportunity_quota_released": opportunity_quota_released,
            "remaining_ticket_quota": (
                max(0, (cumulative_ticket_quota or 0) - len(selected))
                if cumulative_ticket_quota is not None else None
            ),
            "realized_cumulative_profit_yen": (
                realized_cumulative_profit_yen
            ),
            "selections": [
                {"combination": str(row["combination"]),
                 "stake_yen": int(row["stake_yen"])}
                for row in tickets
            ],
        })
        if current_candidate_score is not None:
            observed_candidate_scores.append(current_candidate_score)

    settle_due(datetime.max.replace(tzinfo=timezone.utc))
    stake_yen = sum(int(row["stake_yen"]) for row in selected)
    return_yen = sum(int(row["return_yen"]) for row in selected)
    hit_returns = [
        int(row["return_yen"])
        for row in selected
        if int(row.get("return_yen") or 0) > 0
    ]
    return {
        "race_date": race_date, "initial_bankroll_yen": initial_bankroll_yen,
        "closing_bankroll_yen": cash_yen, "available_cash_yen": cash_yen,
        "outstanding_stake_yen": 0,
        "evaluated_races": len(set(str(value) for value in evaluated_races)),
        "candidate_tickets": len(decisions), "tickets": len(selected),
        "races_bet": len({str(row["race_id"]) for row in selected}),
        "hit_races": len({
            str(row["race_id"])
            for row in selected
            if int(row.get("return_yen") or 0) > 0
        }),
        "hit_tickets": sum(bool(row["hit"]) for row in selected),
        "stake_yen": stake_yen, "return_yen": return_yen,
        "profit_yen": cash_yen - initial_bankroll_yen,
        "roi": return_yen / stake_yen if stake_yen else None,
        "max_drawdown_yen": max_drawdown_yen,
        "largest_hit_return_yen": max(hit_returns, default=0),
        "hit_return_square_sum_yen2": sum(
            value * value for value in hit_returns
        ),
        "initial_gross_stake_allowance_yen": (
            initial_gross_stake_allowance_yen
        ),
        "final_gross_stake_allowance_yen": (
            initial_gross_stake_allowance_yen
            + max(0, realized_cumulative_profit_yen)
        ),
        "gross_stake_yen": gross_stake_yen,
        "realized_cumulative_profit_yen": realized_cumulative_profit_yen,
        "daily_stake_limit_fraction": daily_stake_limit_fraction,
        "learned_daily_ticket_limit": max_daily_tickets,
        "schedule_races_total": len(ordered_schedule_times),
        "schedule_quota_rounding": (
            schedule_quota_rounding if max_daily_tickets is not None else None
        ),
        "schedule_quota_opportunity": opportunity_policy,
        "schedule_quota_rule": (
            f"{schedule_quota_rounding}(learned_daily_ticket_limit*"
            "scheduled_races_elapsed/scheduled_races_total)"
            if max_daily_tickets is not None else None
        ),
        "gross_stake_allowance_rule": (
            "initial_allowance_plus_positive_part_of_cumulative_net_realized_profit"
        ),
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
        "hit_races": sum(int(row.get("hit_races") or 0) for row in rows),
        "hit_tickets": sum(
            int(row.get("hit_tickets") or 0) for row in rows
        ),
        "largest_hit_return_yen": max(
            (int(row.get("largest_hit_return_yen") or 0) for row in rows),
            default=0,
        ),
        "hit_return_square_sum_yen2": sum(
            int(row.get("hit_return_square_sum_yen2") or 0) for row in rows
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
