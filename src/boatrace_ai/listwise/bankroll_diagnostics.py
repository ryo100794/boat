from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from .closing_odds import decision_odds


STAKE_YEN = 100
SETTLEMENT_DELAY_MINUTES = 10


def _decision_time(race: dict[str, Any]) -> datetime:
    raw = race.get("odds_deadline_at")
    if not raw:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(raw))
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def sequential_top5_ev_kelly_diagnostic(
    races: list[dict[str, Any]],
    *,
    daily_budget_yen: int,
) -> dict[str, Any]:
    if daily_budget_yen < STAKE_YEN:
        raise ValueError("daily budget must fund at least one 100-yen unit")
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for race in races:
        by_day[str(race["race_date"])].append(race)

    daily = []
    total_stake = total_return = total_units = total_bets = total_hits = 0
    for race_date in sorted(by_day):
        opening_balance = int(daily_budget_yen)
        balance = opening_balance
        peak_balance = balance
        max_drawdown = 0
        stake_yen = return_yen = units = bets = hits = 0
        ordered = sorted(
            by_day[race_date],
            key=lambda race: (
                _decision_time(race),
                str(race.get("jcd") or ""),
                int(race.get("rno") or 0),
            ),
        )
        pending: list[tuple[datetime, int]] = []
        for race in ordered:
            decision_at = _decision_time(race)
            still_pending = []
            for settles_at, payout in pending:
                if settles_at <= decision_at:
                    balance += payout
                    peak_balance = max(peak_balance, balance)
                else:
                    still_pending.append((settles_at, payout))
            pending = still_pending
            probabilities = race["model_probabilities"]
            odds = decision_odds(race)
            top5 = sorted(probabilities, key=probabilities.get, reverse=True)[:5]
            candidates = []
            for combination in top5:
                probability = float(probabilities[combination])
                price = float(odds[combination])
                expected_value = probability * price
                if price <= 1.0 or expected_value < 1.0:
                    continue
                full_kelly = (expected_value - 1.0) / (price - 1.0)
                desired_fraction = min(0.02, 0.25 * max(0.0, full_kelly))
                desired_stake = (
                    int(balance * desired_fraction) // STAKE_YEN * STAKE_YEN
                )
                if desired_stake >= STAKE_YEN:
                    candidates.append(
                        (expected_value, probability, combination, desired_stake)
                    )
            candidates.sort(reverse=True)
            race_cap = int(balance * 0.05) // STAKE_YEN * STAKE_YEN
            remaining_cap = min(balance, race_cap)
            allocations: dict[str, int] = {}
            for _ev, _probability, combination, desired_stake in candidates:
                stake = min(desired_stake, remaining_cap)
                stake = stake // STAKE_YEN * STAKE_YEN
                if stake < STAKE_YEN:
                    continue
                allocations[combination] = stake
                remaining_cap -= stake
                if remaining_cap < STAKE_YEN:
                    break
            race_stake = sum(allocations.values())
            if not race_stake:
                continue
            balance -= race_stake
            max_drawdown = max(max_drawdown, peak_balance - balance)
            race_return = 0
            actual = str(race["actual_combination"])
            if actual in allocations:
                actual_units = allocations[actual] // STAKE_YEN
                race_return = actual_units * int(race["actual_payout_yen"])
                hits += 1
            pending.append(
                (
                    decision_at + timedelta(minutes=SETTLEMENT_DELAY_MINUTES),
                    race_return,
                )
            )
            stake_yen += race_stake
            return_yen += race_return
            units += race_stake // STAKE_YEN
            bets += len(allocations)
        for _settles_at, payout in sorted(pending):
            balance += payout
            peak_balance = max(peak_balance, balance)
            max_drawdown = max(max_drawdown, peak_balance - balance)
        row = {
            "race_date": race_date,
            "evaluated_races": len(ordered),
            "opening_balance_yen": opening_balance,
            "closing_balance_yen": balance,
            "stake_yen": stake_yen,
            "return_yen": return_yen,
            "profit_yen": balance - opening_balance,
            "roi": return_yen / stake_yen if stake_yen else None,
            "bets": bets,
            "units": units,
            "hit_bets": hits,
            "max_drawdown_yen": max_drawdown,
        }
        daily.append(row)
        total_stake += stake_yen
        total_return += return_yen
        total_units += units
        total_bets += bets
        total_hits += hits
    return {
        "name": "top5_ev_gte_1_quarter_kelly_sequential",
        "daily_budget_yen": int(daily_budget_yen),
        "profit_reinvestment": True,
        "settlement_delay_minutes": SETTLEMENT_DELAY_MINUTES,
        "fractional_kelly": 0.25,
        "ticket_cap_fraction": 0.02,
        "race_cap_fraction": 0.05,
        "stake_granularity_yen": STAKE_YEN,
        "evaluated_races": len(races),
        "evaluation_days": len(daily),
        "bets": total_bets,
        "units": total_units,
        "hit_bets": total_hits,
        "stake_yen": total_stake,
        "return_yen": total_return,
        "profit_yen": sum(int(row["profit_yen"]) for row in daily),
        "roi": total_return / total_stake if total_stake else None,
        "winning_days": sum(int(row["profit_yen"] > 0) for row in daily),
        "max_drawdown_yen": max(
            (int(row["max_drawdown_yen"]) for row in daily),
            default=0,
        ),
        "daily": daily,
    }
