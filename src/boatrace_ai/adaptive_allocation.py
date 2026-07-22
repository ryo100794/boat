from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from .roi_attribution import update_roi_attribution


PAYOUT_UNIT_YEN = 100
MIN_STAKE_UNIT_YEN = 100


def folds_by_full_day(
    race_keys: list[tuple[str, str, str, int]],
    *,
    folds: int,
    min_train_races: int,
) -> list[tuple[set[str], set[str], set[str]]]:
    dates = sorted({race_date for _race_id, race_date, _jcd, _rno in race_keys})
    race_ids_by_date: dict[str, list[str]] = defaultdict(list)
    for race_id, race_date, _jcd, _rno in race_keys:
        race_ids_by_date[race_date].append(race_id)

    first_test_date_index = 0
    race_count = 0
    for index, race_date in enumerate(dates):
        race_count += len(race_ids_by_date[race_date])
        if race_count >= min_train_races:
            first_test_date_index = index + 1
            break
    if first_test_date_index >= len(dates):
        raise ValueError("min_train_races leaves no full test day")

    test_dates_all = dates[first_test_date_index:]
    window = max(1, math.ceil(len(test_dates_all) / folds))
    specs = []
    for fold in range(folds):
        start = first_test_date_index + fold * window
        end = min(len(dates), start + window)
        if start >= end:
            continue
        train_dates = set(dates[:start])
        test_dates = set(dates[start:end])
        train_races = {
            race_id
            for race_id, race_date, _jcd, _rno in race_keys
            if race_date in train_dates
        }
        test_races = {
            race_id
            for race_id, race_date, _jcd, _rno in race_keys
            if race_date in test_dates
        }
        specs.append((train_races, test_races, test_dates))
    return specs


def allocate_adaptive_day(
    race_date: str,
    candidates: list[dict[str, Any]],
    evaluated_races: set[str],
    *,
    daily_budget_yen: int,
    fractional_kelly: float,
    max_daily_exposure_fraction: float,
    min_daily_exposure_fraction: float,
    race_cap_fraction: float,
    ticket_cap_fraction: float,
    max_daily_tickets: int | None,
    allocation_mode: str,
    stake_granularity_yen: int,
    min_stake_yen: int,
    roi_attribution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prepared = []
    for item in candidates:
        estimated_odds = float(item["estimated_odds"])
        edge = float(item["estimated_ev"]) - 1.0
        if estimated_odds <= 1.0 or edge <= 0.0:
            continue
        kelly_fraction = edge / (estimated_odds - 1.0)
        if kelly_fraction <= 0.0 or not math.isfinite(kelly_fraction):
            continue
        stake_fraction = min(ticket_cap_fraction, fractional_kelly * kelly_fraction)
        if stake_fraction <= 0.0:
            continue
        prepared.append(
            {
                **item,
                "edge": edge,
                "kelly_fraction": kelly_fraction,
                "stake_fraction": stake_fraction,
            }
        )

    raw_positive_edge_tickets = len(prepared)
    if max_daily_tickets is not None and max_daily_tickets > 0:
        prepared = sorted(
            prepared,
            key=lambda row: (
                row["estimated_ev"],
                row["probability"],
                row["stake_fraction"],
            ),
            reverse=True,
        )[:max_daily_tickets]

    if allocation_mode == "normalized_kelly":
        total_fraction = sum(float(item["stake_fraction"]) for item in prepared)
        if total_fraction > 0.0 and total_fraction < min_daily_exposure_fraction:
            target_fraction = min(
                max_daily_exposure_fraction,
                min_daily_exposure_fraction,
            )
            scale = target_fraction / total_fraction
            for item in prepared:
                item["stake_fraction"] = min(
                    ticket_cap_fraction,
                    float(item["stake_fraction"]) * scale,
                )

    apply_fraction_caps(
        prepared,
        max_daily_exposure_fraction=max_daily_exposure_fraction,
        race_cap_fraction=race_cap_fraction,
    )

    planned_stakes = plan_stakes(
        prepared,
        daily_budget_yen=daily_budget_yen,
        min_daily_exposure_fraction=min_daily_exposure_fraction,
        max_daily_exposure_fraction=max_daily_exposure_fraction,
        race_cap_fraction=race_cap_fraction,
        ticket_cap_fraction=ticket_cap_fraction,
        allocation_mode=allocation_mode,
        stake_granularity_yen=stake_granularity_yen,
    )
    selected = []
    stake_yen = 0
    return_yen = 0
    hit_tickets = 0
    ranked_indices = sorted(
        range(len(prepared)),
        key=lambda index: (
            prepared[index]["stake_fraction"],
            prepared[index]["estimated_ev"],
        ),
        reverse=True,
    )
    for index in ranked_indices:
        item = prepared[index]
        ticket_stake = planned_stakes[index]
        if ticket_stake < min_stake_yen:
            continue
        ticket_return = (
            int(
                round(
                    ticket_stake
                    * int(item["actual_payout_yen"])
                    / PAYOUT_UNIT_YEN
                )
            )
            if item["hit"]
            else 0
        )
        stake_yen += ticket_stake
        return_yen += ticket_return
        if item["hit"]:
            hit_tickets += 1
        selected.append(
            {
                **item,
                "stake_yen": ticket_stake,
                "return_yen": ticket_return,
                "profit_yen": ticket_return - ticket_stake,
            }
        )

    if roi_attribution is not None:
        for item in selected:
            update_roi_attribution(roi_attribution, item)

    profit_yen = return_yen - stake_yen
    selected_races = {str(item["race_id"]) for item in selected}
    hit_races = {str(item["race_id"]) for item in selected if item["hit"]}
    return {
        "race_date": race_date,
        "evaluated_races": len(evaluated_races),
        "candidate_tickets": len(candidates),
        "positive_edge_tickets": raw_positive_edge_tickets,
        "allocation_candidate_tickets": len(prepared),
        "tickets": len(selected),
        "races_bet": len(selected_races),
        "hit_tickets": hit_tickets,
        "hit_races": len(hit_races),
        "stake_yen": stake_yen,
        "return_yen": return_yen,
        "profit_yen": profit_yen,
        "roi": return_yen / stake_yen if stake_yen else None,
        "budget_used_fraction": (
            stake_yen / daily_budget_yen if daily_budget_yen else 0.0
        ),
        "avg_stake_yen": stake_yen / len(selected) if selected else 0.0,
        "max_stake_yen": max(
            (item["stake_yen"] for item in selected),
            default=0,
        ),
        "selected_sample": selection_sample(selected),
    }


def plan_stakes(
    prepared: list[dict[str, Any]],
    *,
    daily_budget_yen: int,
    min_daily_exposure_fraction: float,
    max_daily_exposure_fraction: float,
    race_cap_fraction: float,
    ticket_cap_fraction: float,
    allocation_mode: str,
    stake_granularity_yen: int,
) -> list[int]:
    stakes = [
        floor_to_granularity(
            daily_budget_yen * float(item["stake_fraction"]),
            stake_granularity_yen,
        )
        for item in prepared
    ]
    if allocation_mode != "normalized_kelly" or not prepared:
        return stakes

    daily_cap = floor_to_granularity(
        daily_budget_yen * max_daily_exposure_fraction,
        stake_granularity_yen,
    )
    target = min(
        daily_cap,
        floor_to_granularity(
            daily_budget_yen * min_daily_exposure_fraction,
            stake_granularity_yen,
        ),
    )
    ticket_cap = floor_to_granularity(
        daily_budget_yen * ticket_cap_fraction,
        stake_granularity_yen,
    )
    race_cap = floor_to_granularity(
        daily_budget_yen * race_cap_fraction,
        stake_granularity_yen,
    )
    race_stakes: dict[str, int] = defaultdict(int)
    for index, item in enumerate(prepared):
        race_stakes[str(item["race_id"])] += stakes[index]

    while sum(stakes) < target:
        eligible = [
            index
            for index, item in enumerate(prepared)
            if stakes[index] + stake_granularity_yen <= ticket_cap
            and race_stakes[str(item["race_id"])] + stake_granularity_yen
            <= race_cap
            and sum(stakes) + stake_granularity_yen <= daily_cap
        ]
        if not eligible:
            break
        index = max(
            eligible,
            key=lambda candidate: (
                daily_budget_yen * float(prepared[candidate]["stake_fraction"])
                - stakes[candidate],
                float(prepared[candidate]["estimated_ev"]),
                float(prepared[candidate]["probability"]),
            ),
        )
        stakes[index] += stake_granularity_yen
        race_stakes[str(prepared[index]["race_id"])] += stake_granularity_yen
    return stakes


def apply_fraction_caps(
    prepared: list[dict[str, Any]],
    *,
    max_daily_exposure_fraction: float,
    race_cap_fraction: float,
) -> None:
    by_race: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in prepared:
        by_race[str(item["race_id"])].append(item)
    for race_items in by_race.values():
        race_total = sum(float(item["stake_fraction"]) for item in race_items)
        if race_total > race_cap_fraction:
            scale = race_cap_fraction / race_total
            for item in race_items:
                item["stake_fraction"] = float(item["stake_fraction"]) * scale

    total_fraction = sum(float(item["stake_fraction"]) for item in prepared)
    if total_fraction > max_daily_exposure_fraction:
        scale = max_daily_exposure_fraction / total_fraction
        for item in prepared:
            item["stake_fraction"] = float(item["stake_fraction"]) * scale


def selection_sample(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = sorted(
        selected,
        key=lambda item: (item["stake_yen"], item["estimated_ev"]),
        reverse=True,
    )[:12]
    sample = []
    for item in rows:
        row = {
            "race_id": item["race_id"],
            "combination": item["combination"],
            "stake_yen": item["stake_yen"],
            "estimated_ev": round(float(item["estimated_ev"]), 6),
            "kelly_fraction": round(float(item["kelly_fraction"]), 6),
            "hit": bool(item["hit"]),
            "return_yen": item["return_yen"],
        }
        for key in (
            "odds_source",
            "real_odds_snapshot_id",
            "real_odds_captured_at",
            "real_odds_deadline_at",
            "real_odds_combinations",
        ):
            if item.get(key) is not None:
                row[key] = item[key]
        sample.append(row)
    return sample


def append_day_result(
    daily_rows: list[dict[str, Any]],
    totals: dict[str, float],
    day_result: dict[str, Any],
    *,
    cumulative_profit: int,
    peak_profit: int,
    max_drawdown: int,
) -> tuple[int, int, int]:
    cumulative_profit += int(day_result["profit_yen"])
    peak_profit = max(peak_profit, cumulative_profit)
    max_drawdown = max(max_drawdown, peak_profit - cumulative_profit)
    day_result["cumulative_profit_yen"] = cumulative_profit
    daily_rows.append(day_result)
    for key in (
        "candidate_tickets",
        "positive_edge_tickets",
        "allocation_candidate_tickets",
        "tickets",
        "races_bet",
        "hit_tickets",
        "hit_races",
        "stake_yen",
        "return_yen",
        "profit_yen",
        "evaluated_races",
    ):
        totals[key] += float(day_result.get(key, 0) or 0)
    if day_result["stake_yen"] > 0:
        totals["days_with_bets"] += 1.0
        if day_result["profit_yen"] > 0:
            totals["winning_days"] += 1.0
        elif day_result["profit_yen"] < 0:
            totals["losing_days"] += 1.0
        else:
            totals["breakeven_days"] += 1.0
    return cumulative_profit, peak_profit, max_drawdown


def zero_totals() -> dict[str, float]:
    return {
        "candidate_tickets": 0.0,
        "positive_edge_tickets": 0.0,
        "allocation_candidate_tickets": 0.0,
        "tickets": 0.0,
        "races_bet": 0.0,
        "hit_tickets": 0.0,
        "hit_races": 0.0,
        "stake_yen": 0.0,
        "return_yen": 0.0,
        "profit_yen": 0.0,
        "evaluated_races": 0.0,
        "days_with_bets": 0.0,
        "winning_days": 0.0,
        "losing_days": 0.0,
        "breakeven_days": 0.0,
    }


def floor_to_granularity(value: float, granularity: int) -> int:
    if not math.isfinite(value) or value <= 0:
        return 0
    granularity = max(1, int(granularity))
    return int(math.floor(value / granularity) * granularity)


def validate_policy(
    *,
    daily_budget_yen: int,
    fractional_kelly: float,
    max_daily_exposure_fraction: float,
    min_daily_exposure_fraction: float,
    race_cap_fraction: float,
    ticket_cap_fraction: float,
    max_daily_tickets: int | None,
    allocation_mode: str,
    stake_granularity_yen: int,
    min_stake_yen: int,
) -> None:
    if daily_budget_yen <= 0:
        raise ValueError("daily_budget_yen must be positive")
    for name, value in (
        ("fractional_kelly", fractional_kelly),
        ("max_daily_exposure_fraction", max_daily_exposure_fraction),
        ("race_cap_fraction", race_cap_fraction),
        ("ticket_cap_fraction", ticket_cap_fraction),
    ):
        if value <= 0.0 or not math.isfinite(value):
            raise ValueError(f"{name} must be positive")
    if min_daily_exposure_fraction < 0.0 or not math.isfinite(
        min_daily_exposure_fraction
    ):
        raise ValueError("min_daily_exposure_fraction must be non-negative")
    if max_daily_exposure_fraction > 1.0:
        raise ValueError("max_daily_exposure_fraction must not exceed 1.0")
    if min_daily_exposure_fraction > max_daily_exposure_fraction:
        raise ValueError(
            "min_daily_exposure_fraction must not exceed max_daily_exposure_fraction"
        )
    if allocation_mode not in {"kelly_floor", "normalized_kelly"}:
        raise ValueError("allocation_mode must be kelly_floor or normalized_kelly")
    if max_daily_tickets is not None and max_daily_tickets < 1:
        raise ValueError("max_daily_tickets must be positive when set")
    if race_cap_fraction > max_daily_exposure_fraction:
        raise ValueError(
            "race_cap_fraction must not exceed max_daily_exposure_fraction"
        )
    if ticket_cap_fraction > race_cap_fraction:
        raise ValueError("ticket_cap_fraction must not exceed race_cap_fraction")
    if stake_granularity_yen < MIN_STAKE_UNIT_YEN:
        raise ValueError("stake_granularity_yen must be at least 100")
    if stake_granularity_yen % MIN_STAKE_UNIT_YEN:
        raise ValueError("stake_granularity_yen must be divisible by 100")
    if min_stake_yen < MIN_STAKE_UNIT_YEN:
        raise ValueError("min_stake_yen must be at least 100")
    if min_stake_yen % MIN_STAKE_UNIT_YEN:
        raise ValueError("min_stake_yen must be divisible by 100")
    if daily_budget_yen < min_stake_yen:
        raise ValueError("daily_budget_yen must be at least min_stake_yen")
    if daily_budget_yen % stake_granularity_yen:
        raise ValueError("daily_budget_yen must be divisible by stake_granularity_yen")
