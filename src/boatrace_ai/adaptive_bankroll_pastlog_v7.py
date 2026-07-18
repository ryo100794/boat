from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .bankroll_backtest import (
    PAYOUT_UNIT_YEN,
    _build_payout_model,
    _candidate_tickets,
    _load_trifecta_payouts,
)
from .db import connection, init_db
from .modeling_pastlog_v7_stream_hash import (
    FEATURE_SET,
    iter_scored_races,
    load_complete_race_ids,
    train_bundle,
)


DEFAULT_STAKE_UNIT_YEN = 100


def adaptive_bankroll_streaming(
    conn,
    *,
    output_path: Path,
    daily_budget_yen: int = 10_000,
    folds: int = 5,
    min_train_races: int = 500,
    ev_threshold: float = 1.0,
    payout_prior_weight: float = 30.0,
    fractional_kelly: float = 0.25,
    max_daily_exposure_fraction: float = 1.0,
    race_cap_fraction: float = 0.20,
    ticket_cap_fraction: float = 0.04,
    stake_granularity_yen: int = DEFAULT_STAKE_UNIT_YEN,
    min_stake_yen: int = DEFAULT_STAKE_UNIT_YEN,
    batch_size: int = 24_000,
    epochs: int = 1,
) -> dict[str, Any]:
    _validate_policy(
        daily_budget_yen=daily_budget_yen,
        fractional_kelly=fractional_kelly,
        max_daily_exposure_fraction=max_daily_exposure_fraction,
        race_cap_fraction=race_cap_fraction,
        ticket_cap_fraction=ticket_cap_fraction,
        stake_granularity_yen=stake_granularity_yen,
        min_stake_yen=min_stake_yen,
    )
    race_keys = load_complete_race_ids(conn)
    if len(race_keys) < min_train_races + folds:
        raise ValueError(f"not enough parsed races: {len(race_keys)}")

    fold_specs = _folds_by_full_day(race_keys, folds=folds, min_train_races=min_train_races)
    race_date_by_id = {race_id: race_date for race_id, race_date, _jcd, _rno in race_keys}
    all_race_ids = [race_id for race_id, *_ in race_keys]
    payouts = _load_trifecta_payouts(conn)

    evaluated_races: set[str] = set()
    daily_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    totals = _zero_totals()
    cumulative_profit = 0
    peak_profit = 0
    max_drawdown = 0

    for fold_index, (train_races, test_races, test_dates) in enumerate(fold_specs, start=1):
        if not train_races or not test_races:
            continue
        payout_model = _build_payout_model(
            payouts,
            train_races=train_races,
            prior_weight=payout_prior_weight,
        )
        bundle = train_bundle(conn, include_races=train_races, batch_size=batch_size, epochs=epochs)
        fold_candidate_count = 0
        fold_selected_count = 0
        fold_evaluated = 0
        current_date: str | None = None
        day_candidates: list[dict[str, Any]] = []
        day_evaluated: set[str] = set()

        for rows in iter_scored_races(conn, bundle=bundle, include_races=test_races):
            race_id_value = str(rows[0]["race_id"])
            race_date_value = str(rows[0]["race_date"])
            if current_date is None:
                current_date = race_date_value
            if race_date_value != current_date:
                day_result = _allocate_adaptive_day(
                    current_date,
                    day_candidates,
                    day_evaluated,
                    daily_budget_yen=daily_budget_yen,
                    fractional_kelly=fractional_kelly,
                    max_daily_exposure_fraction=max_daily_exposure_fraction,
                    race_cap_fraction=race_cap_fraction,
                    ticket_cap_fraction=ticket_cap_fraction,
                    stake_granularity_yen=stake_granularity_yen,
                    min_stake_yen=min_stake_yen,
                )
                cumulative_profit, peak_profit, max_drawdown = _append_day_result(
                    daily_rows,
                    totals,
                    day_result,
                    cumulative_profit=cumulative_profit,
                    peak_profit=peak_profit,
                    max_drawdown=max_drawdown,
                )
                fold_selected_count += day_result["tickets"]
                current_date = race_date_value
                day_candidates = []
                day_evaluated = set()

            payout = payouts.get(race_id_value)
            if len(rows) != 6 or not payout:
                continue
            evaluated_races.add(race_id_value)
            day_evaluated.add(race_id_value)
            fold_evaluated += 1
            race_candidates = _candidate_tickets(
                rows,
                actual=payout,
                payout_model=payout_model,
                ev_threshold=ev_threshold,
            )
            fold_candidate_count += len(race_candidates)
            day_candidates.extend(race_candidates)

        if current_date is not None:
            day_result = _allocate_adaptive_day(
                current_date,
                day_candidates,
                day_evaluated,
                daily_budget_yen=daily_budget_yen,
                fractional_kelly=fractional_kelly,
                max_daily_exposure_fraction=max_daily_exposure_fraction,
                race_cap_fraction=race_cap_fraction,
                ticket_cap_fraction=ticket_cap_fraction,
                stake_granularity_yen=stake_granularity_yen,
                min_stake_yen=min_stake_yen,
            )
            cumulative_profit, peak_profit, max_drawdown = _append_day_result(
                daily_rows,
                totals,
                day_result,
                cumulative_profit=cumulative_profit,
                peak_profit=peak_profit,
                max_drawdown=max_drawdown,
            )
            fold_selected_count += day_result["tickets"]

        fold_row = {
            "fold": fold_index,
            "train_races": len(train_races),
            "test_races": len(test_races),
            "test_days": len(test_dates),
            "evaluated_races": fold_evaluated,
            "candidate_tickets": fold_candidate_count,
            "selected_tickets": fold_selected_count,
        }
        fold_rows.append(fold_row)
        print(json.dumps(fold_row, ensure_ascii=False), flush=True)

    result = _summarize(
        daily_rows,
        totals,
        evaluated_races=evaluated_races,
        all_race_count=len(all_race_ids),
        max_drawdown=max_drawdown,
        policy={
            "daily_budget_yen": daily_budget_yen,
            "bet_type": "3連単",
            "include_odds": False,
            "ev_threshold": ev_threshold,
            "stake_model": "adaptive_unit_yen",
            "unit_yen": stake_granularity_yen,
            "max_tickets_per_race": None,
            "fractional_kelly": fractional_kelly,
            "max_daily_exposure_fraction": max_daily_exposure_fraction,
            "race_cap_fraction": race_cap_fraction,
            "ticket_cap_fraction": ticket_cap_fraction,
            "stake_granularity_yen": stake_granularity_yen,
            "min_stake_yen": min_stake_yen,
            "payout_estimator": "train-fold average payout by trifecta combination, blended with train-fold global average",
            "payout_prior_weight": payout_prior_weight,
            "allocation": "stake is proportional to positive Kelly edge, capped by daily/race/ticket risk fractions; each ticket stake is floored to stake_granularity_yen and tickets below min_stake_yen are skipped",
            "feature_set": FEATURE_SET,
            "model": "win_model_pastlog_v7_stream_hash",
        },
        folds=fold_rows,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _folds_by_full_day(
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
        train_races = {race_id for race_id, race_date, _jcd, _rno in race_keys if race_date in train_dates}
        test_races = {race_id for race_id, race_date, _jcd, _rno in race_keys if race_date in test_dates}
        specs.append((train_races, test_races, test_dates))
    return specs


def _allocate_adaptive_day(
    race_date: str,
    candidates: list[dict[str, Any]],
    evaluated_races: set[str],
    *,
    daily_budget_yen: int,
    fractional_kelly: float,
    max_daily_exposure_fraction: float,
    race_cap_fraction: float,
    ticket_cap_fraction: float,
    stake_granularity_yen: int,
    min_stake_yen: int,
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
    max_fraction = max_daily_exposure_fraction
    if total_fraction > max_fraction:
        scale = max_fraction / total_fraction
        for item in prepared:
            item["stake_fraction"] = float(item["stake_fraction"]) * scale

    selected = []
    stake_yen = 0
    return_yen = 0
    hit_tickets = 0
    for item in sorted(prepared, key=lambda row: (row["stake_fraction"], row["estimated_ev"]), reverse=True):
        ticket_stake = _floor_to_granularity(
            daily_budget_yen * float(item["stake_fraction"]),
            stake_granularity_yen,
        )
        if ticket_stake < min_stake_yen:
            continue
        ticket_return = (
            int(round(ticket_stake * int(item["actual_payout_yen"]) / PAYOUT_UNIT_YEN))
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

    profit_yen = return_yen - stake_yen
    selected_races = {str(item["race_id"]) for item in selected}
    hit_races = {str(item["race_id"]) for item in selected if item["hit"]}
    return {
        "race_date": race_date,
        "evaluated_races": len(evaluated_races),
        "candidate_tickets": len(candidates),
        "positive_edge_tickets": len(prepared),
        "tickets": len(selected),
        "races_bet": len(selected_races),
        "hit_tickets": hit_tickets,
        "hit_races": len(hit_races),
        "stake_yen": stake_yen,
        "return_yen": return_yen,
        "profit_yen": profit_yen,
        "roi": return_yen / stake_yen if stake_yen else None,
        "budget_used_fraction": stake_yen / daily_budget_yen if daily_budget_yen else 0.0,
        "avg_stake_yen": stake_yen / len(selected) if selected else 0.0,
        "max_stake_yen": max((item["stake_yen"] for item in selected), default=0),
        "selected_sample": _selection_sample(selected),
    }


def _selection_sample(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = sorted(selected, key=lambda item: (item["stake_yen"], item["estimated_ev"]), reverse=True)[:12]
    return [
        {
            "race_id": item["race_id"],
            "combination": item["combination"],
            "stake_yen": item["stake_yen"],
            "estimated_ev": round(float(item["estimated_ev"]), 6),
            "kelly_fraction": round(float(item["kelly_fraction"]), 6),
            "hit": bool(item["hit"]),
            "return_yen": item["return_yen"],
        }
        for item in rows
    ]


def _append_day_result(
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


def _summarize(
    daily_rows: list[dict[str, Any]],
    totals: dict[str, float],
    *,
    evaluated_races: set[str],
    all_race_count: int,
    max_drawdown: int,
    policy: dict[str, Any],
    folds: list[dict[str, Any]],
) -> dict[str, Any]:
    stake_yen = int(totals["stake_yen"])
    return_yen = int(totals["return_yen"])
    tickets = int(totals["tickets"])
    races_bet = int(totals["races_bet"])
    hit_tickets = int(totals["hit_tickets"])
    hit_races = int(totals["hit_races"])
    days_with_bets = int(totals["days_with_bets"])
    race_days = len(daily_rows)
    return {
        "generated_at": _now(),
        "policy": policy,
        "folds": folds,
        "examples": 0,
        "races": all_race_count,
        "evaluated_races": len(evaluated_races),
        "candidate_tickets": int(totals["candidate_tickets"]),
        "positive_edge_tickets": int(totals["positive_edge_tickets"]),
        "race_days": race_days,
        "days_with_bets": days_with_bets,
        "winning_days": int(totals["winning_days"]),
        "losing_days": int(totals["losing_days"]),
        "breakeven_days": int(totals["breakeven_days"]),
        "selected_races": races_bet,
        "hit_races": hit_races,
        "tickets": tickets,
        "hit_tickets": hit_tickets,
        "ticket_hit_rate": hit_tickets / tickets if tickets else 0.0,
        "race_hit_rate": hit_races / races_bet if races_bet else 0.0,
        "stake_yen": stake_yen,
        "return_yen": return_yen,
        "profit_yen": return_yen - stake_yen,
        "roi": return_yen / stake_yen if stake_yen else 0.0,
        "max_drawdown_yen": max_drawdown,
        "budget_utilization": stake_yen / (policy["daily_budget_yen"] * race_days) if race_days else 0.0,
        "avg_stake_yen_per_ticket": stake_yen / tickets if tickets else 0.0,
        "avg_tickets_per_betting_day": tickets / days_with_bets if days_with_bets else 0.0,
        "avg_tickets_per_selected_race": tickets / races_bet if races_bet else 0.0,
        "daily": daily_rows,
        "best_days": sorted(daily_rows, key=lambda row: row["profit_yen"], reverse=True)[:10],
        "worst_days": sorted(daily_rows, key=lambda row: row["profit_yen"])[:10],
        "feature_set": FEATURE_SET,
        "model": "win_model_pastlog_v7_stream_hash",
    }


def _zero_totals() -> dict[str, float]:
    return {
        "candidate_tickets": 0.0,
        "positive_edge_tickets": 0.0,
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


def _floor_to_granularity(value: float, granularity: int) -> int:
    if not math.isfinite(value) or value <= 0:
        return 0
    granularity = max(1, int(granularity))
    return int(math.floor(value / granularity) * granularity)


def _validate_policy(
    *,
    daily_budget_yen: int,
    fractional_kelly: float,
    max_daily_exposure_fraction: float,
    race_cap_fraction: float,
    ticket_cap_fraction: float,
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
    if max_daily_exposure_fraction > 1.0:
        raise ValueError("max_daily_exposure_fraction must not exceed 1.0")
    if race_cap_fraction > max_daily_exposure_fraction:
        raise ValueError("race_cap_fraction must not exceed max_daily_exposure_fraction")
    if ticket_cap_fraction > race_cap_fraction:
        raise ValueError("ticket_cap_fraction must not exceed race_cap_fraction")
    if stake_granularity_yen < DEFAULT_STAKE_UNIT_YEN:
        raise ValueError("stake_granularity_yen must be at least 100")
    if stake_granularity_yen % DEFAULT_STAKE_UNIT_YEN:
        raise ValueError("stake_granularity_yen must be divisible by 100")
    if min_stake_yen < DEFAULT_STAKE_UNIT_YEN:
        raise ValueError("min_stake_yen must be at least 100")
    if min_stake_yen % DEFAULT_STAKE_UNIT_YEN:
        raise ValueError("min_stake_yen must be divisible by 100")
    if daily_budget_yen < min_stake_yen:
        raise ValueError("daily_budget_yen must be at least min_stake_yen")
    if daily_budget_yen % stake_granularity_yen:
        raise ValueError("daily_budget_yen must be divisible by stake_granularity_yen")


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Adaptive bankroll backtest for past-log v7 streaming model.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--output", default="data/models/bankroll_backtest_pastlog_v7_adaptive_10000.json")
    parser.add_argument("--daily-budget-yen", type=int, default=10_000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-train-races", type=int, default=500)
    parser.add_argument("--ev-threshold", type=float, default=1.0)
    parser.add_argument("--payout-prior-weight", type=float, default=30.0)
    parser.add_argument("--fractional-kelly", type=float, default=0.25)
    parser.add_argument("--max-daily-exposure-fraction", type=float, default=1.0)
    parser.add_argument("--race-cap-fraction", type=float, default=0.20)
    parser.add_argument("--ticket-cap-fraction", type=float, default=0.04)
    parser.add_argument("--stake-granularity-yen", type=int, default=DEFAULT_STAKE_UNIT_YEN)
    parser.add_argument("--min-stake-yen", type=int, default=DEFAULT_STAKE_UNIT_YEN)
    parser.add_argument("--batch-size", type=int, default=24_000)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args(argv)
    init_db(args.db)
    with connection(args.db) as conn:
        result = adaptive_bankroll_streaming(
            conn,
            output_path=Path(args.output),
            daily_budget_yen=args.daily_budget_yen,
            folds=args.folds,
            min_train_races=args.min_train_races,
            ev_threshold=args.ev_threshold,
            payout_prior_weight=args.payout_prior_weight,
            fractional_kelly=args.fractional_kelly,
            max_daily_exposure_fraction=args.max_daily_exposure_fraction,
            race_cap_fraction=args.race_cap_fraction,
            ticket_cap_fraction=args.ticket_cap_fraction,
            stake_granularity_yen=args.stake_granularity_yen,
            min_stake_yen=args.min_stake_yen,
            batch_size=args.batch_size,
            epochs=args.epochs,
        )
    compact = {key: value for key, value in result.items() if key != "daily"}
    compact["daily_rows"] = len(result.get("daily", []))
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
