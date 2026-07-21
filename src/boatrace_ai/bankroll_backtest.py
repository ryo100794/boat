from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from sklearn.metrics import brier_score_loss

from .bankroll_policy import race_confidence
from .db import connection, init_db
from .features import latest_trifecta_odds_before_deadline, load_training_examples
from .modeling import (
    _make_pipeline,
    _normalize_lane_probs,
    _positive_probs,
    _race_level_metrics,
    _safe_log_loss,
    trifecta_predictions,
)

PAYOUT_BET_TYPE = "3連単"
PAYOUT_UNIT_YEN = 100


def bankroll_backtest(
    conn,
    *,
    output_path: Path,
    daily_budget_yen: int = 10_000,
    unit_yen: int = 100,
    folds: int = 5,
    min_train_races: int = 500,
    include_odds: bool = False,
    from_date: str | None = None,
    min_odds_snapshots: int = 0,
    ev_threshold: float = 1.0,
    min_ticket_probability: float = 0.0,
    max_estimated_odds: float | None = None,
    max_tickets_per_race: int = 5,
    payout_prior_weight: float = 30.0,
    require_real_odds: bool = False,
) -> dict[str, Any]:
    if daily_budget_yen < unit_yen:
        raise ValueError("daily_budget_yen must be at least one betting unit")
    if daily_budget_yen % unit_yen:
        raise ValueError("daily_budget_yen must be divisible by unit_yen")
    if not 0.0 <= min_ticket_probability <= 1.0:
        raise ValueError("min_ticket_probability must be between zero and one")
    if max_estimated_odds is not None and max_estimated_odds <= 0.0:
        raise ValueError("max_estimated_odds must be positive")
    if max_tickets_per_race < 1:
        raise ValueError("max_tickets_per_race must be positive")

    X, y, meta = load_training_examples(
        conn,
        from_date=from_date,
        include_odds=include_odds,
        min_odds_snapshots=min_odds_snapshots,
        complete_results_only=include_odds,
    )
    if len(X) < 100:
        raise ValueError(f"not enough parsed historical examples: {len(X)}")
    races = sorted({row["race_id"] for row in meta})
    if len(races) < max(10, min_train_races + folds):
        raise ValueError(
            f"not enough parsed historical races: {len(races)} "
            f"(need at least {min_train_races + folds})"
        )

    payouts = _load_trifecta_payouts(conn)
    race_index = {race: idx for idx, race in enumerate(races)}
    test_window = max(1, (len(races) - min_train_races) // folds)
    candidates: list[dict[str, Any]] = []
    evaluated_races: set[str] = set()
    real_odds_by_race: dict[str, dict[str, Any] | None] = {}
    real_odds_races: set[str] = set()
    skipped_no_real_odds = 0
    fold_results: list[dict[str, Any]] = []
    all_entry_probs: list[float] = []
    all_entry_labels: list[int] = []
    race_predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for fold in range(folds):
        test_start = min_train_races + fold * test_window
        test_end = len(races) if fold == folds - 1 else min(len(races), test_start + test_window)
        train_races = set(races[:test_start])
        test_races = set(races[test_start:test_end])
        train_idx = [i for i, row in enumerate(meta) if row["race_id"] in train_races]
        test_idx = [i for i, row in enumerate(meta) if row["race_id"] in test_races]
        if not train_idx or not test_idx:
            continue

        payout_model = _build_payout_model(
            payouts,
            train_races=train_races,
            prior_weight=payout_prior_weight,
        )
        pipeline = _make_pipeline()
        pipeline.fit([X[i] for i in train_idx], [y[i] for i in train_idx])
        probs = _positive_probs(pipeline, [X[i] for i in test_idx])
        all_entry_probs.extend(probs)
        all_entry_labels.extend(y[i] for i in test_idx)

        by_race: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for local_i, global_i in enumerate(test_idx):
            row = meta[global_i]
            by_race[row["race_id"]].append(
                {
                    "race_id": row["race_id"],
                    "race_date": row["race_date"],
                    "jcd": row["jcd"],
                    "rno": row["rno"],
                    "lane": row["lane"],
                    "rank": row["rank"],
                    "probability": probs[local_i],
                }
            )

        for race_id_value, rows in by_race.items():
            race_predictions[race_id_value].extend(rows)

        fold_candidates = 0
        fold_evaluated = 0
        fold_real_odds_races = 0
        fold_skipped_no_real_odds = 0
        for race_id_value, rows in by_race.items():
            payout = payouts.get(race_id_value)
            if len(rows) != 6 or not payout:
                continue
            real_odds_snapshot = None
            if require_real_odds:
                if race_id_value not in real_odds_by_race:
                    real_odds_by_race[race_id_value] = latest_trifecta_odds_before_deadline(conn, race_id_value)
                real_odds_snapshot = real_odds_by_race[race_id_value]
                if real_odds_snapshot is None:
                    skipped_no_real_odds += 1
                    fold_skipped_no_real_odds += 1
                    continue
                real_odds_races.add(race_id_value)
                fold_real_odds_races += 1
            evaluated_races.add(race_id_value)
            fold_evaluated += 1
            race_candidates = _candidate_tickets(
                rows,
                actual=payout,
                payout_model=payout_model,
                ev_threshold=ev_threshold,
                min_ticket_probability=min_ticket_probability,
                max_estimated_odds=max_estimated_odds,
                real_odds_snapshot=real_odds_snapshot,
            )[:max_tickets_per_race]
            fold_candidates += len(race_candidates)
            candidates.extend(race_candidates)

        fold_results.append(
            {
                "fold": fold + 1,
                "train_races": test_start,
                "test_races": test_end - test_start,
                "evaluated_races": fold_evaluated,
                "candidate_tickets": fold_candidates,
                "real_odds_races": fold_real_odds_races,
                "skipped_no_real_odds": fold_skipped_no_real_odds,
            }
        )

    allocated = _allocate_daily_budget(
        candidates,
        evaluated_races=evaluated_races,
        daily_budget_yen=daily_budget_yen,
        unit_yen=unit_yen,
    )
    prediction_metrics = _race_level_metrics(race_predictions)

    result = {
        "generated_at": _now(),
        "policy": {
            "daily_budget_yen": daily_budget_yen,
            "unit_yen": unit_yen,
            "bet_type": PAYOUT_BET_TYPE,
            "include_odds": include_odds,
            "require_real_odds": require_real_odds,
            "ev_threshold": ev_threshold,
            "min_ticket_probability": min_ticket_probability,
            "max_estimated_odds": max_estimated_odds,
            "max_tickets_per_race": max_tickets_per_race,
            "payout_estimator": "deadline real odds" if require_real_odds else "train-fold average payout by trifecta combination, blended with train-fold global average",
            "payout_prior_weight": payout_prior_weight,
            "allocation": "each day, rank positive-EV tickets by estimated EV; buy within daily budget and split stake in 100-yen units",
        },
        "folds": fold_results,
        "examples": len(X),
        "races": len(races),
        "from_date": from_date,
        "min_odds_snapshots": min_odds_snapshots,
        "entry_log_loss": _safe_log_loss(all_entry_labels, all_entry_probs),
        "entry_brier": (
            float(brier_score_loss(all_entry_labels, all_entry_probs))
            if all_entry_labels
            else None
        ),
        **prediction_metrics,
        "evaluated_races": len(evaluated_races),
        "real_odds_races": len(real_odds_races),
        "skipped_no_real_odds": skipped_no_real_odds,
        "candidate_tickets": len(candidates),
        "candidate_hit_tickets": sum(1 for item in candidates if item["hit"]),
        **allocated,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _load_trifecta_payouts(conn) -> dict[str, dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT r.race_id, r.race_date, r.jcd, r.rno, p.combination, p.payout_yen, p.popularity
        FROM payouts p
        JOIN races r ON r.race_id = p.race_id
        WHERE p.bet_type = ? AND p.payout_yen IS NOT NULL
        """,
        (PAYOUT_BET_TYPE,),
    ).fetchall()
    return {
        row["race_id"]: {
            "race_id": row["race_id"],
            "race_date": row["race_date"],
            "jcd": row["jcd"],
            "rno": int(row["rno"]),
            "combination": row["combination"],
            "payout_yen": int(row["payout_yen"]),
            "popularity": row["popularity"],
        }
        for row in rows
    }


def _build_payout_model(
    payouts: dict[str, dict[str, Any]],
    *,
    train_races: set[str],
    prior_weight: float,
) -> dict[str, dict[str, float]]:
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    global_sum = 0.0
    global_count = 0
    for race_id_value in train_races:
        payout = payouts.get(race_id_value)
        if not payout:
            continue
        combo = str(payout["combination"])
        yen = float(payout["payout_yen"])
        sums[combo] += yen
        counts[combo] += 1
        global_sum += yen
        global_count += 1
    global_mean = global_sum / global_count if global_count else 10_000.0
    estimates: dict[str, dict[str, float]] = {}
    for combo in {f"{a}-{b}-{c}" for a in range(1, 7) for b in range(1, 7) for c in range(1, 7) if len({a, b, c}) == 3}:
        count = counts.get(combo, 0)
        mean = (
            (sums.get(combo, 0.0) + global_mean * prior_weight)
            / (count + prior_weight)
        )
        estimates[combo] = {
            "estimated_payout_yen": mean,
            "estimated_odds": mean / PAYOUT_UNIT_YEN,
            "history_count": float(count),
        }
    return estimates


def _candidate_tickets(
    rows: list[dict[str, Any]],
    *,
    actual: dict[str, Any],
    payout_model: dict[str, dict[str, float]],
    ev_threshold: float,
    real_odds_snapshot: dict[str, Any] | None = None,
    min_ticket_probability: float = 0.0,
    max_estimated_odds: float | None = None,
) -> list[dict[str, Any]]:
    lane_probs = _normalize_lane_probs(
        {int(row["lane"]): float(row["probability"]) for row in rows}
    )
    confidence = race_confidence(lane_probs)
    race = rows[0]
    real_odds = real_odds_snapshot.get("odds", {}) if real_odds_snapshot else None
    feature_context_by_combination: dict[str, dict[str, Any]] = {}
    candidates = []
    for prediction in trifecta_predictions(lane_probs):
        combo = prediction["combination"]
        if real_odds is not None:
            try:
                estimated_odds = float(real_odds[combo])
            except (KeyError, TypeError, ValueError):
                continue
            if estimated_odds <= 0.0:
                continue
            estimated_payout_yen = estimated_odds * PAYOUT_UNIT_YEN
            payout_history_count = 0
            odds_source = "real"
        else:
            payout_estimate = payout_model.get(combo)
            if not payout_estimate:
                continue
            estimated_odds = float(payout_estimate["estimated_odds"])
            estimated_payout_yen = float(payout_estimate["estimated_payout_yen"])
            payout_history_count = int(payout_estimate["history_count"])
            odds_source = "payout_model"
        ticket_probability = float(prediction["probability"])
        if ticket_probability < min_ticket_probability:
            continue
        if max_estimated_odds is not None and estimated_odds > max_estimated_odds:
            continue
        estimated_ev = float(prediction["probability"]) * estimated_odds
        if estimated_ev < ev_threshold:
            continue
        item = {
            **confidence,
            "race_id": race["race_id"],
            "race_date": race["race_date"],
            "jcd": race["jcd"],
            "rno": int(race["rno"]),
            "combination": combo,
            "probability": float(prediction["probability"]),
            "estimated_odds": estimated_odds,
            "estimated_payout_yen": estimated_payout_yen,
            "estimated_ev": estimated_ev,
            "payout_history_count": payout_history_count,
            "odds_source": odds_source,
            "actual_combination": actual["combination"],
            "actual_payout_yen": int(actual["payout_yen"]),
            "hit": combo == actual["combination"],
        }
        feature_context = feature_context_by_combination.get(combo)
        if feature_context is None:
            feature_context = _ticket_feature_context(rows, combo)
            feature_context_by_combination[combo] = feature_context
        if feature_context:
            item["feature_context"] = feature_context
        if real_odds_snapshot is not None:
            item.update(
                {
                    "real_odds_snapshot_id": real_odds_snapshot.get("snapshot_id"),
                    "real_odds_captured_at": real_odds_snapshot.get("captured_at"),
                    "real_odds_deadline_at": real_odds_snapshot.get("odds_deadline_at"),
                    "real_odds_combinations": real_odds_snapshot.get("odds_count"),
                }
            )
        candidates.append(item)
    return sorted(
        candidates,
        key=lambda item: (item["estimated_ev"], item["probability"]),
        reverse=True,
    )


def _ticket_feature_context(rows: list[dict[str, Any]], combination: str) -> dict[str, Any]:
    diagnostics_by_lane = {
        int(row["lane"]): row.get("diagnostic_features") or {}
        for row in rows
        if row.get("lane") is not None
    }
    if not any(diagnostics_by_lane.values()):
        return {}
    try:
        lanes = [int(value) for value in combination.split("-")]
    except (TypeError, ValueError):
        return {}
    if len(lanes) != 3:
        return {}

    context: dict[str, Any] = {}
    common_keys = ("race_month", "race_weekday", "race_rno_bucket")
    first_diagnostics = diagnostics_by_lane.get(lanes[0], {})
    for key in common_keys:
        if first_diagnostics.get(key) is not None:
            context[key] = first_diagnostics[key]

    runner_keys = (
        "racer_class",
        "origin",
        "class_rank",
        "national_win_rate_rank",
        "local_win_rate_rank",
        "motor_2_rate_rank",
        "boat_2_rate_rank",
        "hist_racer_win_rate_s",
        "hist_racer_venue_win_rate_s",
        "hist_motor_win_rate_s",
        "hist_boat_win_rate_s",
        "series_win_rate",
        "series_avg_finish",
    )
    for position, lane in zip(("first", "second", "third"), lanes):
        diagnostics = diagnostics_by_lane.get(lane, {})
        for key in runner_keys:
            if diagnostics.get(key) is not None:
                context[f"{position}_{key}"] = diagnostics[key]
    return context


def _allocate_daily_budget(
    candidates: list[dict[str, Any]],
    *,
    evaluated_races: set[str],
    daily_budget_yen: int,
    unit_yen: int,
) -> dict[str, Any]:
    units_per_day = daily_budget_yen // unit_yen
    candidates_by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        candidates_by_day[item["race_date"]].append(item)

    evaluated_days = sorted({race_id[:10] for race_id in evaluated_races})
    selected: list[dict[str, Any]] = []
    daily_rows = []
    cumulative_profit = 0
    peak_profit = 0
    max_drawdown = 0

    for race_date in evaluated_days:
        day_candidates = sorted(
            candidates_by_day.get(race_date, []),
            key=lambda item: (item["estimated_ev"], item["probability"]),
            reverse=True,
        )
        day_selected = day_candidates[:units_per_day]
        stake_yen = 0
        return_yen = 0
        hit_tickets = 0
        if day_selected:
            base_units = units_per_day // len(day_selected)
            extra_units = units_per_day % len(day_selected)
            for index, item in enumerate(day_selected):
                units = base_units + (1 if index < extra_units else 0)
                ticket_stake = units * unit_yen
                ticket_return = (
                    int(round(ticket_stake * item["actual_payout_yen"] / PAYOUT_UNIT_YEN))
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
        cumulative_profit += profit_yen
        peak_profit = max(peak_profit, cumulative_profit)
        max_drawdown = max(max_drawdown, peak_profit - cumulative_profit)
        daily_rows.append(
            {
                "race_date": race_date,
                "candidate_tickets": len(day_candidates),
                "tickets": len(day_selected),
                "races_bet": len({item["race_id"] for item in day_selected}),
                "stake_yen": stake_yen,
                "return_yen": return_yen,
                "profit_yen": profit_yen,
                "hit_tickets": hit_tickets,
                "roi": return_yen / stake_yen if stake_yen else None,
                "cumulative_profit_yen": cumulative_profit,
            }
        )

    stake_total = sum(item["stake_yen"] for item in selected)
    return_total = sum(item["return_yen"] for item in selected)
    profit_total = return_total - stake_total
    hit_tickets = sum(1 for item in selected if item["hit"])
    selected_races = {item["race_id"] for item in selected}
    hit_races = {item["race_id"] for item in selected if item["hit"]}
    selected_odds = [float(item["estimated_odds"]) for item in selected]
    selected_probabilities = [float(item["probability"]) for item in selected]
    selected_evs = [float(item["estimated_ev"]) for item in selected]
    betting_days = [row for row in daily_rows if row["stake_yen"] > 0]

    return {
        "race_days": len(evaluated_days),
        "days_with_bets": len(betting_days),
        "winning_days": sum(1 for row in betting_days if row["profit_yen"] > 0),
        "losing_days": sum(1 for row in betting_days if row["profit_yen"] < 0),
        "breakeven_days": sum(1 for row in betting_days if row["profit_yen"] == 0),
        "selected_races": len(selected_races),
        "hit_races": len(hit_races),
        "tickets": len(selected),
        "hit_tickets": hit_tickets,
        "ticket_hit_rate": hit_tickets / len(selected) if selected else 0.0,
        "race_hit_rate": len(hit_races) / len(selected_races) if selected_races else 0.0,
        "avg_selected_odds": (
            sum(selected_odds) / len(selected_odds) if selected_odds else None
        ),
        "max_selected_odds": max(selected_odds) if selected_odds else None,
        "avg_selected_probability": (
            sum(selected_probabilities) / len(selected_probabilities)
            if selected_probabilities
            else None
        ),
        "avg_selected_ev": (
            sum(selected_evs) / len(selected_evs) if selected_evs else None
        ),
        "stake_yen": stake_total,
        "return_yen": return_total,
        "profit_yen": profit_total,
        "roi": return_total / stake_total if stake_total else 0.0,
        "max_drawdown_yen": max_drawdown,
        "daily": daily_rows,
        "best_days": sorted(daily_rows, key=lambda row: row["profit_yen"], reverse=True)[:10],
        "worst_days": sorted(daily_rows, key=lambda row: row["profit_yen"])[:10],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backtest daily bankroll-based trifecta buying.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--output", default="data/models/bankroll_backtest_10000.json")
    parser.add_argument("--daily-budget-yen", type=int, default=10_000)
    parser.add_argument("--unit-yen", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-train-races", type=int, default=500)
    parser.add_argument("--include-odds", action="store_true")
    parser.add_argument("--from-date")
    parser.add_argument("--min-odds-snapshots", type=int, default=0)
    parser.add_argument("--require-real-odds", action="store_true")
    parser.add_argument("--ev-threshold", type=float, default=1.0)
    parser.add_argument("--min-ticket-probability", type=float, default=0.0)
    parser.add_argument("--max-estimated-odds", type=float)
    parser.add_argument("--max-tickets-per-race", type=int, default=5)
    parser.add_argument("--payout-prior-weight", type=float, default=30.0)
    args = parser.parse_args(argv)
    init_db(args.db)
    with connection(args.db) as conn:
        result = bankroll_backtest(
            conn,
            output_path=Path(args.output),
            daily_budget_yen=args.daily_budget_yen,
            unit_yen=args.unit_yen,
            folds=args.folds,
            min_train_races=args.min_train_races,
            include_odds=args.include_odds,
            from_date=args.from_date,
            min_odds_snapshots=args.min_odds_snapshots,
            ev_threshold=args.ev_threshold,
            min_ticket_probability=args.min_ticket_probability,
            max_estimated_odds=args.max_estimated_odds,
            max_tickets_per_race=args.max_tickets_per_race,
            payout_prior_weight=args.payout_prior_weight,
            require_real_odds=args.require_real_odds,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
