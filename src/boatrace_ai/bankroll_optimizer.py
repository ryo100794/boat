from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .adaptive_allocation import (
    allocate_adaptive_day as _allocate_adaptive_day,
    append_day_result as _append_day_result,
    folds_by_full_day as _folds_by_full_day,
    validate_policy as _validate_policy,
    zero_totals as _zero_totals,
)

from .bankroll_backtest import (
    _build_payout_model,
    _candidate_tickets,
    _load_trifecta_payouts,
)
from .db import connection, init_db
from .features import latest_trifecta_odds_before_deadline
from .feature_tuning import (
    FEATURE_SET,
    iter_scored_races,
    load_complete_race_ids,
    normalize_drop_feature_groups,
    train_bundle,
)
from .standard_evaluation import race_set_sha256
from .roi_attribution import (
    merge_roi_attribution,
    new_roi_attribution,
    summarize_fold_signal_stability,
    summarize_roi_attribution,
)


DEFAULT_STAKE_UNIT_YEN = 100


def adaptive_bankroll_streaming(
    conn,
    *,
    output_path: Path,
    drop_feature_groups: tuple[str, ...] | str | None = None,
    daily_budget_yen: int = 10_000,
    folds: int = 5,
    min_train_races: int = 500,
    ev_threshold: float = 1.0,
    payout_prior_weight: float = 30.0,
    require_real_odds: bool = False,
    fractional_kelly: float = 0.25,
    max_daily_exposure_fraction: float = 1.0,
    min_daily_exposure_fraction: float = 0.0,
    race_cap_fraction: float = 0.20,
    ticket_cap_fraction: float = 0.04,
    max_daily_tickets: int | None = None,
    allocation_mode: str = "kelly_floor",
    stake_granularity_yen: int = DEFAULT_STAKE_UNIT_YEN,
    min_stake_yen: int = DEFAULT_STAKE_UNIT_YEN,
    batch_size: int = 24_000,
    epochs: int = 1,
) -> dict[str, Any]:
    drop_feature_groups = normalize_drop_feature_groups(drop_feature_groups)
    _validate_policy(
        daily_budget_yen=daily_budget_yen,
        fractional_kelly=fractional_kelly,
        max_daily_exposure_fraction=max_daily_exposure_fraction,
        min_daily_exposure_fraction=min_daily_exposure_fraction,
        race_cap_fraction=race_cap_fraction,
        ticket_cap_fraction=ticket_cap_fraction,
        max_daily_tickets=max_daily_tickets,
        allocation_mode=allocation_mode,
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

    real_odds_by_race: dict[str, dict[str, Any] | None] = {}
    real_odds_races: set[str] = set()
    skipped_no_real_odds = 0
    evaluated_races: set[str] = set()
    daily_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    totals = _zero_totals()
    cumulative_profit = 0
    peak_profit = 0
    max_drawdown = 0
    roi_attribution = new_roi_attribution()

    for fold_index, (train_races, test_races, test_dates) in enumerate(fold_specs, start=1):
        if not train_races or not test_races:
            continue
        payout_model = _build_payout_model(
            payouts,
            train_races=train_races,
            prior_weight=payout_prior_weight,
        )
        bundle = train_bundle(
            conn,
            include_races=train_races,
            drop_feature_groups=drop_feature_groups,
            batch_size=batch_size,
            epochs=epochs,
        )
        fold_candidate_count = 0
        fold_selected_count = 0
        fold_evaluated = 0
        fold_real_odds_races = 0
        fold_skipped_no_real_odds = 0
        fold_roi_attribution = new_roi_attribution()
        current_date: str | None = None
        day_candidates: list[dict[str, Any]] = []
        day_evaluated: set[str] = set()

        for rows in iter_scored_races(
            conn,
            bundle=bundle,
            include_races=test_races,
            drop_feature_groups=drop_feature_groups,
        ):
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
                    min_daily_exposure_fraction=min_daily_exposure_fraction,
                    race_cap_fraction=race_cap_fraction,
                    ticket_cap_fraction=ticket_cap_fraction,
                    max_daily_tickets=max_daily_tickets,
                    allocation_mode=allocation_mode,
                    stake_granularity_yen=stake_granularity_yen,
                    min_stake_yen=min_stake_yen,
                    roi_attribution=fold_roi_attribution,
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
            day_evaluated.add(race_id_value)
            fold_evaluated += 1
            race_candidates = _candidate_tickets(
                rows,
                actual=payout,
                payout_model=payout_model,
                ev_threshold=ev_threshold,
                real_odds_snapshot=real_odds_snapshot,
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
                min_daily_exposure_fraction=min_daily_exposure_fraction,
                race_cap_fraction=race_cap_fraction,
                ticket_cap_fraction=ticket_cap_fraction,
                max_daily_tickets=max_daily_tickets,
                allocation_mode=allocation_mode,
                stake_granularity_yen=stake_granularity_yen,
                min_stake_yen=min_stake_yen,
                roi_attribution=fold_roi_attribution,
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

        merge_roi_attribution(roi_attribution, fold_roi_attribution)
        fold_roi_summary = _compact_roi_attribution(summarize_roi_attribution(fold_roi_attribution))
        fold_row = {
            "fold": fold_index,
            "train_races": len(train_races),
            "test_races": len(test_races),
            "test_days": len(test_dates),
            "evaluated_races": fold_evaluated,
            "candidate_tickets": fold_candidate_count,
            "selected_tickets": fold_selected_count,
            "real_odds_races": fold_real_odds_races,
            "skipped_no_real_odds": fold_skipped_no_real_odds,
            "ticket_roi_attribution": fold_roi_summary,
        }
        fold_rows.append(fold_row)
        print(json.dumps(fold_row, ensure_ascii=False), flush=True)

    result = _summarize(
        daily_rows,
        totals,
        evaluated_races=evaluated_races,
        real_odds_races=real_odds_races,
        skipped_no_real_odds=skipped_no_real_odds,
        all_race_count=len(all_race_ids),
        max_drawdown=max_drawdown,
        roi_attribution=roi_attribution,
        policy={
            "daily_budget_yen": daily_budget_yen,
            "bet_type": "3連単",
            "include_odds": False,
            "require_real_odds": require_real_odds,
            "ev_threshold": ev_threshold,
            "stake_model": "adaptive_unit_yen",
            "unit_yen": stake_granularity_yen,
            "max_tickets_per_race": None,
            "fractional_kelly": fractional_kelly,
            "max_daily_exposure_fraction": max_daily_exposure_fraction,
            "min_daily_exposure_fraction": min_daily_exposure_fraction,
            "race_cap_fraction": race_cap_fraction,
            "ticket_cap_fraction": ticket_cap_fraction,
            "max_daily_tickets": max_daily_tickets,
            "allocation_mode": allocation_mode,
            "stake_granularity_yen": stake_granularity_yen,
            "min_stake_yen": min_stake_yen,
            "payout_estimator": "deadline real odds" if require_real_odds else "train-fold average payout by trifecta combination, blended with train-fold global average",
            "payout_prior_weight": payout_prior_weight,
            "allocation": "stake is proportional to positive Kelly edge; normalized-kelly can scale ranked positive edges up to min_daily_exposure_fraction before daily/race/ticket caps; each ticket stake is floored to stake_granularity_yen and tickets below min_stake_yen are skipped",
            "feature_set": FEATURE_SET,
            "drop_feature_groups": list(drop_feature_groups),
            "model": (
                "win_model_pastlog_v7_stream_hash"
                if "research_correlates" in drop_feature_groups
                else "win_model_pastlog_v9_research"
            ),
        },
        folds=fold_rows,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _summarize(
    daily_rows: list[dict[str, Any]],
    totals: dict[str, float],
    *,
    evaluated_races: set[str],
    real_odds_races: set[str],
    skipped_no_real_odds: int,
    all_race_count: int,
    max_drawdown: int,
    roi_attribution: dict[str, Any],
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
    ticket_roi_attribution = summarize_roi_attribution(roi_attribution)
    ticket_roi_attribution["fold_stability"] = summarize_fold_signal_stability(
        [row.get("ticket_roi_attribution") or {} for row in folds]
    )
    return {
        "generated_at": _now(),
        "policy": policy,
        "folds": folds,
        "examples": 0,
        "races": all_race_count,
        "evaluated_races": len(evaluated_races),
        "evaluation_race_set_sha256": race_set_sha256(evaluated_races),
        "real_odds_races": len(real_odds_races),
        "skipped_no_real_odds": skipped_no_real_odds,
        "candidate_tickets": int(totals["candidate_tickets"]),
        "positive_edge_tickets": int(totals["positive_edge_tickets"]),
        "allocation_candidate_tickets": int(totals["allocation_candidate_tickets"]),
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
        "ticket_roi_attribution": ticket_roi_attribution,
        "feature_set": FEATURE_SET,
        "model": str(policy.get("model") or "win_model_pastlog_v9_research"),
    }


def _compact_roi_attribution(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": value.get("method"),
        "minimum_evidence": value.get("minimum_evidence") or {},
        "top_signals": (value.get("top_signals") or [])[:16],
        "diagnosis": value.get("diagnosis"),
    }


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Adaptive bankroll backtest for the streaming past-log model.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument(
        "--output",
        default="data/models/bankroll_backtest_pastlog_v9_research_adaptive_10000.json",
    )
    parser.add_argument("--drop-feature-groups", default="")
    parser.add_argument("--daily-budget-yen", type=int, default=10_000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-train-races", type=int, default=500)
    parser.add_argument("--ev-threshold", type=float, default=1.0)
    parser.add_argument("--payout-prior-weight", type=float, default=30.0)
    parser.add_argument("--require-real-odds", action="store_true")
    parser.add_argument("--fractional-kelly", type=float, default=0.25)
    parser.add_argument("--max-daily-exposure-fraction", type=float, default=1.0)
    parser.add_argument("--min-daily-exposure-fraction", type=float, default=0.0)
    parser.add_argument("--race-cap-fraction", type=float, default=0.20)
    parser.add_argument("--ticket-cap-fraction", type=float, default=0.04)
    parser.add_argument("--max-daily-tickets", type=int, default=0, help="0 means no daily ticket limit before allocation")
    parser.add_argument("--allocation-mode", choices=["kelly_floor", "normalized_kelly"], default="kelly_floor")
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
            drop_feature_groups=args.drop_feature_groups,
            daily_budget_yen=args.daily_budget_yen,
            folds=args.folds,
            min_train_races=args.min_train_races,
            ev_threshold=args.ev_threshold,
            payout_prior_weight=args.payout_prior_weight,
            require_real_odds=args.require_real_odds,
            fractional_kelly=args.fractional_kelly,
            max_daily_exposure_fraction=args.max_daily_exposure_fraction,
            min_daily_exposure_fraction=args.min_daily_exposure_fraction,
            race_cap_fraction=args.race_cap_fraction,
            ticket_cap_fraction=args.ticket_cap_fraction,
            max_daily_tickets=args.max_daily_tickets or None,
            allocation_mode=args.allocation_mode,
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
