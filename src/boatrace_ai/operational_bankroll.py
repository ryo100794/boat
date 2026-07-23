from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
from typing import Any

import joblib

from .bankroll_policy import (
    filter_candidates,
    select_temporal_policy,
    split_calibration_dates,
)
from .bankroll_backtest import (
    _build_payout_model,
    _candidate_tickets,
    _load_trifecta_payouts,
)
from .adaptive_allocation import (
    allocate_adaptive_day as _allocate_adaptive_day,
    append_day_result as _append_day_result,
    folds_by_full_day as _folds_by_full_day,
    validate_policy as _validate_policy,
    zero_totals as _zero_totals,
)
from .base_features import load_training_examples
from .db import connection, init_db
from .historical_model import FEATURE_SET, iter_scored_entries, make_pipeline
from .feature_tuning import load_complete_race_ids
from .model_core import positive_probs
from .standard_evaluation import race_set_sha256
from .roi_attribution import (
    merge_roi_attribution,
    new_roi_attribution,
    summarize_fold_signal_stability,
    summarize_roi_attribution,
)


MODEL_NAME = "win_model_no_odds_v8"


def operational_adaptive_bankroll(
    conn,
    *,
    output_path: Path,
    daily_budget_yen: int = 10_000,
    folds: int = 5,
    min_train_races: int = 500,
    ev_threshold: float = 1.20,
    payout_prior_weight: float = 30.0,
    fractional_kelly: float = 0.25,
    max_daily_exposure_fraction: float = 0.60,
    min_daily_exposure_fraction: float = 0.40,
    race_cap_fraction: float = 0.10,
    ticket_cap_fraction: float = 0.03,
    max_daily_tickets: int | None = 30,
    allocation_mode: str = "normalized_kelly",
    stake_granularity_yen: int = 100,
    min_stake_yen: int = 100,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    adaptive_no_bet: bool = False,
    calibration_fraction: float = 0.25,
    model_input_path: Path | None = None,
) -> dict[str, Any]:
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

    if adaptive_no_bet and not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between zero and one")

    pretrained_bundle: dict[str, Any] | None = None
    features: list[dict[str, Any]] = []
    labels: list[int] = []
    meta: list[dict[str, Any]] = []
    indices_by_race: dict[str, list[int]] = defaultdict(list)
    if model_input_path is not None:
        if folds != 1:
            raise ValueError("model_input_path requires folds=1")
        pretrained_bundle = joblib.load(model_input_path)
        race_keys = load_complete_race_ids(conn)
    else:
        features, labels, meta = load_training_examples(
            conn,
            include_odds=False,
            include_research=False,
        )
        race_keys = race_keys_from_meta(meta)
        for index, row in enumerate(meta):
            indices_by_race[str(row["race_id"])].append(index)
    if len(race_keys) < min_train_races + folds:
        raise ValueError(f"not enough parsed races: {len(race_keys)}")

    fold_specs = _folds_by_full_day(
        race_keys,
        folds=folds,
        min_train_races=min_train_races,
    )

    payouts = _load_trifecta_payouts(conn)
    all_races = {race_id for race_id, *_ in race_keys}
    policy = operational_policy(
        daily_budget_yen=daily_budget_yen,
        ev_threshold=ev_threshold,
        payout_prior_weight=payout_prior_weight,
        fractional_kelly=fractional_kelly,
        max_daily_exposure_fraction=max_daily_exposure_fraction,
        min_daily_exposure_fraction=min_daily_exposure_fraction,
        race_cap_fraction=race_cap_fraction,
        ticket_cap_fraction=ticket_cap_fraction,
        max_daily_tickets=max_daily_tickets,
        allocation_mode=allocation_mode,
        stake_granularity_yen=stake_granularity_yen,
        min_stake_yen=min_stake_yen,
        adaptive_no_bet=adaptive_no_bet,
        calibration_fraction=calibration_fraction,
    )
    checkpoint_file = checkpoint_path or output_path.with_suffix(
        output_path.suffix + ".checkpoint"
    )
    checkpoint_signature = {
        "version": 1,
        "model": MODEL_NAME,
        "feature_set": FEATURE_SET,
        "policy": policy,
        "folds": folds,
        "min_train_races": min_train_races,
        "race_count": len(race_keys),
        "first_race": race_keys[0][0],
        "last_race": race_keys[-1][0],
    }
    evaluated_races: set[str] = set()
    daily_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    totals = _zero_totals()
    cumulative_profit = 0
    peak_profit = 0
    max_drawdown = 0
    roi_attribution = new_roi_attribution()
    start_fold = 1
    if resume and checkpoint_file.exists():
        checkpoint = json.loads(checkpoint_file.read_text(encoding="utf-8"))
        if checkpoint.get("signature") != checkpoint_signature:
            raise ValueError("checkpoint does not match the current model, data, or policy")
        start_fold = max(1, int(checkpoint.get("next_fold") or 1))
        daily_rows = list(checkpoint.get("daily_rows") or [])
        fold_rows = list(checkpoint.get("fold_rows") or [])
        totals.update(checkpoint.get("totals") or {})
        evaluated_races = set(checkpoint.get("evaluated_races") or [])
        cumulative_profit = int(checkpoint.get("cumulative_profit") or 0)
        peak_profit = int(checkpoint.get("peak_profit") or 0)
        max_drawdown = int(checkpoint.get("max_drawdown") or 0)
        roi_attribution = dict(checkpoint.get("roi_attribution") or {})

    for fold_index, (train_races, test_races, test_dates) in enumerate(
        fold_specs,
        start=1,
    ):
        if fold_index < start_fold:
            continue
        if pretrained_bundle is not None:
            _validate_pretrained_bundle(
                pretrained_bundle,
                train_races=train_races,
            )
        else:
            train_indices = _indices_for_races(indices_by_race, train_races)
            test_indices = _indices_for_races(indices_by_race, test_races)
            if not train_indices or not test_indices:
                continue

        payout_model = _build_payout_model(
            payouts,
            train_races=train_races,
            prior_weight=payout_prior_weight,
        )
        pipeline = None
        rows_by_race: dict[str, list[dict[str, Any]]] = defaultdict(list)
        if pretrained_bundle is not None:
            pipeline = pretrained_bundle["pipeline"]
            scored_rows = iter_scored_entries(
                conn,
                pipeline=pipeline,
                include_races=test_races,
                from_date=min(test_dates),
                through_date=max(test_dates),
            )
            for probability, row in scored_rows:
                rows_by_race[str(row["race_id"])].append(
                    {
                        "race_id": str(row["race_id"]),
                        "race_date": str(row["race_date"]),
                        "jcd": str(row["jcd"]),
                        "rno": int(row["rno"]),
                        "lane": int(row["lane"]),
                        "rank": int(row["rank"]),
                        "probability": float(probability),
                    }
                )
        else:
            pipeline = make_pipeline()
            pipeline.fit(
                [features[index] for index in train_indices],
                [labels[index] for index in train_indices],
            )
            probabilities = positive_probs(
                pipeline,
                [features[index] for index in test_indices],
            )
            for probability, index in zip(probabilities, test_indices):
                row = meta[index]
                rows_by_race[str(row["race_id"])].append(
                    {
                        "race_id": str(row["race_id"]),
                        "race_date": str(row["race_date"]),
                        "jcd": str(row["jcd"]),
                        "rno": int(row["rno"]),
                        "lane": int(row["lane"]),
                        "rank": int(row["rank"]),
                        "probability": float(probability),
                    }
                )

        candidates_by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
        evaluated_by_day: dict[str, set[str]] = defaultdict(set)
        for race_id, rows in rows_by_race.items():
            payout = payouts.get(race_id)
            if len(rows) != 6 or not payout:
                continue
            race_date = str(rows[0]["race_date"])
            evaluated_by_day[race_date].add(race_id)
            race_candidates = _candidate_tickets(
                rows,
                actual=payout,
                payout_model=payout_model,
                ev_threshold=ev_threshold,
            )
            candidates_by_day[race_date].extend(race_candidates)

        allocation_kwargs = {
            "daily_budget_yen": daily_budget_yen,
            "fractional_kelly": fractional_kelly,
            "max_daily_exposure_fraction": max_daily_exposure_fraction,
            "min_daily_exposure_fraction": min_daily_exposure_fraction,
            "race_cap_fraction": race_cap_fraction,
            "ticket_cap_fraction": ticket_cap_fraction,
            "max_daily_tickets": max_daily_tickets,
            "allocation_mode": allocation_mode,
            "stake_granularity_yen": stake_granularity_yen,
            "min_stake_yen": min_stake_yen,
        }
        calibration_dates: list[str] = []
        evaluation_dates = sorted(test_dates)
        selected_candidate_policy = {"name": "baseline"}
        calibration_policy_results: list[dict[str, Any]] = []
        if adaptive_no_bet:
            calibration_dates, evaluation_dates = split_calibration_dates(
                test_dates,
                calibration_fraction=calibration_fraction,
            )
            selected_candidate_policy, calibration_policy_results = (
                select_temporal_policy(
                    calibration_dates,
                    candidates_by_day,
                    evaluated_by_day,
                    allocate_day=_allocate_adaptive_day,
                    allocation_kwargs=allocation_kwargs,
                )
            )

        fold_evaluated = set().union(
            *(evaluated_by_day.get(date, set()) for date in evaluation_dates)
        )
        evaluated_races.update(fold_evaluated)
        fold_candidate_count = sum(
            len(candidates_by_day.get(date, [])) for date in evaluation_dates
        )
        fold_roi_attribution = new_roi_attribution()
        fold_stake = 0
        fold_return = 0
        fold_selected = 0
        for race_date in evaluation_dates:
            filtered_candidates = filter_candidates(
                candidates_by_day.get(race_date, []),
                selected_candidate_policy,
            )
            day_result = _allocate_adaptive_day(
                race_date,
                filtered_candidates,
                evaluated_by_day.get(race_date, set()),
                **allocation_kwargs,
                roi_attribution=fold_roi_attribution,
            )
            fold_stake += int(day_result["stake_yen"])
            fold_return += int(day_result["return_yen"])
            fold_selected += int(day_result["tickets"])
            cumulative_profit, peak_profit, max_drawdown = _append_day_result(
                daily_rows,
                totals,
                day_result,
                cumulative_profit=cumulative_profit,
                peak_profit=peak_profit,
                max_drawdown=max_drawdown,
            )

        merge_roi_attribution(roi_attribution, fold_roi_attribution)
        fold_row = {
            "fold": fold_index,
            "train_races": len(train_races),
            "test_races": len(test_races),
            "test_days": len(test_dates),
            "calibration_days": len(calibration_dates),
            "evaluation_days": len(evaluation_dates),
            "selected_candidate_policy": selected_candidate_policy,
            "calibration_policy_results": calibration_policy_results,
            "evaluated_races": len(fold_evaluated),
            "candidate_tickets": fold_candidate_count,
            "selected_tickets": fold_selected,
            "stake_yen": fold_stake,
            "return_yen": fold_return,
            "profit_yen": fold_return - fold_stake,
            "roi": fold_return / fold_stake if fold_stake else 0.0,
            "ticket_roi_attribution": _compact_roi_attribution(
                summarize_roi_attribution(fold_roi_attribution)
            ),
        }
        fold_rows.append(fold_row)
        _write_json_atomic(
            checkpoint_file,
            {
                "version": 1,
                "signature": checkpoint_signature,
                "next_fold": fold_index + 1,
                "daily_rows": daily_rows,
                "fold_rows": fold_rows,
                "totals": totals,
                "evaluated_races": sorted(evaluated_races),
                "cumulative_profit": cumulative_profit,
                "peak_profit": peak_profit,
                "max_drawdown": max_drawdown,
                "roi_attribution": roi_attribution,
            },
        )
        print(json.dumps(fold_row, ensure_ascii=False), flush=True)
        del pipeline, rows_by_race, candidates_by_day, payout_model
        _release_fold_memory()

    result = _summarize_operational(
        daily_rows,
        totals,
        evaluated_races=evaluated_races,
        real_odds_races=set(),
        skipped_no_real_odds=0,
        all_race_count=len(all_races),
        max_drawdown=max_drawdown,
        roi_attribution=roi_attribution,
        policy=policy,
        folds=fold_rows,
    )
    result.update(
        {
            "examples": len(race_keys) * 6,
            "model": MODEL_NAME,
            "feature_set": FEATURE_SET,
            "comparison_role": "operational_model_same_policy_backtest",
        }
    )
    _write_json_atomic(output_path, result)
    checkpoint_file.unlink(missing_ok=True)
    return result


def _validate_pretrained_bundle(
    bundle: dict[str, Any],
    *,
    train_races: set[str],
) -> None:
    metadata = bundle.get("metadata") or {}
    if metadata.get("feature_set") != FEATURE_SET:
        raise ValueError("pretrained model feature set mismatch")
    trained_races = int(
        metadata.get("train_races") or metadata.get("races") or 0
    )
    if trained_races != len(train_races):
        raise ValueError("pretrained model training race count mismatch")
    expected_hash = race_set_sha256(train_races)
    if metadata.get("train_race_set_sha256") != expected_hash:
        raise ValueError("pretrained model training race set mismatch")
    if "pipeline" not in bundle:
        raise ValueError("pretrained model pipeline missing")


def _release_fold_memory() -> None:
    gc.collect()
    try:
        import ctypes

        malloc_trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
        if malloc_trim is not None:
            malloc_trim(0)
    except (AttributeError, OSError):
        pass


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _summarize_operational(
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
        "policy": policy,
        "folds": folds,
        "races": all_race_count,
        "evaluated_races": len(evaluated_races),
        "evaluation_race_set_sha256": race_set_sha256(evaluated_races),
        "real_odds_races": len(real_odds_races),
        "skipped_no_real_odds": skipped_no_real_odds,
        "candidate_tickets": int(totals["candidate_tickets"]),
        "positive_edge_tickets": int(totals["positive_edge_tickets"]),
        "allocation_candidate_tickets": int(
            totals["allocation_candidate_tickets"]
        ),
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
        "budget_utilization": (
            stake_yen / (policy["daily_budget_yen"] * race_days)
            if race_days
            else 0.0
        ),
        "avg_stake_yen_per_ticket": stake_yen / tickets if tickets else 0.0,
        "avg_tickets_per_betting_day": (
            tickets / days_with_bets if days_with_bets else 0.0
        ),
        "avg_tickets_per_selected_race": (
            tickets / races_bet if races_bet else 0.0
        ),
        "daily": daily_rows,
        "best_days": sorted(
            daily_rows,
            key=lambda row: row["profit_yen"],
            reverse=True,
        )[:10],
        "worst_days": sorted(
            daily_rows,
            key=lambda row: row["profit_yen"],
        )[:10],
        "ticket_roi_attribution": ticket_roi_attribution,
    }


def _compact_roi_attribution(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": value.get("method"),
        "minimum_evidence": value.get("minimum_evidence") or {},
        "top_signals": (value.get("top_signals") or [])[:16],
        "diagnosis": value.get("diagnosis"),
    }


def race_keys_from_meta(meta: list[dict[str, Any]]) -> list[tuple[str, str, str, int]]:
    by_race: dict[str, tuple[str, str, str, int]] = {}
    for row in meta:
        race_id = str(row["race_id"])
        by_race[race_id] = (
            race_id,
            str(row["race_date"]),
            str(row["jcd"]),
            int(row["rno"]),
        )
    return sorted(by_race.values(), key=lambda row: (row[1], row[2], row[3]))


def _indices_for_races(
    indices_by_race: dict[str, list[int]],
    races: set[str],
) -> list[int]:
    return [index for race_id in sorted(races) for index in indices_by_race[race_id]]


def operational_policy(
    *,
    daily_budget_yen: int,
    ev_threshold: float,
    payout_prior_weight: float,
    fractional_kelly: float,
    max_daily_exposure_fraction: float,
    min_daily_exposure_fraction: float,
    race_cap_fraction: float,
    ticket_cap_fraction: float,
    max_daily_tickets: int | None,
    allocation_mode: str,
    stake_granularity_yen: int,
    min_stake_yen: int,
    adaptive_no_bet: bool = False,
    calibration_fraction: float = 0.25,
) -> dict[str, Any]:
    return {
        "daily_budget_yen": daily_budget_yen,
        "bet_type": "3連単",
        "include_odds": False,
        "require_real_odds": False,
        "ev_threshold": ev_threshold,
        "payout_prior_weight": payout_prior_weight,
        "payout_estimator": "train-fold average payout by trifecta combination, blended with train-fold global average",
        "stake_model": "adaptive_unit_yen",
        "unit_yen": stake_granularity_yen,
        "fractional_kelly": fractional_kelly,
        "max_daily_exposure_fraction": max_daily_exposure_fraction,
        "min_daily_exposure_fraction": min_daily_exposure_fraction,
        "race_cap_fraction": race_cap_fraction,
        "ticket_cap_fraction": ticket_cap_fraction,
        "max_daily_tickets": max_daily_tickets,
        "allocation_mode": allocation_mode,
        "stake_granularity_yen": stake_granularity_yen,
        "min_stake_yen": min_stake_yen,
        "adaptive_no_bet": adaptive_no_bet,
        "calibration_fraction": calibration_fraction if adaptive_no_bet else 0.0,
        "candidate_policy_selection": (
            "calibration-prefix profit with no-bet option"
            if adaptive_no_bet
            else "fixed baseline"
        ),
        "model": MODEL_NAME,
        "feature_set": FEATURE_SET,
        "model_pipeline": "DictVectorizer + SparseIndex32 + MaxAbsScaler + LogisticRegression(liblinear,C=0.20,class_weight=None)",
        "allocation": "same normalized-Kelly policy used by the past-log comparison",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Adaptive bankroll backtest for the deployed no-odds v8 model profile."
    )
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument(
        "--output",
        default="data/models/bankroll_no_odds_v8_normalized_kelly.json",
    )
    parser.add_argument("--daily-budget-yen", type=int, default=10_000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-train-races", type=int, default=500)
    parser.add_argument("--model-input", type=Path)
    parser.add_argument("--ev-threshold", type=float, default=1.20)
    parser.add_argument("--payout-prior-weight", type=float, default=30.0)
    parser.add_argument("--fractional-kelly", type=float, default=0.25)
    parser.add_argument("--max-daily-exposure-fraction", type=float, default=0.60)
    parser.add_argument("--min-daily-exposure-fraction", type=float, default=0.40)
    parser.add_argument("--race-cap-fraction", type=float, default=0.10)
    parser.add_argument("--ticket-cap-fraction", type=float, default=0.03)
    parser.add_argument("--max-daily-tickets", type=int, default=30)
    parser.add_argument(
        "--allocation-mode",
        choices=["kelly_floor", "normalized_kelly"],
        default="normalized_kelly",
    )
    parser.add_argument("--stake-granularity-yen", type=int, default=100)
    parser.add_argument("--min-stake-yen", type=int, default=100)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--adaptive-no-bet", action="store_true")
    parser.add_argument("--calibration-fraction", type=float, default=0.25)
    args = parser.parse_args(argv)

    init_db(args.db)
    with connection(args.db) as conn:
        result = operational_adaptive_bankroll(
            conn,
            output_path=Path(args.output),
            daily_budget_yen=args.daily_budget_yen,
            folds=args.folds,
            min_train_races=args.min_train_races,
            ev_threshold=args.ev_threshold,
            payout_prior_weight=args.payout_prior_weight,
            fractional_kelly=args.fractional_kelly,
            max_daily_exposure_fraction=args.max_daily_exposure_fraction,
            min_daily_exposure_fraction=args.min_daily_exposure_fraction,
            race_cap_fraction=args.race_cap_fraction,
            ticket_cap_fraction=args.ticket_cap_fraction,
            max_daily_tickets=args.max_daily_tickets or None,
            allocation_mode=args.allocation_mode,
            stake_granularity_yen=args.stake_granularity_yen,
            min_stake_yen=args.min_stake_yen,
            checkpoint_path=args.checkpoint,
            resume=args.resume,
            adaptive_no_bet=args.adaptive_no_bet,
            calibration_fraction=args.calibration_fraction,
            model_input_path=args.model_input,
        )
    compact = {key: value for key, value in result.items() if key != "daily"}
    compact["daily_rows"] = len(result.get("daily") or [])
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
