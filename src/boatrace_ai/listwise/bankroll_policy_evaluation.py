from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from sklearn.feature_extraction import FeatureHasher

from ..bankroll_backtest import (
    _build_payout_model,
    _candidate_tickets,
    _load_trifecta_payouts,
)
from ..bankroll_policy_search import (
    promotion_gate,
    recent_allocation_diagnostics,
    temporal_stability,
    successive_halving_search,
)
from ..db import connection, init_db
from ..feature_tuning import load_complete_race_ids
from ..hashed_feature_dataset import load_hashed_dataset, race_ids_sha256
from ..modeling import _normalize_lane_probs, trifecta_predictions
from ..packed_bankroll import (
    candidate_ev_calibration,
    evaluate_packed_policy,
    pack_candidates,
)
from .direct_bankroll import bootstrap_daily_bankroll
from .feature_search import _write_json_atomic
from .model import evaluate_range, fit_scaler, train_listwise_model
from .newton_refine import (
    refine_newton_cg,
    search_race_date_through,
    validate_search_race_universe,
)


def packed_candidates_from_rows(
    rows_by_race: dict[str, list[dict[str, Any]]],
    *,
    payouts: dict[str, dict[str, Any]],
    train_races: set[str],
    payout_prior_weight: float,
) -> Any:
    payout_model = _build_payout_model(
        payouts,
        train_races=train_races,
        prior_weight=payout_prior_weight,
    )
    candidates_by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    evaluated_by_date: dict[str, int] = defaultdict(int)
    for race_id, rows in rows_by_race.items():
        actual = payouts.get(race_id)
        if len(rows) != 6 or actual is None:
            continue
        race_date = str(rows[0]["race_date"])
        evaluated_by_date[race_date] += 1
        candidates_by_date[race_date].extend(
            _candidate_tickets(
                rows,
                actual=actual,
                payout_model=payout_model,
                ev_threshold=1.0,
            )
        )
    return pack_candidates(candidates_by_date, evaluated_by_date)


def flat_top_k_diagnostic(
    rows_by_race: dict[str, list[dict[str, Any]]],
    *,
    payouts: dict[str, dict[str, Any]],
    top_k: int = 5,
    unit_yen: int = 100,
) -> dict[str, Any]:
    if not 1 <= top_k <= 120 or unit_yen <= 0:
        raise ValueError("flat top-k diagnostic requires valid top_k and unit_yen")
    races = hits = returned = 0
    for race_id, rows in rows_by_race.items():
        payout = payouts.get(race_id)
        if len(rows) != 6 or payout is None:
            continue
        lane_probs = _normalize_lane_probs({
            int(row["lane"]): float(row["probability"]) for row in rows
        })
        combinations = {
            item["combination"]
            for item in trifecta_predictions(lane_probs)[:top_k]
        }
        races += 1
        if str(payout["combination"]) in combinations:
            hits += 1
            returned += int(payout["payout_yen"])
    tickets = races * top_k
    stake = tickets * unit_yen
    return {
        "role": "diagnostic_all_races_flat_stake_not_daily_budget",
        "top_k": top_k,
        "unit_yen": unit_yen,
        "evaluated_races": races,
        "tickets": tickets,
        "hit_races": hits,
        "hit_rate": hits / races if races else None,
        "stake_yen": stake,
        "return_yen": returned,
        "profit_yen": returned - stake,
        "roi": returned / stake if stake else None,
        "average_hit_payout_yen": returned / hits if hits else None,
        "breakeven_average_hit_payout_yen": (
            top_k * unit_yen * races / hits if hits else None
        ),
    }


def prior_selection_key(row: dict[str, Any]) -> tuple[Any, ...]:
    selected = row["selected"]
    stability = selected["temporal_stability"]
    confidence = selected["confidence"]
    metrics = selected["metrics"]
    return (
        bool(stability["all_minimum_evidence"]),
        float(stability["minimum_roi"]),
        float(stability["mean_roi_minus_std"]),
        float(confidence["roi_ci95_lower"]),
        float(confidence["probability_roi_above_one"]),
        int(metrics["profit_yen"]),
        int(metrics["hit_tickets"]),
    )


def run(conn: Any, *, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    search_path = Path(args.search_result).resolve()
    search = json.loads(search_path.read_text(encoding="utf-8"))
    selected = search["selected"]
    race_date_through = search_race_date_through(search)
    race_keys = [
        row
        for row in load_complete_race_ids(conn)
        if race_date_through is None or str(row[1]) <= race_date_through
    ]
    validate_search_race_universe(search, race_keys)
    train_end = int(search["train_races"])
    selection_end = train_end + int(search["selection_races"])
    if race_date_through is None:
        raise ValueError("search result lacks race_date_through")
    evaluation_through = date.fromisoformat(str(race_date_through))
    evaluation_from = evaluation_through - timedelta(
        days=args.evaluation_days - 1
    )
    holdout_start = next(
        (
            index for index in range(selection_end, len(race_keys))
            if date.fromisoformat(str(race_keys[index][1])) >= evaluation_from
        ),
        len(race_keys),
    )
    if holdout_start >= len(race_keys):
        raise ValueError("standard evaluation window contains no holdout races")
    dropped = tuple(
        str(value) for value in selected.get("drop_feature_groups") or ()
    )
    n_features = int(search["n_features"])
    hasher = FeatureHasher(
        n_features=n_features,
        input_type="dict",
        alternate_sign=False,
    )
    dataset = load_hashed_dataset(
        Path(args.cache_prefix).resolve(),
        race_keys=race_keys,
        n_features=n_features,
        drop_feature_groups=dropped,
        hasher=hasher,
    )
    if dataset is None:
        raise ValueError("strict selected feature cache validation failed")

    selection_model, selection_history = _train_model(
        dataset,
        race_end=train_end,
        selected=selected,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_races=args.batch_races,
        coefficient_optimizer=args.coefficient_optimizer,
        max_newton_iterations=args.max_newton_iterations,
        max_cg_iterations=args.max_cg_iterations,
        gradient_tolerance=args.gradient_tolerance,
        cg_tolerance=args.cg_tolerance,
    )
    selection_metrics, selection_rows = evaluate_range(
        dataset,
        selection_model,
        race_start=train_end,
        race_end=selection_end,
        batch_races=args.batch_races,
        keep_rows=True,
    )
    payouts = _load_trifecta_payouts(conn)
    selection_top5_flat = flat_top_k_diagnostic(
        selection_rows, payouts=payouts
    )
    selection_train_races = {
        race_id for race_id, *_rest in race_keys[:train_end]
    }
    base_policy = {
        "daily_budget_yen": args.daily_budget_yen,
        "ev_threshold": 1.20,
        "min_ticket_probability": 0.0,
        "max_estimated_odds": None,
        "payout_prior_weight": 30.0,
        "fractional_kelly": 0.25,
        "max_daily_exposure_fraction": 0.60,
        "min_daily_exposure_fraction": 0.40,
        "race_cap_fraction": 0.10,
        "ticket_cap_fraction": 0.03,
        "max_daily_tickets": 30,
        "allocation_mode": "normalized_kelly",
        "stake_granularity_yen": 100,
        "min_stake_yen": 100,
    }
    prior_results = []
    for index, prior_weight in enumerate(args.payout_prior_weights):
        packed = packed_candidates_from_rows(
            selection_rows,
            payouts=payouts,
            train_races=selection_train_races,
            payout_prior_weight=prior_weight,
        )
        policy = {**base_policy, "payout_prior_weight": prior_weight}
        search_result = successive_halving_search(
            packed,
            policy,
            candidate_count=args.candidate_count,
            finalists=args.finalists,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + index,
        )
        prior_results.append({
            "payout_prior_weight": prior_weight,
            "packed_tickets": packed.tickets,
            "candidate_ev_calibration": candidate_ev_calibration(packed),
            **search_result,
        })
        print(json.dumps({
            "phase": "policy_selection",
            "payout_prior_weight": prior_weight,
            "selected": search_result["selected"],
        }, ensure_ascii=False), flush=True)
    selected_search = max(
        prior_results,
        key=prior_selection_key,
    )
    selected_policy = dict(selected_search["selected"]["policy"])

    final_model, final_history = _train_model(
        dataset,
        race_end=selection_end,
        selected=selected,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_races=args.batch_races,
        coefficient_optimizer=args.coefficient_optimizer,
        max_newton_iterations=args.max_newton_iterations,
        max_cg_iterations=args.max_cg_iterations,
        gradient_tolerance=args.gradient_tolerance,
        cg_tolerance=args.cg_tolerance,
    )
    holdout_metrics, holdout_rows = evaluate_range(
        dataset,
        final_model,
        race_start=holdout_start,
        race_end=len(race_keys),
        batch_races=args.batch_races,
        keep_rows=True,
    )
    holdout_top5_flat = flat_top_k_diagnostic(
        holdout_rows, payouts=payouts
    )
    holdout_packed = packed_candidates_from_rows(
        holdout_rows,
        payouts=payouts,
        train_races={
            race_id for race_id, *_rest in race_keys[:selection_end]
        },
        payout_prior_weight=float(selected_policy["payout_prior_weight"]),
    )
    holdout_bankroll = evaluate_packed_policy(holdout_packed, selected_policy)
    holdout_ev_calibration = candidate_ev_calibration(holdout_packed)
    holdout_confidence = bootstrap_daily_bankroll(
        holdout_bankroll["daily"],
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    recent_allocation = recent_allocation_diagnostics(holdout_bankroll["daily"])
    holdout_temporal_stability = temporal_stability(
        holdout_packed, selected_policy
    )
    holdout_gate = promotion_gate({
        "metrics": holdout_bankroll,
        "confidence": holdout_confidence,
        "recent_allocation": recent_allocation,
        "temporal_stability": holdout_temporal_stability,
    })
    result = {
        "model": (
            "bankroll_policy_optimized_newton_v2"
            if args.coefficient_optimizer == "newton_cg"
            else "bankroll_policy_optimized_v1"
        ),
        "comparison_role": "bankroll_policy_model",
        "source_search_result": str(search_path),
        "feature_schema_version": search["feature_schema_version"],
        "race_universe_sha256": race_ids_sha256(race_keys),
        "selected_prediction_model": selected,
        "coefficient_optimizer": args.coefficient_optimizer,
        "evaluation_from": evaluation_from.isoformat(),
        "evaluation_through": evaluation_through.isoformat(),
        "evaluation_days": args.evaluation_days,
        "selection_protocol": (
            "prediction model trained before selection interval; policy selected "
            "on out-of-fold chronological predictions; final model retrained "
            "through selection interval; holdout used once"
        ),
        "selection_races": len(selection_rows),
        "selection_prediction_metrics": selection_metrics,
        "selection_top5_flat_diagnostic": selection_top5_flat,
        "selection_training_history": selection_history,
        "policy_searches": prior_results,
        "selected_policy": selected_policy,
        "holdout_races": len(holdout_rows),
        "holdout_prediction_metrics": holdout_metrics,
        "holdout_top5_flat_diagnostic": holdout_top5_flat,
        "holdout_training_history": final_history,
        "evaluated_races": holdout_bankroll["evaluated_races"],
        "selected_races": holdout_bankroll["selected_races"],
        "tickets": holdout_bankroll["tickets"],
        "hit_tickets": holdout_bankroll["hit_tickets"],
        "hit_races": holdout_bankroll["hit_races"],
        "stake_yen": holdout_bankroll["stake_yen"],
        "return_yen": holdout_bankroll["return_yen"],
        "profit_yen": holdout_bankroll["profit_yen"],
        "roi": holdout_bankroll["roi"],
        "max_drawdown_yen": holdout_bankroll["max_drawdown_yen"],
        "ticket_hit_rate": holdout_bankroll["ticket_hit_rate"],
        "race_hit_rate": holdout_bankroll["race_hit_rate"],
        "bankroll": holdout_bankroll,
        "holdout_candidate_ev_calibration": holdout_ev_calibration,
        "ev_calibration_usage": "reporting_only_not_used_for_selection",
        "bankroll_confidence": holdout_confidence,
        "promotion_gate": holdout_gate,
        "recent_allocation": recent_allocation,
        "holdout_temporal_stability": holdout_temporal_stability,
        "research_only": args.research_only == "true",
        "promotion_eligible": (
            args.research_only != "true" and all(holdout_gate.values())
        ),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    _write_json_atomic(Path(args.output), result)
    return result


def _train_model(
    dataset: Any,
    *,
    race_end: int,
    selected: dict[str, Any],
    learning_rate: float,
    epochs: int,
    batch_races: int,
    coefficient_optimizer: str = "adam",
    max_newton_iterations: int = 10,
    max_cg_iterations: int = 75,
    gradient_tolerance: float = 0.0001,
    cg_tolerance: float = 0.001,
) -> tuple[Any, Any]:
    scaler = fit_scaler(dataset, race_end=race_end, batch_rows=batch_races * 6)
    model, adam_history = train_listwise_model(
        dataset,
        train_race_end=race_end,
        target=str(selected["target"]),
        alpha=float(selected["alpha"]),
        learning_rate=learning_rate,
        epochs=epochs,
        batch_races=batch_races,
        scaler=scaler,
    )
    if coefficient_optimizer == "adam":
        return model, adam_history
    if coefficient_optimizer != "newton_cg":
        raise ValueError(f"unsupported coefficient optimizer: {coefficient_optimizer}")
    refined, convergence = refine_newton_cg(
        dataset,
        model,
        train_race_end=race_end,
        batch_races=batch_races,
        max_newton_iterations=max_newton_iterations,
        max_cg_iterations=max_cg_iterations,
        gradient_tolerance=gradient_tolerance,
        cg_tolerance=cg_tolerance,
    )
    return refined, {
        "coefficient_optimizer": "adam_warm_start_matrix_free_newton_cg",
        "adam_history": adam_history,
        "newton_convergence": convergence,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Leakage-safe listwise bankroll policy optimization."
    )
    parser.add_argument("--evaluation-days", type=int, default=365)
    parser.add_argument("--db", required=True)
    parser.add_argument(
        "--research-only",
        choices=("true", "false"),
        default="false",
    )
    parser.add_argument("--search-result", required=True)
    parser.add_argument("--cache-prefix", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-races", type=int, default=1_000)
    parser.add_argument(
        "--coefficient-optimizer",
        choices=("adam", "newton_cg"),
        default="adam",
    )
    parser.add_argument("--max-newton-iterations", type=int, default=10)
    parser.add_argument("--max-cg-iterations", type=int, default=75)
    parser.add_argument("--gradient-tolerance", type=float, default=0.0001)
    parser.add_argument("--cg-tolerance", type=float, default=0.001)
    parser.add_argument("--daily-budget-yen", type=int, default=10_000)
    parser.add_argument("--candidate-count", type=int, default=24)
    parser.add_argument("--finalists", type=int, default=6)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--payout-prior-weights",
        type=lambda value: tuple(float(item) for item in value.split(",")),
        default=(10.0, 30.0, 100.0),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    init_db(args.db)
    with connection(args.db) as conn:
        result = run(conn, args=args)
    print(json.dumps({
        key: value for key, value in result.items()
        if key not in {
            "bankroll", "policy_searches", "selection_training_history",
            "holdout_training_history",
        }
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
