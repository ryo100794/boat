from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import brier_score_loss, log_loss

from .adaptive_allocation import zero_totals
from .bankroll_backtest import _load_trifecta_payouts
from .db import connection, init_db
from .feature_tuning import (
    _ensure_sparse_index32,
    iter_race_feature_rows,
    load_complete_race_ids,
    normalize_drop_feature_groups,
    to_hashable,
)
from .hashed_feature_dataset import load_or_build_hashed_dataset
from .listwise_ranking_model import (
    FEATURE_SET,
    MODEL_NAME,
    TARGETS,
    evaluate_range,
    fit_scaler,
    train_listwise_model,
)
from .listwise_validation import (
    default_policy,
    evaluate_bankroll_fold,
    full_day_fold_boundaries,
    nested_select_candidate,
)
from .modeling import _race_level_metrics


def run_backtest(conn, *, output_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    race_keys = load_complete_race_ids(conn)
    if len(race_keys) < args.min_train_races + args.folds:
        raise ValueError(f"not enough complete races: {len(race_keys)}")
    targets = tuple(dict.fromkeys(value.strip() for value in args.targets.split(",") if value.strip()))
    alphas = tuple(dict.fromkeys(float(value) for value in args.alphas.split(",") if value.strip()))
    if not targets or any(value not in TARGETS for value in targets):
        raise ValueError(f"targets must be selected from {TARGETS}")
    if not alphas or any(value < 0 for value in alphas):
        raise ValueError("alphas must be non-negative")
    drop_groups = normalize_drop_feature_groups(())
    hasher = FeatureHasher(
        n_features=max(256, int(args.n_features)), input_type="dict", alternate_sign=False
    )
    dataset, cache_source = load_or_build_hashed_dataset(
        cache_prefix=Path(args.feature_cache) if args.feature_cache else None,
        race_keys=race_keys,
        race_rows=lambda: iter_race_feature_rows(
            conn,
            include_races={race_id for race_id, *_rest in race_keys},
            drop_feature_groups=drop_groups,
        ),
        hasher=hasher,
        to_hashable=to_hashable,
        ensure_sparse_index32=_ensure_sparse_index32,
        drop_feature_groups=drop_groups,
        batch_size=args.batch_races * 6,
    )
    boundaries = full_day_fold_boundaries(
        race_keys, folds=args.folds, min_train_races=args.min_train_races
    )
    payouts = _load_trifecta_payouts(conn)
    policy = default_policy(
        daily_budget_yen=args.daily_budget_yen, ev_threshold=args.ev_threshold
    )
    all_predictions: dict[str, list[dict[str, Any]]] = {}
    all_labels: list[int] = []
    all_probabilities: list[float] = []
    ranking_loss_sum = 0.0
    evaluated_races = 0
    fold_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    totals = zero_totals()
    profit_state = (0, 0, 0)

    for fold_number, (train_end, test_end, test_dates) in enumerate(boundaries, start=1):
        fold_started = time.perf_counter()
        selected, candidates = nested_select_candidate(
            dataset,
            outer_train_end=train_end,
            targets=targets,
            alphas=alphas,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            batch_races=args.batch_races,
            validation_fraction=args.validation_fraction,
            min_validation_races=args.min_validation_races,
        )
        scaler = fit_scaler(dataset, race_end=train_end, batch_rows=args.batch_races * 6)
        model, history = train_listwise_model(
            dataset,
            train_race_end=train_end,
            target=str(selected["target"]),
            alpha=float(selected["alpha"]),
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            batch_races=args.batch_races,
            scaler=scaler,
        )
        metrics, rows_by_race = evaluate_range(
            dataset,
            model,
            race_start=train_end,
            race_end=test_end,
            batch_races=args.batch_races,
            keep_rows=True,
        )
        bankroll, profit_state = evaluate_bankroll_fold(
            rows_by_race=rows_by_race,
            train_races={race_id for race_id, *_rest in race_keys[:train_end]},
            test_dates=test_dates,
            payouts=payouts,
            policy=policy,
            totals=totals,
            daily_rows=daily_rows,
            profit_state=profit_state,
        )
        for race_id, rows in rows_by_race.items():
            all_predictions[race_id] = rows
            for row in rows:
                all_labels.append(int(row["rank"] == 1))
                all_probabilities.append(float(row["probability"]))
        ranking_loss_sum += float(metrics["ranking_log_loss"]) * int(metrics["evaluated_races"])
        evaluated_races += int(metrics["evaluated_races"])
        fold_row = {
            "fold": fold_number,
            "train_races": train_end,
            "test_races": test_end - train_end,
            "test_days": len(test_dates),
            "selected_candidate": {
                "target": selected["target"],
                "alpha": selected["alpha"],
                "validation_ranking_log_loss": selected["ranking_log_loss"],
                "validation_trifecta_top5_hit_rate": selected["trifecta_top5_hit_rate"],
            },
            "candidate_results": candidates,
            "training_history": history,
            **metrics,
            "bankroll": bankroll,
            "elapsed_seconds": round(time.perf_counter() - fold_started, 3),
        }
        fold_rows.append(fold_row)
        compact = {key: value for key, value in fold_row.items() if key not in {"candidate_results", "training_history"}}
        print(json.dumps(compact, ensure_ascii=False), flush=True)

    stake_yen = int(totals["stake_yen"])
    return_yen = int(totals["return_yen"])
    tickets = int(totals["tickets"])
    fold_rois = [float(row["bankroll"]["roi"]) for row in fold_rows]
    required_profitable_folds = math.ceil(len(fold_rows) / 2)
    result = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "model": MODEL_NAME,
        "comparison_role": "redesign_shadow_nested_walk_forward",
        "feature_set": FEATURE_SET,
        "include_odds": False,
        "model_structure": "shared linear lane utility; race softmax; PL top-three or winner loss",
        "validation_design": "nested expanding-window; complete-day folds; train-prefix-only scaler",
        "feature_cache_source": cache_source,
        "feature_cache_prefix": args.feature_cache or None,
        "matrix_shape": list(dataset.matrix.shape),
        "matrix_nnz": int(dataset.matrix.nnz),
        "n_features": dataset.n_features,
        "targets": list(targets),
        "alphas": list(alphas),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "races": dataset.race_count,
        "evaluated_races": evaluated_races,
        "examples": len(all_labels),
        "entry_log_loss": float(log_loss(all_labels, all_probabilities, labels=[0, 1])),
        "entry_brier": float(brier_score_loss(all_labels, all_probabilities)),
        "ranking_log_loss": ranking_loss_sum / max(1, evaluated_races),
        **_race_level_metrics(all_predictions),
        "policy": policy,
        "candidate_tickets": int(totals["candidate_tickets"]),
        "selected_races": int(totals["races_bet"]),
        "tickets": tickets,
        "hit_tickets": int(totals["hit_tickets"]),
        "ticket_hit_rate": int(totals["hit_tickets"]) / tickets if tickets else 0.0,
        "stake_yen": stake_yen,
        "return_yen": return_yen,
        "profit_yen": return_yen - stake_yen,
        "roi": return_yen / stake_yen if stake_yen else 0.0,
        "max_drawdown_yen": profit_state[2],
        "race_days": len(daily_rows),
        "winning_days": int(totals["winning_days"]),
        "losing_days": int(totals["losing_days"]),
        "profitable_folds": sum(value >= 1.0 for value in fold_rois),
        "folds": fold_rows,
        "daily": daily_rows,
        "promotion_gate": {
            "minimum_roi": 1.0,
            "minimum_profitable_folds": required_profitable_folds,
            "roi_pass": return_yen > stake_yen,
            "fold_stability_pass": sum(value >= 1.0 for value in fold_rois) >= required_profitable_folds,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    result["promotion_eligible"] = bool(
        result["promotion_gate"]["roi_pass"] and result["promotion_gate"]["fold_stability_pass"]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output_path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Nested walk-forward listwise past-log model.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--output", default="data/models/listwise_pl_v1_2fold.json")
    parser.add_argument("--feature-cache", default="data/models/calibrated_shadow_features_16384")
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--min-train-races", type=int, default=500)
    parser.add_argument("--n-features", type=int, default=1 << 14)
    parser.add_argument("--batch-races", type=int, default=1_000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--targets", default="winner,top3_pl")
    parser.add_argument("--alphas", default="0.00001,0.0001")
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--min-validation-races", type=int, default=250)
    parser.add_argument("--daily-budget-yen", type=int, default=10_000)
    parser.add_argument("--ev-threshold", type=float, default=1.20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    init_db(args.db)
    with connection(args.db) as conn:
        result = run_backtest(conn, output_path=Path(args.output), args=args)
    compact = {key: value for key, value in result.items() if key not in {"folds", "daily"}}
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
