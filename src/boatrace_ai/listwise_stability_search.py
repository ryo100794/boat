from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .adaptive_allocation import zero_totals
from .bankroll_backtest import _load_trifecta_payouts
from .db import connection, init_db
from .feature_tuning import load_complete_race_ids
from .listwise_feature_search import day_boundary, feature_variants, load_variant_dataset
from .listwise_ranking_model import evaluate_range, fit_scaler, train_listwise_model
from .listwise_validation import default_policy, evaluate_bankroll_fold


def candidate_key(row: dict[str, Any]) -> tuple[str, str, float]:
    return str(row["feature_variant"]), str(row["target"]), float(row["alpha"])


def summarize_candidate(rows: list[dict[str, Any]], *, baseline_top1: float) -> dict[str, Any]:
    ranking = [float(row["ranking_log_loss"]) for row in rows]
    top1 = [float(row["winner_top1_accuracy"]) for row in rows]
    top5 = [float(row["trifecta_top5_hit_rate"]) for row in rows]
    mean_ranking = statistics.fmean(ranking)
    ranking_std = statistics.pstdev(ranking)
    mean_top1 = statistics.fmean(top1)
    min_top1 = min(top1)
    constraint_pass = mean_top1 >= baseline_top1 - 0.005 and min_top1 >= 0.54
    return {
        "feature_variant": rows[0]["feature_variant"],
        "drop_feature_groups": rows[0]["drop_feature_groups"],
        "target": rows[0]["target"],
        "alpha": rows[0]["alpha"],
        "folds": rows,
        "mean_ranking_log_loss": mean_ranking,
        "ranking_log_loss_std": ranking_std,
        "worst_ranking_log_loss": max(ranking),
        "mean_winner_top1_accuracy": mean_top1,
        "min_winner_top1_accuracy": min_top1,
        "mean_trifecta_top5_hit_rate": statistics.fmean(top5),
        "min_trifecta_top5_hit_rate": min(top5),
        "top1_stability_constraint_pass": constraint_pass,
        "stability_score": mean_ranking + ranking_std,
    }


def run(conn, *, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    prior = json.loads(Path(args.initial_search_result).read_text(encoding="utf-8"))
    race_keys = load_complete_race_ids(conn)
    boundaries = [
        (
            day_boundary(race_keys, int(len(race_keys) * train_fraction)),
            day_boundary(race_keys, int(len(race_keys) * test_fraction)),
        )
        for train_fraction, test_fraction in ((0.45, 0.60), (0.60, 0.75))
    ]
    prior_train_end = int(prior["train_races"])
    prior_test_end = prior_train_end + int(prior["selection_races"])
    prior_rows = {candidate_key(row): row for row in prior["search_results"]}
    targets = tuple(str(value) for value in prior["teacher_targets"])
    alphas = tuple(float(value) for value in prior["alphas"])
    by_candidate: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)

    for variant_name, dropped in feature_variants():
        dataset, cache_source = load_variant_dataset(
            conn,
            race_keys=race_keys,
            cache_dir=Path(args.cache_dir),
            name=variant_name,
            dropped=dropped,
            n_features=int(prior["n_features"]),
            batch_races=args.batch_races,
        )
        for fold_number, (train_end, test_end) in enumerate(boundaries, start=1):
            scaler = fit_scaler(dataset, race_end=train_end, batch_rows=args.batch_races * 6)
            for target in targets:
                for alpha in alphas:
                    model, _history = train_listwise_model(
                        dataset,
                        train_race_end=train_end,
                        target=target,
                        alpha=alpha,
                        learning_rate=args.learning_rate,
                        epochs=args.epochs,
                        batch_races=args.batch_races,
                        scaler=scaler,
                    )
                    metrics, _ = evaluate_range(
                        dataset,
                        model,
                        race_start=train_end,
                        race_end=test_end,
                        batch_races=args.batch_races,
                    )
                    row = {
                        "fold": fold_number,
                        "train_races": train_end,
                        "test_races": test_end - train_end,
                        "feature_variant": variant_name,
                        "drop_feature_groups": list(dropped),
                        "target": target,
                        "alpha": alpha,
                        "cache_source": cache_source,
                        **metrics,
                    }
                    by_candidate[candidate_key(row)].append(row)
                    print(json.dumps(row, ensure_ascii=False), flush=True)
        for target in targets:
            for alpha in alphas:
                key = (variant_name, target, alpha)
                previous = prior_rows[key]
                by_candidate[key].append({
                    "fold": 3,
                    "train_races": prior_train_end,
                    "test_races": prior_test_end - prior_train_end,
                    "feature_variant": variant_name,
                    "drop_feature_groups": list(dropped),
                    "target": target,
                    "alpha": alpha,
                    "cache_source": previous.get("cache_source"),
                    **{
                        metric: previous[metric]
                        for metric in (
                            "evaluated_races",
                            "entry_log_loss",
                            "entry_brier",
                            "ranking_log_loss",
                            "winner_top1_accuracy",
                            "trifecta_top1_hit_rate",
                            "trifecta_top5_hit_rate",
                        )
                    },
                })
        del dataset
        gc.collect()

    summaries = [
        summarize_candidate(rows, baseline_top1=args.baseline_top1)
        for rows in by_candidate.values()
    ]
    eligible = [row for row in summaries if row["top1_stability_constraint_pass"]]
    pool = eligible or summaries
    selected = min(pool, key=lambda row: (
        float(row["stability_score"]),
        -float(row["mean_winner_top1_accuracy"]),
        -float(row["mean_trifecta_top5_hit_rate"]),
    ))

    dataset, cache_source = load_variant_dataset(
        conn,
        race_keys=race_keys,
        cache_dir=Path(args.cache_dir),
        name=str(selected["feature_variant"]),
        dropped=tuple(str(value) for value in selected["drop_feature_groups"]),
        n_features=int(prior["n_features"]),
        batch_races=args.batch_races,
    )
    scaler = fit_scaler(dataset, race_end=prior_test_end, batch_rows=args.batch_races * 6)
    model, history = train_listwise_model(
        dataset,
        train_race_end=prior_test_end,
        target=str(selected["target"]),
        alpha=float(selected["alpha"]),
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_races=args.batch_races,
        scaler=scaler,
    )
    diagnostic_metrics, diagnostic_rows = evaluate_range(
        dataset,
        model,
        race_start=prior_test_end,
        race_end=len(race_keys),
        batch_races=args.batch_races,
        keep_rows=True,
    )
    policy = default_policy(daily_budget_yen=args.daily_budget_yen, ev_threshold=args.ev_threshold)
    policy.update({
        "feature_variant": selected["feature_variant"],
        "drop_feature_groups": selected["drop_feature_groups"],
        "target": selected["target"],
        "selection": "three temporal validation windows; mean loss plus fold dispersion; top1 stability constraint",
    })
    totals = zero_totals()
    daily_rows: list[dict[str, Any]] = []
    bankroll, profit_state = evaluate_bankroll_fold(
        rows_by_race=diagnostic_rows,
        train_races={race_id for race_id, *_rest in race_keys[:prior_test_end]},
        test_dates={date for _race_id, date, _jcd, _rno in race_keys[prior_test_end:]},
        payouts=_load_trifecta_payouts(conn),
        policy=policy,
        totals=totals,
        daily_rows=daily_rows,
        profit_state=(0, 0, 0),
    )
    result = {
        "model": "pastlog_listwise_temporal_stability_v1",
        "comparison_role": "multi_window_feature_teacher_stability_search",
        "source_initial_search": args.initial_search_result,
        "selection_windows": [
            {"train_races": train_end, "test_races": test_end - train_end}
            for train_end, test_end in boundaries + [(prior_train_end, prior_test_end)]
        ],
        "selection_rule": "minimize mean top3 PL ranking loss + population std; require mean top1 >= baseline-0.5pp and every fold >=54% when feasible",
        "candidates": sorted(summaries, key=lambda row: row["stability_score"]),
        "selected": {key: selected[key] for key in (
            "feature_variant",
            "drop_feature_groups",
            "target",
            "alpha",
            "mean_ranking_log_loss",
            "ranking_log_loss_std",
            "mean_winner_top1_accuracy",
            "min_winner_top1_accuracy",
            "mean_trifecta_top5_hit_rate",
            "top1_stability_constraint_pass",
        )},
        "cache_source": cache_source,
        "final_training_history": history,
        "latest_interval_role": "diagnostic reuse after initial holdout failure; not an untouched promotion holdout",
        "latest_interval": {**diagnostic_metrics, "bankroll": bankroll},
        "roi": bankroll["roi"],
        "profit_yen": bankroll["profit_yen"],
        "stake_yen": bankroll["stake_yen"],
        "return_yen": bankroll["return_yen"],
        "max_drawdown_yen": profit_state[2],
        "promotion_gate": {
            "independent_holdout": False,
            "minimum_roi": 1.0,
            "minimum_top1_accuracy": args.baseline_top1,
            "roi_pass": bankroll["roi"] > 1.0,
            "top1_pass": diagnostic_metrics["winner_top1_accuracy"] >= args.baseline_top1,
            "temporal_stability_pass": selected["top1_stability_constraint_pass"],
        },
        "promotion_eligible": False,
        "daily": daily_rows,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-window stability search for listwise features and teacher.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--initial-search-result", default="data/models/listwise_feature_teacher_search_v1.json")
    parser.add_argument("--output", default="data/models/listwise_temporal_stability_v1.json")
    parser.add_argument("--cache-dir", default="data/models/listwise_search_cache")
    parser.add_argument("--batch-races", type=int, default=1_000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--baseline-top1", type=float, default=0.5642)
    parser.add_argument("--daily-budget-yen", type=int, default=10_000)
    parser.add_argument("--ev-threshold", type=float, default=1.20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    init_db(args.db)
    with connection(args.db) as conn:
        result = run(conn, args=args)
    compact = {key: value for key, value in result.items() if key not in {"candidates", "daily"}}
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
