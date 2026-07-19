from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sklearn.feature_extraction import FeatureHasher

from .adaptive_allocation import zero_totals
from .bankroll_backtest import _load_trifecta_payouts
from .db import connection, init_db
from .feature_tuning import (
    FEATURE_GROUPS,
    _ensure_sparse_index32,
    iter_race_feature_rows,
    load_complete_race_ids,
    normalize_drop_feature_groups,
    to_hashable,
)
from .hashed_feature_dataset import HashedRaceDataset, load_or_build_hashed_dataset
from .listwise_ranking_model import (
    TARGETS,
    evaluate_range,
    fit_scaler,
    train_listwise_model,
)
from .listwise_validation import default_policy, evaluate_bankroll_fold


def day_boundary(race_keys: list[tuple[str, str, str, int]], approximate: int) -> int:
    index = min(len(race_keys) - 1, max(1, int(approximate)))
    current_date = race_keys[index][1]
    while index < len(race_keys) and race_keys[index][1] == current_date:
        index += 1
    if index >= len(race_keys):
        raise ValueError("requested boundary leaves no future races")
    return index


def feature_variants() -> list[tuple[str, tuple[str, ...]]]:
    return [("full", ())] + [(f"drop_{group}", (group,)) for group in FEATURE_GROUPS]


def load_variant_dataset(
    conn,
    *,
    race_keys: list[tuple[str, str, str, int]],
    cache_dir: Path,
    name: str,
    dropped: tuple[str, ...],
    n_features: int,
    batch_races: int,
) -> tuple[HashedRaceDataset, str]:
    hasher = FeatureHasher(
        n_features=n_features,
        input_type="dict",
        alternate_sign=False,
    )
    normalized = normalize_drop_feature_groups(dropped)
    return load_or_build_hashed_dataset(
        cache_prefix=cache_dir / f"listwise_search_{n_features}_{name}",
        race_keys=race_keys,
        race_rows=lambda: iter_race_feature_rows(
            conn,
            include_races={race_id for race_id, *_rest in race_keys},
            drop_feature_groups=normalized,
        ),
        hasher=hasher,
        to_hashable=to_hashable,
        ensure_sparse_index32=_ensure_sparse_index32,
        drop_feature_groups=normalized,
        batch_size=batch_races * 6,
    )


def search(conn, *, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    race_keys = load_complete_race_ids(conn)
    train_end = day_boundary(race_keys, int(len(race_keys) * args.train_fraction))
    selection_end = day_boundary(race_keys, int(len(race_keys) * args.selection_fraction))
    if selection_end <= train_end:
        raise ValueError("selection boundary must be after training boundary")
    targets = tuple(value.strip() for value in args.targets.split(",") if value.strip())
    alphas = tuple(float(value) for value in args.alphas.split(",") if value.strip())
    if not targets or any(value not in TARGETS for value in targets):
        raise ValueError(f"targets must be selected from {TARGETS}")
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    search_rows: list[dict[str, Any]] = []

    for variant_name, dropped in feature_variants():
        variant_started = time.perf_counter()
        dataset, cache_source = load_variant_dataset(
            conn,
            race_keys=race_keys,
            cache_dir=cache_dir,
            name=variant_name,
            dropped=dropped,
            n_features=args.n_features,
            batch_races=args.batch_races,
        )
        scaler = fit_scaler(dataset, race_end=train_end, batch_rows=args.batch_races * 6)
        for target in targets:
            for alpha in alphas:
                model, history = train_listwise_model(
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
                    race_end=selection_end,
                    batch_races=args.batch_races,
                )
                row = {
                    "feature_variant": variant_name,
                    "drop_feature_groups": list(dropped),
                    "target": target,
                    "alpha": alpha,
                    "cache_source": cache_source,
                    "matrix_nnz": int(dataset.matrix.nnz),
                    "training_history": history,
                    **metrics,
                }
                search_rows.append(row)
                print(json.dumps({key: value for key, value in row.items() if key != "training_history"}, ensure_ascii=False), flush=True)
        print(json.dumps({
            "feature_variant_complete": variant_name,
            "elapsed_seconds": round(time.perf_counter() - variant_started, 3),
        }), flush=True)
        del dataset, scaler
        gc.collect()

    selected = min(search_rows, key=lambda row: (
        float(row["ranking_log_loss"]),
        float(row["entry_log_loss"]),
        -float(row["trifecta_top5_hit_rate"]),
    ))
    selected_drops = tuple(str(value) for value in selected["drop_feature_groups"])
    dataset, cache_source = load_variant_dataset(
        conn,
        race_keys=race_keys,
        cache_dir=cache_dir,
        name=str(selected["feature_variant"]),
        dropped=selected_drops,
        n_features=args.n_features,
        batch_races=args.batch_races,
    )
    scaler = fit_scaler(dataset, race_end=selection_end, batch_rows=args.batch_races * 6)
    final_model, final_history = train_listwise_model(
        dataset,
        train_race_end=selection_end,
        target=str(selected["target"]),
        alpha=float(selected["alpha"]),
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_races=args.batch_races,
        scaler=scaler,
    )
    holdout_metrics, holdout_rows = evaluate_range(
        dataset,
        final_model,
        race_start=selection_end,
        race_end=len(race_keys),
        batch_races=args.batch_races,
        keep_rows=True,
    )
    policy = default_policy(
        daily_budget_yen=args.daily_budget_yen,
        ev_threshold=args.ev_threshold,
    )
    policy["feature_variant"] = selected["feature_variant"]
    policy["drop_feature_groups"] = list(selected_drops)
    policy["target"] = selected["target"]
    totals = zero_totals()
    daily_rows: list[dict[str, Any]] = []
    bankroll, profit_state = evaluate_bankroll_fold(
        rows_by_race=holdout_rows,
        train_races={race_id for race_id, *_rest in race_keys[:selection_end]},
        test_dates={race_date for _race_id, race_date, _jcd, _rno in race_keys[selection_end:]},
        payouts=_load_trifecta_payouts(conn),
        policy=policy,
        totals=totals,
        daily_rows=daily_rows,
        profit_state=(0, 0, 0),
    )
    holdout_pass = bankroll["roi"] > 1.0 and holdout_metrics["winner_top1_accuracy"] >= args.min_top1
    result = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "model": "pastlog_listwise_feature_teacher_search_v1",
        "comparison_role": "feature_teacher_selection_then_untouched_holdout",
        "races": len(race_keys),
        "train_races": train_end,
        "selection_races": selection_end - train_end,
        "holdout_races": len(race_keys) - selection_end,
        "n_features": args.n_features,
        "feature_variants": [name for name, _drops in feature_variants()],
        "teacher_targets": list(targets),
        "alphas": list(alphas),
        "selection_metric": "minimum top3 PL ranking log loss; entry log loss and top5 as tie breaks",
        "search_results": search_rows,
        "selected": {
            key: selected[key]
            for key in (
                "feature_variant",
                "drop_feature_groups",
                "target",
                "alpha",
                "ranking_log_loss",
                "entry_log_loss",
                "winner_top1_accuracy",
                "trifecta_top5_hit_rate",
            )
        },
        "selected_cache_source": cache_source,
        "final_training_history": final_history,
        "holdout": {**holdout_metrics, "bankroll": bankroll},
        "policy": policy,
        "roi": bankroll["roi"],
        "profit_yen": bankroll["profit_yen"],
        "stake_yen": bankroll["stake_yen"],
        "return_yen": bankroll["return_yen"],
        "max_drawdown_yen": profit_state[2],
        "promotion_gate": {
            "minimum_roi": 1.0,
            "minimum_top1_accuracy": args.min_top1,
            "roi_pass": bankroll["roi"] > 1.0,
            "top1_pass": holdout_metrics["winner_top1_accuracy"] >= args.min_top1,
        },
        "promotion_eligible": holdout_pass,
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
    parser = argparse.ArgumentParser(description="Past-log feature-group and teacher search.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--output", default="data/models/listwise_feature_teacher_search_v1.json")
    parser.add_argument("--cache-dir", default="data/models/listwise_search_cache")
    parser.add_argument("--n-features", type=int, default=1 << 12)
    parser.add_argument("--batch-races", type=int, default=1_000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--targets", default="winner,top3_pl")
    parser.add_argument("--alphas", default="0.00001,0.0001")
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--selection-fraction", type=float, default=0.90)
    parser.add_argument("--daily-budget-yen", type=int, default=10_000)
    parser.add_argument("--ev-threshold", type=float, default=1.20)
    parser.add_argument("--min-top1", type=float, default=0.5642)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    init_db(args.db)
    with connection(args.db) as conn:
        result = search(conn, args=args)
    compact = {key: value for key, value in result.items() if key not in {"search_results", "daily"}}
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
