from __future__ import annotations

import argparse
import json
import os
import time

import joblib
from pathlib import Path
from typing import Any

from sklearn.feature_extraction import FeatureHasher

from ..adaptive_allocation import zero_totals
from ..bankroll_backtest import _load_trifecta_payouts
from ..db import connection, init_db
from ..feature_tuning import load_complete_race_ids
from ..hashed_feature_dataset import (
    CACHE_VERSION,
    FEATURE_SCHEMA_VERSION,
    race_ids_sha256,
)
from .feature_search import (
    _write_json_atomic,
    load_variant_dataset_with_cache,
    variant_cache_prefix,
)
from .newton import refine_newton_cg
from .model import evaluate_range, fit_scaler, train_listwise_model
from .validation import default_policy, evaluate_bankroll_fold
from ..standard_evaluation import race_set_sha256


def validate_search_race_universe(
    search_result: dict[str, Any],
    race_keys: list[tuple[str, str, str, int]],
) -> None:
    expected_count = len(race_keys)
    expected_sha256 = race_ids_sha256(race_keys)
    if search_result.get("races") != expected_count:
        raise ValueError(
            "search result race count does not match the current race universe: "
            f"search={search_result.get('races')!r} current={expected_count}"
        )
    if search_result.get("race_universe_sha256") != expected_sha256:
        raise ValueError(
            "search result race universe hash does not match the current race universe; "
            "legacy results without race_universe_sha256 must be regenerated"
        )
    if (
        search_result.get("hashed_cache_version") != CACHE_VERSION
        or search_result.get("feature_schema_version") != FEATURE_SCHEMA_VERSION
    ):
        raise ValueError(
            "search result cache/schema version is incompatible; regenerate the search result"
        )


def dump_joblib_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            joblib.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def run(conn, *, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    search_result = json.loads(Path(args.search_result).read_text(encoding="utf-8"))
    selected = search_result["selected"]
    race_keys = load_complete_race_ids(conn)
    validate_search_race_universe(search_result, race_keys)
    train_end = int(search_result["train_races"])
    selection_end = train_end + int(search_result["selection_races"])
    if not 0 < train_end < selection_end < len(race_keys):
        raise ValueError(
            "search result train/selection boundaries are invalid for the current race universe"
        )
    dropped = tuple(str(value) for value in selected.get("drop_feature_groups") or ())
    primary_prefix = variant_cache_prefix(
        Path(args.cache_dir),
        n_features=int(search_result["n_features"]),
        name=str(selected["feature_variant"]),
    ).resolve()
    recorded_prefix_value = search_result.get("selected_cache_prefix")
    recorded_prefix = Path(recorded_prefix_value) if recorded_prefix_value else None
    fallback_prefixes = (recorded_prefix,) if recorded_prefix is not None else ()
    dataset, cache_source, cache_prefix = load_variant_dataset_with_cache(
        conn,
        race_keys=race_keys,
        cache_dir=Path(args.cache_dir),
        name=str(selected["feature_variant"]),
        dropped=dropped,
        n_features=int(search_result["n_features"]),
        batch_races=args.batch_races,
        write_cache=args.cache_write_mode == "always",
        fallback_cache_prefixes=fallback_prefixes,
    )
    if cache_source == "disk" and cache_prefix != primary_prefix:
        print(json.dumps({
            "cache_resume": "persistent_fallback",
            "cache_prefix": str(cache_prefix),
        }), flush=True)
    elif cache_source == "built":
        print(json.dumps({
            "cache_resume": "cache_missing_building_explicitly",
            "requested_cache_prefix": str(primary_prefix),
            "recorded_cache_prefix": str(recorded_prefix)
            if recorded_prefix is not None
            else None,
        }), flush=True)
    scaler = fit_scaler(dataset, race_end=selection_end, batch_rows=args.batch_races * 6)
    initial, adam_history = train_listwise_model(
        dataset,
        train_race_end=selection_end,
        target=str(selected["target"]),
        alpha=float(selected["alpha"]),
        learning_rate=args.learning_rate,
        epochs=args.adam_epochs,
        batch_races=args.batch_races,
        scaler=scaler,
    )
    before_metrics, _ = evaluate_range(
        dataset,
        initial,
        race_start=selection_end,
        race_end=len(race_keys),
        batch_races=args.batch_races,
    )
    refined, convergence = refine_newton_cg(
        dataset,
        initial,
        train_race_end=selection_end,
        batch_races=args.batch_races,
        max_newton_iterations=args.max_newton_iterations,
        max_cg_iterations=args.max_cg_iterations,
        gradient_tolerance=args.gradient_tolerance,
        cg_tolerance=args.cg_tolerance,
    )
    after_metrics, holdout_rows = evaluate_range(
        dataset,
        refined,
        race_start=selection_end,
        race_end=len(race_keys),
        batch_races=args.batch_races,
        keep_rows=True,
    )
    policy = default_policy(
        daily_budget_yen=args.daily_budget_yen,
        ev_threshold=args.ev_threshold,
    )
    policy.update({
        "feature_variant": selected["feature_variant"],
        "drop_feature_groups": list(dropped),
        "target": selected["target"],
        "coefficient_optimizer": "Adam warm start + matrix-free Newton-CG",
    })
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
    evaluation_hash = race_set_sha256(holdout_rows)
    bankroll["evaluation_race_set_sha256"] = evaluation_hash
    after_metrics["evaluation_race_set_sha256"] = evaluation_hash
    result: dict[str, Any] = {
        "model": "pastlog_listwise_newton_cg_v1",
        "comparison_role": "selected_feature_teacher_newton_refinement_holdout",
        "source_search_result": args.search_result,
        "selected": selected,
        "cache_source": cache_source,
        "cache_prefix": str(cache_prefix) if cache_prefix is not None else None,
        "race_universe_sha256": race_ids_sha256(race_keys),
        "train_races": selection_end,
        "holdout_races": len(race_keys) - selection_end,
        "evaluation_race_set_sha256": evaluation_hash,
        "adam_history": adam_history,
        "newton_convergence": convergence,
        "holdout_before_newton": before_metrics,
        "holdout_after_newton": after_metrics,
        "policy": policy,
        "roi": bankroll["roi"],
        "profit_yen": bankroll["profit_yen"],
        "stake_yen": bankroll["stake_yen"],
        "return_yen": bankroll["return_yen"],
        "max_drawdown_yen": profit_state[2],
        "bankroll": bankroll,
        "daily": daily_rows,
        "promotion_gate": {
            "optimizer_converged": convergence["converged"],
            "minimum_roi": 1.0,
            "minimum_top1_accuracy": args.min_top1,
            "roi_pass": bankroll["roi"] > 1.0,
            "top1_pass": after_metrics["winner_top1_accuracy"] >= args.min_top1,
            "ranking_loss_not_worse": after_metrics["ranking_log_loss"] <= before_metrics["ranking_log_loss"],
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    gate = result["promotion_gate"]
    result["promotion_eligible"] = all((
        gate["optimizer_converged"],
        gate["roi_pass"],
        gate["top1_pass"],
        gate["ranking_loss_not_worse"],
    ))
    artifact_path = Path(args.model_output)
    dump_joblib_atomic(
        artifact_path,
        {
            "model": refined,
            "hasher": FeatureHasher(
                n_features=int(search_result["n_features"]),
                input_type="dict",
                alternate_sign=False,
            ),
            "feature_variant": selected["feature_variant"],
            "drop_feature_groups": dropped,
            "n_features": int(search_result["n_features"]),
            "trained_races": selection_end,
            "trained_through": race_keys[selection_end - 1],
            "race_universe_sha256": race_ids_sha256(race_keys),
        },
    )
    result["model_artifact"] = str(artifact_path)
    output = Path(args.output)
    _write_json_atomic(output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Newton-CG refinement of selected listwise model.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--search-result", default="data/models/listwise_feature_teacher_search_v1.json")
    parser.add_argument("--output", default="data/models/listwise_newton_cg_v1.json")
    parser.add_argument(
        "--model-output",
        default="data/models/listwise_newton_cg_v1.joblib",
    )
    parser.add_argument("--cache-dir", default="data/models/listwise_search_cache")
    parser.add_argument(
        "--cache-write-mode",
        choices=("always", "never"),
        default="always",
    )
    parser.add_argument("--batch-races", type=int, default=1_000)
    parser.add_argument("--adam-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--max-newton-iterations", type=int, default=5)
    parser.add_argument("--max-cg-iterations", type=int, default=20)
    parser.add_argument("--gradient-tolerance", type=float, default=1e-4)
    parser.add_argument("--cg-tolerance", type=float, default=1e-3)
    parser.add_argument("--daily-budget-yen", type=int, default=10_000)
    parser.add_argument("--ev-threshold", type=float, default=1.20)
    parser.add_argument("--min-top1", type=float, default=0.5642)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    init_db(args.db)
    with connection(args.db) as conn:
        result = run(conn, args=args)
    compact = {key: value for key, value in result.items() if key not in {"daily", "adam_history"}}
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
