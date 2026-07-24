from __future__ import annotations

import argparse
import gc
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from sklearn.feature_extraction import FeatureHasher

from ..adaptive_allocation import zero_totals
from ..bankroll_backtest import _load_trifecta_payouts
from ..db import connection, init_db
from ..feature_tuning import (
    FEATURE_GROUPS,
    _ensure_sparse_index32,
    iter_race_feature_rows,
    load_complete_race_ids,
    normalize_drop_feature_groups,
    to_hashable,
)
from ..hashed_feature_dataset import (
    CACHE_VERSION,
    FEATURE_SCHEMA_VERSION,
    HashedRaceDataset,
    cache_paths,
    load_hashed_dataset,
    load_or_build_hashed_dataset,
    race_ids_sha256,
    save_hashed_dataset,
)
from .model import (
    TARGETS,
    evaluate_range,
    fit_scaler,
    train_listwise_model,
)
from .validation import default_policy, evaluate_bankroll_fold
from ..standard_evaluation import race_set_sha256


FeatureVariants = tuple[tuple[str, tuple[str, ...]], ...]


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


def _resolved_variants(
    variants: FeatureVariants | None,
) -> FeatureVariants:
    return tuple(feature_variants()) if variants is None else variants


def load_variant_dataset(
    conn,
    *,
    race_keys: list[tuple[str, str, str, int]],
    cache_dir: Path,
    name: str,
    dropped: tuple[str, ...],
    n_features: int,
    batch_races: int,
    write_cache: bool = True,
) -> tuple[HashedRaceDataset, str]:
    dataset, source, _cache_prefix = load_variant_dataset_with_cache(
        conn,
        race_keys=race_keys,
        cache_dir=cache_dir,
        name=name,
        dropped=dropped,
        n_features=n_features,
        batch_races=batch_races,
        write_cache=write_cache,
    )
    return dataset, source


def variant_cache_prefix(cache_dir: Path, *, n_features: int, name: str) -> Path:
    return cache_dir / f"listwise_search_{int(n_features)}_{name}"


def load_variant_dataset_with_cache(
    conn,
    *,
    race_keys: list[tuple[str, str, str, int]],
    cache_dir: Path,
    name: str,
    dropped: tuple[str, ...],
    n_features: int,
    batch_races: int,
    write_cache: bool = True,
    fallback_cache_prefixes: tuple[Path, ...] = (),
) -> tuple[HashedRaceDataset, str, Path | None]:
    hasher = FeatureHasher(
        n_features=n_features,
        input_type="dict",
        alternate_sign=False,
    )
    normalized = normalize_drop_feature_groups(dropped)
    primary_prefix = variant_cache_prefix(
        cache_dir,
        n_features=n_features,
        name=name,
    ).resolve()
    normalized_fallbacks = tuple(prefix.resolve() for prefix in fallback_cache_prefixes)
    read_prefixes = list(dict.fromkeys((primary_prefix, *normalized_fallbacks)))
    for read_prefix in read_prefixes:
        loaded = load_hashed_dataset(
            read_prefix,
            race_keys=race_keys,
            n_features=n_features,
            drop_feature_groups=normalized,
            hasher=hasher,
        )
        if loaded is not None:
            return loaded, "disk", read_prefix

    dataset, source = load_or_build_hashed_dataset(
        cache_prefix=primary_prefix,
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
        write_cache=write_cache,
    )
    return dataset, source, primary_prefix if write_cache else None


def cleanup_selected_cache_family(
    cache_dir: Path,
    *,
    n_features: int,
    variants: FeatureVariants | None = None,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for variant_name, _dropped in _resolved_variants(variants):
        prefix = variant_cache_prefix(
            cache_dir,
            n_features=n_features,
            name=variant_name,
        )
        for path in cache_paths(prefix).values():
            path.unlink(missing_ok=True)
        for path in cache_dir.glob(f".{prefix.name}.*.tmp"):
            if path.is_file():
                path.unlink()


def selected_cache_candidates(
    cache_dir: Path,
    *,
    n_features: int,
    variants: FeatureVariants | None = None,
) -> list[Path]:
    candidates: list[Path] = []
    for variant_name, _dropped in _resolved_variants(variants):
        prefix = variant_cache_prefix(
            cache_dir,
            n_features=n_features,
            name=variant_name,
        )
        if cache_paths(prefix)["manifest"].exists():
            candidates.append(prefix)
    return candidates


def _candidate_key(variant_name: str, target: str, alpha: float) -> str:
    return json.dumps([variant_name, target, float(alpha)], separators=(",", ":"))


def _checkpoint_signature(
    *,
    args: argparse.Namespace,
    race_keys: list[tuple[str, str, str, int]],
    train_end: int,
    selection_end: int,
    targets: tuple[str, ...],
    alphas: tuple[float, ...],
    variants: FeatureVariants | None = None,
) -> dict[str, Any]:
    return {
        "checkpoint_version": 1,
        "cache_version": CACHE_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "as_of_date": getattr(args, "as_of_date", None),
        "race_count": len(race_keys),
        "race_universe_sha256": race_ids_sha256(race_keys),
        "train_end": train_end,
        "selection_end": selection_end,
        "n_features": int(args.n_features),
        "batch_races": int(args.batch_races),
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "targets": list(targets),
        "alphas": list(alphas),
        "feature_variants": [
            [name, list(dropped)] for name, dropped in _resolved_variants(variants)
        ],
    }


def _load_checkpoint(path: Path, signature: dict[str, Any]) -> dict[str, dict[str, Any]]:
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    if checkpoint.get("signature") != signature:
        return {}
    rows = checkpoint.get("search_results")
    if not isinstance(rows, list):
        return {}
    allowed_drops = {
        str(name): list(dropped)
        for name, dropped in signature.get("feature_variants", [])
    }
    allowed_keys = {
        _candidate_key(name, target, alpha)
        for name in allowed_drops
        for target in signature.get("targets", [])
        for alpha in signature.get("alphas", [])
    }
    required_fields = (
        "drop_feature_groups",
        "entry_log_loss",
        "ranking_log_loss",
        "winner_top1_accuracy",
        "trifecta_top5_hit_rate",
        "training_history",
    )
    completed: dict[str, dict[str, Any]] = {}
    try:
        for row in rows:
            variant_name = str(row["feature_variant"])
            key = _candidate_key(
                variant_name,
                str(row["target"]),
                float(row["alpha"]),
            )
            if (
                key not in allowed_keys
                or key in completed
                or any(field not in row for field in required_fields)
                or row["drop_feature_groups"] != allowed_drops.get(variant_name)
            ):
                return {}
            completed[key] = row
    except (KeyError, TypeError, ValueError):
        return {}
    return completed


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
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


def _ordered_rows(
    completed: dict[str, dict[str, Any]],
    *,
    targets: tuple[str, ...],
    alphas: tuple[float, ...],
    variants: FeatureVariants | None = None,
) -> list[dict[str, Any]]:
    return [
        completed[_candidate_key(variant_name, target, alpha)]
        for variant_name, _dropped in _resolved_variants(variants)
        for target in targets
        for alpha in alphas
        if _candidate_key(variant_name, target, alpha) in completed
    ]


def _selected_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return min(rows, key=lambda row: (
        float(row["ranking_log_loss"]),
        float(row["entry_log_loss"]),
        -float(row["trifecta_top5_hit_rate"]),
    ))


def _evaluate_variant(
    conn,
    *,
    request: dict[str, Any],
    candidate_workers: int,
    on_candidate_complete: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[HashedRaceDataset, dict[str, Any]]:
    variant_started = time.perf_counter()
    variant_name = str(request["variant_name"])
    dropped = tuple(str(value) for value in request["dropped"])
    targets = tuple(str(value) for value in request["targets"])
    alphas = tuple(float(value) for value in request["alphas"])
    completed = {
        _candidate_key(variant_name, str(row["target"]), float(row["alpha"])): row
        for row in request["completed_rows"]
    }
    dataset, cache_source = load_variant_dataset(
        conn,
        race_keys=request["race_keys"],
        cache_dir=Path(request["cache_dir"]),
        name=variant_name,
        dropped=dropped,
        n_features=int(request["n_features"]),
        batch_races=int(request["batch_races"]),
        write_cache=bool(request["write_cache"]),
    )
    missing = [
        (target, alpha)
        for target in targets
        for alpha in alphas
        if _candidate_key(variant_name, target, alpha) not in completed
    ]
    scaler = (
        fit_scaler(
            dataset,
            race_end=int(request["train_end"]),
            batch_rows=int(request["batch_races"]) * 6,
        )
        if missing
        else None
    )

    def evaluate_candidate(target: str, alpha: float) -> dict[str, Any]:
        model, history = train_listwise_model(
            dataset,
            train_race_end=int(request["train_end"]),
            target=target,
            alpha=alpha,
            learning_rate=float(request["learning_rate"]),
            epochs=int(request["epochs"]),
            batch_races=int(request["batch_races"]),
            scaler=scaler,
        )
        metrics, _ = evaluate_range(
            dataset,
            model,
            race_start=int(request["train_end"]),
            race_end=int(request["selection_end"]),
            batch_races=int(request["batch_races"]),
        )
        return {
            "feature_variant": variant_name,
            "drop_feature_groups": list(dropped),
            "target": target,
            "alpha": alpha,
            "cache_source": cache_source,
            "matrix_nnz": int(dataset.matrix.nnz),
            "training_history": history,
            **metrics,
        }

    if missing:
        with ThreadPoolExecutor(max_workers=candidate_workers) as executor:
            futures = {
                executor.submit(evaluate_candidate, target, alpha): (target, alpha)
                for target, alpha in missing
            }
            for future in as_completed(futures):
                target, alpha = futures[future]
                row = future.result()
                completed[_candidate_key(variant_name, target, alpha)] = row
                if on_candidate_complete is not None:
                    on_candidate_complete(row)
    rows = [
        completed[_candidate_key(variant_name, target, alpha)]
        for target in targets
        for alpha in alphas
    ]
    return dataset, {
        "feature_variant": variant_name,
        "rows": rows,
        "elapsed_seconds": round(time.perf_counter() - variant_started, 3),
    }


def _checkpoint_payload(
    signature: dict[str, Any],
    completed: dict[str, dict[str, Any]],
    *,
    targets: tuple[str, ...],
    alphas: tuple[float, ...],
    variants: FeatureVariants | None = None,
) -> dict[str, Any]:
    return {
        "signature": signature,
        "search_results": _ordered_rows(
            completed,
            targets=targets,
            alphas=alphas,
            variants=variants,
        ),
    }


def search(
    conn,
    *,
    args: argparse.Namespace,
    variants: FeatureVariants | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    run_variants = _resolved_variants(variants)
    race_keys = [
        row
        for row in load_complete_race_ids(conn)
        if not args.as_of_date or str(row[1]) <= args.as_of_date
    ]
    if not race_keys:
        raise ValueError("no complete races exist on or before as-of date")
    train_end = day_boundary(race_keys, int(len(race_keys) * args.train_fraction))
    selection_end = day_boundary(race_keys, int(len(race_keys) * args.selection_fraction))
    if selection_end <= train_end:
        raise ValueError("selection boundary must be after training boundary")
    targets = tuple(value.strip() for value in args.targets.split(",") if value.strip())
    alphas = tuple(float(value) for value in args.alphas.split(",") if value.strip())
    if not targets or any(value not in TARGETS for value in targets):
        raise ValueError(f"targets must be selected from {TARGETS}")
    variant_workers = int(args.variant_workers)
    if variant_workers != 1:
        raise ValueError("variant workers must be 1 to avoid dataset matrix duplication")
    candidate_workers = int(args.candidate_workers)
    if candidate_workers not in (1, 2, 3, 4):
        raise ValueError("candidate workers must be between 1 and 4")
    db = str(args.db)
    if not db:
        raise ValueError("database DSN is required for feature search")
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    selected_cache_dir = Path(args.selected_cache_dir) if args.selected_cache_dir else None
    if selected_cache_dir is not None:
        if selected_cache_dir.resolve() == cache_dir.resolve():
            raise ValueError("selected cache dir must differ from the persistent cache dir")
    output = Path(args.output)
    checkpoint_path = (
        Path(args.checkpoint)
        if getattr(args, "checkpoint", None)
        else output.with_name(f".{output.name}.checkpoint.json")
    )
    checkpoint_signature = _checkpoint_signature(
        args=args,
        race_keys=race_keys,
        train_end=train_end,
        selection_end=selection_end,
        targets=targets,
        alphas=alphas,
        variants=run_variants,
    )
    completed = _load_checkpoint(checkpoint_path, checkpoint_signature)
    if completed:
        print(json.dumps({"checkpoint_resumed_candidates": len(completed)}), flush=True)
    resumed_rows = _ordered_rows(
        completed,
        targets=targets,
        alphas=alphas,
        variants=run_variants,
    )
    resumed_selected = _selected_row(resumed_rows) if resumed_rows else None
    active_cache_variant: str | None = None
    if selected_cache_dir is not None:
        candidates = selected_cache_candidates(
            selected_cache_dir,
            n_features=args.n_features,
            variants=run_variants,
        )
        expected_prefix = (
            variant_cache_prefix(
                selected_cache_dir,
                n_features=args.n_features,
                name=str(resumed_selected["feature_variant"]),
            )
            if resumed_selected is not None
            else None
        )
        if expected_prefix is not None and candidates == [expected_prefix]:
            active_cache_variant = str(resumed_selected["feature_variant"])
        else:
            cleanup_selected_cache_family(
                selected_cache_dir,
                n_features=args.n_features,
                variants=run_variants,
            )

    requests: list[dict[str, Any]] = []
    for variant_name, dropped in run_variants:
        candidate_keys = [
            _candidate_key(variant_name, target, alpha)
            for target in targets
            for alpha in alphas
        ]
        checkpoint_complete = all(key in completed for key in candidate_keys)
        needs_best_cache = (
            selected_cache_dir is not None
            and resumed_selected is not None
            and str(resumed_selected["feature_variant"]) == variant_name
            and active_cache_variant != variant_name
        )
        if checkpoint_complete and not needs_best_cache:
            print(json.dumps({
                "feature_variant_checkpoint_complete": variant_name,
                "candidates": len(candidate_keys),
            }), flush=True)
            continue
        requests.append({
            "db": db,
            "race_keys": race_keys,
            "cache_dir": str(cache_dir),
            "variant_name": variant_name,
            "dropped": dropped,
            "n_features": int(args.n_features),
            "batch_races": int(args.batch_races),
            "write_cache": args.cache_write_mode == "always",
            "train_end": train_end,
            "selection_end": selection_end,
            "targets": targets,
            "alphas": alphas,
            "learning_rate": float(args.learning_rate),
            "epochs": int(args.epochs),
            "completed_rows": [
                completed[key] for key in candidate_keys if key in completed
            ],
        })

    for request in requests:
        dataset: HashedRaceDataset | None = None
        variant_name = str(request["variant_name"])
        existing_keys = set(completed)

        def record_candidate(row: dict[str, Any]) -> None:
            key = _candidate_key(
                variant_name,
                str(row["target"]),
                float(row["alpha"]),
            )
            completed[key] = row
            _write_json_atomic(
                checkpoint_path,
                _checkpoint_payload(
                    checkpoint_signature,
                    completed,
                    targets=targets,
                    alphas=alphas,
                    variants=run_variants,
                ),
            )

        try:
            dataset, payload = _evaluate_variant(
                conn,
                request=request,
                candidate_workers=candidate_workers,
                on_candidate_complete=record_candidate,
            )
            for row in payload["rows"]:
                key = _candidate_key(
                    variant_name,
                    str(row["target"]),
                    float(row["alpha"]),
                )
                completed[key] = row
                if key not in existing_keys:
                    print(json.dumps({
                        name: value
                        for name, value in row.items()
                        if name != "training_history"
                    }, ensure_ascii=False), flush=True)
            _write_json_atomic(
                checkpoint_path,
                _checkpoint_payload(
                    checkpoint_signature,
                    completed,
                    targets=targets,
                    alphas=alphas,
                    variants=run_variants,
                ),
            )
            current_selected = _selected_row(
                _ordered_rows(
                    completed,
                    targets=targets,
                    alphas=alphas,
                    variants=run_variants,
                )
            )
            if (
                selected_cache_dir is not None
                and str(current_selected["feature_variant"]) == variant_name
                and active_cache_variant != variant_name
            ):
                cleanup_selected_cache_family(
                    selected_cache_dir,
                    n_features=args.n_features,
                    variants=run_variants,
                )
                save_prefix = variant_cache_prefix(
                    selected_cache_dir,
                    n_features=args.n_features,
                    name=variant_name,
                )
                save_hashed_dataset(save_prefix, dataset)
                active_cache_variant = variant_name
            print(json.dumps({
                "feature_variant_complete": variant_name,
                "elapsed_seconds": payload["elapsed_seconds"],
            }), flush=True)
        finally:
            if dataset is not None:
                del dataset
            gc.collect()

    search_rows = _ordered_rows(
        completed,
        targets=targets,
        alphas=alphas,
        variants=run_variants,
    )
    expected_candidates = len(run_variants) * len(targets) * len(alphas)
    if len(search_rows) != expected_candidates:
        raise RuntimeError(
            f"incomplete feature search: {len(search_rows)} of {expected_candidates}"
        )
    selected = _selected_row(search_rows)
    selected_drops = tuple(str(value) for value in selected["drop_feature_groups"])
    if selected_cache_dir is not None:
        selected_cache_prefix = variant_cache_prefix(
            selected_cache_dir,
            n_features=args.n_features,
            name=str(selected["feature_variant"]),
        )
        candidates = selected_cache_candidates(
            selected_cache_dir,
            n_features=args.n_features,
            variants=run_variants,
        )
        if candidates != [selected_cache_prefix]:
            raise RuntimeError("selected cache directory must contain exactly one candidate")
        hasher = FeatureHasher(
            n_features=args.n_features,
            input_type="dict",
            alternate_sign=False,
        )
        dataset = load_hashed_dataset(
            selected_cache_prefix,
            race_keys=race_keys,
            n_features=args.n_features,
            drop_feature_groups=selected_drops,
            hasher=hasher,
        )
        if dataset is None:
            raise RuntimeError("selected cache is missing or invalid")
        cache_source = "selected_cache"
    else:
        dataset, cache_source, selected_cache_prefix = load_variant_dataset_with_cache(
            conn,
            race_keys=race_keys,
            cache_dir=cache_dir,
            name=str(selected["feature_variant"]),
            dropped=selected_drops,
            n_features=args.n_features,
            batch_races=args.batch_races,
            write_cache=args.cache_write_mode == "always",
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
    evaluation_hash = race_set_sha256(holdout_rows)
    bankroll["evaluation_race_set_sha256"] = evaluation_hash
    result = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "model": "pastlog_listwise_feature_teacher_search_v1",
        "comparison_role": "feature_teacher_selection_then_untouched_holdout",
        "races": len(race_keys),
        "race_universe_sha256": race_ids_sha256(race_keys),
        "as_of_date": args.as_of_date,
        "race_date_from": str(race_keys[0][1]),
        "race_date_through": str(race_keys[-1][1]),
        "train_races": train_end,
        "selection_races": selection_end - train_end,
        "holdout_races": len(race_keys) - selection_end,
        "evaluation_race_set_sha256": evaluation_hash,
        "n_features": args.n_features,
        "hashed_cache_version": CACHE_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_variants": [name for name, _drops in run_variants],
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
        "selected_cache_prefix": str(selected_cache_prefix)
        if selected_cache_prefix is not None
        else None,
        "selected_cache_dir": str(selected_cache_prefix.parent)
        if selected_cache_prefix is not None
        else None,
        "selected_cache_persistent": cache_source == "disk",
        "final_training_history": final_history,
        "holdout": {**holdout_metrics, "evaluation_race_set_sha256": evaluation_hash, "bankroll": bankroll},
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
    _write_json_atomic(output, result)
    checkpoint_path.unlink(missing_ok=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Past-log feature-group and teacher search.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--output", default="data/models/listwise_feature_teacher_search_v1.json")
    parser.add_argument("--cache-dir", default="data/models/listwise_search_cache")
    parser.add_argument(
        "--cache-write-mode",
        choices=("always", "never"),
        default="always",
    )
    parser.add_argument("--selected-cache-dir")
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--variant-workers",
        type=int,
        choices=(1,),
        default=1,
        help="Feature variants are sequential to avoid dataset duplication (fixed at 1).",
    )
    parser.add_argument(
        "--candidate-workers",
        type=int,
        choices=(1, 2, 3, 4),
        default=1,
        help="Candidates sharing one read-only variant dataset (1-4).",
    )
    parser.add_argument("--as-of-date")
    parser.add_argument("--n-features", type=int, default=1 << 13)
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
