from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time

import joblib
import numpy as np
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
    load_hashed_dataset,
    promote_legacy_hashed_dataset,
    race_ids_sha256,
)
from .feature_search import (
    _write_json_atomic,
    load_variant_dataset_with_cache,
    variant_cache_prefix,
)
from .newton import refine_newton_cg
from .model import ListwiseLinearModel, evaluate_range, fit_scaler, train_listwise_model
from .validation import default_policy, evaluate_bankroll_fold
from ..standard_evaluation import race_set_sha256


def validate_search_race_universe(
    search_result: dict[str, Any],
    race_keys: list[tuple[str, str, str, int]],
    *,
    allow_legacy: bool = False,
) -> None:
    expected_count = len(race_keys)
    expected_sha256 = race_ids_sha256(race_keys)
    if search_result.get("races") != expected_count:
        raise ValueError(
            "search result race count does not match the current race universe: "
            f"search={search_result.get('races')!r} current={expected_count}"
        )
    train_end = int(search_result.get("train_races", 0))
    selection_end = train_end + int(search_result.get("selection_races", 0))
    if not 0 < train_end < selection_end < expected_count:
        raise ValueError(
            "search result train/selection boundaries are invalid for the current race universe"
        )
    expected_holdout = expected_count - selection_end
    if search_result.get("holdout_races") != expected_holdout:
        raise ValueError(
            "search result holdout race count does not match the current race universe: "
            f"search={search_result.get('holdout_races')!r} current={expected_holdout}"
        )
    expected_evaluation_sha256 = race_set_sha256(
        race_id for race_id, *_rest in race_keys[selection_end:]
    )
    if search_result.get("evaluation_race_set_sha256") != expected_evaluation_sha256:
        raise ValueError(
            "search result evaluation holdout hash does not match the current race universe"
        )
    recorded_sha256 = search_result.get("race_universe_sha256")
    cache_version = search_result.get("hashed_cache_version")
    schema_version = search_result.get("feature_schema_version")
    metadata = (recorded_sha256, cache_version, schema_version)
    if allow_legacy and all(value is None for value in metadata):
        return
    if recorded_sha256 != expected_sha256:
        raise ValueError(
            "search result race universe hash does not match the current race universe; "
            "legacy results without race_universe_sha256 must be regenerated"
        )
    if cache_version != CACHE_VERSION or schema_version != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            "search result cache/schema version is incompatible; regenerate the search result"
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_resume_model_artifact(
    path: Path,
    *,
    feature_variant: str,
    drop_feature_groups: tuple[str, ...],
    n_features: int,
    trained_races: int,
    trained_through: tuple[str, str, str, int],
    target: str,
    alpha: float,
    race_universe_sha256: str,
    allow_legacy: bool = False,
) -> tuple[ListwiseLinearModel, str]:
    """Load a Newton checkpoint only when its training contract is identical."""
    artifact_path = Path(path)
    payload = joblib.load(artifact_path)
    if not isinstance(payload, dict):
        raise ValueError("resume model artifact must contain a mapping payload")
    model = payload.get("model")
    if not isinstance(model, ListwiseLinearModel):
        raise ValueError("resume model artifact does not contain a ListwiseLinearModel")

    checks = (
        ("feature_variant", payload.get("feature_variant"), feature_variant),
        (
            "drop_feature_groups",
            tuple(str(value) for value in payload.get("drop_feature_groups", ())),
            tuple(drop_feature_groups),
        ),
        ("n_features", payload.get("n_features"), int(n_features)),
        ("trained_races", payload.get("trained_races"), int(trained_races)),
        (
            "trained_through",
            tuple(payload.get("trained_through", ())),
            tuple(trained_through),
        ),
    )
    for name, actual, expected in checks:
        if actual != expected:
            raise ValueError(
                f"resume model artifact {name} mismatch: artifact={actual!r} expected={expected!r}"
            )

    payload_target = payload.get("target", model.target)
    if payload_target != target or model.target != target:
        raise ValueError(
            "resume model artifact target mismatch: "
            f"payload={payload_target!r} model={model.target!r} expected={target!r}"
        )
    payload_alpha = float(payload.get("alpha", model.alpha))
    if not (
        math.isclose(payload_alpha, float(alpha), rel_tol=1e-12, abs_tol=0.0)
        and math.isclose(float(model.alpha), float(alpha), rel_tol=1e-12, abs_tol=0.0)
    ):
        raise ValueError(
            "resume model artifact alpha mismatch: "
            f"payload={payload_alpha!r} model={model.alpha!r} expected={alpha!r}"
        )

    weights = getattr(model, "weights", None)
    if weights is None or getattr(weights, "ndim", None) != 1 or len(weights) != n_features:
        raise ValueError(
            "resume model artifact weights length does not match n_features: "
            f"weights={None if weights is None else len(weights)} expected={n_features}"
        )
    if not np.isfinite(np.asarray(weights, dtype=np.float64)).all():
        raise ValueError("resume model artifact weights contain non-finite values")
    scaler_features = getattr(model.scaler, "n_features_in_", None)
    if scaler_features != n_features:
        raise ValueError(
            "resume model artifact scaler n_features does not match: "
            f"scaler={scaler_features!r} expected={n_features}"
        )

    scaler_scale = np.asarray(getattr(model.scaler, "scale_", ()), dtype=np.float64)
    if (
        getattr(model.scaler, "with_mean", None) is not False
        or scaler_scale.shape != (n_features,)
        or not np.isfinite(scaler_scale).all()
        or np.any(scaler_scale <= 0.0)
    ):
        raise ValueError("resume model artifact scaler is invalid for sparse inference")

    recorded_sha256 = payload.get("race_universe_sha256")
    if recorded_sha256 is None:
        if not allow_legacy:
            raise ValueError(
                "resume model artifact lacks race_universe_sha256; "
                "use --allow-legacy-model-artifact explicitly"
            )
    elif recorded_sha256 != race_universe_sha256:
        raise ValueError(
            "resume model artifact race_universe_sha256 mismatch: "
            f"artifact={recorded_sha256!r} expected={race_universe_sha256!r}"
        )
    return model, _file_sha256(artifact_path)


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
    validate_search_race_universe(
        search_result,
        race_keys,
        allow_legacy=args.allow_legacy_search_result,
    )
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
    hasher = FeatureHasher(
        n_features=int(search_result["n_features"]),
        input_type="dict",
        alternate_sign=False,
    )
    if args.promote_legacy_cache:
        promoted = promote_legacy_hashed_dataset(
            primary_prefix,
            race_keys=race_keys,
            n_features=int(search_result["n_features"]),
            drop_feature_groups=dropped,
            hasher=hasher,
        )
        dataset = load_hashed_dataset(
            primary_prefix,
            race_keys=race_keys,
            n_features=int(search_result["n_features"]),
            drop_feature_groups=dropped,
            hasher=hasher,
        )
        if dataset is None:
            raise ValueError("legacy cache promotion or strict v2 validation failed")
        cache_source, cache_prefix = "disk", primary_prefix
        print(json.dumps({
            "cache_promotion": "promoted" if promoted else "already_v2",
            "cache_prefix": str(primary_prefix),
        }), flush=True)
    else:
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
    universe_sha256 = race_ids_sha256(race_keys)
    resume_source: str | None = None
    resume_source_sha256: str | None = None
    if args.resume_model:
        resume_path = Path(args.resume_model).resolve()
        initial, resume_source_sha256 = load_resume_model_artifact(
            resume_path,
            feature_variant=str(selected["feature_variant"]),
            drop_feature_groups=dropped,
            n_features=int(search_result["n_features"]),
            trained_races=selection_end,
            trained_through=race_keys[selection_end - 1],
            target=str(selected["target"]),
            alpha=float(selected["alpha"]),
            race_universe_sha256=universe_sha256,
            allow_legacy=args.allow_legacy_model_artifact,
        )
        resume_source = str(resume_path)
        adam_history: list[dict[str, float]] = []
        optimizer = "resumed matrix-free Newton-CG"
    else:
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
        optimizer = "Adam warm start + matrix-free Newton-CG"
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
        "coefficient_optimizer": optimizer,
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
        "race_universe_sha256": universe_sha256,
        "resume_source": resume_source,
        "resume_source_sha256": resume_source_sha256,
        "coefficient_optimizer": optimizer,
        "train_races": selection_end,
        "holdout_races": len(race_keys) - selection_end,
        "evaluation_race_set_sha256": evaluation_hash,
        "adam_history": adam_history,
        "newton_convergence": convergence,
        "holdout_before_newton": before_metrics,
        "holdout_after_newton": after_metrics,
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
            "hasher": hasher,
            "feature_variant": selected["feature_variant"],
            "drop_feature_groups": dropped,
            "n_features": int(search_result["n_features"]),
            "feature_schema_version": search_result["feature_schema_version"],
            "trained_races": selection_end,
            "trained_through": race_keys[selection_end - 1],
            "target": selected["target"],
            "alpha": float(selected["alpha"]),
            "race_universe_sha256": universe_sha256,
            "coefficient_optimizer": optimizer,
            "resume_source": resume_source,
            "resume_source_sha256": resume_source_sha256,
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
    parser.add_argument("--max-newton-iterations", type=int, default=10)
    parser.add_argument("--max-cg-iterations", type=int, default=50)
    parser.add_argument("--promote-legacy-cache", action="store_true")
    parser.add_argument("--resume-model")
    parser.add_argument("--allow-legacy-model-artifact", action="store_true")
    parser.add_argument("--allow-legacy-search-result", action="store_true")
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
