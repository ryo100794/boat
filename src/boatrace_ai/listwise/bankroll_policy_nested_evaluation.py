from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from sklearn.feature_extraction import FeatureHasher

from ..bankroll_backtest import _load_trifecta_payouts
from ..db import connection, init_db
from ..feature_schema import FEATURE_SCHEMA_VERSION
from ..feature_tuning import load_complete_race_ids
from ..hashed_feature_dataset import (
    CACHE_VERSION,
    HashedRaceDataset,
    cache_paths,
    load_hashed_dataset,
    race_ids_sha256,
)
from .bankroll_policy_evaluation import packed_candidates_from_rows
from .bankroll_policy_walk_forward import (
    PROTOCOL_VERSION,
    build_annual_walk_forward_folds,
    evaluate_annual_walk_forward,
)
from .feature_search import _write_json_atomic
from .model import evaluate_range, fit_scaler, train_listwise_model
from .newton_refine import search_race_date_through, validate_search_race_universe
from .validation import default_policy, nested_select_candidate


FORBIDDEN_SOURCE_JOB_IDS = frozenset({3995})


def build_fold_inputs(
    dataset: HashedRaceDataset,
    *,
    boundaries: Sequence[Mapping[str, Any]],
    payouts: Mapping[str, Mapping[str, Any]],
    targets: Sequence[str],
    alphas: Sequence[float],
    base_policy: Mapping[str, Any],
    learning_rate: float,
    epochs: int,
    batch_races: int,
    validation_fraction: float,
    min_validation_races: int,
    provenance: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fit fold-local prediction models and return policy-layer inputs.

    Every index is the first race of a calendar day. A training end is therefore
    exclusive and cannot contain labels from the prediction boundary day.
    """
    race_keys = dataset.race_keys
    first_index = _first_index_by_date(race_keys)
    fold_inputs: list[dict[str, Any]] = []
    model_audits: list[dict[str, Any]] = []
    payout_prior_weight = float(base_policy["payout_prior_weight"])
    payout_lookup = payouts if isinstance(payouts, dict) else dict(payouts)

    for expected_fold, boundary in enumerate(boundaries, start=1):
        fold = int(boundary.get("fold", expected_fold))
        selection_from = str(boundary["selection_date_from"])
        selection_through = str(boundary["selection_date_through"])
        holdout_from = str(boundary["holdout_date_from"])
        holdout_through = str(boundary["holdout_date_through"])
        selection_start = _required_day_index(first_index, selection_from)
        selection_end = _index_after_day(race_keys, first_index, selection_through)
        holdout_start = _required_day_index(first_index, holdout_from)
        holdout_end = _index_after_day(race_keys, first_index, holdout_through)

        if not 0 < selection_start < selection_end <= holdout_start < holdout_end:
            raise ValueError(f"fold {fold} has invalid chronological race boundaries")

        selected, candidates = nested_select_candidate(
            dataset,
            outer_train_end=selection_start,
            targets=targets,
            alphas=alphas,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_races=batch_races,
            validation_fraction=validation_fraction,
            min_validation_races=min_validation_races,
        )

        selection_model, selection_history = _fit_selected_model(
            dataset,
            race_end=selection_start,
            selected=selected,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_races=batch_races,
        )
        selection_metrics, selection_rows = evaluate_range(
            dataset,
            selection_model,
            race_start=selection_start,
            race_end=selection_end,
            batch_races=batch_races,
            keep_rows=True,
        )
        selection_prior_races = {
            race_id for race_id, *_rest in race_keys[:selection_start]
        }
        selection_packed = packed_candidates_from_rows(
            selection_rows,
            payouts=payout_lookup,
            train_races=selection_prior_races,
            payout_prior_weight=payout_prior_weight,
        )

        holdout_model, holdout_history = _fit_selected_model(
            dataset,
            race_end=holdout_start,
            selected=selected,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_races=batch_races,
        )
        holdout_metrics, holdout_rows = evaluate_range(
            dataset,
            holdout_model,
            race_start=holdout_start,
            race_end=holdout_end,
            batch_races=batch_races,
            keep_rows=True,
        )
        holdout_prior_races = {
            race_id for race_id, *_rest in race_keys[:holdout_start]
        }
        holdout_packed = packed_candidates_from_rows(
            holdout_rows,
            payouts=payout_lookup,
            train_races=holdout_prior_races,
            payout_prior_weight=payout_prior_weight,
        )

        audit = _fold_boundary_audit(
            race_keys,
            boundary=boundary,
            selection_start=selection_start,
            selection_end=selection_end,
            holdout_start=holdout_start,
            holdout_end=holdout_end,
            selection_rows=selection_rows,
            holdout_rows=holdout_rows,
            selection_prior_races=selection_prior_races,
            holdout_prior_races=holdout_prior_races,
        )
        if not audit["passed"]:
            failed = ", ".join(key for key, passed in audit.items() if passed is False)
            raise ValueError(f"fold {fold} leakage/boundary audit failed: {failed}")

        model_audits.append({
            "fold": fold,
            "selected_candidate": {
                "target": str(selected["target"]),
                "alpha": float(selected["alpha"]),
                "inner_train_races": int(selected["inner_train_races"]),
                "validation_races": int(selected["validation_races"]),
            },
            "candidate_results": candidates,
            "selection_training_history": selection_history,
            "holdout_training_history": holdout_history,
            "selection_prediction_metrics": selection_metrics,
            "holdout_prediction_metrics": holdout_metrics,
            "indices": {
                "selection_train_end": selection_start,
                "selection_prediction_start": selection_start,
                "selection_prediction_end": selection_end,
                "holdout_train_end": holdout_start,
                "holdout_prediction_start": holdout_start,
                "holdout_prediction_end": holdout_end,
            },
            "payout_prior": {
                "selection_teacher_races": len(selection_prior_races),
                "selection_teacher_date_through": str(race_keys[selection_start - 1][1]),
                "holdout_teacher_races": len(holdout_prior_races),
                "holdout_teacher_date_through": str(race_keys[holdout_start - 1][1]),
                "weight": payout_prior_weight,
            },
            "boundary_audit": audit,
        })
        fold_inputs.append({
            "fold": fold,
            "selection": selection_packed,
            "holdout": holdout_packed,
            "boundary_audit": {**dict(boundary["boundary_audit"]), **audit},
            "provenance": dict(provenance),
        })
    return fold_inputs, model_audits


def run(conn: Any, *, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    search_path = Path(args.search_result).resolve()
    cache_prefix = Path(args.cache_prefix).resolve()
    search_bytes = search_path.read_bytes()
    search = json.loads(search_bytes.decode("utf-8"))
    _reject_legacy_source(search, explicit_source_job_id=args.source_job_id)
    _require_schema_v6(search)

    race_date_through = search_race_date_through(search)
    if race_date_through is None:
        raise ValueError("search result lacks race_date_through")
    race_keys = [
        row for row in load_complete_race_ids(conn)
        if str(row[1]) <= race_date_through
    ]
    validate_search_race_universe(search, race_keys)

    selected = search["selected"]
    drops = tuple(str(value) for value in selected.get("drop_feature_groups") or ())
    n_features = int(search["n_features"])
    hasher = FeatureHasher(
        n_features=n_features, input_type="dict", alternate_sign=False
    )
    dataset = load_hashed_dataset(
        cache_prefix,
        race_keys=race_keys,
        n_features=n_features,
        drop_feature_groups=drops,
        hasher=hasher,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
    )
    if dataset is None:
        raise ValueError("strict schema-v6 hashed cache validation failed")

    cache_metadata = _cache_metadata(cache_prefix)
    unique_dates = sorted({str(row[1]) for row in race_keys})
    boundaries, evaluation_mode = _build_available_boundaries(
        unique_dates,
        folds=int(args.folds),
        selection_days=int(args.selection_days),
        outer_days=int(args.outer_days),
        embargo_days=int(args.embargo_days),
        allow_research_three_folds=bool(args.allow_research_three_folds),
    )
    targets = tuple(args.targets or search.get("teacher_targets") or ("winner", "top3_pl"))
    alphas = tuple(float(value) for value in (
        args.alphas or search.get("alphas") or (1e-5, 1e-4, 1e-3)
    ))
    base_policy = default_policy(
        daily_budget_yen=int(args.daily_budget_yen),
        ev_threshold=float(args.ev_threshold),
    )
    provenance = {
        "protocol_version": PROTOCOL_VERSION,
        "source_job_id": args.source_job_id,
        "source_schema_version": search["feature_schema_version"],
        "source_race_universe_sha256": race_ids_sha256(race_keys),
        "search_result_sha256": _sha256(search_bytes),
        **cache_metadata,
    }
    payouts = _load_trifecta_payouts(conn)
    fold_inputs, model_audits = build_fold_inputs(
        dataset,
        boundaries=boundaries,
        payouts=payouts,
        targets=targets,
        alphas=alphas,
        base_policy=base_policy,
        learning_rate=float(args.learning_rate),
        epochs=int(args.epochs),
        batch_races=int(args.batch_races),
        validation_fraction=float(args.validation_fraction),
        min_validation_races=int(args.min_validation_races),
        provenance=provenance,
    )
    evaluation = evaluate_annual_walk_forward(
        fold_inputs,
        base_policy,
        candidate_count=int(args.candidate_count),
        finalists=int(args.finalists),
        selection_bootstrap_samples=int(args.selection_bootstrap_samples),
        aggregate_bootstrap_samples=int(args.aggregate_bootstrap_samples),
        outer_days=int(args.outer_days),
        selection_days=int(args.selection_days),
        embargo_days=int(args.embargo_days),
        seed=int(args.seed),
    )
    five_fold = len(boundaries) == 5
    promotion_eligible = bool(five_fold and evaluation["promotion_eligible"])
    result = {
        "model": "bankroll_policy_nested_annual_v1",
        "protocol_version": PROTOCOL_VERSION,
        "evaluation_mode": evaluation_mode,
        "research_only": not five_fold,
        "promotion_eligible": promotion_eligible,
        "promotion_ineligible_reason": (
            None if five_fold else "five annual outer folds are required for promotion"
        ),
        "fold_count": len(boundaries),
        "targets": list(targets),
        "alphas": list(alphas),
        "boundaries": [dict(value) for value in boundaries],
        "fold_model_audits": model_audits,
        "provenance": {
            **provenance,
            "policy_candidates_sha256": evaluation["policy_candidates_sha256"],
        },
        "evaluation": {**evaluation, "promotion_eligible": promotion_eligible},
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    _write_json_atomic(Path(args.output), result)
    return result


def _fit_selected_model(
    dataset: HashedRaceDataset,
    *,
    race_end: int,
    selected: Mapping[str, Any],
    learning_rate: float,
    epochs: int,
    batch_races: int,
) -> tuple[Any, list[dict[str, Any]]]:
    scaler = fit_scaler(dataset, race_end=race_end, batch_rows=batch_races * 6)
    return train_listwise_model(
        dataset,
        train_race_end=race_end,
        target=str(selected["target"]),
        alpha=float(selected["alpha"]),
        learning_rate=learning_rate,
        epochs=epochs,
        batch_races=batch_races,
        scaler=scaler,
    )


def _first_index_by_date(
    race_keys: Sequence[tuple[str, str, str, int]],
) -> dict[str, int]:
    result: dict[str, int] = {}
    previous: str | None = None
    for index, row in enumerate(race_keys):
        race_date = str(row[1])
        if previous is not None and race_date < previous:
            raise ValueError("race keys must be ordered chronologically")
        result.setdefault(race_date, index)
        previous = race_date
    return result


def _required_day_index(indices: Mapping[str, int], value: str) -> int:
    try:
        return int(indices[value])
    except KeyError as exc:
        raise ValueError(f"race date is absent from dataset: {value}") from exc


def _index_after_day(
    race_keys: Sequence[tuple[str, str, str, int]],
    indices: Mapping[str, int],
    value: str,
) -> int:
    start = _required_day_index(indices, value)
    index = start
    while index < len(race_keys) and str(race_keys[index][1]) == value:
        index += 1
    return index


def _fold_boundary_audit(
    race_keys: Sequence[tuple[str, str, str, int]],
    *,
    boundary: Mapping[str, Any],
    selection_start: int,
    selection_end: int,
    holdout_start: int,
    holdout_end: int,
    selection_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    holdout_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    selection_prior_races: set[str],
    holdout_prior_races: set[str],
) -> dict[str, bool]:
    selection_dates = {str(row[1]) for row in race_keys[selection_start:selection_end]}
    holdout_dates = {str(row[1]) for row in race_keys[holdout_start:holdout_end]}
    selection_prediction_dates = {
        str(entry["race_date"]) for rows in selection_rows.values() for entry in rows
    }
    holdout_prediction_dates = {
        str(entry["race_date"]) for rows in holdout_rows.values() for entry in rows
    }
    selection_train_ids = {str(row[0]) for row in race_keys[:selection_start]}
    holdout_train_ids = {str(row[0]) for row in race_keys[:holdout_start]}
    audit = {
        "declared_boundary_passed": bool(boundary["boundary_audit"]["passed"]),
        "selection_training_strictly_prior_day": (
            str(race_keys[selection_start - 1][1])
            < str(race_keys[selection_start][1])
        ),
        "holdout_training_strictly_prior_day": (
            str(race_keys[holdout_start - 1][1])
            < str(race_keys[holdout_start][1])
        ),
        "selection_predictions_only_selection_dates": (
            selection_prediction_dates == selection_dates
        ),
        "holdout_predictions_only_holdout_dates": (
            holdout_prediction_dates == holdout_dates
        ),
        "prediction_periods_disjoint": not selection_dates & holdout_dates,
        "selection_payout_prior_exactly_training_races": (
            selection_prior_races == selection_train_ids
        ),
        "holdout_payout_prior_exactly_training_races": (
            holdout_prior_races == holdout_train_ids
        ),
        "selection_payout_prior_excludes_predictions": not (
            selection_prior_races & set(selection_rows)
        ),
        "holdout_payout_prior_excludes_predictions": not (
            holdout_prior_races & set(holdout_rows)
        ),
    }
    audit["passed"] = all(audit.values())
    return audit


def _build_available_boundaries(
    dates: Sequence[str],
    *,
    folds: int,
    selection_days: int,
    outer_days: int,
    embargo_days: int,
    allow_research_three_folds: bool,
) -> tuple[tuple[dict[str, Any], ...], str]:
    if folds not in (3, 5):
        raise ValueError("folds must be 5, or 3 for explicitly research-only evaluation")
    try:
        boundaries = build_annual_walk_forward_folds(
            dates,
            folds=folds,
            selection_days=selection_days,
            outer_days=outer_days,
            embargo_days=embargo_days,
        )
        return boundaries, "promotion_five_folds" if folds == 5 else "research_only_three_folds"
    except ValueError:
        if folds != 5 or not allow_research_three_folds:
            raise
    boundaries = build_annual_walk_forward_folds(
        dates,
        folds=3,
        selection_days=selection_days,
        outer_days=outer_days,
        embargo_days=embargo_days,
    )
    return boundaries, "research_only_three_folds_insufficient_for_five"


def _reject_legacy_source(
    search: Mapping[str, Any], *, explicit_source_job_id: int | None
) -> None:
    values: list[Any] = [explicit_source_job_id]
    stack: list[Any] = [search]
    while stack:
        value = stack.pop()
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in {"job_id", "source_job_id", "policy_source_job_id", "checkpoint_job_id"}:
                    values.append(item)
                elif isinstance(item, (Mapping, list, tuple)):
                    stack.append(item)
        elif isinstance(value, (list, tuple)):
            stack.extend(value)
    for value in values:
        try:
            normalized = int(value) if value is not None else None
        except (TypeError, ValueError):
            continue
        if normalized in FORBIDDEN_SOURCE_JOB_IDS:
            raise ValueError(f"legacy job {normalized} cannot source nested evaluation")


def _require_schema_v6(search: Mapping[str, Any]) -> None:
    schema = search.get("feature_schema_version")
    if schema != FEATURE_SCHEMA_VERSION or "-v6-" not in str(schema):
        raise ValueError(
            f"source feature schema <6 or unsupported: {schema!r}; "
            f"required={FEATURE_SCHEMA_VERSION!r}"
        )
    if int(search.get("hashed_cache_version", 0)) != CACHE_VERSION:
        raise ValueError("source hashed cache schema is legacy or unsupported")


def _cache_metadata(prefix: Path) -> dict[str, str]:
    paths = cache_paths(prefix)
    manifest_bytes = paths["manifest"].read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    bundle = {
        "manifest_sha256": _sha256(manifest_bytes),
        "matrix_file_sha256": manifest.get("matrix_file_sha256"),
        "ranks_file_sha256": manifest.get("ranks_file_sha256"),
        "race_ids_sha256": manifest.get("race_ids_sha256"),
    }
    return {
        "cache_manifest_sha256": bundle["manifest_sha256"],
        "cache_bundle_sha256": _sha256(
            json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
    }


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build leakage-audited real-data nested annual bankroll folds."
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--search-result", required=True)
    parser.add_argument("--cache-prefix", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-job-id", type=int)
    parser.add_argument("--folds", type=int, choices=(3, 5), default=5)
    parser.add_argument("--allow-research-three-folds", action="store_true")
    parser.add_argument("--selection-days", type=int, default=365)
    parser.add_argument("--outer-days", type=int, default=365)
    parser.add_argument("--embargo-days", type=int, default=0)
    parser.add_argument("--targets", type=lambda value: tuple(value.split(",")))
    parser.add_argument(
        "--alphas", type=lambda value: tuple(float(item) for item in value.split(","))
    )
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-races", type=int, default=1_000)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--min-validation-races", type=int, default=1_000)
    parser.add_argument("--daily-budget-yen", type=int, default=10_000)
    parser.add_argument("--ev-threshold", type=float, default=1.20)
    parser.add_argument("--candidate-count", type=int, default=64)
    parser.add_argument("--finalists", type=int, default=8)
    parser.add_argument("--selection-bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--aggregate-bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    init_db(args.db)
    with connection(args.db) as conn:
        result = run(conn, args=args)
    print(json.dumps({
        "model": result["model"],
        "evaluation_mode": result["evaluation_mode"],
        "fold_count": result["fold_count"],
        "promotion_eligible": result["promotion_eligible"],
        "provenance": result["provenance"],
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
