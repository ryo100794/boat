from __future__ import annotations

import argparse
import gc
import hashlib
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
from ..cache_entry_series_features import ensure_series_cache_table
from ..db import connection, init_db
from ..feature_tuning import (
    DEFAULT_ABLATION_FEATURE_GROUPS,
    _ensure_sparse_index32,
    iter_race_feature_rows,
    load_complete_race_ids,
    normalize_drop_feature_groups,
    to_hashable,
)
from ..feature_schema import (
    DECAYED_HISTORY_FEATURE_SCHEMA_VERSION,
    uses_official_series_features,
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

SELECTION_RULE_VERSION = "ranking-loss-top5-slack-top1-v3"
SELECTION_RANKING_LOSS_RELATIVE_TOLERANCE = 0.01
SELECTION_TOP5_ABSOLUTE_TOLERANCE = 0.001
DEFAULT_EV_THRESHOLDS = (1.00, 1.10, 1.20, 1.35, 1.50)
CHECKPOINT_VERSION = 2
SOURCE_DATA_SNAPSHOT_VERSION = 1



def _include_decayed_history(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "include_decayed_history", False))


def _effective_feature_schema_version(args: argparse.Namespace) -> str:
    return (
        DECAYED_HISTORY_FEATURE_SCHEMA_VERSION
        if _include_decayed_history(args)
        else FEATURE_SCHEMA_VERSION
    )


def parse_ev_thresholds(value: str | None, fallback: float) -> tuple[float, ...]:
    values = (
        [float(item) for item in value.split(",") if item.strip()]
        if value
        else [float(fallback)]
    )
    if not 1 <= len(values) <= 10 or not all(0.8 <= item <= 3.0 for item in values):
        raise ValueError("ev thresholds must contain 1-10 values between 0.8 and 3.0")
    return tuple(dict.fromkeys(values))


def select_ev_policy(
    *,
    rows_by_race: dict[str, Any],
    race_keys: list[tuple[str, str, str, int]],
    train_end: int,
    selection_end: int,
    payouts: dict[str, Any],
    daily_budget_yen: int,
    thresholds: tuple[float, ...],
) -> tuple[float, list[dict[str, Any]]]:
    train_races = {race_id for race_id, *_rest in race_keys[:train_end]}
    selection_dates = {
        race_date for _race_id, race_date, _jcd, _rno
        in race_keys[train_end:selection_end]
    }
    results: list[dict[str, Any]] = []
    for threshold in thresholds:
        totals = zero_totals()
        daily_rows: list[dict[str, Any]] = []
        bankroll, profit_state = evaluate_bankroll_fold(
            rows_by_race=rows_by_race,
            train_races=train_races,
            test_dates=selection_dates,
            payouts=payouts,
            policy=default_policy(
                daily_budget_yen=daily_budget_yen,
                ev_threshold=threshold,
            ),
            totals=totals,
            daily_rows=daily_rows,
            profit_state=(0, 0, 0),
        )
        results.append({
            "ev_threshold": threshold,
            "roi": bankroll["roi"],
            "profit_yen": bankroll["profit_yen"],
            "stake_yen": bankroll["stake_yen"],
            "max_drawdown_yen": profit_state[2],
        })
    eligible = [row for row in results if float(row["stake_yen"] or 0) > 0]
    selected = max(
        eligible or results,
        key=lambda row: (
            float(row["roi"] or 0),
            float(row["profit_yen"] or 0),
            -float(row["max_drawdown_yen"] or 0),
        ),
    )
    return float(selected["ev_threshold"]), results


def day_boundary(race_keys: list[tuple[str, str, str, int]], approximate: int) -> int:
    index = min(len(race_keys) - 1, max(1, int(approximate)))
    current_date = race_keys[index][1]
    while index < len(race_keys) and race_keys[index][1] == current_date:
        index += 1
    if index >= len(race_keys):
        raise ValueError("requested boundary leaves no future races")
    return index


def feature_variants() -> list[tuple[str, tuple[str, ...]]]:
    active_groups = DEFAULT_ABLATION_FEATURE_GROUPS
    if not uses_official_series_features(FEATURE_SCHEMA_VERSION):
        active_groups = tuple(
            group
            for group in active_groups
            if group not in {"series_cached", "series_relative"}
        )
    return [("full", ())] + [
        (f"drop_{group}", (group,)) for group in active_groups
    ]


def _feature_variant_catalog() -> dict[str, tuple[str, ...]]:
    backward_compatible_groups = {
        f"drop_{group}": (group,)
        for group in DEFAULT_ABLATION_FEATURE_GROUPS
    }
    return dict(feature_variants()) | backward_compatible_groups | {
        "drop_base_pastlog_rolling_history": (
            "base_pastlog",
            "rolling_history",
        ),
        "drop_card_numeric": ("card_numeric",),
        "drop_card_relative": ("card_relative",),
        "drop_card_numeric_card_relative": ("card_numeric", "card_relative"),
    }


def parse_feature_variants(value: str | None) -> FeatureVariants | None:
    if not value:
        return None
    available = _feature_variant_catalog()
    names = tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    unknown = tuple(name for name in names if name not in available)
    if not names or unknown:
        choices = ", ".join(available)
        raise argparse.ArgumentTypeError(
            f"feature variants must be selected from: {choices}"
        )
    return tuple((name, available[name]) for name in names)


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
    include_decayed_history: bool = False,
    feature_schema_version: str = FEATURE_SCHEMA_VERSION,
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
        include_decayed_history=include_decayed_history,
        feature_schema_version=feature_schema_version,
    )
    return dataset, source


def variant_cache_prefix(
    cache_dir: Path,
    *,
    n_features: int,
    name: str,
    include_decayed_history: bool = False,
) -> Path:
    history_suffix = "_decayed_history" if include_decayed_history else ""
    return cache_dir / f"listwise_search_{int(n_features)}_{name}{history_suffix}"


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
    include_decayed_history: bool = False,
    feature_schema_version: str = FEATURE_SCHEMA_VERSION,
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
        include_decayed_history=include_decayed_history,
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
            feature_schema_version=feature_schema_version,
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
            include_decayed_history=include_decayed_history,
        ),
        hasher=hasher,
        to_hashable=to_hashable,
        ensure_sparse_index32=_ensure_sparse_index32,
        drop_feature_groups=normalized,
        batch_size=batch_races * 6,
        write_cache=write_cache,
        feature_schema_version=feature_schema_version,
    )
    return dataset, source, primary_prefix if write_cache else None


def cleanup_selected_cache_family(
    cache_dir: Path,
    *,
    n_features: int,
    variants: FeatureVariants | None = None,
    include_decayed_history: bool = False,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for variant_name, _dropped in _resolved_variants(variants):
        prefix = variant_cache_prefix(
            cache_dir,
            n_features=n_features,
            name=variant_name,
            include_decayed_history=include_decayed_history,
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
    include_decayed_history: bool = False,
) -> list[Path]:
    candidates: list[Path] = []
    for variant_name, _dropped in _resolved_variants(variants):
        prefix = variant_cache_prefix(
            cache_dir,
            n_features=n_features,
            name=variant_name,
            include_decayed_history=include_decayed_history,
        )
        if cache_paths(prefix)["manifest"].exists():
            candidates.append(prefix)
    return candidates


def _candidate_key(variant_name: str, target: str, alpha: float) -> str:
    return json.dumps([variant_name, target, float(alpha)], separators=(",", ":"))



def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _selected_cache_manifest_sha256(
    cache_dir: Path | None,
    *,
    n_features: int,
    variants: FeatureVariants | None,
    include_decayed_history: bool = False,
) -> str:
    manifests: list[dict[str, Any]] = []
    if cache_dir is None:
        return _canonical_sha256(manifests)
    for prefix in selected_cache_candidates(
        cache_dir,
        n_features=n_features,
        variants=variants,
        include_decayed_history=include_decayed_history,
    ):
        path = cache_paths(prefix)["manifest"]
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"selected cache manifest is invalid: {path}") from exc
        fields = {
            name: manifest.get(name)
            for name in (
                "cache_version",
                "feature_schema_version",
                "race_count",
                "race_ids_sha256",
                "n_features",
                "drop_feature_groups",
                "matrix_shape",
                "matrix_nnz",
                "matrix_dtype",
                "matrix_file_sha256",
                "ranks_shape",
                "ranks_dtype",
                "ranks_sha256",
                "ranks_file_sha256",
            )
        }
        for name in (
            "race_ids_sha256",
            "matrix_file_sha256",
            "ranks_sha256",
            "ranks_file_sha256",
        ):
            if not _is_sha256(fields[name]):
                raise ValueError(
                    f"selected cache manifest lacks valid {name}: {path}"
                )
        manifests.append({"prefix": prefix.name, **fields})
    return _canonical_sha256(manifests)


def _source_table_watermark(
    conn: Any,
    *,
    as_of_date: str,
    feature_schema_version: str = FEATURE_SCHEMA_VERSION,
) -> str:
    ensure_series_cache_table(conn)
    statements = (
        (
            "races",
            """
            SELECT COUNT(*) AS row_count, MIN(r.updated_at) AS min_updated_at,
                   MAX(r.updated_at) AS max_updated_at
            FROM races r
            WHERE r.race_date <= ?
            """,
        ),
        (
            "entries",
            """
            SELECT COUNT(*) AS row_count, MIN(e.updated_at) AS min_updated_at,
                   MAX(e.updated_at) AS max_updated_at
            FROM entries e
            JOIN races r ON r.race_id = e.race_id
            WHERE r.race_date <= ?
            """,
        ),
        (
            "race_results",
            """
            SELECT COUNT(*) AS row_count, MIN(rr.updated_at) AS min_updated_at,
                   MAX(rr.updated_at) AS max_updated_at
            FROM race_results rr
            JOIN races r ON r.race_id = rr.race_id
            WHERE r.race_date <= ?
            """,
        ),
        (
            "entry_series_features",
            """
            SELECT COUNT(*) AS row_count, MIN(sf.updated_at) AS min_updated_at,
                   MAX(sf.updated_at) AS max_updated_at
            FROM entry_series_features sf
            JOIN races r ON r.race_id = sf.race_id
            WHERE r.race_date <= ?
            """,
        ),
    )
    tables: list[dict[str, Any]] = []
    for name, statement in statements:
        row = conn.execute(statement, (as_of_date,)).fetchone()
        tables.append({
            "table": name,
            "row_count": int(row["row_count"] or 0),
            "min_updated_at": (
                None
                if row["min_updated_at"] is None
                else str(row["min_updated_at"])
            ),
            "max_updated_at": (
                None
                if row["max_updated_at"] is None
                else str(row["max_updated_at"])
            ),
        })

    period_digest = hashlib.sha256()
    period_count = 0
    period_rows = conn.execute(
        """
        SELECT year, half, racer_no, raw_json
        FROM racer_period_stats
        ORDER BY year, half, racer_no
        """
    ).fetchall()
    for row in period_rows:
        period_count += 1
        period_digest.update(
            json.dumps(
                [
                    int(row["year"]),
                    int(row["half"]),
                    int(row["racer_no"]),
                    str(row["raw_json"]),
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        period_digest.update(b"\n")
    return _canonical_sha256({
        "watermark_version": 1,
        "as_of_date": as_of_date,
        "feature_schema_version": feature_schema_version,
        "tables": tables,
        "racer_period_stats_count": period_count,
        "racer_period_stats_sha256": period_digest.hexdigest(),
    })


def source_data_snapshot(
    conn: Any,
    *,
    race_keys: list[tuple[str, str, str, int]],
    payouts: dict[str, dict[str, Any]],
    selected_cache_dir: Path | None = None,
    n_features: int = 4096,
    variants: FeatureVariants | None = None,
    as_of_date: str | None = None,
    include_decayed_history: bool = False,
    feature_schema_version: str = FEATURE_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Build a cheap source watermark plus immutable payout/cache identities."""
    expected_ids = [str(race_id) for race_id, *_rest in race_keys]
    if not expected_ids:
        raise ValueError("source snapshot requires at least one race")
    as_of_date = str(as_of_date or race_keys[-1][1])
    payout_digest = hashlib.sha256()
    payout_digest.update(b"trifecta-payouts-v1\n")
    for race_id in expected_ids:
        payout = payouts.get(race_id)
        settlements = tuple(payout.get("settlements") or ()) if payout else ()
        canonical = sorted(
            (
                str(row["combination"]),
                int(row["payout_yen"]),
                (
                    None
                    if row.get("popularity") is None
                    else int(row["popularity"])
                ),
            )
            for row in settlements
        )
        payout_digest.update(
            json.dumps(
                [race_id, canonical],
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        payout_digest.update(b"\n")

    identity = {
        "snapshot_version": SOURCE_DATA_SNAPSHOT_VERSION,
        "race_count": len(expected_ids),
        "race_universe_sha256": race_ids_sha256(race_keys),
        "source_watermark_sha256": _source_table_watermark(
            conn,
            as_of_date=as_of_date,
            feature_schema_version=feature_schema_version,
        ),
        "trifecta_payouts_sha256": payout_digest.hexdigest(),
        "selected_cache_manifest_sha256": _selected_cache_manifest_sha256(
            selected_cache_dir,
            n_features=n_features,
            variants=variants,
            include_decayed_history=include_decayed_history,
        ),
    }
    identity["snapshot_sha256"] = _canonical_sha256(identity)
    return identity


def _validated_source_data_snapshot(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("source data snapshot identity is required")
    required = {
        "snapshot_version",
        "race_count",
        "race_universe_sha256",
        "source_watermark_sha256",
        "trifecta_payouts_sha256",
        "selected_cache_manifest_sha256",
        "snapshot_sha256",
    }
    if set(value) != required:
        raise ValueError("source data snapshot identity fields are invalid")
    if (
        value.get("snapshot_version") != SOURCE_DATA_SNAPSHOT_VERSION
        or isinstance(value.get("race_count"), bool)
        or not isinstance(value.get("race_count"), int)
        or int(value["race_count"]) <= 0
        or any(
            not _is_sha256(value.get(name))
            for name in (
                "race_universe_sha256",
                "source_watermark_sha256",
                "trifecta_payouts_sha256",
                "selected_cache_manifest_sha256",
                "snapshot_sha256",
            )
        )
    ):
        raise ValueError("source data snapshot identity is invalid")
    expected = _canonical_sha256({
        name: field
        for name, field in value.items()
        if name != "snapshot_sha256"
    })
    if value["snapshot_sha256"] != expected:
        raise ValueError("source data snapshot aggregate hash is invalid")
    return dict(value)


def _refresh_selected_cache_identity(
    signature: dict[str, Any],
    *,
    selected_cache_dir: Path | None,
    n_features: int,
    variants: FeatureVariants | None,
    include_decayed_history: bool = False,
) -> bool:
    snapshot = dict(signature["source_data_snapshot"])
    current = _selected_cache_manifest_sha256(
        selected_cache_dir,
        n_features=n_features,
        variants=variants,
        include_decayed_history=include_decayed_history,
    )
    if snapshot["selected_cache_manifest_sha256"] == current:
        return False
    snapshot["selected_cache_manifest_sha256"] = current
    snapshot["snapshot_sha256"] = _canonical_sha256({
        name: value
        for name, value in snapshot.items()
        if name != "snapshot_sha256"
    })
    signature["source_data_snapshot"] = snapshot
    return True


def _checkpoint_signature(
    *,
    args: argparse.Namespace,
    race_keys: list[tuple[str, str, str, int]],
    train_end: int,
    selection_end: int,
    targets: tuple[str, ...],
    alphas: tuple[float, ...],
    data_snapshot: dict[str, Any],
    variants: FeatureVariants | None = None,
) -> dict[str, Any]:
    verified_snapshot = _validated_source_data_snapshot(data_snapshot)
    if verified_snapshot["race_count"] != len(race_keys):
        raise ValueError("source data snapshot race count mismatch")
    if verified_snapshot["race_universe_sha256"] != race_ids_sha256(race_keys):
        raise ValueError("source data snapshot race universe mismatch")
    signature = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "cache_version": CACHE_VERSION,
        "feature_schema_version": _effective_feature_schema_version(args),
        "as_of_date": getattr(args, "as_of_date", None),
        "race_count": len(race_keys),
        "race_universe_sha256": race_ids_sha256(race_keys),
        "source_data_snapshot": verified_snapshot,
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
    if _include_decayed_history(args):
        signature["include_decayed_history"] = True
    loss_blend = getattr(args, "loss_blend", None)
    if loss_blend is not None:
        signature["loss_blend"] = float(loss_blend)
    return signature


def _validated_candidate_rows(
    rows: Any,
    signature: dict[str, Any],
) -> dict[str, dict[str, Any]]:
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
                or (
                    signature.get("loss_blend") is not None
                    and row.get("loss_blend") != signature["loss_blend"]
                )
            ):
                return {}
            completed[key] = row
    except (KeyError, TypeError, ValueError):
        return {}
    return completed


def _load_checkpoint(path: Path, signature: dict[str, Any]) -> dict[str, dict[str, Any]]:
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    stored_signature = checkpoint.get("signature")
    if (
        not isinstance(stored_signature, dict)
        or stored_signature.get("checkpoint_version") != CHECKPOINT_VERSION
        or "source_data_snapshot" not in stored_signature
        or stored_signature != signature
    ):
        return {}
    return _validated_candidate_rows(checkpoint.get("search_results"), signature)


def _load_reusable_search_results(
    path: Path,
    signature: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"reusable search output is unreadable: {path}") from exc
    expected = {
        "model": "pastlog_listwise_feature_teacher_search_v1",
        "races": int(signature["race_count"]),
        "race_universe_sha256": signature["race_universe_sha256"],
        "as_of_date": signature.get("as_of_date"),
        "train_races": int(signature["train_end"]),
        "selection_races": int(signature["selection_end"] - signature["train_end"]),
        "holdout_races": int(signature["race_count"] - signature["selection_end"]),
        "n_features": int(signature["n_features"]),
        "feature_schema_version": signature.get(
            "feature_schema_version", FEATURE_SCHEMA_VERSION
        ),
        "feature_variants": [name for name, _drops in signature["feature_variants"]],
        "teacher_targets": list(signature["targets"]),
        "loss_blend": signature.get("loss_blend"),
        "alphas": [float(value) for value in signature["alphas"]],
    }
    if signature.get("include_decayed_history"):
        expected["include_decayed_history"] = True
    actual = {name: payload.get(name) for name in expected}
    if actual != expected:
        mismatches = [name for name in expected if actual[name] != expected[name]]
        raise ValueError(
            "reusable search output metadata mismatch: " + ", ".join(mismatches)
        )
    completed = _validated_candidate_rows(payload.get("search_results"), signature)
    expected_count = (
        len(signature["feature_variants"])
        * len(signature["targets"])
        * len(signature["alphas"])
    )
    if len(completed) != expected_count:
        raise ValueError(
            f"reusable search output candidates are incomplete: "
            f"{len(completed)} of {expected_count}"
        )
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
    best_ranking_loss = min(float(row["ranking_log_loss"]) for row in rows)
    ranking_ceiling = best_ranking_loss * (
        1.0 + SELECTION_RANKING_LOSS_RELATIVE_TOLERANCE
    )
    ranking_eligible = [
        row
        for row in rows
        if float(row["ranking_log_loss"]) <= ranking_ceiling + 1e-12
    ]
    best_top5 = max(
        float(row["trifecta_top5_hit_rate"])
        for row in ranking_eligible
    )
    top5_floor = best_top5 - SELECTION_TOP5_ABSOLUTE_TOLERANCE
    eligible = [
        row
        for row in ranking_eligible
        if float(row["trifecta_top5_hit_rate"]) >= top5_floor - 1e-12
    ]
    return min(eligible, key=lambda row: (
        -float(row["winner_top1_accuracy"]),
        float(row["entry_log_loss"]),
        float(row["ranking_log_loss"]),
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
    loss_blend = request.get("loss_blend")
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
        include_decayed_history=bool(request.get("include_decayed_history", False)),
        feature_schema_version=str(
            request.get("feature_schema_version", FEATURE_SCHEMA_VERSION)
        ),
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
            loss_blend=loss_blend,
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
            "loss_blend": loss_blend,
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
    last_completed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_variants = _resolved_variants(variants)
    completed_variants = sum(
        all(
            _candidate_key(variant_name, target, alpha) in completed
            for target in targets
            for alpha in alphas
        )
        for variant_name, _dropped in run_variants
    )
    total_candidates = len(run_variants) * len(targets) * len(alphas)
    return {
        "signature": signature,
        "progress": {
            "completed_candidates": len(completed),
            "total_candidates": total_candidates,
            "completed_variants": completed_variants,
            "total_variants": len(run_variants),
            "last_completed": last_completed,
        },
        "search_results": _ordered_rows(
            completed,
            targets=targets,
            alphas=alphas,
            variants=variants,
        ),
    }


def _persist_checkpoint_progress(
    path: Path,
    signature: dict[str, Any],
    completed: dict[str, dict[str, Any]],
    *,
    targets: tuple[str, ...],
    alphas: tuple[float, ...],
    variants: FeatureVariants | None,
    last_completed: dict[str, Any],
) -> None:
    payload = _checkpoint_payload(
        signature,
        completed,
        targets=targets,
        alphas=alphas,
        variants=variants,
        last_completed=last_completed,
    )
    _write_json_atomic(path, payload)
    print(json.dumps({
        "feature_search_progress": {
            **payload["progress"],
            "checkpoint": str(path),
        }
    }, ensure_ascii=False), flush=True)


def search(
    conn,
    *,
    args: argparse.Namespace,
    variants: FeatureVariants | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    run_variants = _resolved_variants(variants)
    include_decayed_history = _include_decayed_history(args)
    feature_schema_version = _effective_feature_schema_version(args)
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
    loss_blend = getattr(args, "loss_blend", None)
    if loss_blend is not None and not 0.0 <= float(loss_blend) <= 1.0:
        raise ValueError("loss_blend must be between 0 and 1")
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
    payouts = _load_trifecta_payouts(conn)
    data_snapshot = source_data_snapshot(
        conn,
        race_keys=race_keys,
        payouts=payouts,
        selected_cache_dir=selected_cache_dir,
        n_features=int(args.n_features),
        variants=run_variants,
        as_of_date=args.as_of_date,
        include_decayed_history=include_decayed_history,
        feature_schema_version=feature_schema_version,
    )
    checkpoint_signature = _checkpoint_signature(
        args=args,
        race_keys=race_keys,
        train_end=train_end,
        selection_end=selection_end,
        targets=targets,
        alphas=alphas,
        data_snapshot=data_snapshot,
        variants=run_variants,
    )
    completed = _load_checkpoint(checkpoint_path, checkpoint_signature)
    reuse_search_output = getattr(args, "reuse_search_output", None)
    if reuse_search_output:
        reuse_path = Path(reuse_search_output)
        if reuse_path.resolve() == output.resolve():
            raise ValueError("reusable search output must differ from output")
        completed = _load_reusable_search_results(reuse_path, checkpoint_signature)
        print(json.dumps({"reused_search_candidates": len(completed)}), flush=True)
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
            include_decayed_history=include_decayed_history,
        )
        expected_prefix = (
            variant_cache_prefix(
                selected_cache_dir,
                n_features=args.n_features,
                name=str(resumed_selected["feature_variant"]),
                include_decayed_history=include_decayed_history,
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
                include_decayed_history=include_decayed_history,
            )
            if completed and _refresh_selected_cache_identity(
                checkpoint_signature,
                selected_cache_dir=selected_cache_dir,
                n_features=int(args.n_features),
                variants=run_variants,
                include_decayed_history=include_decayed_history,
            ):
                _persist_checkpoint_progress(
                    checkpoint_path,
                    checkpoint_signature,
                    completed,
                    targets=targets,
                    alphas=alphas,
                    variants=run_variants,
                    last_completed={"kind": "selected_cache_reset"},
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
            "include_decayed_history": include_decayed_history,
            "feature_schema_version": feature_schema_version,
            "train_end": train_end,
            "selection_end": selection_end,
            "targets": targets,
            "alphas": alphas,
            "learning_rate": float(args.learning_rate),
            "loss_blend": loss_blend,
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
            _persist_checkpoint_progress(
                checkpoint_path,
                checkpoint_signature,
                completed,
                targets=targets,
                alphas=alphas,
                variants=run_variants,
                last_completed={
                    "kind": "candidate",
                    "feature_variant": variant_name,
                    "target": str(row["target"]),
                    "alpha": float(row["alpha"]),
                },
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
            _persist_checkpoint_progress(
                checkpoint_path,
                checkpoint_signature,
                completed,
                targets=targets,
                alphas=alphas,
                variants=run_variants,
                last_completed={
                    "kind": "variant",
                    "feature_variant": variant_name,
                },
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
                    include_decayed_history=include_decayed_history,
                )
                save_prefix = variant_cache_prefix(
                    selected_cache_dir,
                    n_features=args.n_features,
                    name=variant_name,
                    include_decayed_history=include_decayed_history,
                )
                save_hashed_dataset(save_prefix, dataset)
                active_cache_variant = variant_name
                if _refresh_selected_cache_identity(
                    checkpoint_signature,
                    selected_cache_dir=selected_cache_dir,
                    n_features=int(args.n_features),
                    variants=run_variants,
                    include_decayed_history=include_decayed_history,
                ):
                    _persist_checkpoint_progress(
                        checkpoint_path,
                        checkpoint_signature,
                        completed,
                        targets=targets,
                        alphas=alphas,
                        variants=run_variants,
                        last_completed={
                            "kind": "selected_cache_manifest",
                            "feature_variant": variant_name,
                        },
                    )
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
            include_decayed_history=include_decayed_history,
        )
        candidates = selected_cache_candidates(
            selected_cache_dir,
            n_features=args.n_features,
            variants=run_variants,
            include_decayed_history=include_decayed_history,
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
            feature_schema_version=feature_schema_version,
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
            include_decayed_history=include_decayed_history,
            feature_schema_version=feature_schema_version,
        )
    policy_scaler = fit_scaler(
        dataset, race_end=train_end, batch_rows=args.batch_races * 6
    )
    policy_model, _policy_history = train_listwise_model(
        dataset,
        train_race_end=train_end,
        target=str(selected["target"]),
        loss_blend=loss_blend,
        alpha=float(selected["alpha"]),
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_races=args.batch_races,
        scaler=policy_scaler,
    )
    _selection_metrics, selection_rows = evaluate_range(
        dataset,
        policy_model,
        race_start=train_end,
        race_end=selection_end,
        batch_races=args.batch_races,
        keep_rows=True,
    )
    selected_ev_threshold, ev_policy_results = select_ev_policy(
        rows_by_race=selection_rows,
        race_keys=race_keys,
        train_end=train_end,
        selection_end=selection_end,
        payouts=payouts,
        daily_budget_yen=args.daily_budget_yen,
        thresholds=parse_ev_thresholds(args.ev_thresholds, args.ev_threshold),
    )
    del policy_model, policy_scaler, selection_rows
    gc.collect()
    scaler = fit_scaler(dataset, race_end=selection_end, batch_rows=args.batch_races * 6)
    final_model, final_history = train_listwise_model(
        dataset,
        train_race_end=selection_end,
        target=str(selected["target"]),
        loss_blend=loss_blend,
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
        ev_threshold=selected_ev_threshold,
    )
    policy["feature_variant"] = selected["feature_variant"]
    policy["drop_feature_groups"] = list(selected_drops)
    policy["target"] = selected["target"]
    policy["loss_blend"] = loss_blend
    totals = zero_totals()
    daily_rows: list[dict[str, Any]] = []
    bankroll, profit_state = evaluate_bankroll_fold(
        rows_by_race=holdout_rows,
        train_races={race_id for race_id, *_rest in race_keys[:selection_end]},
        test_dates={race_date for _race_id, race_date, _jcd, _rno in race_keys[selection_end:]},
        payouts=payouts,
        policy=policy,
        totals=totals,
        daily_rows=daily_rows,
        profit_state=(0, 0, 0),
    )
    holdout_pass = bankroll["roi"] > 1.0 and holdout_metrics["winner_top1_accuracy"] >= args.min_top1
    evaluation_hash = race_set_sha256(holdout_rows)
    bankroll["evaluation_race_set_sha256"] = evaluation_hash
    selected_payload = {
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
    }
    if loss_blend is not None:
        selected_payload["loss_blend"] = float(loss_blend)
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
        "feature_schema_version": feature_schema_version,
        "include_decayed_history": include_decayed_history,
        "selection_rule_version": SELECTION_RULE_VERSION,
        "selection_ranking_loss_relative_tolerance": (
            SELECTION_RANKING_LOSS_RELATIVE_TOLERANCE
        ),
        "selection_top5_absolute_tolerance": (
            SELECTION_TOP5_ABSOLUTE_TOLERANCE
        ),
        "feature_variants": [name for name, _drops in run_variants],
        "teacher_targets": list(targets),
        "loss_blend": loss_blend,
        "alphas": list(alphas),
        "selection_metric": (
            "ranking log loss within 1% of best and 3T5 within 0.1 percentage "
            "points of best, then maximum winner top1; entry and ranking log "
            "loss as tie breaks"
        ),
        "search_results": search_rows,
        "selected": selected_payload,
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
        "ev_policy_selection": {
            "scope": "selection_window_only_before_untouched_holdout",
            "selected_ev_threshold": selected_ev_threshold,
            "candidates": ev_policy_results,
        },
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
    parser.add_argument(
        "--feature-variants",
        type=parse_feature_variants,
        help="Comma-separated feature variants to evaluate; defaults to all.",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--reuse-search-output",
        help=(
            "Reuse a completed, identity-matched candidate search and rerun only "
            "selection, final training, and untouched holdout evaluation."
        ),
    )
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
    parser.add_argument(
        "--include-decayed-history",
        action="store_true",
        help=(
            "Add leakage-safe 30/90/365-day entity history features using a "
            "separate experimental cache schema."
        ),
    )
    parser.add_argument("--n-features", type=int, default=1 << 13)
    parser.add_argument("--batch-races", type=int, default=1_000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--targets", default="winner,top3_pl")
    parser.add_argument("--loss-blend", type=float)
    parser.add_argument("--alphas", default="0.00001,0.0001")
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--selection-fraction", type=float, default=0.90)
    parser.add_argument("--daily-budget-yen", type=int, default=10_000)
    parser.add_argument("--ev-threshold", type=float, default=1.20)
    parser.add_argument("--ev-thresholds")
    parser.add_argument("--min-top1", type=float, default=0.5642)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    init_db(args.db)
    with connection(args.db) as conn:
        result = search(conn, args=args, variants=args.feature_variants)
    compact = {key: value for key, value in result.items() if key not in {"search_results", "daily"}}
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
