from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

import joblib

from .nonlinear_market_residual_v38 import (
    MODEL_NAME,
    fit_temporal_nonlinear_market_residual,
)


DECISION_MODEL_NAME = "decision_time_nonlinear_market_residual_v38"
DEFAULT_MINIMUM_TRAINING_DAYS = 30
DEFAULT_MINIMUM_TRAINING_RACES = 3_000
MINIMUM_SELECTION_HOLDOUT_DAYS = 7
MAXIMUM_TOP5_HIT_RATE_DEGRADATION = 0.005
REQUIRED_KEYS = (
    "race_id",
    "race_date",
    "jcd",
    "rno",
    "actual_combination",
    "actual_payout_yen",
    "odds",
    "model_probabilities",
    "market_probabilities",
    "lane_context",
)
OPTIONAL_AUDIT_KEYS = (
    "snapshot_id",
    "captured_at",
    "odds_deadline_at",
    "input_snapshot_age_seconds",
    "odds_path",
    "odds_path_points",
    "odds_checkpoints",
    "historical_return_multipliers",
)
FORBIDDEN_SOURCE_PREFIXES = (
    "official_closing_",
    "closing_",
)


def _iso_date(value: object, name: str) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError as exc:
        raise ValueError(f"{name} must start with an ISO date") from exc


def decision_time_race(race: Mapping[str, Any]) -> dict[str, Any]:
    """Copy only fields available at the declared purchase decision."""
    missing = [key for key in REQUIRED_KEYS if key not in race]
    if missing:
        raise ValueError(
            "decision-time V38 race is missing fields: " + ", ".join(missing)
        )
    result = {key: race[key] for key in REQUIRED_KEYS}
    for key in OPTIONAL_AUDIT_KEYS:
        if key in race:
            result[key] = race[key]
    result["market_probability_source"] = "decision_snapshot_odds"
    return result


def fit_decision_time_market_residual(
    races: list[dict[str, Any]],
    *,
    calibration_through: str,
    minimum_training_days: int = DEFAULT_MINIMUM_TRAINING_DAYS,
    minimum_training_races: int = DEFAULT_MINIMUM_TRAINING_RACES,
    num_threads: int = 4,
) -> dict[str, Any]:
    cutoff = _iso_date(calibration_through, "calibration_through")
    if minimum_training_days < 5:
        raise ValueError("minimum_training_days must be at least five")
    if minimum_training_races < 1:
        raise ValueError("minimum_training_races must be positive")
    sanitized = [decision_time_race(race) for race in races]
    calibration = [
        race for race in sanitized if _iso_date(race["race_date"], "race_date") <= cutoff
    ]
    evaluation = [
        race for race in sanitized if _iso_date(race["race_date"], "race_date") > cutoff
    ]
    training_dates = sorted({str(race["race_date"]) for race in calibration})
    status = (
        "ready"
        if len(training_dates) >= minimum_training_days
        and len(calibration) >= minimum_training_races
        else "insufficient_training_history"
    )
    audit = {
        "model": DECISION_MODEL_NAME,
        "probability_model": MODEL_NAME,
        "status": status,
        "market_probability_source": "decision_snapshot_odds",
        "feature_time_boundary": "at_or_before_odds_deadline_at",
        "official_closing_fields_used": False,
        "forbidden_source_prefixes": list(FORBIDDEN_SOURCE_PREFIXES),
        "calibration_through": cutoff,
        "training_from": training_dates[0] if training_dates else None,
        "training_through": training_dates[-1] if training_dates else None,
        "training_days": len(training_dates),
        "training_races": len(calibration),
        "minimum_training_days": int(minimum_training_days),
        "minimum_training_races": int(minimum_training_races),
        "evaluation_from": (
            min(str(race["race_date"]) for race in evaluation)
            if evaluation
            else None
        ),
        "evaluation_through": (
            max(str(race["race_date"]) for race in evaluation)
            if evaluation
            else None
        ),
        "evaluation_races": len(evaluation),
    }
    if status != "ready":
        audit["ready_reasons"] = [
            *(
                ["training_days_below_minimum"]
                if len(training_dates) < minimum_training_days
                else []
            ),
            *(
                ["training_races_below_minimum"]
                if len(calibration) < minimum_training_races
                else []
            ),
        ]
        return audit
    fitted = fit_temporal_nonlinear_market_residual(
        calibration,
        evaluation,
        num_threads=num_threads,
    )
    return {
        **audit,
        "ready_reasons": [],
        "market_is_exact_nested_null": fitted["market_is_exact_nested_null"],
        "inner_fit_through": fitted["inner_fit_through"],
        "inner_validation_from": fitted["inner_validation_from"],
        "selected_tree_preset": fitted["selected_tree_preset"],
        "selected_shrinkage": fitted["selected_shrinkage"],
        "selection_candidates": fitted["candidates"],
        "artifact": fitted["artifact"],
        "holdout_metrics": fitted["metrics"],
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def train_from_scored_cache(
    cache_path: Path,
    *,
    calibration_through: str,
    minimum_training_days: int = DEFAULT_MINIMUM_TRAINING_DAYS,
    minimum_training_races: int = DEFAULT_MINIMUM_TRAINING_RACES,
    num_threads: int = 4,
) -> dict[str, Any]:
    payload = joblib.load(cache_path)
    if not isinstance(payload, Mapping):
        raise ValueError("decision-time V38 cache must contain a mapping")
    races = payload.get("races")
    contract = payload.get("contract")
    if not isinstance(races, list) or not isinstance(contract, Mapping):
        raise ValueError("decision-time V38 cache is missing races or contract")
    fitted = fit_decision_time_market_residual(
        races,
        calibration_through=calibration_through,
        minimum_training_days=minimum_training_days,
        minimum_training_races=minimum_training_races,
        num_threads=num_threads,
    )
    training_status = str(fitted.pop("status"))
    return {
        "status": "completed",
        "model": DECISION_MODEL_NAME,
        "evaluation_version": 1,
        "training_status": training_status,
        "decision": (
            "research_only_frozen_artifact"
            if training_status == "ready"
            else "insufficient_data"
        ),
        "promotion_eligible": False,
        "source_scored_cache": str(cache_path),
        "source_scored_cache_sha256": _file_sha256(cache_path),
        "source_cache_contract": dict(contract),
        **fitted,
    }


def decision_v38_challenger_eligible(payload: Mapping[str, Any]) -> bool:
    """Gate a probability challenger before any prospective purchase ledger."""
    metrics = payload.get("holdout_metrics")
    artifact = payload.get("artifact")
    if not isinstance(metrics, Mapping) or not isinstance(artifact, Mapping):
        return False
    try:
        evaluated_days = int(metrics.get("evaluated_days") or 0)
        days_better = int(metrics.get("days_better_than_market") or 0)
        loss_delta = float(metrics["log_loss_delta_vs_market"])
        top5 = float(metrics["trifecta_top5_hit_rate"])
        market_top5 = float(metrics["market_trifecta_top5_hit_rate"])
    except (KeyError, TypeError, ValueError):
        return False
    digest = str(artifact.get("booster_sha256") or "")
    return bool(
        payload.get("training_status") == "ready"
        and payload.get("official_closing_fields_used") is False
        and payload.get("market_is_exact_nested_null") is True
        and evaluated_days >= MINIMUM_SELECTION_HOLDOUT_DAYS
        and days_better * 2 > evaluated_days
        and loss_delta < 0.0
        and top5 >= market_top5 - MAXIMUM_TOP5_HIT_RATE_DEGRADATION
        and float(payload.get("selected_shrinkage") or 0.0) > 0.0
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest.lower())
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a frozen nonlinear market residual from decision-time snapshots only"
        )
    )
    parser.add_argument("--scored-cache", type=Path, required=True)
    parser.add_argument("--calibration-through", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--minimum-training-days",
        type=int,
        default=DEFAULT_MINIMUM_TRAINING_DAYS,
    )
    parser.add_argument(
        "--minimum-training-races",
        type=int,
        default=DEFAULT_MINIMUM_TRAINING_RACES,
    )
    parser.add_argument("--num-threads", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = train_from_scored_cache(
        args.scored_cache,
        calibration_through=args.calibration_through,
        minimum_training_days=args.minimum_training_days,
        minimum_training_races=args.minimum_training_races,
        num_threads=args.num_threads,
    )
    _write_json_atomic(args.output, result)
    print(json.dumps({
        "output": str(args.output),
        "training_status": result["training_status"],
        "training_days": result["training_days"],
        "training_races": result["training_races"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
