from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import joblib

from .decision_market_residual_v38 import (
    DEFAULT_MINIMUM_TRAINING_DAYS,
    DEFAULT_MINIMUM_TRAINING_RACES,
    FORBIDDEN_SOURCE_PREFIXES,
    MAXIMUM_INPUT_SNAPSHOT_AGE_SECONDS,
    MAXIMUM_TOP5_HIT_RATE_DEGRADATION,
    MINIMUM_SELECTION_HOLDOUT_DAYS,
    REQUIRED_MINIMUM_DECISION_LEAD_SECONDS,
    _iso_date,
    decision_time_race,
    validate_decision_scored_cache_contract,
)
from .stacked_market_residual_v42 import (
    fit_temporal_stacked_market_residual,
)


MODEL_NAME = "decision_time_stacked_market_residual_v44"


def fit_decision_time_stacked_market(
    races: list[dict[str, Any]],
    *,
    calibration_through: str,
    minimum_training_days: int = DEFAULT_MINIMUM_TRAINING_DAYS,
    minimum_training_races: int = DEFAULT_MINIMUM_TRAINING_RACES,
    num_threads: int = 4,
) -> dict[str, Any]:
    cutoff = _iso_date(calibration_through, "calibration_through")
    if minimum_training_days < 10:
        raise ValueError("V44 minimum_training_days must be at least ten")
    if minimum_training_races < 1:
        raise ValueError("minimum_training_races must be positive")
    sanitized = [decision_time_race(race) for race in races]
    calibration = [
        race
        for race in sanitized
        if _iso_date(race["race_date"], "race_date") <= cutoff
    ]
    evaluation = [
        race
        for race in sanitized
        if _iso_date(race["race_date"], "race_date") > cutoff
    ]
    training_dates = sorted({str(race["race_date"]) for race in calibration})
    status = (
        "ready"
        if len(training_dates) >= minimum_training_days
        and len(calibration) >= minimum_training_races
        else "insufficient_training_history"
    )
    audit = {
        "model": MODEL_NAME,
        "status": status,
        "market_probability_source": "decision_snapshot_odds",
        "feature_time_boundary": "at_or_before_odds_deadline_at",
        "official_closing_fields_used": False,
        "forbidden_source_prefixes": list(FORBIDDEN_SOURCE_PREFIXES),
        "calibration_through": cutoff,
        "training_from": training_dates[0] if training_dates else None,
        "decision_time_boundary_all_passed": True,
        "decision_time_boundary_violations": 0,
        "minimum_decision_lead_seconds": (
            min(float(race["decision_lead_seconds"]) for race in sanitized)
            if sanitized else None
        ),
        "required_minimum_decision_lead_seconds": (
            REQUIRED_MINIMUM_DECISION_LEAD_SECONDS
        ),
        "maximum_input_snapshot_age_seconds": (
            max(
                float(race["input_snapshot_age_seconds"])
                for race in sanitized
            )
            if sanitized else None
        ),
        "allowed_input_snapshot_age_seconds": MAXIMUM_INPUT_SNAPSHOT_AGE_SECONDS,
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
    fitted = fit_temporal_stacked_market_residual(
        calibration,
        evaluation,
        num_threads=num_threads,
    )
    return {
        **audit,
        "ready_reasons": [],
        "market_is_exact_nested_null": fitted["market_is_exact_nested_null"],
        "base_training_through": fitted["base_training_through"],
        "stack_validation_from": fitted["stack_validation_from"],
        "stack_candidates": fitted["stack_candidates"],
        "selected_stack": fitted["selected_stack"],
        "selected_weights": fitted["selected_weights"],
        "component_selection": fitted["component_selection"],
        "artifact": fitted["artifact"],
        "holdout_metrics": fitted["metrics"],
    }


def decision_v44_challenger_eligible(payload: Mapping[str, Any]) -> bool:
    metrics = payload.get("holdout_metrics")
    artifact = payload.get("artifact")
    weights = payload.get("selected_weights")
    if not all(isinstance(value, Mapping) for value in (metrics, artifact, weights)):
        return False
    try:
        evaluated_days = int(metrics.get("evaluated_days") or 0)
        days_better = int(metrics.get("days_better_than_market") or 0)
        loss_delta = float(metrics["log_loss_delta_vs_market"])
        top5 = float(metrics["trifecta_top5_hit_rate"])
        market_top5 = float(metrics["market_trifecta_top5_hit_rate"])
        residual_weight = float(weights.get("linear") or 0.0) + float(
            weights.get("nonlinear") or 0.0
        )
    except (KeyError, TypeError, ValueError):
        return False
    digest = str(artifact.get("artifact_sha256") or "")
    return bool(
        payload.get("training_status") == "ready"
        and payload.get("official_closing_fields_used") is False
        and payload.get("market_is_exact_nested_null") is True
        and evaluated_days >= MINIMUM_SELECTION_HOLDOUT_DAYS
        and days_better * 2 > evaluated_days
        and loss_delta < 0.0
        and top5 >= market_top5 - MAXIMUM_TOP5_HIT_RATE_DEGRADATION
        and residual_weight > 0.0
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest.lower())
    )


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
        raise ValueError("decision-time V44 cache must contain a mapping")
    races = payload.get("races")
    contract = payload.get("contract")
    if not isinstance(races, list) or not isinstance(contract, Mapping):
        raise ValueError("decision-time V44 cache is missing races or contract")
    validate_decision_scored_cache_contract(contract)
    fitted = fit_decision_time_stacked_market(
        races,
        calibration_through=calibration_through,
        minimum_training_days=minimum_training_days,
        minimum_training_races=minimum_training_races,
        num_threads=num_threads,
    )
    training_status = str(fitted.pop("status"))
    return {
        "status": "completed",
        "model": MODEL_NAME,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a frozen market/linear/nonlinear stack from decision-time "
            "snapshots only"
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
        "selected_stack": result.get("selected_stack"),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
