#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from boatrace_ai.standard_evaluation import (
    MODEL_SOURCES,
    POLICY,
    PROTOCOL_ID,
    evaluate_promotions,
    protocol_sha256,
)


def audit(root: Path) -> dict[str, Any]:
    manifest = _read_json(root / "manifest.json")
    expected_ids = [source.model_id for source in MODEL_SOURCES]
    expected_hash = str(manifest.get("race_set_sha256") or "")
    expected_protocol_hash = str(manifest.get("protocol_sha256") or "")
    expected_prediction_races = int(manifest.get("prediction_races") or 0)
    expected_bankroll_races = int(manifest.get("bankroll_evaluable_races") or 0)
    expected_date_count = int(manifest.get("holdout_date_count") or 0)
    errors: list[str] = []

    _check(manifest.get("protocol_id") == PROTOCOL_ID, "protocol_id mismatch", errors)
    _check(
        expected_protocol_hash == protocol_sha256(manifest),
        "protocol fingerprint mismatch",
        errors,
    )
    _check(bool(manifest.get("full_day_boundary")), "holdout is not full-day bounded", errors)
    _check(manifest.get("policy") == POLICY, "manifest policy mismatch", errors)
    _check(
        manifest.get("comparison_model_ids") == expected_ids,
        "comparison model registry mismatch",
        errors,
    )
    _check(bool(manifest.get("registry_coverage_passed")), "registry coverage failed", errors)
    _check(int(manifest.get("model_count") or 0) == len(expected_ids), "model count mismatch", errors)
    _check(int(manifest.get("valid_model_count") or 0) == len(expected_ids), "valid model count mismatch", errors)
    _check(not manifest.get("failed_models"), "manifest contains failed models", errors)
    _check(bool(manifest.get("comparison_ready")), "comparison is not ready", errors)

    models: list[dict[str, Any]] = []
    common_daily_dates: list[str] | None = None
    for model_id in expected_ids:
        result = _read_json(root / f"{model_id}.json")
        validation = result.get("validation") or {}
        daily = result.get("daily") or []
        daily_dates = [str(row.get("race_date")) for row in daily if row.get("race_date")]
        unique_daily_dates = sorted(set(daily_dates))
        daily_evaluated_races = sum(int(row.get("evaluated_races") or 0) for row in daily)

        prefix = f"{model_id}: "
        _check(result.get("model_id") == model_id, prefix + "model_id mismatch", errors)
        _check(result.get("protocol_id") == PROTOCOL_ID, prefix + "protocol_id mismatch", errors)
        _check(
            result.get("protocol_sha256") == expected_protocol_hash,
            prefix + "protocol fingerprint mismatch",
            errors,
        )
        _check(result.get("policy") == POLICY, prefix + "policy mismatch", errors)
        _check(bool(validation.get("passed")), prefix + "validation failed", errors)
        _check(not validation.get("policy_mismatches"), prefix + "raw policy mismatch", errors)
        for field in (
            "prediction_race_set_sha256",
            "bankroll_race_set_sha256",
            "expected_race_set_sha256",
        ):
            _check(validation.get(field) == expected_hash, prefix + field + " mismatch", errors)
        _check(
            int(validation.get("prediction_race_count") or 0) == expected_prediction_races,
            prefix + "prediction race count mismatch",
            errors,
        )
        _check(
            int(validation.get("bankroll_race_count") or 0) == expected_bankroll_races,
            prefix + "bankroll race count mismatch",
            errors,
        )
        _check(len(daily_dates) == len(unique_daily_dates), prefix + "duplicate daily rows", errors)
        _check(len(unique_daily_dates) == expected_date_count, prefix + "daily date count mismatch", errors)
        _check(daily_dates == unique_daily_dates, prefix + "daily rows are not sorted", errors)
        _check(
            unique_daily_dates[0] == manifest.get("holdout_start") if unique_daily_dates else False,
            prefix + "daily start mismatch",
            errors,
        )
        _check(
            unique_daily_dates[-1] == manifest.get("holdout_end") if unique_daily_dates else False,
            prefix + "daily end mismatch",
            errors,
        )
        _check(
            daily_evaluated_races == expected_bankroll_races,
            prefix + "daily evaluated race aggregate mismatch",
            errors,
        )
        if common_daily_dates is None:
            common_daily_dates = unique_daily_dates
        else:
            _check(unique_daily_dates == common_daily_dates, prefix + "daily date axis mismatch", errors)

        models.append(
            {
                "model_id": model_id,
                "entry_log_loss": result.get("entry_log_loss"),
                "winner_top1_accuracy": result.get("winner_top1_accuracy"),
                "trifecta_top5_hit_rate": result.get("trifecta_top5_hit_rate"),
                "roi": result.get("roi"),
                "profit_yen": result.get("profit_yen"),
                "stake_yen": result.get("stake_yen"),
                "max_drawdown_yen": result.get("max_drawdown_yen"),
                "daily_rows": len(daily),
                "validation_passed": bool(validation.get("passed")),
                "promotion_eligible": bool((result.get("promotion") or {}).get("eligible")),
            }
        )

    recalculated = evaluate_promotions(deepcopy([_read_json(root / f"{model_id}.json") for model_id in expected_ids]), comparison_ready=True)
    recorded = manifest.get("promotion_decision") or {}
    for field in ("status", "selected_model_id", "eligible_candidate_ids"):
        _check(recorded.get(field) == recalculated.get(field), f"promotion decision {field} mismatch", errors)

    return {
        "passed": not errors,
        "errors": errors,
        "protocol": {
            "protocol_id": manifest.get("protocol_id"),
            "protocol_sha256": expected_protocol_hash,
            "holdout_start": manifest.get("holdout_start"),
            "holdout_end": manifest.get("holdout_end"),
            "holdout_date_count": expected_date_count,
            "prediction_races": expected_prediction_races,
            "bankroll_evaluable_races": expected_bankroll_races,
            "race_set_sha256": expected_hash,
            "policy": POLICY,
        },
        "models": models,
        "promotion_decision": recorded,
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit unified standardized model evaluation artifacts.")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
