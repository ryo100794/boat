#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

from boatrace_ai.db import connection
from boatrace_ai.standard_evaluation import (
    MODEL_SOURCES,
    POLICY,
    PROTOCOL_ID,
    evaluate_promotions,
    load_protocol,
    verify_protocol_against_database,
)


def audit(root: Path, *, db: str | Path | None = None) -> dict[str, Any]:
    errors: list[str] = []
    try:
        protocol = load_protocol(root / "protocol.json")
    except ValueError as exc:
        protocol = _read_json(root / "protocol.json")
        errors.append(f"protocol validation failed: {exc}")
    manifest = _read_json(root / "manifest.json")
    expected_ids = [source.model_id for source in MODEL_SOURCES]
    expected_hash = str(protocol.get("race_set_sha256") or "")
    expected_protocol_hash = str(protocol.get("protocol_sha256") or "")
    expected_prediction_races = _integer(
        protocol.get("prediction_races"), "protocol prediction_races", errors
    )
    expected_bankroll_races = _integer(
        protocol.get("bankroll_evaluable_races"),
        "protocol bankroll_evaluable_races",
        errors,
    )
    expected_date_count = _integer(
        protocol.get("holdout_date_count"), "protocol holdout_date_count", errors
    )

    for field, expected in protocol.items():
        _check(
            manifest.get(field) == expected,
            f"manifest {field} mismatch",
            errors,
        )
    _check(protocol.get("protocol_id") == PROTOCOL_ID, "protocol_id mismatch", errors)
    _check(protocol.get("policy") == POLICY, "protocol policy mismatch", errors)
    _check(
        manifest.get("protocol_sha256") == expected_protocol_hash,
        "protocol fingerprint mismatch",
        errors,
    )
    _check(
        manifest.get("comparison_model_ids") == expected_ids,
        "comparison model registry mismatch",
        errors,
    )
    registry_coverage = bool(manifest.get("registry_coverage_passed"))
    _check(registry_coverage, "registry coverage failed", errors)
    _check(
        _integer(manifest.get("model_count"), "manifest model_count", errors)
        == len(expected_ids),
        "model count mismatch",
        errors,
    )
    _check(
        _integer(
            manifest.get("valid_model_count"), "manifest valid_model_count", errors
        )
        == len(expected_ids),
        "valid model count mismatch",
        errors,
    )
    _check(not manifest.get("failed_models"), "manifest contains failed models", errors)
    _check(bool(manifest.get("comparison_ready")), "comparison is not ready", errors)

    if db is not None:
        try:
            with connection(db) as conn:
                verify_protocol_against_database(conn, protocol)
        except Exception as exc:
            errors.append(f"database protocol verification failed: {exc}")

    models: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    validations_passed: list[bool] = []
    common_daily_dates: list[str] | None = None
    for model_id in expected_ids:
        result = _read_json(root / f"{model_id}.json")
        results.append(result)
        validation = result.get("validation") or {}
        validation_passed = bool(validation.get("passed"))
        validations_passed.append(validation_passed)
        daily = result.get("daily") or []
        prefix = f"{model_id}: "

        _check(result.get("model_id") == model_id, prefix + "model_id mismatch", errors)
        _check(
            result.get("protocol_id") == protocol.get("protocol_id"),
            prefix + "protocol_id mismatch",
            errors,
        )
        _check(
            result.get("protocol_sha256") == expected_protocol_hash,
            prefix + "protocol fingerprint mismatch",
            errors,
        )
        _check(
            result.get("policy") == protocol.get("policy"),
            prefix + "policy mismatch",
            errors,
        )
        _check(validation_passed, prefix + "validation failed", errors)
        _check(
            not validation.get("policy_mismatches"),
            prefix + "raw policy mismatch",
            errors,
        )
        for field in (
            "prediction_race_set_sha256",
            "bankroll_race_set_sha256",
            "expected_race_set_sha256",
        ):
            _check(
                validation.get(field) == expected_hash,
                prefix + field + " mismatch",
                errors,
            )
        _check(
            _integer(
                validation.get("prediction_race_count"),
                prefix + "prediction_race_count",
                errors,
            )
            == expected_prediction_races,
            prefix + "prediction race count mismatch",
            errors,
        )
        _check(
            _integer(
                validation.get("bankroll_race_count"),
                prefix + "bankroll_race_count",
                errors,
            )
            == expected_bankroll_races,
            prefix + "bankroll race count mismatch",
            errors,
        )
        _check(
            _integer(result.get("evaluated_races"), prefix + "evaluated_races", errors)
            == expected_prediction_races,
            prefix + "model prediction race count mismatch",
            errors,
        )

        daily_dates = [
            str(row.get("race_date"))
            for row in daily
            if isinstance(row, dict) and row.get("race_date")
        ]
        unique_daily_dates = sorted(set(daily_dates))
        _check(
            len(daily_dates) == len(unique_daily_dates),
            prefix + "duplicate daily rows",
            errors,
        )
        _check(
            len(unique_daily_dates) == expected_date_count,
            prefix + "daily date count mismatch",
            errors,
        )
        _check(
            daily_dates == unique_daily_dates,
            prefix + "daily rows are not sorted",
            errors,
        )
        _check(
            unique_daily_dates[0] == protocol.get("holdout_start")
            if unique_daily_dates
            else False,
            prefix + "daily start mismatch",
            errors,
        )
        _check(
            unique_daily_dates[-1] == protocol.get("holdout_end")
            if unique_daily_dates
            else False,
            prefix + "daily end mismatch",
            errors,
        )
        if common_daily_dates is None:
            common_daily_dates = unique_daily_dates
        else:
            _check(
                unique_daily_dates == common_daily_dates,
                prefix + "daily date axis mismatch",
                errors,
            )

        aggregate = _audit_daily(daily, prefix, errors)
        _check(
            aggregate["evaluated_races"] == expected_bankroll_races,
            prefix + "daily evaluated race aggregate mismatch",
            errors,
        )
        _check(
            _integer(
                result.get("bankroll_evaluated_races"),
                prefix + "bankroll_evaluated_races",
                errors,
            )
            == aggregate["evaluated_races"],
            prefix + "bankroll evaluated race total mismatch",
            errors,
        )
        bankroll = result.get("bankroll") or {}
        for field in ("stake_yen", "return_yen", "profit_yen", "tickets"):
            expected = aggregate[field]
            _check(
                _integer(result.get(field), prefix + field, errors) == expected,
                prefix + field + " aggregate mismatch",
                errors,
            )
            _check(
                _integer(bankroll.get(field), prefix + "bankroll." + field, errors)
                == expected,
                prefix + "bankroll " + field + " aggregate mismatch",
                errors,
            )
        _check(
            _same_number(result.get("roi"), aggregate["roi"]),
            prefix + "roi aggregate mismatch",
            errors,
        )
        _check(
            _same_number(bankroll.get("roi"), aggregate["roi"]),
            prefix + "bankroll roi aggregate mismatch",
            errors,
        )

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
                "validation_passed": validation_passed,
                "promotion_eligible": bool(
                    (result.get("promotion") or {}).get("eligible")
                ),
            }
        )

    comparison_ready = (
        len(results) == len(expected_ids)
        and all(validations_passed)
        and registry_coverage
    )
    recalculated_results = deepcopy(results)
    recalculated = evaluate_promotions(
        recalculated_results,
        comparison_ready=comparison_ready,
    )
    recorded = manifest.get("promotion_decision") or {}
    _check(
        recorded == recalculated,
        "promotion decision mismatch",
        errors,
    )
    for recorded_model, recalculated_model in zip(results, recalculated_results):
        model_id = str(recorded_model.get("model_id") or "-")
        _check(
            recorded_model.get("promotion") == recalculated_model.get("promotion"),
            f"{model_id}: model promotion mismatch",
            errors,
        )

    return {
        "passed": not errors,
        "errors": errors,
        "protocol": {
            "protocol_id": protocol.get("protocol_id"),
            "protocol_sha256": expected_protocol_hash,
            "holdout_start": protocol.get("holdout_start"),
            "holdout_end": protocol.get("holdout_end"),
            "holdout_date_count": expected_date_count,
            "prediction_races": expected_prediction_races,
            "bankroll_evaluable_races": expected_bankroll_races,
            "race_set_sha256": expected_hash,
            "policy": protocol.get("policy"),
        },
        "models": models,
        "promotion_decision": recorded,
    }


def _audit_daily(
    daily: list[dict[str, Any]],
    prefix: str,
    errors: list[str],
) -> dict[str, int | float]:
    totals = {
        "stake_yen": 0,
        "return_yen": 0,
        "profit_yen": 0,
        "tickets": 0,
        "evaluated_races": 0,
    }
    cumulative_profit = 0
    for index, row in enumerate(daily):
        label = f"{prefix}daily[{index}]"
        if not isinstance(row, dict):
            errors.append(label + " is not an object")
            continue
        values = {
            field: _integer(row.get(field), f"{label}.{field}", errors)
            for field in totals
        }
        for field, value in values.items():
            totals[field] += value
        _check(
            values["profit_yen"] == values["return_yen"] - values["stake_yen"],
            label + " profit does not equal return minus stake",
            errors,
        )
        daily_roi = (
            values["return_yen"] / values["stake_yen"]
            if values["stake_yen"]
            else 0.0
        )
        if row.get("roi") is not None:
            _check(
                _same_number(row.get("roi"), daily_roi),
                label + " roi mismatch",
                errors,
            )
        cumulative_profit += values["profit_yen"]
        _check(
            "cumulative_profit_yen" in row,
            label + " cumulative profit missing",
            errors,
        )
        if "cumulative_profit_yen" in row:
            _check(
                _integer(
                    row.get("cumulative_profit_yen"),
                    label + ".cumulative_profit_yen",
                    errors,
                )
                == cumulative_profit,
                label + " cumulative profit mismatch",
                errors,
            )
    _check(
        totals["profit_yen"] == totals["return_yen"] - totals["stake_yen"],
        prefix + "daily total profit does not equal return minus stake",
        errors,
    )
    totals["roi"] = (
        totals["return_yen"] / totals["stake_yen"]
        if totals["stake_yen"]
        else 0.0
    )
    return totals


def _integer(value: Any, label: str, errors: list[str]) -> int:
    if isinstance(value, bool):
        errors.append(f"{label} is not an integer")
        return 0
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError):
        errors.append(f"{label} is not an integer")
        return 0
    if isinstance(value, float) and not value.is_integer():
        errors.append(f"{label} is not an integer")
        return 0
    return converted


def _same_number(actual: Any, expected: float) -> bool:
    try:
        return math.isclose(
            float(actual),
            float(expected),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    except (TypeError, ValueError, OverflowError):
        return False


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit unified standardized model evaluation artifacts."
    )
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--db",
        help="optionally rebuild and verify protocol fields against this database",
    )
    args = parser.parse_args()
    result = audit(args.root, db=args.db)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
