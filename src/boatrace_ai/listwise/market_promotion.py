from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from .market_calibration import (
    MARKET_EVALUATION_VERSION,
    file_sha256,
    write_json_atomic,
)


MANIFEST_VERSION = 1
REQUIRED_PASS_GATES = (
    "sample_size_pass",
    "positive_profit_pass",
    "roi_pass",
    "fold_stability_pass",
    "calibration_pass",
    "market_confidence_pass",
    "no_lookahead_pass",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _resolved_path(raw: Any) -> Path:
    path = Path(str(raw or "")).expanduser()
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def _finite_number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def validate_candidate(path: str | Path) -> dict[str, Any]:
    candidate_path = Path(path).resolve()
    errors: list[str] = []
    try:
        data = _read_json(candidate_path)
    except ValueError as exc:
        return {
            "candidate_path": str(candidate_path),
            "valid": False,
            "errors": [str(exc)],
            "data": {},
        }

    if data.get("evaluation_version") != MARKET_EVALUATION_VERSION:
        errors.append("evaluation_version mismatch")
    gate = data.get("promotion_gate")
    if not isinstance(gate, dict):
        gate = {}
        errors.append("promotion_gate missing")
    for key in REQUIRED_PASS_GATES:
        if gate.get(key) is not True:
            errors.append(f"promotion gate failed: {key}")
    if data.get("promotion_eligible") is not True:
        errors.append("promotion_eligible is not true")

    evaluated_races = int(data.get("evaluation_races") or 0)
    evaluation_days = int(data.get("evaluation_days") or 0)
    if evaluated_races < int(gate.get("minimum_evaluation_races") or 1000):
        errors.append("evaluation race minimum not met")
    if evaluation_days < int(gate.get("minimum_evaluation_days") or 30):
        errors.append("evaluation day minimum not met")
    if int(data.get("stake_yen") or 0) <= 0:
        errors.append("evaluated stake is zero")
    if int(data.get("profit_yen") or 0) <= 0:
        errors.append("evaluated profit is not positive")
    roi = _finite_number(data.get("roi"))
    if roi is None or roi <= 1.0:
        errors.append("evaluated ROI does not exceed one")

    source_path = _resolved_path(data.get("source_model"))
    expected_source_hash = str(data.get("source_model_sha256") or "")
    actual_source_hash = None
    if not source_path.is_file():
        errors.append("source model missing")
    else:
        actual_source_hash = file_sha256(source_path)
        if not expected_source_hash or actual_source_hash != expected_source_hash:
            errors.append("source model SHA-256 mismatch")

    from_date = str(data.get("from_date") or "")
    through_date = str(data.get("through_date") or "")
    trained_through = data.get("source_model_trained_through") or ()
    trained_date = str(trained_through[1]) if len(trained_through) > 1 else ""
    if not from_date or not trained_date or trained_date >= from_date:
        errors.append("source model training overlaps evaluation")

    deployment = data.get("deployment_configuration")
    if not isinstance(deployment, dict):
        deployment = {}
        errors.append("deployment configuration missing")
    if deployment.get("role") != "next_day_refit_not_evaluation":
        errors.append("deployment configuration role mismatch")
    if str(deployment.get("trained_through_date") or "") != through_date:
        errors.append("deployment configuration is not refit through evaluation end")
    if int(deployment.get("training_races") or 0) < evaluated_races:
        errors.append("deployment training race count is incomplete")
    calibrator = deployment.get("calibrator")
    if not isinstance(calibrator, dict):
        errors.append("deployment calibrator missing")
    elif (
        deployment.get("calibrator_strategy") == "newton_residual"
        and calibrator.get("converged") is not True
    ):
        errors.append("deployment Newton calibrator did not converge")
    policy = deployment.get("selected_policy")
    if not isinstance(policy, dict) or policy.get("no_bet") is True:
        errors.append("deployment policy is no-bet or missing")

    metrics = data.get("probability_metrics") or {}
    log_loss = _finite_number(metrics.get("calibrated_trifecta_log_loss"))
    if log_loss is None:
        errors.append("calibrated LogLoss missing")

    return {
        "candidate_id": candidate_path.stem,
        "candidate_path": str(candidate_path),
        "candidate_sha256": file_sha256(candidate_path),
        "source_model_path": str(source_path),
        "source_model_sha256": actual_source_hash,
        "valid": not errors,
        "errors": errors,
        "metrics": {
            "evaluation_races": evaluated_races,
            "evaluation_days": evaluation_days,
            "profit_yen": int(data.get("profit_yen") or 0),
            "roi": roi,
            "calibrated_trifecta_log_loss": log_loss,
            "trifecta_top5_hit_rate": _finite_number(
                metrics.get("calibrated_trifecta_top5_hit_rate")
            ),
        },
        "data": data,
    }


def _selection_key(candidate: dict[str, Any]) -> tuple[float, float, float, str]:
    metrics = candidate["metrics"]
    return (
        float(metrics["profit_yen"]),
        float(metrics["roi"]),
        -float(metrics["calibrated_trifecta_log_loss"]),
        str(candidate["candidate_id"]),
    )


def promote_best_candidate(
    candidate_paths: Iterable[str | Path],
    *,
    output_path: str | Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    output = Path(output_path)
    candidates = [validate_candidate(path) for path in candidate_paths]
    eligible = [candidate for candidate in candidates if candidate["valid"]]
    summary = [
        {
            key: value
            for key, value in candidate.items()
            if key != "data"
        }
        for candidate in candidates
    ]
    if not eligible:
        return {
            "status": "no_eligible_candidate",
            "manifest_path": str(output),
            "manifest_preserved": output.exists(),
            "candidates": summary,
        }

    selected = max(eligible, key=_selection_key)
    data = selected["data"]
    deployment = data["deployment_configuration"]
    trained_through = date.fromisoformat(str(deployment["trained_through_date"]))
    activated_at = (now or datetime.now(timezone.utc)).replace(
        microsecond=0
    ).isoformat()
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "status": "active",
        "activated_at": activated_at,
        "valid_from_date": (trained_through + timedelta(days=1)).isoformat(),
        "selected_candidate_id": selected["candidate_id"],
        "evaluation_path": selected["candidate_path"],
        "evaluation_sha256": selected["candidate_sha256"],
        "evaluation_version": data["evaluation_version"],
        "evaluation_from_date": data["from_date"],
        "evaluation_through_date": data["through_date"],
        "evaluation_metrics": selected["metrics"],
        "source_model_path": selected["source_model_path"],
        "source_model_sha256": selected["source_model_sha256"],
        "source_model_trained_through": data.get("source_model_trained_through"),
        "deployment_configuration": deployment,
        "promotion_gate": data["promotion_gate"],
    }
    if output.is_file():
        try:
            existing = _read_json(output)
        except ValueError:
            existing = {}
        immutable_keys = (
            "evaluation_sha256",
            "source_model_sha256",
            "deployment_configuration",
        )
        if all(existing.get(key) == manifest.get(key) for key in immutable_keys):
            return {
                "status": "already_active",
                "manifest_path": str(output),
                "selected_candidate_id": selected["candidate_id"],
                "candidates": summary,
            }
    write_json_atomic(output, manifest)
    return {
        "status": "promoted",
        "manifest_path": str(output),
        "selected_candidate_id": selected["candidate_id"],
        "candidates": summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Promote only fully verified market-model evaluation artifacts."
    )
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument(
        "--output", default="data/models/active_market_model.json"
    )
    args = parser.parse_args(argv)
    result = promote_best_candidate(args.candidate, output_path=args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
