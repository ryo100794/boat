from __future__ import annotations

import argparse
import copy
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import joblib

from ..listwise.closing_odds_t300_nonlinear_v12 import (
    MODEL_NAME as CLOSING_MODEL_NAME,
    MODEL_VERSION as CLOSING_MODEL_VERSION,
    fit_closing_odds_t300_nonlinear_v12,
)


ARTIFACT_TYPE = "boatrace_v12_t300_shadow_bundle"
ARTIFACT_SCHEMA_VERSION = 1
STRATEGY_NAME = "v12_role_t300"
INTEGRATED_MODEL_NAME = "odds_path_role_integrated_t300_nonlinear_v12"
DEPLOYMENT_ROLE = "next_day_refit_not_evaluation"


def _iso_date(value: object, name: str) -> str:
    text = str(value or "").strip()
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO date") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load evaluation JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("evaluation JSON must contain an object")
    return payload


def _load_scored_cache(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        payload = joblib.load(path)
    except Exception as exc:
        raise ValueError(f"cannot load scored cache: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("scored cache must contain a mapping")
    contract = payload.get("contract")
    races = payload.get("races")
    if not isinstance(contract, dict):
        raise ValueError("scored cache contract is missing")
    if not isinstance(races, list) or not races:
        raise ValueError("scored cache must contain a non-empty race list")
    normalized: list[dict[str, Any]] = []
    for index, race in enumerate(races):
        if not isinstance(race, dict):
            raise ValueError(f"scored cache race {index} must be a mapping")
        item = dict(race)
        item["race_date"] = _iso_date(
            item.get("race_date"), f"scored cache race {index} race_date"
        )
        normalized.append(item)
    normalized.sort(
        key=lambda race: (
            str(race["race_date"]),
            str(race.get("race_id") or ""),
            str(race.get("jcd") or race.get("venue_code") or ""),
            str(race.get("rno") or race.get("race_no") or ""),
        )
    )
    return contract, normalized


def _without_estimator(model: Mapping[str, Any]) -> dict[str, Any]:
    report = copy.deepcopy(dict(model))
    point_model = report.get("point_model")
    if isinstance(point_model, dict):
        point_model.pop("estimator", None)
    return report


def _component_identity(component: object) -> dict[str, Any] | None:
    if not isinstance(component, Mapping):
        return None
    report = _without_estimator(component)
    return {
        "model_name": report.get("model_name") or report.get("model_type"),
        "model_version": report.get("model_version"),
        "method": report.get("method"),
        "ready": report.get("ready"),
        "trained_through_date": report.get("trained_through_date"),
        "sha256": _canonical_sha256(report),
    }


def _validate_evaluation(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    if evaluation.get("model") != INTEGRATED_MODEL_NAME:
        raise ValueError("evaluation model identity is not V12 role-integrated")
    if evaluation.get("calibrator_strategy") != INTEGRATED_MODEL_NAME:
        raise ValueError("evaluation calibrator identity is not V12 role-integrated")
    folds = evaluation.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError("V12 evaluation is incomplete: no completed folds")
    if int(evaluation.get("evaluation_days") or 0) != len(folds):
        raise ValueError("V12 evaluation is incomplete: fold count mismatch")
    if int(evaluation.get("evaluated_races") or 0) <= 0:
        raise ValueError("V12 evaluation is incomplete: no evaluated races")
    for index, fold in enumerate(folds):
        if not isinstance(fold, Mapping):
            raise ValueError(f"V12 evaluation fold {index} is invalid")
        leakage = fold.get("leakage_guard")
        if not isinstance(leakage, Mapping) or leakage.get("pass") is not True:
            raise ValueError(f"V12 evaluation fold {index} did not pass date checks")
    coverage = evaluation.get("benchmark_evaluation_coverage")
    if coverage is not None and float(coverage) < 1.0:
        raise ValueError("V12 evaluation is incomplete: benchmark coverage below 100%")

    deployment = evaluation.get("deployment_configuration")
    if not isinstance(deployment, dict):
        raise ValueError("V12 deployment configuration is missing")
    if deployment.get("role") != DEPLOYMENT_ROLE:
        raise ValueError("V12 deployment role is not a next-day refit")
    if deployment.get("calibrator_strategy") != INTEGRATED_MODEL_NAME:
        raise ValueError("deployment calibrator identity does not match V12")
    for key in ("probability_lcb", "selection_conformal"):
        if not isinstance(deployment.get(key), Mapping):
            raise ValueError(f"deployment component is missing: {key}")
    if not isinstance(
        deployment.get("probability_model") or deployment.get("operational_model"),
        Mapping,
    ):
        raise ValueError("deployment probability model is missing")

    closing = deployment.get("closing_t300_v12_model")
    if not isinstance(closing, dict):
        raise ValueError("report-only V12 closing model is missing")
    if closing.get("model_name") != CLOSING_MODEL_NAME:
        raise ValueError("closing model identity does not match V12 T300")
    if int(closing.get("model_version") or 0) != CLOSING_MODEL_VERSION:
        raise ValueError("closing model version does not match V12")
    if not closing.get("ready"):
        raise ValueError("V12 closing model is not ready")
    if not closing.get("challenger_adopted"):
        raise ValueError("V12 closing model was not adopted")
    if closing.get("selected_mode") != "nonlinear_model":
        raise ValueError("V12 closing model did not select the nonlinear estimator")
    point_model = closing.get("point_model")
    if not isinstance(point_model, dict) or not point_model:
        raise ValueError("report-only V12 point model metadata is missing")
    if "estimator" in point_model:
        raise ValueError("evaluation JSON must not contain a live estimator")

    identity = deployment.get("closing_model_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("deployment closing model identity is missing")
    if identity.get("requested_model") != CLOSING_MODEL_NAME:
        raise ValueError("deployment requested closing model does not match V12")
    if identity.get("selected_model") != CLOSING_MODEL_NAME:
        raise ValueError("deployment did not select V12 closing odds")
    if identity.get("v12_ready") is not True or identity.get("v12_adopted") is not True:
        raise ValueError("deployment V12 closing identity is not ready and adopted")
    return deployment


def _validate_source_identity(
    evaluation: Mapping[str, Any],
    contract: Mapping[str, Any],
    races: Sequence[Mapping[str, Any]],
    *,
    prediction_date: str,
) -> tuple[str, list[str]]:
    evaluation_source_hash = str(evaluation.get("source_model_sha256") or "")
    cache_source_hash = str(contract.get("model_sha256") or "")
    if len(evaluation_source_hash) != 64 or evaluation_source_hash != cache_source_hash:
        raise ValueError("source model SHA256 mismatch between evaluation and scored cache")
    if evaluation.get("source_model_trained_through") != contract.get("trained_through"):
        raise ValueError("source model trained-through identity mismatch")
    for key in ("from_date", "through_date"):
        left = _iso_date(evaluation.get(key), f"evaluation {key}")
        right = _iso_date(contract.get(key), f"scored cache contract {key}")
        if left != right:
            raise ValueError(f"{key} mismatch between evaluation and scored cache")
    if evaluation.get("odds_data_signature") != contract.get("odds_data_signature"):
        raise ValueError("odds data signature mismatch between evaluation and scored cache")
    if (
        evaluation.get("dataset") is not None
        and contract.get("dataset") is not None
        and evaluation.get("dataset") != contract.get("dataset")
    ):
        raise ValueError("dataset identity mismatch between evaluation and scored cache")

    race_dates = sorted({str(race["race_date"]) for race in races})
    from_date = _iso_date(contract.get("from_date"), "scored cache contract from_date")
    through_date = _iso_date(contract.get("through_date"), "scored cache contract through_date")
    if any(value < from_date or value > through_date for value in race_dates):
        raise ValueError("scored cache contains races outside its date boundaries")
    if race_dates[-1] != through_date:
        raise ValueError("scored cache does not reach its declared through_date")
    expected_prediction_date = (
        date.fromisoformat(through_date) + timedelta(days=1)
    ).isoformat()
    if prediction_date != expected_prediction_date:
        raise ValueError(
            "prediction_date must be exactly one day after the scored cache through_date"
        )
    return through_date, race_dates


def _iter_trained_through_dates(
    value: object, path: str = "deployment"
) -> Sequence[tuple[str, object]]:
    found: list[tuple[str, object]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key == "trained_through_date":
                found.append((child_path, child))
            found.extend(_iter_trained_through_dates(child, child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(_iter_trained_through_dates(child, f"{path}[{index}]"))
    return found


def _validate_training_boundaries(deployment: Mapping[str, Any], prediction_date: str) -> None:
    for path, value in _iter_trained_through_dates(deployment):
        if value is None:
            continue
        trained = _iso_date(value, path)
        if trained >= prediction_date:
            raise ValueError(
                f"future-date training boundary at {path}: {trained} >= {prediction_date}"
            )


def _temporary_path(destination: Path, suffix: str) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=suffix, dir=destination.parent
    )
    os.close(descriptor)
    return Path(name)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bundle_and_manifest_atomic(
    output: Path,
    bundle: Mapping[str, Any],
    manifest_without_output_hash: Mapping[str, Any],
) -> dict[str, Any]:
    if output.suffix != ".joblib":
        raise ValueError("output must use the .joblib suffix")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_suffix(".manifest.json")
    bundle_temporary = _temporary_path(output, ".joblib.tmp")
    manifest_temporary = _temporary_path(manifest_path, ".json.tmp")
    try:
        joblib.dump(dict(bundle), bundle_temporary)
        _fsync_file(bundle_temporary)
        bundle_hash = _sha256_file(bundle_temporary)
        manifest = copy.deepcopy(dict(manifest_without_output_hash))
        manifest["output"] = {
            "bundle": str(output),
            "bundle_sha256": bundle_hash,
            "bundle_bytes": bundle_temporary.stat().st_size,
            "manifest": str(manifest_path),
        }
        encoded = (
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        with manifest_temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())

        loaded = joblib.load(bundle_temporary)
        estimator = (
            loaded.get("deployment", {})
            .get("closing_t300_v12_model", {})
            .get("point_model", {})
            .get("estimator")
            if isinstance(loaded, Mapping)
            else None
        )
        if estimator is None or not hasattr(estimator, "predict"):
            raise ValueError("temporary V12 bundle does not contain a live estimator")
        json.loads(manifest_temporary.read_text(encoding="utf-8"))

        os.replace(bundle_temporary, output)
        os.replace(manifest_temporary, manifest_path)
        _fsync_directory(output.parent)
        return manifest
    finally:
        bundle_temporary.unlink(missing_ok=True)
        manifest_temporary.unlink(missing_ok=True)


def build_v12_shadow_bundle(
    evaluation_json: Path,
    *,
    scored_cache: Path | None,
    output: Path,
    prediction_date: object,
) -> dict[str, Any]:
    prediction_day = _iso_date(prediction_date, "prediction_date")
    evaluation_json = evaluation_json.resolve()
    evaluation = _load_json(evaluation_json)
    deployment_source = _validate_evaluation(evaluation)
    configured_prediction_date = _iso_date(
        deployment_source.get("prediction_date"),
        "deployment prediction_date",
    )
    if configured_prediction_date != prediction_day:
        raise ValueError("prediction_date does not match the evaluated deployment date")

    cache_value = scored_cache or Path(str(evaluation.get("scored_cache") or ""))
    if not str(cache_value):
        raise ValueError("scored cache path is missing from evaluation JSON")
    cache_path = cache_value.resolve()
    contract, races = _load_scored_cache(cache_path)
    through_date, training_dates = _validate_source_identity(
        evaluation,
        contract,
        races,
        prediction_date=prediction_day,
    )

    report_closing = deployment_source["closing_t300_v12_model"]
    summary = report_closing.get("training_summary") or {}
    if int(summary.get("training_races") or 0) != len(races):
        raise ValueError("V12 report training race count does not match scored cache")
    if list(summary.get("training_dates") or []) != training_dates:
        raise ValueError("V12 report training dates do not match scored cache")
    if _iso_date(report_closing.get("trained_through_date"), "V12 trained_through_date") != through_date:
        raise ValueError("V12 report trained-through date does not match scored cache")

    lower_model = report_closing.get("lower_quantile_model") or {}
    point_model = report_closing.get("point_model") or {}
    refitted = fit_closing_odds_t300_nonlinear_v12(
        races,
        prediction_date=prediction_day,
        minimum_relative_mae_improvement=float(
            report_closing.get("minimum_relative_mae_improvement")
        ),
        lower_quantile=float(lower_model.get("quantile")),
        random_state=int(point_model.get("random_state")),
        engine=str(report_closing.get("actual_engine")),
    )
    if not refitted.get("ready") or not refitted.get("challenger_adopted"):
        raise ValueError("refitted V12 closing model is not ready and adopted")
    if refitted.get("point_model", {}).get("estimator") is None:
        raise ValueError("refitted V12 closing model has no estimator")
    refitted_report = _without_estimator(refitted)
    if _canonical_sha256(refitted_report) != _canonical_sha256(report_closing):
        raise ValueError("refitted V12 model identity does not match evaluation report")

    deployment = copy.deepcopy(deployment_source)
    deployment["closing_t300_v12_model"] = refitted
    _validate_training_boundaries(deployment, prediction_day)
    preserved_keys = [
        key
        for key in (
            "probability_model",
            "operational_model",
            "probability_lcb",
            "selection_conformal",
            "closing_v11_fallback_model",
        )
        if key in deployment_source
    ]
    for key in preserved_keys:
        if deployment[key] != deployment_source[key]:
            raise ValueError(f"deployment component changed unexpectedly: {key}")

    evaluation_hash = _sha256_file(evaluation_json)
    cache_hash = _sha256_file(cache_path)
    contract_hash = _canonical_sha256(contract)
    provenance = {
        "evaluation_json_sha256": evaluation_hash,
        "scored_cache_sha256": cache_hash,
        "scored_cache_contract_sha256": contract_hash,
        "source_model_sha256": str(contract["model_sha256"]),
        "source_model_trained_through": contract.get("trained_through"),
        "from_date": str(contract["from_date"]),
        "through_date": str(contract["through_date"]),
        "odds_data_signature_sha256": _canonical_sha256(
            contract.get("odds_data_signature")
        ),
    }
    bundle = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "strategy_name": STRATEGY_NAME,
        "prediction_date": prediction_day,
        "deployment": deployment,
        "provenance": provenance,
    }
    probability = deployment.get("probability_model") or deployment.get(
        "operational_model"
    )
    manifest = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "strategy_name": STRATEGY_NAME,
        "prediction_date": prediction_day,
        "trained_through_date": through_date,
        "inputs": {
            "evaluation_json": {
                "path": str(evaluation_json),
                "sha256": evaluation_hash,
            },
            "scored_cache": {
                "path": str(cache_path),
                "sha256": cache_hash,
                "contract_sha256": contract_hash,
            },
        },
        "source_identity": provenance,
        "training": {
            "dates": training_dates,
            "days": len(training_dates),
            "races": len(races),
            "examples": refitted.get("training_summary", {}).get(
                "training_examples"
            ),
        },
        "model_identities": {
            "integrated_model": INTEGRATED_MODEL_NAME,
            "calibrator_strategy": evaluation["calibrator_strategy"],
            "probability_model": _component_identity(probability),
            "probability_lcb": _component_identity(
                deployment["probability_lcb"]
            ),
            "closing_t300_v12_report": _component_identity(report_closing),
            "closing_t300_v12_refit": _component_identity(refitted),
            "selection_conformal": _component_identity(
                deployment["selection_conformal"]
            ),
            "closing_v11_fallback": _component_identity(
                deployment.get("closing_v11_fallback_model")
            ),
        },
        "preserved_components": preserved_keys,
        "estimator_storage": "joblib_only",
    }
    return _write_bundle_and_manifest_atomic(output.resolve(), bundle, manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Refit the evaluated V12 T300 closing model and build a next-day "
            "shadow joblib bundle."
        )
    )
    parser.add_argument("--evaluation-json", type=Path, required=True)
    parser.add_argument(
        "--scored-cache",
        type=Path,
        help="Defaults to the scored_cache path recorded in the evaluation JSON.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prediction-date", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_v12_shadow_bundle(
        args.evaluation_json,
        scored_cache=args.scored_cache,
        output=args.output,
        prediction_date=args.prediction_date,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
