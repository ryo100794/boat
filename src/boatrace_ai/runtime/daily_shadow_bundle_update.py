from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import tempfile
import time
from typing import Any, Mapping, Sequence

import joblib
import psycopg
from psycopg.rows import dict_row

from . import v12_shadow_bundle


JST = timezone(timedelta(hours=9))
V12_MODEL = v12_shadow_bundle.INTEGRATED_MODEL_NAME
V14_MODEL = "odds_path_role_integrated_registered_band_lcb_v14"
V16_MODEL = "odds_path_role_integrated_fixed_band_passthrough_v16"
V18_MODEL = "odds_path_observed_closing_return_schedule_quota_v18"
V20_MODEL = "odds_path_observed_closing_return_schedule_quota_dual_head_v20"
V21_MODEL = "odds_path_observed_closing_return_schedule_quota_triple_head_v21"
SHADOW_STRATEGY = v12_shadow_bundle.STRATEGY_NAME
V14_SHADOW_STRATEGY = "v14_registered_band_t300"
V16_SHADOW_STRATEGY = "v16_fixed_band_t300"
V18_SHADOW_STRATEGY = "v18_schedule_quota_t300"
V20_SHADOW_STRATEGY = "v20_dual_head_t300"
V21_SHADOW_STRATEGY = "v21_triple_head_t300"
FAMILIES = ("v12", "v14", "v16")
ALL_FAMILIES = (*FAMILIES, "v18")
BUNDLE_FAMILIES = (*ALL_FAMILIES, "v20", "v21")
RECOVERY_FAMILIES = ("v18", "v20", "v21")
RECOVERY_FIRST_RACES = 5
RECOVERY_MAX_DECISION_DELAY_SECONDS = 90.0
FROZEN_SOURCE_JOB_IDS = {
    "v12": 7401,
    "v14": 7430,
    "v16": 7760,
    "v18": 8191,
    "v20": 8458,
    "v21": 8666,
}
FROZEN_TRAINED_THROUGH_DATE = "2026-07-29"


@dataclass(frozen=True)
class CompletedJob:
    job_id: int
    model_key: str
    result_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _iso(value: object, name: str) -> str:
    try:
        return date.fromisoformat(str(value or "")).isoformat()
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO date") from exc


def _inside(root: Path, value: Path, name: str) -> Path:
    root = root.resolve()
    value = value.resolve()
    if value != root and root not in value.parents:
        raise ValueError(f"{name} must be inside {root}")
    return value


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _model(family: str) -> str:
    if family == "v12":
        return V12_MODEL
    if family == "v14":
        return V14_MODEL
    if family == "v16":
        return V16_MODEL
    if family == "v18":
        return V18_MODEL
    if family == "v20":
        return V20_MODEL
    if family == "v21":
        return V21_MODEL
    raise ValueError(f"unsupported family: {family}")


def _manifest_trained_through(family: str, requested_through: str) -> str:
    return (
        FROZEN_TRAINED_THROUGH_DATE
        if family in FROZEN_SOURCE_JOB_IDS
        else requested_through
    )


def find_latest_completed_job(
    conn: Any, *, family: str, through_date: str, app_root: Path
) -> CompletedJob:
    row = conn.execute(
        """
        SELECT job_id, model_key, result_path
        FROM model_evaluation_jobs
        WHERE status = 'completed' AND result_path IS NOT NULL
          AND parameters->>'through_date' = %s
          AND parameters->>'calibrator_strategy' = %s
        ORDER BY priority DESC NULLS LAST,
                 completed_at DESC NULLS LAST, job_id DESC
        LIMIT 1
        """,
        (_iso(through_date, "through_date"), _model(family)),
    ).fetchone()
    if row is None:
        raise LookupError(f"no completed {family} job for {through_date}")
    path = Path(str(row["result_path"]))
    if not path.is_absolute():
        path = app_root / path
    path = _inside(app_root, path, "result_path")
    if not path.is_file():
        raise ValueError(f"evaluation result does not exist: {path}")
    return CompletedJob(int(row["job_id"]), str(row["model_key"]), path)


def find_completed_job(
    conn: Any, *, job_id: int, family: str, through_date: str, app_root: Path
) -> CompletedJob:
    if isinstance(job_id, bool) or not isinstance(job_id, int) or job_id <= 0:
        raise ValueError(f"active {family} source job ID is invalid")
    expected_through = _iso(through_date, "through_date")
    expected_strategy = _model(family)
    row = conn.execute(
        """
        SELECT job_id, model_key, result_path, status,
               parameters->>'through_date' AS through_date,
               parameters->>'calibrator_strategy' AS calibrator_strategy
        FROM model_evaluation_jobs
        WHERE job_id = %s
        """,
        (job_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"active {family} source job does not exist: {job_id}")
    actual_through = str(row["through_date"] or "")
    frozen = FROZEN_SOURCE_JOB_IDS.get(family) == job_id
    through_valid = (
        bool(actual_through) and actual_through <= expected_through
        if frozen else actual_through == expected_through
    )
    if (
        row["status"] != "completed"
        or not row["result_path"]
        or not through_valid
        or row["calibrator_strategy"] != expected_strategy
    ):
        raise ValueError(f"active {family} source job failed revalidation: {job_id}")
    path = Path(str(row["result_path"]))
    if not path.is_absolute():
        path = app_root / path
    path = _inside(app_root, path, "result_path")
    if not path.is_file():
        raise ValueError(f"evaluation result does not exist: {path}")
    return CompletedJob(int(row["job_id"]), str(row["model_key"]), path)


def _source_identity(result: Mapping[str, Any], result_path: Path) -> dict[str, Any]:
    cache = Path(str(result.get("scored_cache") or "")).resolve()
    if not cache.is_file():
        raise ValueError(f"scored_cache does not exist: {cache}")
    return {
        "source_model_sha256": str(result.get("source_model_sha256") or ""),
        "source_model_trained_through": result.get("source_model_trained_through"),
        "from_date": _iso(result.get("from_date"), "from_date"),
        "through_date": _iso(result.get("through_date"), "through_date"),
        "odds_data_signature": result.get("odds_data_signature"),
        "scored_cache_path": str(cache),
        "scored_cache_sha256": _sha256(cache),
        "result_path": str(result_path),
    }


def validate_shared_source(
    v12_job: CompletedJob,
    v14_job: CompletedJob,
    v16_job: CompletedJob | None = None,
    v18_job: CompletedJob | None = None,
    v20_job: CompletedJob | None = None,
    v21_job: CompletedJob | None = None,
) -> dict[str, Any]:
    left = _source_identity(_json(v12_job.result_path), v12_job.result_path)
    keys = tuple(key for key in left if key != "result_path")
    for label, job in (
        ("V14", v14_job), ("V16", v16_job), ("V18", v18_job),
        ("V20", v20_job), ("V21", v21_job),
    ):
        if job is None:
            continue
        right = _source_identity(_json(job.result_path), job.result_path)
        mismatches = [key for key in keys if left[key] != right[key]]
        if mismatches:
            raise ValueError(
                f"V12/{label} source mismatch: " + ", ".join(mismatches)
            )
    if len(left["source_model_sha256"]) != 64:
        raise ValueError("invalid shared source model SHA256")
    return left


def _validate_v14(result: Mapping[str, Any]) -> Mapping[str, Any]:
    if result.get("model") != V14_MODEL or result.get("calibrator_strategy") != V14_MODEL:
        raise ValueError("V14 result identity mismatch")
    deployment = result.get("deployment_configuration")
    if not isinstance(deployment, Mapping) or deployment.get("calibrator_strategy") != V14_MODEL:
        raise ValueError("V14 deployment identity mismatch")
    for key in ("operational_model", "probability_lcb", "selection_conformal", "candidate_policy"):
        if not isinstance(deployment.get(key), Mapping):
            raise ValueError(f"V14 deployment component is missing: {key}")
    if "closing_t300_v12_model" in deployment:
        raise ValueError("V14 evaluation must not supply a closing model")
    if not isinstance(result.get("closing_model_identity"), Mapping):
        raise ValueError("V14 closing model identity is missing")
    if not isinstance(deployment.get("closing_model_artifact_audit"), Mapping):
        raise ValueError("V14 closing model artifact audit is missing")
    return deployment


def _validate_v16(result: Mapping[str, Any]) -> Mapping[str, Any]:
    if result.get("model") != V16_MODEL or result.get("calibrator_strategy") != V16_MODEL:
        raise ValueError("V16 result identity mismatch")
    if result.get("real_betting_enabled") is not False:
        raise ValueError("V16 evaluation must disable real betting")
    deployment = result.get("deployment_configuration")
    if not isinstance(deployment, Mapping) or deployment.get("calibrator_strategy") != V16_MODEL:
        raise ValueError("V16 deployment identity mismatch")
    for key in (
        "operational_model", "probability_lcb", "closing_envelope_conformal",
        "candidate_policy",
    ):
        if not isinstance(deployment.get(key), Mapping):
            raise ValueError(f"V16 deployment component is missing: {key}")
    source_closing = deployment.get("closing_t300_v12_model")
    if source_closing is not None:
        if not isinstance(source_closing, Mapping):
            raise ValueError("V16 source closing artifact must be a mapping")
        boundary_audit = source_closing.get("boundary_audit")
        if (
            source_closing.get("model_name")
            != "closing_odds_t300_nonlinear_v12"
            or not isinstance(boundary_audit, Mapping)
            or boundary_audit.get("future_checkpoint_imputation") is not False
        ):
            raise ValueError("V16 source closing artifact is unsafe or inconsistent")
    probability = deployment["probability_lcb"]
    if (
        probability.get("model_name") != "strict_prior_t300_divergence_passthrough_v16"
        or probability.get("artifact_method")
        != "fixed_t300_divergence_raw_probability_passthrough_v16"
        or probability.get("raw_probability_passthrough") is not True
        or probability.get("uses_result") is not False
        or probability.get("uses_payout") is not False
    ):
        raise ValueError("V16 probability artifact is unsafe or inconsistent")
    envelope = deployment["closing_envelope_conformal"]
    if (
        envelope.get("method")
        != "selection_free_strict_prior_daily_q20_closing_ratio_v15"
        or envelope.get("selection_free") is not True
    ):
        raise ValueError("V16 closing envelope is unsafe or inconsistent")
    if deployment.get("real_betting_enabled") is not False:
        raise ValueError("V16 deployment must disable real betting")
    return deployment


def _validate_v18(result: Mapping[str, Any]) -> Mapping[str, Any]:
    if result.get("model") != V18_MODEL or result.get("calibrator_strategy") != V18_MODEL:
        raise ValueError("V18 result identity mismatch")
    if result.get("real_betting_enabled") is not False:
        raise ValueError("V18 evaluation must disable real betting")
    deployment = result.get("deployment_configuration")
    if not isinstance(deployment, Mapping) or deployment.get("calibrator_strategy") != V18_MODEL:
        raise ValueError("V18 deployment identity mismatch")
    if (
        deployment.get("deployment_mode") != "shadow_only"
        or deployment.get("real_betting_enabled") is not False
        or float(deployment.get("daily_stake_limit_fraction", 0.0)) != 1.0
    ):
        raise ValueError("V18 deployment must be 10000-yen shadow-only")
    calibrator = deployment.get("calibrator")
    operational = deployment.get("operational_model")
    policy = deployment.get("candidate_policy")
    selected = deployment.get("selected_policy")
    if not all(isinstance(value, Mapping) for value in (calibrator, operational, policy, selected)):
        raise ValueError("V18 fixed deployment components are missing")
    if (
        calibrator.get("converged") is not True
        or int(calibrator.get("training_races") or 0) <= 0
        or operational.get("model_type") != "odds_path_observed_closing_return_v4"
        or not isinstance(operational.get("weights"), Sequence)
        or not isinstance(operational.get("performance_priors"), Mapping)
    ):
        raise ValueError("V18 probability artifacts are unsafe or inconsistent")
    control = policy.get("v18_ticket_control")
    if (
        policy.get("no_bet") is True
        or not isinstance(control, Mapping)
        or control.get("method") != "strict_prior_daily_ticket_lower_quantile"
        or int(control.get("learned_daily_ticket_limit") or 0) <= 0
        or int(control.get("stake_granularity_yen") or 0) != 100
        or control.get("result_or_payout_fields_used") is not False
    ):
        raise ValueError("V18 schedule-aware ticket control is unsafe or inconsistent")
    if selected.get("no_bet") is not True:
        raise ValueError("V18 formal gate selection must remain fixed at no_bet")
    return deployment


def _validate_v20(result: Mapping[str, Any]) -> Mapping[str, Any]:
    if result.get("model") != V20_MODEL or result.get("calibrator_strategy") != V20_MODEL:
        raise ValueError("V20 result identity mismatch")
    if result.get("real_betting_enabled") is not False:
        raise ValueError("V20 evaluation must disable real betting")
    deployment = result.get("deployment_configuration")
    if not isinstance(deployment, Mapping) or deployment.get("calibrator_strategy") != V20_MODEL:
        raise ValueError("V20 deployment identity mismatch")
    if (
        deployment.get("deployment_mode") != "evaluation_only"
        or deployment.get("real_betting_enabled") is not False
        or float(deployment.get("daily_stake_limit_fraction", 0.0)) != 1.0
        or deployment.get("probability_metrics_head") != "probability_head"
        or deployment.get("chronological_bankroll_head") != "purchase_head"
    ):
        raise ValueError("V20 deployment routing is unsafe or inconsistent")
    dual = deployment.get("dual_head_calibration")
    probability = deployment.get("probability_calibrator")
    purchase = deployment.get("purchase_calibrator")
    operational = deployment.get("operational_model")
    policy = deployment.get("candidate_policy")
    selected = deployment.get("selected_policy")
    if not all(
        isinstance(value, Mapping)
        for value in (dual, probability, purchase, operational, policy, selected)
    ):
        raise ValueError("V20 dual-head deployment components are missing")
    probability_head = dual.get("probability_head")
    purchase_head = dual.get("purchase_head")
    if (
        dual.get("architecture") != "strict_prior_dual_calibrator_heads_v20"
        or dual.get("outer_holdout_used") is not False
        or not isinstance(probability_head, Mapping)
        or not isinstance(purchase_head, Mapping)
        or probability_head.get("role")
        != "probability_reporting_and_promotion_calibration"
        or purchase_head.get("role")
        != "purchase_policy_and_chronological_bankroll"
        or probability_head.get("calibrator") != probability
        or purchase_head.get("calibrator") != purchase
        or deployment.get("calibrator") != probability
    ):
        raise ValueError("V20 dual-head provenance is inconsistent")
    if (
        probability.get("converged") is not True
        or purchase.get("converged") is not True
        or int(probability.get("training_races") or 0) <= 0
        or int(purchase.get("training_races") or 0) <= 0
        or operational.get("model_type") != "odds_path_observed_closing_return_v4"
        or not isinstance(operational.get("weights"), Sequence)
        or not isinstance(operational.get("performance_priors"), Mapping)
    ):
        raise ValueError("V20 head models are unsafe or inconsistent")
    control = policy.get("v18_ticket_control")
    if (
        policy.get("no_bet") is True
        or not isinstance(control, Mapping)
        or control.get("method") != "strict_prior_daily_ticket_lower_quantile"
        or int(control.get("learned_daily_ticket_limit") or 0) <= 0
        or int(control.get("stake_granularity_yen") or 0) != 100
        or control.get("result_or_payout_fields_used") is not False
        or selected.get("no_bet") is not True
    ):
        raise ValueError("V20 purchase policy is unsafe or inconsistent")
    return deployment


def _validate_v21(result: Mapping[str, Any]) -> Mapping[str, Any]:
    if result.get("model") != V21_MODEL or result.get("calibrator_strategy") != V21_MODEL:
        raise ValueError("V21 result identity mismatch")
    if result.get("real_betting_enabled") is not False:
        raise ValueError("V21 evaluation must disable real betting")
    deployment = result.get("deployment_configuration")
    if not isinstance(deployment, Mapping) or deployment.get("calibrator_strategy") != V21_MODEL:
        raise ValueError("V21 deployment identity mismatch")
    triple = deployment.get("triple_head_calibration")
    probability = deployment.get("probability_calibrator")
    ranking = deployment.get("ranking_calibrator")
    purchase = deployment.get("purchase_calibrator")
    if (
        deployment.get("deployment_mode") != "evaluation_only"
        or deployment.get("real_betting_enabled") is not False
        or deployment.get("winner_and_logloss_head") != "probability_head"
        or deployment.get("trifecta_top5_head") != "ranking_head"
        or deployment.get("chronological_bankroll_head") != "purchase_head"
        or not all(isinstance(value, Mapping) for value in (triple, probability, ranking, purchase))
        or triple.get("architecture") != "strict_prior_triple_calibrator_heads_v21"
        or triple.get("outer_holdout_used") is not False
        or triple.get("ranking_purchase_share_v18_selection") is not True
    ):
        raise ValueError("V21 triple-head routing is unsafe or inconsistent")
    for name, calibrator in (("probability", probability), ("ranking", ranking), ("purchase", purchase)):
        if calibrator.get("converged") is not True or int(calibrator.get("training_races") or 0) <= 0:
            raise ValueError(f"V21 {name} calibrator is unsafe")
    if deployment.get("selected_policy", {}).get("no_bet") is not True:
        raise ValueError("V21 formal selection must remain no_bet")
    return deployment


def verify_bundle(
    path: Path, *, family: str, through_date: str, prediction_date: str
) -> dict[str, Any]:
    manifest = _json(path.with_suffix(".manifest.json"))
    expected = str((manifest.get("output") or {}).get("bundle_sha256") or "")
    if len(expected) != 64 or _sha256(path) != expected:
        raise ValueError("bundle hash does not match manifest")
    trained_through = _iso(
        manifest.get("trained_through_date"), "trained_through_date"
    )
    if trained_through > through_date or trained_through >= prediction_date:
        raise ValueError("trained-through mismatch")
    if _iso(manifest.get("prediction_date"), "prediction_date") != prediction_date:
        raise ValueError("prediction date mismatch")
    bundle = joblib.load(path)
    deployment = bundle.get("deployment") if isinstance(bundle, Mapping) else None
    if not isinstance(deployment, Mapping):
        raise ValueError("bundle deployment is missing")
    if deployment.get("calibrator_strategy") != _model(family):
        raise ValueError(f"{family} calibrator identity mismatch")
    estimator = (
        deployment.get("closing_t300_v12_model", {})
        .get("point_model", {})
        .get("estimator")
    )
    if estimator is None or not hasattr(estimator, "predict"):
        raise ValueError("live V12 closing estimator is missing")
    return manifest


def build_v12(
    job: CompletedJob, *, through_date: str, prediction_date: str, output_root: Path
) -> dict[str, Any]:
    output = output_root / prediction_date / f"v12-{prediction_date}-job-{job.job_id}.joblib"
    output.parent.mkdir(parents=True, exist_ok=True)
    source_prediction_date = (
        date.fromisoformat(FROZEN_TRAINED_THROUGH_DATE) + timedelta(days=1)
    ).isoformat()
    if (
        job.job_id == FROZEN_SOURCE_JOB_IDS["v12"]
        and prediction_date != source_prediction_date
        and not output.exists()
    ):
        source = (
            output_root
            / source_prediction_date
            / f"v12-{source_prediction_date}-job-{job.job_id}.joblib"
        )
        if not source.exists():
            source.parent.mkdir(parents=True, exist_ok=True)
            v12_shadow_bundle.build_v12_shadow_bundle(
                job.result_path,
                scored_cache=None,
                output=source,
                prediction_date=source_prediction_date,
            )
        source_manifest = verify_bundle(
            source,
            family="v12",
            through_date=FROZEN_TRAINED_THROUGH_DATE,
            prediction_date=source_prediction_date,
        )
        bundle = joblib.load(source)
        deployment = bundle.get("deployment") if isinstance(bundle, dict) else None
        if (
            not isinstance(deployment, dict)
            or bundle.get("prediction_date") != source_prediction_date
            or deployment.get("prediction_date") != source_prediction_date
        ):
            raise ValueError("frozen V12 source prediction identity mismatch")
        rebound = copy.deepcopy(bundle)
        rebound["prediction_date"] = prediction_date
        rebound["deployment"]["prediction_date"] = prediction_date
        manifest = copy.deepcopy(source_manifest)
        manifest.pop("output", None)
        manifest["prediction_date"] = prediction_date
        manifest["trained_through_date"] = FROZEN_TRAINED_THROUGH_DATE
        manifest["runtime_date_rebind"] = {
            "source_bundle": str(source),
            "source_bundle_sha256": _sha256(source),
            "changed_fields": [
                "prediction_date",
                "deployment.prediction_date",
            ],
            "model_identities_unchanged": True,
            "real_betting_enabled": False,
        }
        v12_shadow_bundle._write_bundle_and_manifest_atomic(
            output, rebound, manifest
        )
    elif not output.exists():
        v12_shadow_bundle.build_v12_shadow_bundle(
            job.result_path, scored_cache=None, output=output,
            prediction_date=prediction_date,
        )
    manifest = verify_bundle(
        output, family="v12", through_date=through_date,
        prediction_date=prediction_date,
    )
    return {"path": str(output), "manifest": manifest}


def build_v14_composite(
    job: CompletedJob,
    *,
    v12_path: Path,
    shared_source: Mapping[str, Any],
    through_date: str,
    prediction_date: str,
    output_root: Path,
) -> dict[str, Any]:
    output = output_root / prediction_date / f"v14-{prediction_date}-job-{job.job_id}.joblib"
    if output.exists():
        manifest = verify_bundle(
            output, family="v14", through_date=through_date,
            prediction_date=prediction_date,
        )
        return {"path": str(output), "manifest": manifest}
    bundle = joblib.load(v12_path)
    if not isinstance(bundle, dict) or not isinstance(bundle.get("deployment"), dict):
        raise ValueError("verified V12 bundle is invalid")
    source = _json(job.result_path)
    v14 = _validate_v14(source)
    deployment = copy.deepcopy(bundle["deployment"])
    merged = ("operational_model", "probability_lcb", "selection_conformal", "candidate_policy")
    for key in merged:
        deployment[key] = copy.deepcopy(v14[key])
    for key in ("selected_policy", "operational_status"):
        if key in v14:
            deployment[key] = copy.deepcopy(v14[key])
    deployment["calibrator_strategy"] = V14_MODEL
    deployment["real_betting_enabled"] = False
    bundle["deployment"] = deployment
    bundle["deployment_model_family"] = "v14"
    bundle["source_evaluation_model"] = V14_MODEL
    base_manifest = _json(v12_path.with_suffix(".manifest.json"))
    manifest = copy.deepcopy(base_manifest)
    manifest.pop("output", None)
    manifest.update({
        "deployment_model_family": "v14",
        "prediction_date": prediction_date,
        "trained_through_date": _manifest_trained_through("v14", through_date),
        "composite": {
            "closing_estimator_source": str(v12_path),
            "closing_estimator_source_sha256": _sha256(v12_path),
            "merged_components": list(merged),
            "shared_source_identity": dict(shared_source),
        },
        "source_evaluation": {
            "job_id": job.job_id,
            "model_key": job.model_key,
            "path": str(job.result_path),
            "sha256": _sha256(job.result_path),
            "model": V14_MODEL,
        },
    })
    identities = dict(manifest.get("model_identities") or {})
    identities.update({"integrated_model": V14_MODEL, "calibrator_strategy": V14_MODEL})
    manifest["model_identities"] = identities
    v12_shadow_bundle._write_bundle_and_manifest_atomic(output, bundle, manifest)
    verified = verify_bundle(
        output, family="v14", through_date=through_date,
        prediction_date=prediction_date,
    )
    return {"path": str(output), "manifest": verified}


def build_v16_composite(
    job: CompletedJob,
    *,
    v12_path: Path,
    shared_source: Mapping[str, Any],
    through_date: str,
    prediction_date: str,
    output_root: Path,
) -> dict[str, Any]:
    output = output_root / prediction_date / f"v16-{prediction_date}-job-{job.job_id}.joblib"
    if output.exists():
        manifest = verify_bundle(
            output, family="v16", through_date=through_date,
            prediction_date=prediction_date,
        )
        return {"path": str(output), "manifest": manifest}
    bundle = joblib.load(v12_path)
    if not isinstance(bundle, dict) or not isinstance(bundle.get("deployment"), dict):
        raise ValueError("verified V12 bundle is invalid")
    source = _json(job.result_path)
    v16 = _validate_v16(source)
    deployment = copy.deepcopy(bundle["deployment"])
    verified_v12_closing = deployment["closing_t300_v12_model"]
    merged = (
        "operational_model", "probability_lcb", "closing_envelope_conformal",
        "candidate_policy",
    )
    for key in merged:
        deployment[key] = copy.deepcopy(v16[key])
    for key in ("selected_policy", "operational_status", "missing_real_t300_action"):
        if key in v16:
            deployment[key] = copy.deepcopy(v16[key])
    if deployment["closing_t300_v12_model"] is not verified_v12_closing:
        raise ValueError("verified V12 closing estimator was replaced")
    deployment["calibrator_strategy"] = V16_MODEL
    deployment["real_betting_enabled"] = False
    bundle["deployment"] = deployment
    bundle["deployment_model_family"] = "v16"
    bundle["source_evaluation_model"] = V16_MODEL
    base_manifest = _json(v12_path.with_suffix(".manifest.json"))
    manifest = copy.deepcopy(base_manifest)
    manifest.pop("output", None)
    manifest.update({
        "deployment_model_family": "v16",
        "prediction_date": prediction_date,
        "trained_through_date": _manifest_trained_through("v16", through_date),
        "real_betting_enabled": False,
        "composite": {
            "closing_estimator_source": str(v12_path),
            "closing_estimator_source_sha256": _sha256(v12_path),
            "closing_estimator_policy": {
                "runtime_estimator": "retain_verified_v12_bundle_estimator",
                "source_evaluation_artifact": "validate_only_never_merge",
            },
            "merged_components": list(merged),
            "shared_source_identity": dict(shared_source),
        },
        "source_evaluation": {
            "job_id": job.job_id,
            "model_key": job.model_key,
            "path": str(job.result_path),
            "sha256": _sha256(job.result_path),
            "model": V16_MODEL,
        },
    })
    identities = dict(manifest.get("model_identities") or {})
    identities.update({
        "integrated_model": V16_MODEL,
        "calibrator_strategy": V16_MODEL,
    })
    manifest["model_identities"] = identities
    v12_shadow_bundle._write_bundle_and_manifest_atomic(output, bundle, manifest)
    verified = verify_bundle(
        output, family="v16", through_date=through_date,
        prediction_date=prediction_date,
    )
    return {"path": str(output), "manifest": verified}


def build_v18_composite(
    job: CompletedJob,
    *,
    v12_path: Path,
    shared_source: Mapping[str, Any],
    through_date: str,
    prediction_date: str,
    output_root: Path,
) -> dict[str, Any]:
    if job.job_id != FROZEN_SOURCE_JOB_IDS["v18"]:
        raise ValueError("V18 frozen source must be formal job 8191")
    output = output_root / prediction_date / f"v18-{prediction_date}-job-{job.job_id}.joblib"
    if output.exists():
        manifest = verify_bundle(
            output, family="v18", through_date=through_date,
            prediction_date=prediction_date,
        )
        return {"path": str(output), "manifest": manifest}
    bundle = joblib.load(v12_path)
    if not isinstance(bundle, dict) or not isinstance(bundle.get("deployment"), dict):
        raise ValueError("verified V12 bundle is invalid")
    source = _json(job.result_path)
    v18 = _validate_v18(source)
    deployment = copy.deepcopy(bundle["deployment"])
    merged = (
        "calibrator", "operational_model", "candidate_policy", "selected_policy",
        "closing_odds_selection",
    )
    for key in merged:
        if key in v18:
            deployment[key] = copy.deepcopy(v18[key])
    for key in (
        "deployment_mode", "daily_stake_limit_fraction", "trained_through_date",
        "operational_status", "policy_selection", "primary_promotion_bankroll",
    ):
        if key in v18:
            deployment[key] = copy.deepcopy(v18[key])
    deployment["calibrator_strategy"] = V18_MODEL
    deployment["real_betting_enabled"] = False
    bundle["deployment"] = deployment
    bundle["deployment_model_family"] = "v18"
    bundle["source_evaluation_model"] = V18_MODEL
    base_manifest = _json(v12_path.with_suffix(".manifest.json"))
    manifest = copy.deepcopy(base_manifest)
    manifest.pop("output", None)
    manifest.update({
        "deployment_model_family": "v18",
        "prediction_date": prediction_date,
        "trained_through_date": _manifest_trained_through("v18", through_date),
        "real_betting_enabled": False,
        "composite": {
            "base_probability_source": str(v12_path),
            "base_probability_source_sha256": _sha256(v12_path),
            "runtime_information_boundary": "t300_or_earlier_no_result_no_payout",
            "merged_components": [key for key in merged if key in v18],
            "shared_source_identity": dict(shared_source),
        },
        "source_evaluation": {
            "job_id": job.job_id,
            "model_key": job.model_key,
            "path": str(job.result_path),
            "sha256": _sha256(job.result_path),
            "model": V18_MODEL,
        },
    })
    identities = dict(manifest.get("model_identities") or {})
    identities.update({"integrated_model": V18_MODEL, "calibrator_strategy": V18_MODEL})
    manifest["model_identities"] = identities
    v12_shadow_bundle._write_bundle_and_manifest_atomic(output, bundle, manifest)
    verified = verify_bundle(
        output, family="v18", through_date=through_date,
        prediction_date=prediction_date,
    )
    return {"path": str(output), "manifest": verified}


def build_v20_composite(
    job: CompletedJob,
    *,
    v12_path: Path,
    shared_source: Mapping[str, Any],
    through_date: str,
    prediction_date: str,
    output_root: Path,
) -> dict[str, Any]:
    if job.job_id != FROZEN_SOURCE_JOB_IDS["v20"]:
        raise ValueError("V20 frozen source must be formal job 8458")
    output = output_root / prediction_date / f"v20-{prediction_date}-job-{job.job_id}.joblib"
    if output.exists():
        manifest = verify_bundle(
            output, family="v20", through_date=through_date,
            prediction_date=prediction_date,
        )
        return {"path": str(output), "manifest": manifest}
    bundle = joblib.load(v12_path)
    if not isinstance(bundle, dict) or not isinstance(bundle.get("deployment"), dict):
        raise ValueError("verified V12 bundle is invalid")
    source = _json(job.result_path)
    v20 = _validate_v20(source)
    deployment = copy.deepcopy(bundle["deployment"])
    merged = (
        "calibrator",
        "operational_model",
        "probability_calibrator",
        "purchase_calibrator",
        "dual_head_calibration",
        "candidate_policy",
        "selected_policy",
        "closing_odds_selection",
    )
    for key in merged:
        if key in v20:
            deployment[key] = copy.deepcopy(v20[key])
    for key in (
        "deployment_mode",
        "daily_stake_limit_fraction",
        "trained_through_date",
        "operational_status",
        "policy_selection",
        "primary_promotion_bankroll",
        "probability_metrics_head",
        "chronological_bankroll_head",
        "comparison_role",
    ):
        if key in v20:
            deployment[key] = copy.deepcopy(v20[key])
    deployment["calibrator_strategy"] = V20_MODEL
    deployment["source_evaluation_job_id"] = FROZEN_SOURCE_JOB_IDS["v20"]
    deployment.pop("daily_refit_job_id", None)
    deployment["real_betting_enabled"] = False
    deployment["outer_result_or_payout_used"] = False
    bundle["deployment"] = deployment
    bundle["deployment_model_family"] = "v20"
    bundle["source_evaluation_model"] = V20_MODEL
    base_manifest = _json(v12_path.with_suffix(".manifest.json"))
    manifest = copy.deepcopy(base_manifest)
    manifest.pop("output", None)
    manifest.update({
        "deployment_model_family": "v20",
        "prediction_date": prediction_date,
        "trained_through_date": _manifest_trained_through("v20", through_date),
        "real_betting_enabled": False,
        "composite": {
            "base_probability_source": str(v12_path),
            "base_probability_source_sha256": _sha256(v12_path),
            "runtime_information_boundary": "t300_or_earlier_no_outer_result_no_outer_payout",
            "probability_output_head": "probability_head",
            "purchase_and_bankroll_head": "purchase_head",
            "merged_components": [key for key in merged if key in v20],
            "shared_source_identity": dict(shared_source),
        },
        "source_evaluation": {
            "job_id": job.job_id,
            "model_key": job.model_key,
            "path": str(job.result_path),
            "sha256": _sha256(job.result_path),
            "model": V20_MODEL,
            "probability_head": "probability_calibrator",
            "purchase_head": "purchase_calibrator",
        },
    })
    identities = dict(manifest.get("model_identities") or {})
    identities.update({"integrated_model": V20_MODEL, "calibrator_strategy": V20_MODEL})
    manifest["model_identities"] = identities
    v12_shadow_bundle._write_bundle_and_manifest_atomic(output, bundle, manifest)
    verified = verify_bundle(
        output, family="v20", through_date=through_date,
        prediction_date=prediction_date,
    )
    return {"path": str(output), "manifest": verified}


def build_v21_composite(
    job: CompletedJob,
    *,
    v12_path: Path,
    shared_source: Mapping[str, Any],
    through_date: str,
    prediction_date: str,
    output_root: Path,
) -> dict[str, Any]:
    if job.job_id != FROZEN_SOURCE_JOB_IDS["v21"]:
        raise ValueError("V21 frozen source must be formal job 8666")
    output = output_root / prediction_date / f"v21-{prediction_date}-job-{job.job_id}.joblib"
    if output.exists():
        manifest = verify_bundle(output, family="v21", through_date=through_date, prediction_date=prediction_date)
        return {"path": str(output), "manifest": manifest}
    bundle = joblib.load(v12_path)
    if not isinstance(bundle, dict) or not isinstance(bundle.get("deployment"), dict):
        raise ValueError("verified V12 bundle is invalid")
    source = _json(job.result_path)
    v21 = _validate_v21(source)
    deployment = copy.deepcopy(bundle["deployment"])
    merged = (
        "calibrator", "operational_model", "probability_calibrator",
        "ranking_calibrator", "purchase_calibrator", "triple_head_calibration",
        "candidate_policy", "selected_policy", "closing_odds_selection",
    )
    for key in merged:
        if key in v21:
            deployment[key] = copy.deepcopy(v21[key])
    for key in (
        "deployment_mode", "daily_stake_limit_fraction", "trained_through_date",
        "operational_status", "policy_selection", "primary_promotion_bankroll",
        "winner_and_logloss_head", "trifecta_top5_head",
        "market_logloss_comparison_head", "market_top5_comparison_head",
        "chronological_bankroll_head", "comparison_role",
    ):
        if key in v21:
            deployment[key] = copy.deepcopy(v21[key])
    deployment.update({
        "calibrator_strategy": V21_MODEL,
        "source_evaluation_job_id": FROZEN_SOURCE_JOB_IDS["v21"],
        "real_betting_enabled": False,
        "outer_result_or_payout_used": False,
    })
    bundle.update({
        "deployment": deployment,
        "deployment_model_family": "v21",
        "source_evaluation_model": V21_MODEL,
    })
    manifest = copy.deepcopy(_json(v12_path.with_suffix(".manifest.json")))
    manifest.pop("output", None)
    manifest.update({
        "deployment_model_family": "v21",
        "prediction_date": prediction_date,
        "trained_through_date": _manifest_trained_through("v21", through_date),
        "real_betting_enabled": False,
        "composite": {
            "base_probability_source": str(v12_path),
            "base_probability_source_sha256": _sha256(v12_path),
            "runtime_information_boundary": "t300_or_earlier_no_outer_result_no_outer_payout",
            "probability_output_head": "probability_head",
            "ranking_output_head": "ranking_head",
            "purchase_and_bankroll_head": "purchase_head",
            "merged_components": [key for key in merged if key in v21],
            "shared_source_identity": dict(shared_source),
        },
        "source_evaluation": {
            "job_id": job.job_id, "model_key": job.model_key,
            "path": str(job.result_path), "sha256": _sha256(job.result_path),
            "model": V21_MODEL,
        },
    })
    identities = dict(manifest.get("model_identities") or {})
    identities.update({"integrated_model": V21_MODEL, "calibrator_strategy": V21_MODEL})
    manifest["model_identities"] = identities
    v12_shadow_bundle._write_bundle_and_manifest_atomic(output, bundle, manifest)
    verified = verify_bundle(output, family="v21", through_date=through_date, prediction_date=prediction_date)
    return {"path": str(output), "manifest": verified}


def first_race_start(conn: Any, prediction_date: str) -> datetime | None:
    row = conn.execute(
        "SELECT MIN(deadline_at) AS first_start FROM races WHERE race_date = %s",
        (prediction_date,),
    ).fetchone()
    if row is None or row["first_start"] is None:
        return None
    value = row["first_start"]
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value))
        except ValueError as exc:
            raise ValueError("races.deadline_at must be an ISO datetime") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=JST)
    return parsed.astimezone(JST)


def _active_state(state_root: Path) -> dict[str, Any] | None:
    try:
        return _json((state_root / "active").resolve(strict=True) / "state.json")
    except FileNotFoundError:
        return None


def _temporary(path: Path) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    return Path(name)


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_text(path: Path, value: str) -> None:
    temporary = _temporary(path)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _timestamp(value: object, name: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value))
        except ValueError as exc:
            raise ValueError(f"{name} must be an ISO datetime") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=JST)
    return parsed.astimezone(JST)


def verify_activation_recovery(
    conn: Any, *, state_root: Path, prediction_date: str, now: datetime,
    race_limit: int = RECOVERY_FIRST_RACES,
    max_delay_seconds: float = RECOVERY_MAX_DECISION_DELAY_SECONDS,
) -> dict[str, Any]:
    if race_limit <= 0 or max_delay_seconds <= 0:
        raise ValueError("recovery thresholds must be positive")
    active = _active_state(state_root)
    required = set(RECOVERY_FAMILIES)
    active_identities = dict((active or {}).get("model_identities") or {})
    active_specs = dict((active or {}).get("model_specs") or {})
    runtime_identities = dict(
        (active or {}).get("runtime_model_identities") or {}
    )
    activation_ready = bool(
        active
        and active.get("prediction_date") == prediction_date
        and active.get("real_betting_enabled") is False
        and required.issubset(active_identities)
        and required.issubset(active_specs)
        and required.issubset(runtime_identities)
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "prediction_date": prediction_date,
        "checked_at": now.isoformat(),
        "required_families": list(RECOVERY_FAMILIES),
        "required_first_races": race_limit,
        "maximum_decision_delay_seconds": max_delay_seconds,
        "activation_ready": activation_ready,
        "identity_freeze_preserved": activation_ready,
        "real_betting_enabled": False,
    }
    if not activation_ready:
        payload.update({
            "status": "activation_pending",
            "missing_families": sorted(required - set(active_identities)),
            "gate_pass": False,
        })
    else:
        race_rows = conn.execute(
            """
            SELECT r.race_id,
                   r.deadline_at::timestamptz - INTERVAL '10 minutes'
                     AS target_t300_at
            FROM races r
            WHERE r.race_date = %s AND r.deadline_at IS NOT NULL
              AND (SELECT COUNT(DISTINCT e.lane) FROM entries e
                   WHERE e.race_id = r.race_id) = 6
            ORDER BY r.deadline_at, r.jcd, r.rno, r.race_id
            LIMIT %s
            """,
            (prediction_date, race_limit),
        ).fetchall()
        race_ids = [str(row["race_id"]) for row in race_rows]
        targets = {
            str(row["race_id"]): _timestamp(row["target_t300_at"], "target_t300_at")
            for row in race_rows
        }
        model_keys = {family: active_specs[family].split(":", 1)[0] for family in RECOVERY_FAMILIES}
        decision_rows = []
        if race_ids:
            decision_rows = conn.execute(
                """
                SELECT race_id, model_key, model_hash, strategy_name,
                       decision_at, target_t300_at
                FROM intraday_t300_shadow_decisions
                WHERE race_date = %s AND race_id = ANY(%s) AND model_key = ANY(%s)
                """,
                (prediction_date, race_ids, list(model_keys.values())),
            ).fetchall()
        decisions = {(str(row["race_id"]), str(row["model_key"])): row for row in decision_rows}
        checks = []
        missing_overdue = []
        identity_mismatches = []
        for race_id in race_ids:
            for family in RECOVERY_FAMILIES:
                key = model_keys[family]
                row = decisions.get((race_id, key))
                if row is None:
                    overdue = now >= targets[race_id] + timedelta(seconds=max_delay_seconds)
                    checks.append({
                        "race_id": race_id, "family": family, "model_key": key,
                        "status": "missing_overdue" if overdue else "pending",
                    })
                    if overdue:
                        missing_overdue.append({"race_id": race_id, "family": family})
                    continue
                decision_at = _timestamp(row["decision_at"], "decision_at")
                target = _timestamp(row["target_t300_at"], "target_t300_at")
                delay = max(0.0, (decision_at - target).total_seconds())
                identity_ok = (
                    str(row["model_hash"]) == str(runtime_identities[family])
                    and str(row["strategy_name"]) == active_specs[family].split(":", 2)[1]
                )
                checks.append({
                    "race_id": race_id, "family": family, "model_key": key,
                    "status": "recorded", "decision_delay_seconds": delay,
                    "within_limit": delay < max_delay_seconds,
                    "identity_ok": identity_ok,
                })
                if not identity_ok:
                    identity_mismatches.append({"race_id": race_id, "family": family})
        recorded = [row for row in checks if row["status"] == "recorded"]
        late = [row for row in recorded if not row["within_limit"]]
        complete = len(race_rows) == race_limit and len(recorded) == race_limit * len(RECOVERY_FAMILIES)
        failed = bool(late or missing_overdue or identity_mismatches)
        payload.update({
            "status": "failed" if failed else "passed" if complete else "monitoring",
            "gate_pass": complete and not failed,
            "race_count": len(race_rows),
            "recorded_decisions": len(recorded),
            "expected_decisions": race_limit * len(RECOVERY_FAMILIES),
            "maximum_observed_delay_seconds": max(
                (float(row["decision_delay_seconds"]) for row in recorded), default=None
            ),
            "late_decisions": late,
            "missing_overdue": missing_overdue,
            "identity_mismatches": identity_mismatches,
            "checks": checks,
            "recovery_action": (
                "retain_shadow_real_betting_disabled_and_raise_latency_incident"
                if failed else "continue_automatic_observation"
                if not complete else "recovery_gate_satisfied"
            ),
        })
    state_root.mkdir(parents=True, exist_ok=True)
    _atomic_text(
        state_root / "activation-recovery.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    return payload


def promote(
    *,
    state_root: Path,
    prediction_date: str,
    bundles: Mapping[str, Mapping[str, Any]],
    jobs: Mapping[str, CompletedJob],
    base_model: Path,
    first_start: datetime | None,
    now: datetime,
) -> str:
    identities = {
        family: str(row["manifest"]["output"]["bundle_sha256"])
        for family, row in bundles.items()
    }
    base_sha256 = _sha256(base_model)
    runtime_identities = {
        family: hashlib.sha256(
            (identities[family] + base_sha256).encode("ascii")
        ).hexdigest()
        for family in identities
    }
    specs = {
        "v12": f"v12_daily:{SHADOW_STRATEGY}:{bundles['v12']['path']}:{base_model}",
        "v14": f"v14_daily:{V14_SHADOW_STRATEGY}:{bundles['v14']['path']}:{base_model}",
        "v16": f"v16_daily:{V16_SHADOW_STRATEGY}:{bundles['v16']['path']}:{base_model}",
    }
    if "v18" in bundles:
        specs["v18"] = (
            f"v18_daily:{V18_SHADOW_STRATEGY}:{bundles['v18']['path']}:{base_model}"
        )
    if "v20" in bundles:
        specs["v20"] = (
            f"v20_daily:{V20_SHADOW_STRATEGY}:{bundles['v20']['path']}:{base_model}"
        )
    if "v21" in bundles:
        specs["v21"] = (
            f"v21_daily:{V21_SHADOW_STRATEGY}:{bundles['v21']['path']}:{base_model}"
        )
    active = _active_state(state_root)
    extension: dict[str, Any] | None = None
    extension_status: str | None = None
    if active and active.get("prediction_date") == prediction_date:
        active_identities = dict(active.get("model_identities") or {})
        if active_identities == identities:
            return "already_active"
        preserved = tuple(family for family in BUNDLE_FAMILIES if family in active_identities)
        added = tuple(
            family for family in BUNDLE_FAMILIES
            if family in identities and family not in active_identities
        )
        additive = (
            set(active_identities).issubset(identities)
            and {"v12", "v14"}.issubset(active_identities)
            and bool(added)
            and set(added).issubset({"v16", "v18", "v20", "v21"})
            and all(active_identities[family] == identities[family] for family in preserved)
            and dict(active.get("model_specs") or {})
            == {family: specs[family] for family in preserved}
            and dict(active.get("source_jobs") or {})
            == {family: jobs[family].job_id for family in preserved}
            and active.get("real_betting_enabled") is False
        )
        if not additive:
            return "same_day_identity_frozen"
        prediction = date.fromisoformat(prediction_date)
        if now.date() > prediction or (first_start is not None and now >= first_start):
            return "first_race_boundary_passed"
        label = "_".join(added)
        extension_status = f"additive_{label}_extended"
        extension = {
            "family": added[0] if len(added) == 1 else list(added),
            "reason": (
                "pre_first_race_additive_v16_shadow_extension"
                if added == ("v16",)
                else "pre_first_race_additive_v18_shadow_extension"
                if added == ("v18",)
                else "pre_first_race_additive_v20_shadow_extension"
                if added == ("v20",)
                else "pre_first_race_additive_v21_shadow_extension"
                if added == ("v21",)
                else "pre_first_race_additive_shadow_extension"
            ),
            "extended_at": now.isoformat(),
            "source_job": (
                jobs[added[0]].job_id if len(added) == 1
                else {family: jobs[family].job_id for family in added}
            ),
            "preserved_families": list(preserved),
            "real_betting_enabled": False,
        }
    if active and str(active.get("prediction_date")) > prediction_date:
        return "older_prediction_date_rejected"
    prediction = date.fromisoformat(prediction_date)
    if now.date() > prediction or (first_start is not None and now >= first_start):
        return "first_race_boundary_passed"
    state_root.mkdir(parents=True, exist_ok=True)
    release_name = prediction_date + "-" + hashlib.sha256(
        json.dumps(identities, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    release = state_root / "releases" / release_name
    release.mkdir(parents=True, exist_ok=True)
    env = "\n".join((
        "BOATRACE_T300_SHADOW_MODEL_SPEC=" + shlex.quote(specs["v12"]),
        "BOATRACE_T300_SHADOW_EXTRA_MODEL_SPECS=" + shlex.quote(
            " ".join(specs[family] for family in BUNDLE_FAMILIES if family in specs and family != "v12")
        ),
        "BOATRACE_T300_SHADOW_DATE=" + shlex.quote(prediction_date),
        "BOATRACE_T300_SHADOW_REAL_BETTING_ENABLED=0", "",
    ))
    state = {
        "schema_version": 1,
        "prediction_date": prediction_date,
        "activated_at": (
            str(active.get("activated_at"))
            if extension is not None and active
            else now.isoformat()
        ),
        "first_race_start": first_start.isoformat() if first_start else None,
        "real_betting_enabled": False,
        "model_identities": identities,
        "runtime_model_identities": runtime_identities,
        "model_specs": specs,
        "source_jobs": {family: job.job_id for family, job in jobs.items()},
    }
    if extension is not None:
        state["extensions"] = [*list(active.get("extensions") or []), extension]
    _atomic_text(release / "model-spec.env", env)
    _atomic_text(release / "state.json", json.dumps(state, indent=2, sort_keys=True) + "\n")
    link = state_root / f".active-{os.getpid()}"
    link.unlink(missing_ok=True)
    os.symlink(os.path.relpath(release, state_root), link)
    os.replace(link, state_root / "active")
    _fsync_dir(state_root)
    return extension_status or "activated"


def run_once(
    conn: Any,
    *,
    app_root: Path,
    output_root: Path,
    state_root: Path,
    base_model: Path,
    through_date: str,
    now: datetime,
) -> dict[str, Any]:
    through = _iso(through_date, "through_date")
    prediction = (date.fromisoformat(through) + timedelta(days=1)).isoformat()
    app_root = app_root.resolve()
    output_root = _inside(app_root, output_root, "output_root")
    state_root = _inside(app_root, state_root, "state_root")
    base_model = _inside(app_root, base_model, "base_model")
    if not base_model.is_file():
        raise ValueError(f"base model does not exist: {base_model}")
    active = _active_state(state_root)
    active_families = set((active or {}).get("model_identities") or {})
    preserve_active = bool(
        active is not None
        and active.get("prediction_date") == prediction
        and {"v12", "v14"}.issubset(active_families)
        and active_families.issubset(BUNDLE_FAMILIES)
    )
    jobs: dict[str, CompletedJob] = {}
    if preserve_active:
        source_jobs = active.get("source_jobs")
        if not isinstance(source_jobs, Mapping) or set(source_jobs) != active_families:
            raise ValueError("active shadow source jobs are invalid")
        for family in BUNDLE_FAMILIES:
            if family in FROZEN_SOURCE_JOB_IDS:
                jobs[family] = find_completed_job(
                    conn, job_id=FROZEN_SOURCE_JOB_IDS[family], family=family,
                    through_date=through, app_root=app_root,
                )
            elif family in active_families:
                jobs[family] = find_completed_job(
                    conn, job_id=source_jobs[family], family=family,
                    through_date=through, app_root=app_root,
                )
            else:
                jobs[family] = find_latest_completed_job(
                    conn, family=family, through_date=through, app_root=app_root,
                )
    else:
        jobs = {}
        for family in BUNDLE_FAMILIES:
            if family in FROZEN_SOURCE_JOB_IDS:
                jobs[family] = find_completed_job(
                    conn, job_id=FROZEN_SOURCE_JOB_IDS[family], family=family,
                    through_date=through, app_root=app_root,
                )
            else:
                jobs[family] = find_latest_completed_job(
                    conn, family=family, through_date=through, app_root=app_root
                )
    shared = validate_shared_source(
        jobs["v12"], jobs["v14"], jobs["v16"], jobs["v18"], jobs["v20"],
        jobs["v21"],
    )
    v12 = build_v12(
        jobs["v12"], through_date=through, prediction_date=prediction,
        output_root=output_root,
    )
    v14 = build_v14_composite(
        jobs["v14"], v12_path=Path(v12["path"]), shared_source=shared,
        through_date=through, prediction_date=prediction, output_root=output_root,
    )
    v16 = build_v16_composite(
        jobs["v16"], v12_path=Path(v12["path"]), shared_source=shared,
        through_date=through, prediction_date=prediction, output_root=output_root,
    )
    v18 = build_v18_composite(
        jobs["v18"], v12_path=Path(v12["path"]), shared_source=shared,
        through_date=through, prediction_date=prediction, output_root=output_root,
    )
    v20 = build_v20_composite(
        jobs["v20"], v12_path=Path(v12["path"]), shared_source=shared,
        through_date=through, prediction_date=prediction, output_root=output_root,
    )
    v21 = build_v21_composite(
        jobs["v21"], v12_path=Path(v12["path"]), shared_source=shared,
        through_date=through, prediction_date=prediction, output_root=output_root,
    )
    bundles = {
        "v12": v12, "v14": v14, "v16": v16, "v18": v18,
        "v20": v20, "v21": v21,
    }
    status = promote(
        state_root=state_root, prediction_date=prediction, bundles=bundles,
        jobs=jobs, base_model=base_model,
        first_start=first_race_start(conn, prediction), now=now,
    )
    recovery = verify_activation_recovery(
        conn, state_root=state_root, prediction_date=prediction, now=now,
    )
    return {
        "status": status, "through_date": through, "prediction_date": prediction,
        "jobs": {family: job.job_id for family, job in jobs.items()},
        "bundles": {family: row["path"] for family, row in bundles.items()},
        "real_betting_enabled": False,
        "activation_recovery": recovery,
    }



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Update verified next-day V12/V14/V16/V18/V20/V21 shadow bundles")
    parser.add_argument("--postgres-dsn", required=True)
    parser.add_argument("--app-root", type=Path, default=Path("/workspace/boat"))
    parser.add_argument("--output-root", type=Path, default=Path("/workspace/boat/data/models/daily-shadow-bundles"))
    parser.add_argument("--state-root", type=Path, default=Path("/workspace/boat/data/runtime/daily-shadow-models"))
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--through-date")
    parser.add_argument("--interval-seconds", type=float, default=300.0)
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    while True:
        now = datetime.now(JST)
        through = args.through_date or (now.date() - timedelta(days=1)).isoformat()
        try:
            with psycopg.connect(args.postgres_dsn, row_factory=dict_row) as conn:
                result = run_once(
                    conn, app_root=args.app_root, output_root=args.output_root,
                    state_root=args.state_root, base_model=args.base_model,
                    through_date=through, now=now,
                )
            print(json.dumps(result, sort_keys=True), flush=True)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            prediction = (
                date.fromisoformat(through) + timedelta(days=1)
            ).isoformat()
            recovery = {
                "schema_version": 1,
                "prediction_date": prediction,
                "checked_at": now.isoformat(),
                "required_families": list(RECOVERY_FAMILIES),
                "required_first_races": RECOVERY_FIRST_RACES,
                "maximum_decision_delay_seconds": (
                    RECOVERY_MAX_DECISION_DELAY_SECONDS
                ),
                "status": "activation_blocked_dependency_or_validation",
                "activation_ready": False,
                "identity_freeze_preserved": True,
                "gate_pass": False,
                "error": error,
                "recovery_action": "retry_before_first_race_retain_previous_release",
                "real_betting_enabled": False,
            }
            try:
                args.state_root.mkdir(parents=True, exist_ok=True)
                _atomic_text(
                    args.state_root / "activation-recovery.json",
                    json.dumps(recovery, indent=2, sort_keys=True) + "\n",
                )
            except OSError:
                pass
            print(json.dumps({
                "status": "retained_previous_verified_bundle",
                "through_date": through,
                "error": error,
                "activation_recovery": recovery,
                "real_betting_enabled": False,
            }, sort_keys=True), flush=True)
            if args.once:
                return 1
        if args.once:
            return 0
        time.sleep(max(30.0, args.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
