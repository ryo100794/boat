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
SHADOW_STRATEGY = v12_shadow_bundle.STRATEGY_NAME
V14_SHADOW_STRATEGY = "v14_registered_band_t300"
V16_SHADOW_STRATEGY = "v16_fixed_band_t300"
V18_SHADOW_STRATEGY = "v18_schedule_quota_t300"
FAMILIES = ("v12", "v14", "v16")
ALL_FAMILIES = (*FAMILIES, "v18")


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
    raise ValueError(f"unsupported family: {family}")


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
        ORDER BY completed_at DESC NULLS LAST, job_id DESC LIMIT 1
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
    if (
        row["status"] != "completed"
        or not row["result_path"]
        or row["through_date"] != expected_through
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
) -> dict[str, Any]:
    left = _source_identity(_json(v12_job.result_path), v12_job.result_path)
    keys = tuple(key for key in left if key != "result_path")
    for label, job in (("V14", v14_job), ("V16", v16_job), ("V18", v18_job)):
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


def verify_bundle(
    path: Path, *, family: str, through_date: str, prediction_date: str
) -> dict[str, Any]:
    manifest = _json(path.with_suffix(".manifest.json"))
    expected = str((manifest.get("output") or {}).get("bundle_sha256") or "")
    if len(expected) != 64 or _sha256(path) != expected:
        raise ValueError("bundle hash does not match manifest")
    if _iso(manifest.get("trained_through_date"), "trained_through_date") != through_date:
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
    if not output.exists():
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
        "trained_through_date": through_date,
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
        "trained_through_date": through_date,
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
        "trained_through_date": through_date,
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
    specs = {
        "v12": f"v12_daily:{SHADOW_STRATEGY}:{bundles['v12']['path']}:{base_model}",
        "v14": f"v14_daily:{V14_SHADOW_STRATEGY}:{bundles['v14']['path']}:{base_model}",
        "v16": f"v16_daily:{V16_SHADOW_STRATEGY}:{bundles['v16']['path']}:{base_model}",
    }
    if "v18" in bundles:
        specs["v18"] = (
            f"v18_daily:{V18_SHADOW_STRATEGY}:{bundles['v18']['path']}:{base_model}"
        )
    active = _active_state(state_root)
    extension: dict[str, Any] | None = None
    extension_status: str | None = None
    if active and active.get("prediction_date") == prediction_date:
        active_identities = dict(active.get("model_identities") or {})
        if active_identities == identities:
            return "already_active"
        preserved = tuple(family for family in ALL_FAMILIES if family in active_identities)
        added = tuple(
            family for family in ALL_FAMILIES
            if family in identities and family not in active_identities
        )
        additive = (
            set(active_identities).issubset(identities)
            and {"v12", "v14"}.issubset(active_identities)
            and bool(added)
            and set(added).issubset({"v16", "v18"})
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
            " ".join(specs[family] for family in ALL_FAMILIES if family in specs and family != "v12")
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
        and active_families.issubset(ALL_FAMILIES)
    )
    jobs: dict[str, CompletedJob] = {}
    if preserve_active:
        source_jobs = active.get("source_jobs")
        if not isinstance(source_jobs, Mapping) or set(source_jobs) != active_families:
            raise ValueError("active shadow source jobs are invalid")
        for family in ALL_FAMILIES:
            if family in active_families:
                jobs[family] = find_completed_job(
                    conn, job_id=source_jobs[family], family=family,
                    through_date=through, app_root=app_root,
                )
            else:
                jobs[family] = find_latest_completed_job(
                    conn, family=family, through_date=through, app_root=app_root,
                )
    else:
        jobs = {
            family: find_latest_completed_job(
                conn, family=family, through_date=through, app_root=app_root
            )
            for family in ALL_FAMILIES
        }
    shared = validate_shared_source(
        jobs["v12"], jobs["v14"], jobs["v16"], jobs["v18"]
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
    bundles = {"v12": v12, "v14": v14, "v16": v16, "v18": v18}
    status = promote(
        state_root=state_root, prediction_date=prediction, bundles=bundles,
        jobs=jobs, base_model=base_model,
        first_start=first_race_start(conn, prediction), now=now,
    )
    return {
        "status": status, "through_date": through, "prediction_date": prediction,
        "jobs": {family: job.job_id for family, job in jobs.items()},
        "bundles": {family: row["path"] for family, row in bundles.items()},
        "real_betting_enabled": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Update verified next-day V12/V14/V16/V18 shadow bundles")
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
            print(json.dumps({
                "status": "retained_previous_verified_bundle",
                "through_date": through,
                "error": f"{type(exc).__name__}: {exc}",
                "real_betting_enabled": False,
            }, sort_keys=True), flush=True)
            if args.once:
                return 1
        if args.once:
            return 0
        time.sleep(max(30.0, args.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
