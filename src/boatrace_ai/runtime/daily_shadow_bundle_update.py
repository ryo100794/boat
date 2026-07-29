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
SHADOW_STRATEGY = v12_shadow_bundle.STRATEGY_NAME
FAMILIES = ("v12", "v14")


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


def validate_shared_source(v12_job: CompletedJob, v14_job: CompletedJob) -> dict[str, Any]:
    left = _source_identity(_json(v12_job.result_path), v12_job.result_path)
    right = _source_identity(_json(v14_job.result_path), v14_job.result_path)
    keys = tuple(key for key in left if key != "result_path")
    mismatches = [key for key in keys if left[key] != right[key]]
    if mismatches:
        raise ValueError("V12/V14 source mismatch: " + ", ".join(mismatches))
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
    closing = deployment.get("closing_t300_v12_model")
    point = closing.get("point_model") if isinstance(closing, Mapping) else None
    if not isinstance(point, Mapping):
        raise ValueError("V14 report-only closing metadata is missing")
    if point.get("estimator") is not None:
        raise ValueError("V14 result must not contain a live closing estimator")
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


def first_race_start(conn: Any, prediction_date: str) -> datetime | None:
    row = conn.execute(
        "SELECT MIN(deadline_at) AS first_start FROM races WHERE race_date = %s",
        (prediction_date,),
    ).fetchone()
    return None if row is None else row["first_start"]


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
    active = _active_state(state_root)
    if active and active.get("prediction_date") == prediction_date:
        return "already_active" if active.get("model_identities") == identities else "same_day_identity_frozen"
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
    specs = {
        family: f"{family}_daily:{SHADOW_STRATEGY}:{row['path']}:{base_model}"
        for family, row in bundles.items()
    }
    env = "\n".join((
        "BOATRACE_T300_SHADOW_MODEL_SPEC=" + shlex.quote(specs["v12"]),
        "BOATRACE_T300_SHADOW_EXTRA_MODEL_SPECS=" + shlex.quote(specs["v14"]),
        "BOATRACE_T300_SHADOW_DATE=" + shlex.quote(prediction_date),
        "BOATRACE_T300_SHADOW_REAL_BETTING_ENABLED=0", "",
    ))
    state = {
        "schema_version": 1,
        "prediction_date": prediction_date,
        "activated_at": now.isoformat(),
        "first_race_start": first_start.isoformat() if first_start else None,
        "real_betting_enabled": False,
        "model_identities": identities,
        "model_specs": specs,
        "source_jobs": {family: job.job_id for family, job in jobs.items()},
    }
    _atomic_text(release / "model-spec.env", env)
    _atomic_text(release / "state.json", json.dumps(state, indent=2, sort_keys=True) + "\n")
    link = state_root / f".active-{os.getpid()}"
    link.unlink(missing_ok=True)
    os.symlink(os.path.relpath(release, state_root), link)
    os.replace(link, state_root / "active")
    _fsync_dir(state_root)
    return "activated"


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
    jobs = {
        family: find_latest_completed_job(
            conn, family=family, through_date=through, app_root=app_root
        ) for family in FAMILIES
    }
    shared = validate_shared_source(jobs["v12"], jobs["v14"])
    v12 = build_v12(
        jobs["v12"], through_date=through, prediction_date=prediction,
        output_root=output_root,
    )
    v14 = build_v14_composite(
        jobs["v14"], v12_path=Path(v12["path"]), shared_source=shared,
        through_date=through, prediction_date=prediction, output_root=output_root,
    )
    bundles = {"v12": v12, "v14": v14}
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
    parser = argparse.ArgumentParser(description="Update verified next-day V12/V14 shadow bundles")
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
