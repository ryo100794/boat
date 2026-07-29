from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import joblib
import pytest

from boatrace_ai.runtime import daily_shadow_bundle_update as updater
from boatrace_ai.runtime import v12_shadow_bundle


class Estimator:
    def predict(self, rows):
        return [1.0 for _ in rows]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def result_payload(cache: Path, *, model: str) -> dict:
    deployment = {
        "calibrator_strategy": model,
        "operational_model": {"name": model + "-operational"},
        "probability_lcb": {"method": model + "-lcb", "ready": True},
        "selection_conformal": {"method": model + "-selection", "ready": True},
        "candidate_policy": {"name": model + "-candidate"},
    }
    return {
        "model": model,
        "calibrator_strategy": model,
        "source_model_sha256": "a" * 64,
        "source_model_trained_through": {"winner": "2026-07-01"},
        "from_date": "2026-07-20",
        "through_date": "2026-07-29",
        "odds_data_signature": {"rows": 120},
        "scored_cache": str(cache),
        "closing_model_identity": {
            "model_name": "closing_odds_t300_nonlinear_v12",
            "selected_model": "closing_odds_t300_nonlinear_v12",
            "source_model_sha256": "a" * 64,
        },
        "closing_model_audit": {
            "prediction_cutoff": "T-300",
            "live_estimator_in_report": False,
        },
        "deployment_configuration": deployment,
    }


def job(path: Path, job_id: int, family: str) -> updater.CompletedJob:
    return updater.CompletedJob(job_id, family + "-job", path)


def make_v12_bundle(path: Path) -> None:
    bundle = {
        "prediction_date": "2026-07-30",
        "deployment": {
            "calibrator_strategy": updater.V12_MODEL,
            "operational_model": {"name": "v12-operational"},
            "probability_lcb": {"method": "v12-lcb"},
            "selection_conformal": {"method": "v12-selection"},
            "candidate_policy": {"name": "v12-candidate"},
            "closing_t300_v12_model": {
                "point_model": {"estimator": Estimator()},
            },
        },
    }
    manifest = {
        "prediction_date": "2026-07-30",
        "trained_through_date": "2026-07-29",
        "model_identities": {
            "integrated_model": updater.V12_MODEL,
            "calibrator_strategy": updater.V12_MODEL,
        },
    }
    v12_shadow_bundle._write_bundle_and_manifest_atomic(path, bundle, manifest)


def test_shared_source_requires_same_hash_date_and_scored_cache(tmp_path: Path) -> None:
    cache = tmp_path / "cache.joblib"
    cache.write_bytes(b"same-cache")
    v12_path = tmp_path / "v12.json"
    v14_path = tmp_path / "v14.json"
    write_json(v12_path, result_payload(cache, model=updater.V12_MODEL))
    write_json(v14_path, result_payload(cache, model=updater.V14_MODEL))

    identity = updater.validate_shared_source(
        job(v12_path, 12, "v12"), job(v14_path, 14, "v14")
    )
    assert identity["scored_cache_sha256"] == hashlib.sha256(b"same-cache").hexdigest()

    changed = result_payload(cache, model=updater.V14_MODEL)
    changed["source_model_sha256"] = "b" * 64
    write_json(v14_path, changed)
    with pytest.raises(ValueError, match="source_model_sha256"):
        updater.validate_shared_source(
            job(v12_path, 12, "v12"), job(v14_path, 14, "v14")
        )


def test_v14_composite_keeps_live_v12_estimator_and_merges_roles(tmp_path: Path) -> None:
    cache = tmp_path / "cache.joblib"
    cache.write_bytes(b"cache")
    v14_path = tmp_path / "v14.json"
    write_json(v14_path, result_payload(cache, model=updater.V14_MODEL))
    v12_path = tmp_path / "v12.joblib"
    make_v12_bundle(v12_path)

    built = updater.build_v14_composite(
        job(v14_path, 7396, "v14"),
        v12_path=v12_path,
        shared_source={"source_model_sha256": "a" * 64},
        through_date="2026-07-29",
        prediction_date="2026-07-30",
        output_root=tmp_path,
    )

    bundle = joblib.load(built["path"])
    deployment = bundle["deployment"]
    assert hasattr(
        deployment["closing_t300_v12_model"]["point_model"]["estimator"],
        "predict",
    )
    assert deployment["operational_model"]["name"].startswith(updater.V14_MODEL)
    assert deployment["probability_lcb"]["method"].startswith(updater.V14_MODEL)
    assert deployment["selection_conformal"]["method"].startswith(updater.V14_MODEL)
    assert deployment["candidate_policy"]["name"].startswith(updater.V14_MODEL)
    assert deployment["real_betting_enabled"] is False
    assert built["manifest"]["composite"]["merged_components"] == [
        "operational_model", "probability_lcb", "selection_conformal", "candidate_policy"
    ]


def test_v14_job_7396_shape_uses_v12_live_closing_estimator(tmp_path: Path) -> None:
    cache = tmp_path / "job-7396-cache.joblib"
    cache.write_bytes(b"job-7396-cache")
    payload = result_payload(cache, model=updater.V14_MODEL)
    deployment = payload["deployment_configuration"]
    assert "closing_t300_v12_model" not in deployment
    payload["closing_model_identity"] = {
        "model_name": "closing_odds_t300_nonlinear_v12",
        "selected_model": "closing_odds_t300_nonlinear_v12",
        "trained_through_date": "2026-07-29",
        "source_model_sha256": "a" * 64,
    }
    payload["closing_model_audit"] = {
        "live_estimator_in_report": False,
        "report_only": True,
        "snapshot_target": "T-300",
    }
    result = tmp_path / "job-00007396.json"
    write_json(result, payload)
    v12_path = tmp_path / "v12-live.joblib"
    make_v12_bundle(v12_path)

    built = updater.build_v14_composite(
        job(result, 7396, "v14"),
        v12_path=v12_path,
        shared_source={"source_model_sha256": "a" * 64},
        through_date="2026-07-29",
        prediction_date="2026-07-30",
        output_root=tmp_path,
    )

    merged = joblib.load(built["path"])["deployment"]
    assert hasattr(
        merged["closing_t300_v12_model"]["point_model"]["estimator"],
        "predict",
    )
    assert merged["calibrator_strategy"] == updater.V14_MODEL


def bundle_rows(tmp_path: Path, suffix: str) -> dict[str, dict]:
    return {
        family: {
            "path": str(tmp_path / f"{family}-{suffix}.joblib"),
            "manifest": {"output": {"bundle_sha256": hashlib.sha256(f"{family}:{suffix}".encode()).hexdigest()}},
        }
        for family in updater.FAMILIES
    }


def test_promotion_is_atomic_shadow_only_and_freezes_same_day(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    base = tmp_path / "base.joblib"
    base.write_bytes(b"base")
    jobs = {
        "v12": job(tmp_path / "v12.json", 12, "v12"),
        "v14": job(tmp_path / "v14.json", 14, "v14"),
    }
    now = datetime(2026, 7, 29, 20, tzinfo=timezone(timedelta(hours=9)))
    status = updater.promote(
        state_root=state_root, prediction_date="2026-07-30",
        bundles=bundle_rows(tmp_path, "a"), jobs=jobs, base_model=base,
        first_start=now + timedelta(hours=12), now=now,
    )
    assert status == "activated"
    active_target = (state_root / "active").resolve()
    env = (active_target / "model-spec.env").read_text()
    assert "REAL_BETTING_ENABLED=0" in env
    assert ":v12_role_t300:" in env
    assert ":v14_registered_band_t300:" in env
    assert (active_target / "model-spec.env").stat().st_mode & 0o777 == 0o600

    status = updater.promote(
        state_root=state_root, prediction_date="2026-07-30",
        bundles=bundle_rows(tmp_path, "changed"), jobs=jobs, base_model=base,
        first_start=now + timedelta(hours=12), now=now,
    )
    assert status == "same_day_identity_frozen"
    assert (state_root / "active").resolve() == active_target


def test_promotion_after_first_race_retains_previous_release(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    state_root.mkdir()
    base = tmp_path / "base.joblib"
    base.write_bytes(b"base")
    now = datetime(2026, 7, 30, 9, tzinfo=timezone(timedelta(hours=9)))
    status = updater.promote(
        state_root=state_root, prediction_date="2026-07-30",
        bundles=bundle_rows(tmp_path, "late"),
        jobs={
            "v12": job(tmp_path / "v12.json", 12, "v12"),
            "v14": job(tmp_path / "v14.json", 14, "v14"),
        },
        base_model=base, first_start=now - timedelta(minutes=1), now=now,
    )
    assert status == "first_race_boundary_passed"
    assert not (state_root / "active").exists()


class FirstRaceConnection:
    def __init__(self, value: object):
        self.value = value

    def execute(self, _query: str, _parameters: tuple[str]):
        return self

    def fetchone(self) -> dict[str, object]:
        return {"first_start": self.value}


def test_first_race_start_parses_postgresql_text_jst_iso() -> None:
    parsed = updater.first_race_start(
        FirstRaceConnection("2026-07-30T09:15:00+09:00"), "2026-07-30"
    )
    assert parsed == datetime(
        2026, 7, 30, 9, 15, tzinfo=timezone(timedelta(hours=9))
    )
    assert parsed is not None and parsed.utcoffset() == timedelta(hours=9)


def test_first_race_start_rejects_naive_text() -> None:
    with pytest.raises(ValueError, match="timezone offset"):
        updater.first_race_start(
            FirstRaceConnection("2026-07-30T09:15:00"), "2026-07-30"
        )


def test_deployment_scripts_are_opt_in_and_shadow_only() -> None:
    root = Path(__file__).resolve().parents[1]
    supervisor = (
        root / "scripts/deployment/supervisor-boatrace-daily-shadow-bundles.ini"
    ).read_text()
    wrapper = (
        root / "scripts/deployment/run-boatrace-intraday-t300-daily-bundles.sh"
    ).read_text()
    assert supervisor.count("autostart=false") == 2
    assert "REAL_BETTING_ENABLED" in wrapper
    assert "run-boatrace-intraday-t300-shadow.sh" in wrapper
    assert "kill -TERM" in wrapper
    assert "wait \"$child_pid\"" in wrapper
    assert "sha256sum" in wrapper
    assert "startretries=20" not in supervisor
