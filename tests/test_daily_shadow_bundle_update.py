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
        "closing_model_artifact_audit": {
            "live_estimator_in_report": False,
            "report_only": True,
        },
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
        "deployment_configuration": deployment,
    }


def job(path: Path, job_id: int, family: str) -> updater.CompletedJob:
    return updater.CompletedJob(job_id, family + "-job", path)


def make_v12_bundle(path: Path) -> None:
    bundle = {
        "prediction_date": "2026-07-30",
        "deployment": {
            "prediction_date": "2026-07-30",
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


def test_frozen_v12_rebinds_only_runtime_prediction_date(tmp_path: Path) -> None:
    output_root = tmp_path / "bundles"
    source = (
        output_root
        / "2026-07-30"
        / "v12-2026-07-30-job-7401.joblib"
    )
    source.parent.mkdir(parents=True)
    make_v12_bundle(source)
    result = tmp_path / "job-00007401.json"
    result.write_text("{}", encoding="utf-8")

    built = updater.build_v12(
        updater.CompletedJob(7401, "v12-frozen", result),
        through_date="2026-07-30",
        prediction_date="2026-07-31",
        output_root=output_root,
    )

    rebound_path = Path(built["path"])
    original = joblib.load(source)
    rebound = joblib.load(rebound_path)
    assert rebound["prediction_date"] == "2026-07-31"
    assert rebound["deployment"]["prediction_date"] == "2026-07-31"
    original["prediction_date"] = rebound["prediction_date"]
    original["deployment"]["prediction_date"] = rebound["deployment"]["prediction_date"]
    assert joblib.hash(rebound) == joblib.hash(original)
    manifest = built["manifest"]
    assert manifest["trained_through_date"] == "2026-07-29"
    assert manifest["model_identities"] == {
        "integrated_model": updater.V12_MODEL,
        "calibrator_strategy": updater.V12_MODEL,
    }
    assert manifest["runtime_date_rebind"] == {
        "source_bundle": str(source),
        "source_bundle_sha256": updater._sha256(source),
        "changed_fields": [
            "prediction_date",
            "deployment.prediction_date",
        ],
        "model_identities_unchanged": True,
        "real_betting_enabled": False,
    }


class PriorityJobConnection:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.query = ""
        self.parameters: tuple[str, str] | None = None

    def execute(self, query: str, parameters: tuple[str, str]):
        self.query = " ".join(query.split())
        self.parameters = parameters
        return self

    def fetchone(self) -> dict:
        assert self.query.index("priority DESC NULLS LAST") < self.query.index(
            "completed_at DESC NULLS LAST"
        )
        return max(
            self.rows,
            key=lambda row: (
                int(row["priority"]),
                str(row["completed_at"]),
                int(row["job_id"]),
            ),
        )


def test_latest_completed_v18_prefers_canonical_priority_over_completion_time(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "job-00008181.json"
    canonical = tmp_path / "job-00008191.json"
    duplicate.write_text("{}", encoding="utf-8")
    canonical.write_text("{}", encoding="utf-8")
    conn = PriorityJobConnection([
        {
            "job_id": 8181,
            "model_key": "v18-duplicate",
            "result_path": str(duplicate),
            "priority": 100,
            "completed_at": "2026-07-30T01:00:00+00:00",
        },
        {
            "job_id": 8191,
            "model_key": "v18-canonical",
            "result_path": str(canonical),
            "priority": 115,
            "completed_at": "2026-07-29T23:00:00+00:00",
        },
    ])

    selected = updater.find_latest_completed_job(
        conn,
        family="v18",
        through_date="2026-07-29",
        app_root=tmp_path,
    )

    assert selected.job_id == 8191
    assert selected.model_key == "v18-canonical"
    assert selected.result_path == canonical
    assert conn.parameters == ("2026-07-29", updater.V18_MODEL)
    assert conn.query.endswith(
        "priority DESC NULLS LAST, completed_at DESC NULLS LAST, "
        "job_id DESC LIMIT 1"
    )


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
    deployment["closing_model_artifact_audit"] = {
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


def test_first_race_start_treats_postgresql_naive_text_as_jst() -> None:
    parsed = updater.first_race_start(
        FirstRaceConnection("2026-07-29T10:38:00"), "2026-07-29"
    )
    assert parsed == datetime(
        2026, 7, 29, 10, 38, tzinfo=timezone(timedelta(hours=9))
    )
    assert parsed is not None and parsed.utcoffset() == timedelta(hours=9)


def test_first_race_start_converts_offset_text_to_jst() -> None:
    parsed = updater.first_race_start(
        FirstRaceConnection("2026-07-29T01:38:00+00:00"), "2026-07-29"
    )
    assert parsed == datetime(
        2026, 7, 29, 10, 38, tzinfo=timezone(timedelta(hours=9))
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


class RecoveryConnection:
    def __init__(self, races, decisions):
        self.races = races
        self.decisions = decisions

    def execute(self, query, parameters):
        self.query = " ".join(query.split())
        self.parameters = parameters
        return self

    def fetchall(self):
        if "FROM races r" in self.query:
            return self.races[: int(self.parameters[1])]
        if "FROM intraday_t300_shadow_decisions" in self.query:
            race_ids = set(self.parameters[1])
            model_keys = set(self.parameters[2])
            return [
                row for row in self.decisions
                if row["race_id"] in race_ids and row["model_key"] in model_keys
            ]
        raise AssertionError(self.query)


def write_recovery_active(state_root: Path, base: Path) -> tuple[dict, dict]:
    release = state_root / "releases" / "next-day"
    release.mkdir(parents=True)
    specs = {
        "v18": f"v18_daily:{updater.V18_SHADOW_STRATEGY}:v18.joblib:{base}",
        "v20": f"v20_daily:{updater.V20_SHADOW_STRATEGY}:v20.joblib:{base}",
        "v21": f"v21_daily:{updater.V21_SHADOW_STRATEGY}:v21.joblib:{base}",
    }
    hashes = {family: hashlib.sha256(family.encode()).hexdigest() for family in specs}
    state = {
        "prediction_date": "2026-07-31",
        "real_betting_enabled": False,
        "model_identities": hashes,
        "runtime_model_identities": hashes,
        "model_specs": specs,
    }
    (release / "state.json").write_text(json.dumps(state))
    (release / "model-spec.env").write_text("BOATRACE_T300_SHADOW_REAL_BETTING_ENABLED=0\n")
    (state_root / "active").symlink_to(release.relative_to(state_root))
    return specs, hashes


def test_recovery_gate_requires_first_five_for_v18_v20_v21_below_90_seconds(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "state"
    base = tmp_path / "base.joblib"
    base.write_bytes(b"base")
    specs, hashes = write_recovery_active(state_root, base)
    target = datetime(2026, 7, 31, 1, 0, tzinfo=timezone.utc)
    races = [
        {"race_id": f"race-{index}", "target_t300_at": target + timedelta(minutes=index)}
        for index in range(5)
    ]
    decisions = []
    for race in races:
        for family in updater.RECOVERY_FAMILIES:
            decisions.append({
                "race_id": race["race_id"],
                "model_key": specs[family].split(":", 1)[0],
                "model_hash": hashes[family],
                "strategy_name": specs[family].split(":", 2)[1],
                "target_t300_at": race["target_t300_at"],
                "decision_at": race["target_t300_at"] + timedelta(seconds=89.9),
            })
    result = updater.verify_activation_recovery(
        RecoveryConnection(races, decisions),
        state_root=state_root, prediction_date="2026-07-31",
        now=target + timedelta(minutes=10),
    )
    assert result["status"] == "passed"
    assert result["gate_pass"] is True
    assert result["recorded_decisions"] == 15
    assert result["maximum_observed_delay_seconds"] == pytest.approx(89.9)
    assert result["real_betting_enabled"] is False
    persisted = json.loads((state_root / "activation-recovery.json").read_text())
    assert persisted["status"] == "passed"


def test_recovery_gate_fails_at_90_seconds_and_on_missing_overdue(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "state"
    base = tmp_path / "base.joblib"
    base.write_bytes(b"base")
    specs, hashes = write_recovery_active(state_root, base)
    target = datetime(2026, 7, 31, 1, 0, tzinfo=timezone.utc)
    races = [
        {"race_id": f"race-{index}", "target_t300_at": target + timedelta(minutes=index)}
        for index in range(5)
    ]
    decisions = [{
        "race_id": "race-0", "model_key": "v18_daily",
        "model_hash": hashes["v18"],
        "strategy_name": updater.V18_SHADOW_STRATEGY,
        "target_t300_at": target,
        "decision_at": target + timedelta(seconds=90),
    }]
    result = updater.verify_activation_recovery(
        RecoveryConnection(races, decisions),
        state_root=state_root, prediction_date="2026-07-31",
        now=target + timedelta(minutes=10),
    )
    assert result["status"] == "failed"
    assert result["gate_pass"] is False
    assert len(result["late_decisions"]) == 1
    assert result["missing_overdue"]
    assert result["recovery_action"] == (
        "retain_shadow_real_betting_disabled_and_raise_latency_incident"
    )


def test_shadow_runner_uses_intraday_module_with_static_v21_registration() -> None:
    root = Path(__file__).resolve().parents[1]
    runner = (
        root / "scripts" / "deployment" / "run-boatrace-intraday-t300-shadow.sh"
    ).read_text()
    assert "-m boatrace_ai.runtime.intraday_t300_shadow" in runner
    assert "register_v21_shadow_adapter" not in runner


def test_main_persists_dependency_failure_and_retries_without_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_root = tmp_path / "state"
    base = tmp_path / "base.joblib"
    base.write_bytes(b"base")

    def unavailable(*args, **kwargs):
        raise LookupError("V21 daily refit is pending")

    monkeypatch.setattr(updater.psycopg, "connect", unavailable)
    result = updater.main([
        "--postgres-dsn", "host=unused",
        "--app-root", str(tmp_path),
        "--output-root", str(tmp_path / "bundles"),
        "--state-root", str(state_root),
        "--base-model", str(base),
        "--through-date", "2026-07-30",
        "--once",
    ])
    assert result == 1
    recovery = json.loads((state_root / "activation-recovery.json").read_text())
    assert recovery["prediction_date"] == "2026-07-31"
    assert recovery["status"] == "activation_blocked_dependency_or_validation"
    assert recovery["identity_freeze_preserved"] is True
    assert recovery["real_betting_enabled"] is False
    assert not (state_root / "active").exists()
