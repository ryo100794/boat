from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import pytest

from boatrace_ai.runtime import daily_shadow_bundle_update as updater
from boatrace_ai.runtime import v12_shadow_bundle
from boatrace_ai.runtime.intraday_t300_shadow import (
    RaceWindow,
    T300Snapshot,
    V16FixedBandModelAdapter,
)


JST = timezone(timedelta(hours=9))
COMBINATIONS = [
    f"{first}-{second}-{third}"
    for first in range(1, 7)
    for second in range(1, 7)
    if second != first
    for third in range(1, 7)
    if third not in (first, second)
]


class Estimator:
    source = "verified-v12-bundle"

    def predict(self, rows):
        return [1.0 for _ in rows]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _job(path: Path, job_id: int, family: str) -> updater.CompletedJob:
    return updater.CompletedJob(job_id, family + "-job", path)


def _v12_bundle(path: Path) -> None:
    bundle = {
        "deployment": {
            "calibrator_strategy": updater.V12_MODEL,
            "operational_model": {"trained_through_date": "2026-07-29"},
            "probability_lcb": {"ready": True},
            "selection_conformal": {"ready": True},
            "candidate_policy": {},
            "closing_t300_v12_model": {
                "model_name": "closing_odds_t300_nonlinear_v12",
                "ready": True,
                "trained_through_date": "2026-07-29",
                "point_model": {"estimator": Estimator()},
            },
        }
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


def _v16_deployment(*, envelope_ready: bool = True) -> dict[str, Any]:
    return {
        "calibrator_strategy": updater.V16_MODEL,
        "operational_model": {"trained_through_date": "2026-07-29"},
        "probability_lcb": {
            "model_name": "strict_prior_t300_divergence_passthrough_v16",
            "artifact_method": "fixed_t300_divergence_raw_probability_passthrough_v16",
            "fixed_filter": True,
            "raw_probability_passthrough": True,
            "uses_result": False,
            "uses_payout": False,
            "trained_through_date": "2026-07-29",
            "registered_divergence_lower_inclusive": 0.5,
            "registered_divergence_upper_exclusive": 1.0,
        },
        "closing_envelope_conformal": {
            "model_name": "closing_envelope_conformal_v15",
            "method": "selection_free_strict_prior_daily_q20_closing_ratio_v15",
            "selection_free": True,
            "ready": envelope_ready,
            "haircut": 0.9 if envelope_ready else None,
            "trained_through_date": "2026-07-29",
        },
        "candidate_policy": {
            "registered_divergence_lower_inclusive": 0.5,
            "registered_divergence_upper_exclusive": 1.0,
            "raw_model_probability_inside_fixed_band": True,
            "real_betting_enabled": False,
        },
        "missing_real_t300_action": "no_bet",
        "real_betting_enabled": False,
        "closing_t300_v12_model": {
            "model_name": "closing_odds_t300_nonlinear_v12",
            "boundary_audit": {"future_checkpoint_imputation": False},
            "point_model": {"source": "evaluation-artifact-must-not-merge"},
        },
    }


def test_v16_completed_evaluation_is_composed_with_v12_live_estimator(
    tmp_path: Path,
) -> None:
    v12_path = tmp_path / "v12.joblib"
    _v12_bundle(v12_path)
    result = tmp_path / "v16.json"
    _write_json(result, {
        "model": updater.V16_MODEL,
        "calibrator_strategy": updater.V16_MODEL,
        "real_betting_enabled": False,
        "deployment_configuration": _v16_deployment(),
    })

    built = updater.build_v16_composite(
        _job(result, 7600, "v16"),
        v12_path=v12_path,
        shared_source={"source_model_sha256": "a" * 64},
        through_date="2026-07-29",
        prediction_date="2026-07-30",
        output_root=tmp_path,
    )

    bundle = joblib.load(built["path"])
    deployment = bundle["deployment"]
    assert deployment["calibrator_strategy"] == updater.V16_MODEL
    estimator = deployment["closing_t300_v12_model"]["point_model"]["estimator"]
    assert hasattr(estimator, "predict")
    assert estimator.source == "verified-v12-bundle"
    assert "source" not in deployment["closing_t300_v12_model"]["point_model"]
    assert deployment["real_betting_enabled"] is False
    assert built["manifest"]["source_evaluation"]["job_id"] == 7600
    assert built["manifest"]["composite"]["merged_components"] == [
        "operational_model",
        "probability_lcb",
        "closing_envelope_conformal",
        "candidate_policy",
    ]
    assert built["manifest"]["composite"]["closing_estimator_policy"] == {
        "runtime_estimator": "retain_verified_v12_bundle_estimator",
        "source_evaluation_artifact": "validate_only_never_merge",
    }


def _bundle_rows(tmp_path: Path) -> dict[str, dict[str, Any]]:
    return {
        family: {
            "path": str(tmp_path / f"{family}.joblib"),
            "manifest": {
                "output": {
                    "bundle_sha256": hashlib.sha256(family.encode()).hexdigest()
                }
            },
        }
        for family in updater.FAMILIES
    }


def _jobs(tmp_path: Path) -> dict[str, updater.CompletedJob]:
    return {
        family: _job(tmp_path / f"{family}.json", number, family)
        for family, number in (("v12", 12), ("v14", 14), ("v16", 16))
    }


def _legacy_active(
    state_root: Path,
    *,
    bundles: dict[str, dict[str, Any]],
    jobs: dict[str, updater.CompletedJob],
    base_model: Path,
    activated_at: datetime,
) -> Path:
    release = state_root / "releases" / "legacy-v12-v14"
    release.mkdir(parents=True)
    specs = {
        "v12": (
            f"v12_daily:{updater.SHADOW_STRATEGY}:"
            f"{bundles['v12']['path']}:{base_model}"
        ),
        "v14": (
            f"v14_daily:{updater.V14_SHADOW_STRATEGY}:"
            f"{bundles['v14']['path']}:{base_model}"
        ),
    }
    state = {
        "schema_version": 1,
        "prediction_date": "2026-07-30",
        "activated_at": activated_at.isoformat(),
        "real_betting_enabled": False,
        "model_identities": {
            family: bundles[family]["manifest"]["output"]["bundle_sha256"]
            for family in ("v12", "v14")
        },
        "model_specs": specs,
        "source_jobs": {family: jobs[family].job_id for family in ("v12", "v14")},
    }
    (release / "state.json").write_text(json.dumps(state), encoding="utf-8")
    (release / "model-spec.env").write_text("legacy", encoding="utf-8")
    os.symlink(os.path.relpath(release, state_root), state_root / "active")
    return release


def test_pre_first_race_additive_v16_extension_is_audited_and_preserves_identity(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "state"
    state_root.mkdir()
    base = tmp_path / "base.joblib"
    base.write_bytes(b"base")
    bundles = _bundle_rows(tmp_path)
    jobs = _jobs(tmp_path)
    now = datetime(2026, 7, 30, 8, tzinfo=JST)
    old_release = _legacy_active(
        state_root,
        bundles=bundles,
        jobs=jobs,
        base_model=base,
        activated_at=now - timedelta(hours=1),
    )
    old_state = json.loads((old_release / "state.json").read_text())

    status = updater.promote(
        state_root=state_root,
        prediction_date="2026-07-30",
        bundles=bundles,
        jobs=jobs,
        base_model=base,
        first_start=now + timedelta(hours=1),
        now=now,
    )

    assert status == "additive_v16_extended"
    active = (state_root / "active").resolve()
    state = json.loads((active / "state.json").read_text())
    assert {
        family: state["model_identities"][family] for family in ("v12", "v14")
    } == old_state["model_identities"]
    assert {
        family: state["model_specs"][family] for family in ("v12", "v14")
    } == old_state["model_specs"]
    assert {
        family: state["source_jobs"][family] for family in ("v12", "v14")
    } == old_state["source_jobs"]
    assert state["real_betting_enabled"] is False
    assert state["extensions"][-1] == {
        "family": "v16",
        "reason": "pre_first_race_additive_v16_shadow_extension",
        "extended_at": now.isoformat(),
        "source_job": 16,
        "preserved_families": ["v12", "v14"],
        "real_betting_enabled": False,
    }
    env = (active / "model-spec.env").read_text()
    assert env.count(":v12_role_t300:") == 1
    assert env.count(":v14_registered_band_t300:") == 1
    assert env.count(":v16_fixed_band_t300:") == 1
    assert "REAL_BETTING_ENABLED=0" in env


def test_run_once_pins_active_v12_v14_jobs_when_newer_jobs_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_root = tmp_path / "state"
    state_root.mkdir()
    output_root = tmp_path / "bundles"
    output_root.mkdir()
    base = tmp_path / "base.joblib"
    base.write_bytes(b"base")
    bundles = _bundle_rows(tmp_path)
    active_jobs = _jobs(tmp_path)
    now = datetime(2026, 7, 30, 8, tzinfo=JST)
    _legacy_active(
        state_root,
        bundles=bundles,
        jobs=active_jobs,
        base_model=base,
        activated_at=now - timedelta(hours=1),
    )
    fixed_calls: list[tuple[str, int]] = []
    latest_calls: list[str] = []

    def fixed_job(conn, *, job_id, family, through_date, app_root):
        fixed_calls.append((family, job_id))
        return active_jobs[family]

    def latest_job(conn, *, family, through_date, app_root):
        latest_calls.append(family)
        if family in ("v12", "v14"):
            return _job(tmp_path / f"new-{family}.json", 1000, family)
        return active_jobs["v16"]

    monkeypatch.setattr(updater, "find_completed_job", fixed_job)
    monkeypatch.setattr(updater, "find_latest_completed_job", latest_job)
    monkeypatch.setattr(updater, "validate_shared_source", lambda *args: {})
    monkeypatch.setattr(
        updater, "build_v12", lambda *args, **kwargs: bundles["v12"]
    )
    monkeypatch.setattr(
        updater, "build_v14_composite", lambda *args, **kwargs: bundles["v14"]
    )
    monkeypatch.setattr(
        updater, "build_v16_composite", lambda *args, **kwargs: bundles["v16"]
    )
    monkeypatch.setattr(
        updater, "first_race_start", lambda *args: now + timedelta(hours=1)
    )

    result = updater.run_once(
        object(),
        app_root=tmp_path,
        output_root=output_root,
        state_root=state_root,
        base_model=base,
        through_date="2026-07-29",
        now=now,
    )

    assert result["status"] == "additive_v16_extended"
    assert fixed_calls == [("v12", 12), ("v14", 14)]
    assert latest_calls == ["v16"]
    assert result["jobs"] == {"v12": 12, "v14": 14, "v16": 16}


def test_additive_extension_rejects_existing_identity_change_and_post_start(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "state"
    state_root.mkdir()
    base = tmp_path / "base.joblib"
    base.write_bytes(b"base")
    bundles = _bundle_rows(tmp_path)
    jobs = _jobs(tmp_path)
    now = datetime(2026, 7, 30, 8, tzinfo=JST)
    original = _legacy_active(
        state_root,
        bundles=bundles,
        jobs=jobs,
        base_model=base,
        activated_at=now - timedelta(hours=1),
    )

    changed = copy.deepcopy(bundles)
    changed["v12"]["manifest"]["output"]["bundle_sha256"] = "f" * 64
    assert updater.promote(
        state_root=state_root,
        prediction_date="2026-07-30",
        bundles=changed,
        jobs=jobs,
        base_model=base,
        first_start=now + timedelta(hours=1),
        now=now,
    ) == "same_day_identity_frozen"
    assert (state_root / "active").resolve() == original

    assert updater.promote(
        state_root=state_root,
        prediction_date="2026-07-30",
        bundles=bundles,
        jobs=jobs,
        base_model=base,
        first_start=now - timedelta(seconds=1),
        now=now,
    ) == "first_race_boundary_passed"
    assert (state_root / "active").resolve() == original


def _adapter_artifacts(
    tmp_path: Path, *, envelope_ready: bool = True,
) -> V16FixedBandModelAdapter:
    tmp_path.mkdir(parents=True, exist_ok=True)
    bundle_path = tmp_path / "v16-runtime.joblib"
    base_path = tmp_path / "base.joblib"
    deployment = _v16_deployment(envelope_ready=envelope_ready)
    deployment["closing_t300_v12_model"] = {
        "model_name": "closing_odds_t300_nonlinear_v12",
        "ready": True,
        "trained_through_date": "2026-07-29",
        "point_model": {"estimator": Estimator()},
    }
    joblib.dump({"deployment": deployment}, bundle_path)
    joblib.dump({"feature_schema_version": 1}, base_path)
    return V16FixedBandModelAdapter(
        model_key="v16-jul30",
        bundle_path=bundle_path,
        base_model_path=base_path,
    )


def _race_and_snapshot() -> tuple[RaceWindow, T300Snapshot]:
    start = datetime(2026, 7, 30, 12, tzinfo=JST)
    race = RaceWindow("20260730-01-01", "2026-07-30", "01", 1, start)
    captured = race.target_t300_at - timedelta(seconds=10)
    snapshot = T300Snapshot(
        300,
        captured,
        (captured - timedelta(seconds=5)).isoformat(),
        {},
        {combination: 10.0 for combination in COMBINATIONS},
    )
    return race, snapshot


def _probabilities(divergence: float) -> dict[str, float]:
    target_probability = (1.0 / 120.0) * math.exp(divergence)
    remainder = (1.0 - target_probability) / 119.0
    return {
        combination: target_probability if index == 0 else remainder
        for index, combination in enumerate(COMBINATIONS)
    }


def _wire_decision(
    monkeypatch: pytest.MonkeyPatch,
    adapter: V16FixedBandModelAdapter,
    *,
    divergence: float,
    seen: list[dict[str, Any]],
) -> None:
    probabilities = _probabilities(divergence)
    monkeypatch.setattr(adapter, "_base_probabilities", lambda conn, race: probabilities)

    def attach(races: list[dict[str, Any]], model: dict[str, Any]):
        seen.append(copy.deepcopy(races[0]))
        row = copy.deepcopy(races[0])
        row["model_probabilities"] = probabilities
        return [row]

    monkeypatch.setattr(
        "boatrace_ai.runtime.intraday_t300_shadow.attach_odds_path_probability_v8",
        attach,
    )
    monkeypatch.setattr(
        "boatrace_ai.runtime.intraday_t300_shadow.forecast_closing_odds_t300_nonlinear_v12",
        lambda race, model, prediction_date: {
            "ready": True,
            "future_checkpoint_offsets_used": [],
            "point_final_odds": {
                combination: 100.0 for combination in COMBINATIONS
            },
        },
    )
    monkeypatch.setattr(
        "boatrace_ai.runtime.intraday_t300_shadow.allocate_discrete_log_day",
        lambda day, candidates, race_ids, **kwargs: {
            "selected_sample": [
                {**candidate, "stake_yen": 100} for candidate in candidates
            ],
            "allocation_candidate_tickets": len(candidates),
        },
    )


def test_v16_decision_uses_fixed_band_raw_probability_and_is_result_invariant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter_artifacts(tmp_path)
    race, snapshot = _race_and_snapshot()
    seen: list[dict[str, Any]] = []
    _wire_decision(monkeypatch, adapter, divergence=0.7, seen=seen)

    first = adapter.decide(
        {"result": "1-2-3", "payout": 100000},
        race,
        snapshot,
        bankroll_yen=10_000,
    )
    second = adapter.decide(
        {"result": "6-5-4", "payout": 0},
        race,
        snapshot,
        bankroll_yen=10_000,
    )

    assert first == second
    assert first.status == "selected"
    assert first.selected_candidates[0]["combination"] == COMBINATIONS[0]
    assert first.selected_candidates[0]["probability_source"] == "v8_raw_probability"
    assert first.selected_candidates[0]["estimated_odds"] == 90.0
    assert first.diagnostics["v16_fixed_band"]["source_snapshot_id"] == 300
    assert first.diagnostics["v16_fixed_band"]["uses_result"] is False
    assert first.diagnostics["v16_fixed_band"]["uses_payout"] is False
    assert all("result" not in payload and "payout" not in payload for payload in seen)
    assert all(payload["snapshot_id"] == 300 for payload in seen)


def test_v16_outside_band_and_missing_envelope_are_no_bet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    race, snapshot = _race_and_snapshot()
    adapter = _adapter_artifacts(tmp_path)
    seen: list[dict[str, Any]] = []
    _wire_decision(monkeypatch, adapter, divergence=0.4, seen=seen)
    outside = adapter.decide(object(), race, snapshot, bankroll_yen=10_000)
    assert outside.status == "no_bet"
    assert outside.diagnostics["v16_fixed_band"]["registered_combination_count"] == 0

    missing = _adapter_artifacts(tmp_path / "missing", envelope_ready=False)
    assert missing.decide(
        object(), race, snapshot, bankroll_yen=10_000
    ).no_bet_reason == "closing_envelope_not_ready"

