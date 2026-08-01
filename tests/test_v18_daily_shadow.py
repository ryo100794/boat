from __future__ import annotations

import copy
import hashlib
import json
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
    V18ScheduleQuotaModelAdapter,
    build_adapter,
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
    source = "verified-v12"

    def predict(self, rows):
        return [1.0 for _ in rows]


def deployment() -> dict[str, Any]:
    return {
        "calibrator_strategy": updater.V18_MODEL,
        "deployment_mode": "shadow_only",
        "real_betting_enabled": False,
        "daily_stake_limit_fraction": 1.0,
        "trained_through_date": "2026-07-29",
        "calibrator": {
            "model_weight": 1.0,
            "temperature": 1.0,
            "converged": True,
            "training_races": 1447,
        },
        "operational_model": {
            "model_type": "odds_path_observed_closing_return_v4",
            "weights": [0.0] * 11,
            "performance_priors": {"buckets": {}},
        },
        "selected_policy": {"name": "no_bet", "no_bet": True},
        "candidate_policy": {
            "name": "ev1.50_oddsnone_r1_ratio1.20_kelly_100",
            "ev_threshold": 1.5,
            "max_estimated_ev": None,
            "max_odds": None,
            "max_tickets_per_race": 1,
            "min_model_market_ratio": 1.2,
            "staking_mode": "kelly_100",
            "v18_ticket_control": {
                "method": "strict_prior_daily_ticket_lower_quantile",
                "learned_daily_ticket_limit": 26,
                "stake_granularity_yen": 100,
                "result_or_payout_fields_used": False,
            },
        },
    }


def v12_bundle(path: Path) -> None:
    value = {
        "deployment": {
            "calibrator_strategy": updater.V12_MODEL,
            "closing_t300_v12_model": {
                "model_name": "closing_odds_t300_nonlinear_v12",
                "ready": True,
                "point_model": {"estimator": Estimator()},
            },
        }
    }
    manifest = {
        "prediction_date": "2026-07-30",
        "trained_through_date": "2026-07-29",
        "model_identities": {},
    }
    v12_shadow_bundle._write_bundle_and_manifest_atomic(path, value, manifest)


def test_job8191_components_are_fixed_in_v18_composite(tmp_path: Path) -> None:
    base = tmp_path / "v12.joblib"
    v12_bundle(base)
    result = tmp_path / "job-00008191.json"
    result.write_text(json.dumps({
        "model": updater.V18_MODEL,
        "calibrator_strategy": updater.V18_MODEL,
        "real_betting_enabled": False,
        "deployment_configuration": deployment(),
    }))
    built = updater.build_v18_composite(
        updater.CompletedJob(8191, "v18-job8191", result),
        v12_path=base,
        shared_source={"source_model_sha256": "a" * 64},
        through_date="2026-07-29",
        prediction_date="2026-07-30",
        output_root=tmp_path,
    )
    merged = joblib.load(built["path"])["deployment"]
    assert merged["calibrator"] == deployment()["calibrator"]
    assert merged["operational_model"] == deployment()["operational_model"]
    assert merged["candidate_policy"] == deployment()["candidate_policy"]
    assert merged["selected_policy"] == {"name": "no_bet", "no_bet": True}
    assert merged["real_betting_enabled"] is False
    assert merged["closing_t300_v12_model"]["point_model"]["estimator"].source == "verified-v12"
    assert built["manifest"]["source_evaluation"]["job_id"] == 8191
    assert built["manifest"]["composite"]["runtime_information_boundary"] == (
        "t300_or_earlier_no_result_no_payout"
    )


def adapter(tmp_path: Path) -> V18ScheduleQuotaModelAdapter:
    bundle = tmp_path / "v18.joblib"
    base = tmp_path / "base.joblib"
    joblib.dump({"deployment": deployment()}, bundle)
    joblib.dump({"feature_schema_version": 1}, base)
    return V18ScheduleQuotaModelAdapter(
        model_key="v18_daily", bundle_path=bundle, base_model_path=base
    )


def race_snapshot() -> tuple[RaceWindow, T300Snapshot]:
    start = datetime(2026, 7, 30, 12, tzinfo=JST)
    race = RaceWindow("20260730-01-01", "2026-07-30", "01", 1, start)
    captured = race.target_t300_at - timedelta(seconds=5)
    return race, T300Snapshot(
        8191,
        captured,
        captured.isoformat(),
        {},
        {combination: 100.0 for combination in COMBINATIONS},
    )


def test_v18_decision_uses_only_t300_features_and_fixed_quota(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = adapter(tmp_path)
    race, snapshot = race_snapshot()
    probabilities = {
        combination: (0.02 if index == 0 else 0.98 / 119)
        for index, combination in enumerate(COMBINATIONS)
    }
    monkeypatch.setattr(model, "_base_probabilities", lambda conn, race: probabilities)
    monkeypatch.setattr(model, "_runtime_limits", lambda conn, race, bankroll_yen: {
        "schedule_races_elapsed": 10,
        "schedule_races_total": 100,
        "cumulative_ticket_quota": 2,
        "used_tickets": 1,
        "remaining_ticket_quota": 1,
        "gross_stake_yen": 100,
        "realized_cumulative_profit_yen": 0,
        "gross_stake_allowance_yen": 10_000,
        "remaining_gross_stake_allowance_yen": 9_900,
        "allocatable_bankroll_yen": 9_900,
    })
    seen: list[dict[str, Any]] = []

    def attach(rows, operational):
        seen.append(copy.deepcopy(rows[0]))
        value = copy.deepcopy(rows[0])
        value["model_probabilities"] = probabilities
        value["historical_return_multipliers"] = {
            combination: 1.0 for combination in COMBINATIONS
        }
        return [value]

    monkeypatch.setattr(
        "boatrace_ai.runtime.intraday_t300_shadow.attach_odds_path_model", attach
    )
    monkeypatch.setattr(
        "boatrace_ai.runtime.intraday_t300_shadow.allocate_adaptive_day",
        lambda day, candidates, races, **kwargs: {
            "selected_sample": [{**candidates[0], "stake_yen": 100, "hit": False, "return_yen": 0}],
            "allocation_candidate_tickets": len(candidates),
        },
    )
    first = model.decide(
        {"result": "1-2-3", "payout": 999_999}, race, snapshot,
        bankroll_yen=10_000,
    )
    second = model.decide(
        {"result": "6-5-4", "payout": 0}, race, snapshot,
        bankroll_yen=10_000,
    )
    assert first == second
    assert first.status == "selected"
    assert first.selected_candidates[0]["combination"] == COMBINATIONS[0]
    assert first.selected_candidates[0]["odds_source"] == "real_t300_job8191_v18"
    diagnostic = first.diagnostics["v18_schedule_quota"]
    assert diagnostic["learned_daily_ticket_limit"] == 26
    assert diagnostic["uses_result_as_model_feature"] is False
    assert diagnostic["uses_payout_as_model_feature"] is False
    assert diagnostic["real_betting_enabled"] is False
    assert all("actual_combination" not in row and "actual_payout_yen" not in row for row in seen)
    assert all(row["odds_path_points"] == 1 for row in seen)


def test_v18_runtime_applies_min_raw_ev_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured = deployment()
    configured["candidate_policy"]["min_raw_ev"] = 1.1
    bundle = tmp_path / "v18-raw-guard.joblib"
    base = tmp_path / "base-raw-guard.joblib"
    joblib.dump({"deployment": configured}, bundle)
    joblib.dump({"feature_schema_version": 1}, base)
    model = V18ScheduleQuotaModelAdapter(
        model_key="v18_raw_guard", bundle_path=bundle, base_model_path=base
    )
    race, source = race_snapshot()
    snapshot = T300Snapshot(
        source.snapshot_id, source.captured_at, source.source_update_time, {},
        {combination: 50.0 for combination in COMBINATIONS},
    )
    probabilities = {
        combination: (0.02 if index == 0 else 0.98 / 119)
        for index, combination in enumerate(COMBINATIONS)
    }
    monkeypatch.setattr(model, "_base_probabilities", lambda conn, row: probabilities)
    monkeypatch.setattr(
        "boatrace_ai.runtime.intraday_t300_shadow.attach_odds_path_model",
        lambda rows, operational: [{
            **copy.deepcopy(rows[0]),
            "model_probabilities": probabilities,
            "historical_return_multipliers": {
                combination: 2.0 for combination in COMBINATIONS
            },
        }],
    )
    monkeypatch.setattr(model, "_runtime_limits", lambda conn, row, bankroll_yen: {
        "schedule_races_elapsed": 100,
        "schedule_races_total": 100,
        "cumulative_ticket_quota": 1,
        "used_tickets": 0,
        "remaining_ticket_quota": 1,
        "observed_candidate_scores": [],
        "gross_stake_yen": 0,
        "realized_cumulative_profit_yen": 0,
        "gross_stake_allowance_yen": 10_000,
        "remaining_gross_stake_allowance_yen": 10_000,
        "allocatable_bankroll_yen": 10_000,
    })

    decision = model.decide(object(), race, snapshot, bankroll_yen=10_000)

    assert decision.selected_candidates == ()
    assert decision.no_bet_reason == "no_safe_ev_threshold_candidate"


def test_v18_registry_and_quota_zero_are_no_bet(tmp_path: Path, monkeypatch) -> None:
    model = adapter(tmp_path)
    bundle = tmp_path / "v18.joblib"
    base = tmp_path / "base.joblib"
    built = build_adapter(f"v18_daily:v18_schedule_quota_t300:{bundle}:{base}")
    assert isinstance(built, V18ScheduleQuotaModelAdapter)
    race, snapshot = race_snapshot()
    probabilities = {
        combination: (0.02 if index == 0 else 0.98 / 119)
        for index, combination in enumerate(COMBINATIONS)
    }
    monkeypatch.setattr(model, "_base_probabilities", lambda conn, race: probabilities)
    monkeypatch.setattr(
        "boatrace_ai.runtime.intraday_t300_shadow.attach_odds_path_model",
        lambda rows, operational: [{
            **copy.deepcopy(rows[0]),
            "model_probabilities": probabilities,
            "historical_return_multipliers": {
                combination: 1.0 for combination in COMBINATIONS
            },
        }],
    )
    monkeypatch.setattr(model, "_runtime_limits", lambda conn, race, bankroll_yen: {
        "schedule_races_elapsed": 1,
        "schedule_races_total": 100,
        "cumulative_ticket_quota": 0,
        "used_tickets": 0,
        "remaining_ticket_quota": 0,
        "observed_candidate_scores": [],
        "gross_stake_yen": 0,
        "realized_cumulative_profit_yen": 0,
        "gross_stake_allowance_yen": 10_000,
        "remaining_gross_stake_allowance_yen": 10_000,
        "allocatable_bankroll_yen": 10_000,
    })
    decision = model.decide(object(), race, snapshot, bankroll_yen=10_000)
    assert decision.no_bet_reason == "v18_schedule_ticket_quota_not_released"
    assert len(decision.probabilities) == 120
    assert decision.closing_lower_odds == snapshot.odds
    assert decision.selected_candidates == ()
    diagnostic = decision.diagnostics["v18_schedule_quota"]
    assert diagnostic["allocation_candidates"] == 0
    assert diagnostic["real_betting_enabled"] is False


def test_additive_v18_extension_preserves_existing_three_identities(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    release = state_root / "releases" / "existing"
    release.mkdir(parents=True)
    base = tmp_path / "base.joblib"
    base.write_bytes(b"base")
    bundles = {
        family: {
            "path": str(tmp_path / f"{family}.joblib"),
            "manifest": {"output": {"bundle_sha256": hashlib.sha256(family.encode()).hexdigest()}},
        }
        for family in updater.ALL_FAMILIES
    }
    jobs = {
        family: updater.CompletedJob(number, family, tmp_path / f"{family}.json")
        for family, number in (("v12", 12), ("v14", 14), ("v16", 16), ("v18", 8191))
    }
    specs = {
        "v12": f"v12_daily:{updater.SHADOW_STRATEGY}:{bundles['v12']['path']}:{base}",
        "v14": f"v14_daily:{updater.V14_SHADOW_STRATEGY}:{bundles['v14']['path']}:{base}",
        "v16": f"v16_daily:{updater.V16_SHADOW_STRATEGY}:{bundles['v16']['path']}:{base}",
    }
    existing = {
        "prediction_date": "2026-07-30",
        "activated_at": "2026-07-30T07:00:00+09:00",
        "real_betting_enabled": False,
        "model_identities": {
            family: bundles[family]["manifest"]["output"]["bundle_sha256"]
            for family in updater.FAMILIES
        },
        "model_specs": specs,
        "source_jobs": {family: jobs[family].job_id for family in updater.FAMILIES},
    }
    (release / "state.json").write_text(json.dumps(existing))
    (release / "model-spec.env").write_text("existing")
    (state_root / "active").symlink_to(release.relative_to(state_root))
    now = datetime(2026, 7, 30, 8, tzinfo=JST)
    status = updater.promote(
        state_root=state_root,
        prediction_date="2026-07-30",
        bundles=bundles,
        jobs=jobs,
        base_model=base,
        first_start=now + timedelta(hours=1),
        now=now,
    )
    assert status == "additive_v18_extended"
    state = json.loads(((state_root / "active").resolve() / "state.json").read_text())
    for family in updater.FAMILIES:
        assert state["model_identities"][family] == existing["model_identities"][family]
        assert state["model_specs"][family] == existing["model_specs"][family]
        assert state["source_jobs"][family] == existing["source_jobs"][family]
    assert state["source_jobs"]["v18"] == 8191
    assert state["real_betting_enabled"] is False
    assert ":v18_schedule_quota_t300:" in (
        (state_root / "active").resolve() / "model-spec.env"
    ).read_text()
