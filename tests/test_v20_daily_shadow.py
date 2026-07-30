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
    V20DualHeadModelAdapter,
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


def calibrator(*, model_weight: float, temperature: float = 1.0) -> dict[str, Any]:
    return {
        "model_weight": model_weight,
        "temperature": temperature,
        "converged": True,
        "training_races": 1447,
    }


def policy() -> dict[str, Any]:
    return {
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
    }


def operational_model() -> dict[str, Any]:
    return {
        "model_type": "odds_path_observed_closing_return_v4",
        "weights": [0.0] * 11,
        "performance_priors": {"buckets": {}},
    }


def v20_deployment() -> dict[str, Any]:
    probability = calibrator(model_weight=1.0)
    purchase = calibrator(model_weight=0.8)
    return {
        "calibrator_strategy": updater.V20_MODEL,
        "comparison_role": "strict_prior_dual_head_probability_v19_purchase_v18_evaluation_only",
        "deployment_mode": "evaluation_only",
        "real_betting_enabled": False,
        "outer_result_or_payout_used": False,
        "daily_stake_limit_fraction": 1.0,
        "trained_through_date": "2026-07-29",
        "source_evaluation_job_id": 8458,
        "probability_metrics_head": "probability_head",
        "chronological_bankroll_head": "purchase_head",
        "calibrator": copy.deepcopy(probability),
        "probability_calibrator": probability,
        "purchase_calibrator": purchase,
        "dual_head_calibration": {
            "architecture": "strict_prior_dual_calibrator_heads_v20",
            "selection_data": "strict_prior_training_and_inner_prequential_folds_only",
            "outer_holdout_used": False,
            "probability_head": {
                "role": "probability_reporting_and_promotion_calibration",
                "calibrator": copy.deepcopy(probability),
            },
            "purchase_head": {
                "role": "purchase_policy_and_chronological_bankroll",
                "calibrator": copy.deepcopy(purchase),
            },
        },
        "operational_model": operational_model(),
        "candidate_policy": policy(),
        "selected_policy": {"name": "no_bet", "no_bet": True},
    }


def v18_deployment() -> dict[str, Any]:
    return {
        "calibrator_strategy": updater.V18_MODEL,
        "deployment_mode": "shadow_only",
        "real_betting_enabled": False,
        "daily_stake_limit_fraction": 1.0,
        "trained_through_date": "2026-07-29",
        "calibrator": calibrator(model_weight=0.8),
        "operational_model": operational_model(),
        "candidate_policy": policy(),
        "selected_policy": {"name": "no_bet", "no_bet": True},
    }


def write_v12_bundle(path: Path) -> None:
    bundle = {
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
    v12_shadow_bundle._write_bundle_and_manifest_atomic(path, bundle, manifest)


def test_v20_bundle_rejects_noncanonical_source_job(tmp_path: Path) -> None:
    base = tmp_path / "v12.joblib"
    write_v12_bundle(base)
    with pytest.raises(ValueError, match="formal job 8458"):
        updater.build_v20_composite(
            updater.CompletedJob(8448, "duplicate", tmp_path / "duplicate.json"),
            v12_path=base,
            shared_source={},
            through_date="2026-07-29",
            prediction_date="2026-07-30",
            output_root=tmp_path,
        )


def test_job8458_dual_heads_and_provenance_are_fixed_in_bundle(tmp_path: Path) -> None:
    base = tmp_path / "v12.joblib"
    write_v12_bundle(base)
    result = tmp_path / "job-00008458.json"
    source_deployment = v20_deployment()
    source_deployment.pop("source_evaluation_job_id")
    source_deployment.pop("outer_result_or_payout_used")
    result.write_text(json.dumps({
        "model": updater.V20_MODEL,
        "calibrator_strategy": updater.V20_MODEL,
        "real_betting_enabled": False,
        "deployment_configuration": source_deployment,
    }))
    built = updater.build_v20_composite(
        updater.CompletedJob(8458, "v20-job8458", result),
        v12_path=base,
        shared_source={"source_model_sha256": "a" * 64},
        through_date="2026-07-29",
        prediction_date="2026-07-30",
        output_root=tmp_path,
    )
    merged = joblib.load(built["path"])["deployment"]
    assert merged["probability_calibrator"] == v20_deployment()["probability_calibrator"]
    assert merged["purchase_calibrator"] == v20_deployment()["purchase_calibrator"]
    assert merged["probability_metrics_head"] == "probability_head"
    assert merged["chronological_bankroll_head"] == "purchase_head"
    assert merged["source_evaluation_job_id"] == 8458
    assert merged["outer_result_or_payout_used"] is False
    assert merged["real_betting_enabled"] is False
    assert built["manifest"]["source_evaluation"]["job_id"] == 8458
    assert built["manifest"]["source_evaluation"]["probability_head"] == (
        "probability_calibrator"
    )
    assert built["manifest"]["source_evaluation"]["purchase_head"] == (
        "purchase_calibrator"
    )


class PriorityConnection:
    def __init__(self, rows):
        self.rows = rows

    def execute(self, query, parameters):
        self.query = " ".join(query.split())
        self.parameters = parameters
        return self

    def fetchone(self):
        return max(
            self.rows,
            key=lambda row: (
                int(row["priority"]),
                str(row["completed_at"]),
                int(row["job_id"]),
            ),
        )


def test_v20_source_job_selection_prefers_formal_job8458(tmp_path: Path) -> None:
    duplicate = tmp_path / "job-00008448.json"
    canonical = tmp_path / "job-00008458.json"
    duplicate.write_text("{}")
    canonical.write_text("{}")
    conn = PriorityConnection([
        {
            "job_id": 8448,
            "model_key": "duplicate",
            "result_path": str(duplicate),
            "priority": 100,
            "completed_at": "2026-07-30T02:00:00+00:00",
        },
        {
            "job_id": 8458,
            "model_key": "canonical",
            "result_path": str(canonical),
            "priority": 116,
            "completed_at": "2026-07-30T01:00:00+00:00",
        },
    ])
    selected = updater.find_latest_completed_job(
        conn,
        family="v20",
        through_date="2026-07-29",
        app_root=tmp_path,
    )
    assert selected.job_id == 8458
    assert conn.parameters == ("2026-07-29", updater.V20_MODEL)
    assert conn.query.index("priority DESC") < conn.query.index("completed_at DESC")


def adapters(
    tmp_path: Path,
) -> tuple[V18ScheduleQuotaModelAdapter, V20DualHeadModelAdapter]:
    base = tmp_path / "base.joblib"
    v18_bundle = tmp_path / "v18.joblib"
    v20_bundle = tmp_path / "v20.joblib"
    joblib.dump({"feature_schema_version": 1}, base)
    joblib.dump({"deployment": v18_deployment()}, v18_bundle)
    joblib.dump({"deployment": v20_deployment()}, v20_bundle)
    return (
        V18ScheduleQuotaModelAdapter(
            model_key="v18_daily", bundle_path=v18_bundle, base_model_path=base
        ),
        V20DualHeadModelAdapter(
            model_key="v20_daily", bundle_path=v20_bundle, base_model_path=base
        ),
    )


def race_snapshot() -> tuple[RaceWindow, T300Snapshot]:
    start = datetime(2026, 7, 30, 12, tzinfo=JST)
    race = RaceWindow("20260730-01-01", "2026-07-30", "01", 1, start)
    captured = race.target_t300_at - timedelta(seconds=5)
    return race, T300Snapshot(
        8458,
        captured,
        captured.isoformat(),
        {},
        {combination: 100.0 for combination in COMBINATIONS},
    )


def test_v20_routes_probability_and_purchase_heads_without_outer_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    v18, v20 = adapters(tmp_path)
    race, snapshot = race_snapshot()
    raw = {
        combination: (0.02 if index == 0 else 0.98 / 119)
        for index, combination in enumerate(COMBINATIONS)
    }
    limits = {
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
    }
    for model in (v18, v20):
        monkeypatch.setattr(model, "_base_probabilities", lambda conn, race: raw)
        monkeypatch.setattr(
            model,
            "_runtime_limits",
            lambda conn, race, bankroll_yen: dict(limits),
        )

    def attach(rows, operational):
        item = copy.deepcopy(rows[0])
        item["model_probabilities"] = raw
        item["historical_return_multipliers"] = {
            combination: 1.0 for combination in COMBINATIONS
        }
        assert "actual_combination" not in item
        assert "actual_payout_yen" not in item
        return [item]

    monkeypatch.setattr(
        "boatrace_ai.runtime.intraday_t300_shadow.attach_odds_path_model", attach
    )
    monkeypatch.setattr(
        "boatrace_ai.runtime.intraday_t300_shadow.allocate_adaptive_day",
        lambda day, candidates, races, **kwargs: {
            "selected_sample": [
                {**candidates[0], "stake_yen": 100, "hit": False, "return_yen": 0}
            ],
            "allocation_candidate_tickets": len(candidates),
        },
    )
    v18_decision = v18.decide(object(), race, snapshot, bankroll_yen=10_000)
    first = v20.decide(
        {"result": "1-2-3", "payout": 999_999},
        race,
        snapshot,
        bankroll_yen=10_000,
    )
    second = v20.decide(
        {"result": "6-5-4", "payout": 0},
        race,
        snapshot,
        bankroll_yen=10_000,
    )
    assert first == second
    assert first.selected_candidates == v18_decision.selected_candidates
    assert first.probabilities == pytest.approx(raw)
    assert first.probabilities != v18_decision.probabilities
    dual = first.diagnostics["v20_dual_head"]
    assert dual["source_evaluation_job_id"] == 8458
    assert dual["probability_output_head"] == "probability_head"
    assert dual["candidate_selection_head"] == "purchase_head"
    assert dual["chronological_bankroll_head"] == "purchase_head"
    assert dual["outer_result_used"] is False
    assert dual["outer_payout_used"] is False
    assert dual["real_betting_enabled"] is False
    assert len(dual["probability_output_sha256"]) == 64
    assert len(dual["purchase_decisions_sha256"]) == 64
    built = build_adapter(
        f"v20_daily:v20_dual_head_t300:{tmp_path / 'v20.joblib'}:"
        f"{tmp_path / 'base.joblib'}"
    )
    assert isinstance(built, V20DualHeadModelAdapter)


def test_pre_first_race_v20_addition_preserves_existing_four_identities(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "state"
    release = state_root / "releases" / "existing"
    release.mkdir(parents=True)
    base = tmp_path / "base.joblib"
    base.write_bytes(b"base")
    bundles = {
        family: {
            "path": str(tmp_path / f"{family}.joblib"),
            "manifest": {
                "output": {
                    "bundle_sha256": hashlib.sha256(family.encode()).hexdigest()
                }
            },
        }
        for family in updater.BUNDLE_FAMILIES
    }
    jobs = {
        family: updater.CompletedJob(number, family, tmp_path / f"{family}.json")
        for family, number in (
            ("v12", 12), ("v14", 14), ("v16", 16), ("v18", 8191), ("v20", 8458)
        )
    }
    specs = {
        "v12": f"v12_daily:{updater.SHADOW_STRATEGY}:{bundles['v12']['path']}:{base}",
        "v14": f"v14_daily:{updater.V14_SHADOW_STRATEGY}:{bundles['v14']['path']}:{base}",
        "v16": f"v16_daily:{updater.V16_SHADOW_STRATEGY}:{bundles['v16']['path']}:{base}",
        "v18": f"v18_daily:{updater.V18_SHADOW_STRATEGY}:{bundles['v18']['path']}:{base}",
    }
    existing = {
        "prediction_date": "2026-07-30",
        "activated_at": "2026-07-30T07:00:00+09:00",
        "real_betting_enabled": False,
        "model_identities": {
            family: bundles[family]["manifest"]["output"]["bundle_sha256"]
            for family in updater.ALL_FAMILIES
        },
        "model_specs": specs,
        "source_jobs": {
            family: jobs[family].job_id for family in updater.ALL_FAMILIES
        },
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
    assert status == "additive_v20_extended"
    state = json.loads(((state_root / "active").resolve() / "state.json").read_text())
    for family in updater.ALL_FAMILIES:
        assert state["model_identities"][family] == existing["model_identities"][family]
        assert state["model_specs"][family] == existing["model_specs"][family]
        assert state["source_jobs"][family] == existing["source_jobs"][family]
    assert state["source_jobs"]["v20"] == 8458
    assert state["real_betting_enabled"] is False
    assert ":v20_dual_head_t300:" in (
        (state_root / "active").resolve() / "model-spec.env"
    ).read_text()
    assert updater.promote(
        state_root=state_root,
        prediction_date="2026-07-30",
        bundles=bundles,
        jobs=jobs,
        base_model=base,
        first_start=now - timedelta(seconds=1),
        now=now,
    ) == "already_active"



def test_post_first_race_v20_addition_is_rejected(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    release = state_root / "releases" / "existing"
    release.mkdir(parents=True)
    base = tmp_path / "base.joblib"
    base.write_bytes(b"base")
    bundles = {
        family: {
            "path": str(tmp_path / f"{family}.joblib"),
            "manifest": {
                "output": {
                    "bundle_sha256": hashlib.sha256(family.encode()).hexdigest()
                }
            },
        }
        for family in updater.BUNDLE_FAMILIES
    }
    jobs = {
        family: updater.CompletedJob(number, family, tmp_path / f"{family}.json")
        for family, number in (
            ("v12", 12), ("v14", 14), ("v16", 16), ("v18", 8191), ("v20", 8458)
        )
    }
    existing = {
        "prediction_date": "2026-07-30",
        "activated_at": "2026-07-30T07:00:00+09:00",
        "real_betting_enabled": False,
        "model_identities": {
            family: bundles[family]["manifest"]["output"]["bundle_sha256"]
            for family in updater.ALL_FAMILIES
        },
        "model_specs": {
            "v12": "v12_daily:{}:{}:{}".format(updater.SHADOW_STRATEGY, bundles["v12"]["path"], base),
            "v14": "v14_daily:{}:{}:{}".format(updater.V14_SHADOW_STRATEGY, bundles["v14"]["path"], base),
            "v16": "v16_daily:{}:{}:{}".format(updater.V16_SHADOW_STRATEGY, bundles["v16"]["path"], base),
            "v18": "v18_daily:{}:{}:{}".format(updater.V18_SHADOW_STRATEGY, bundles["v18"]["path"], base),
        },
        "source_jobs": {
            family: jobs[family].job_id for family in updater.ALL_FAMILIES
        },
    }
    (release / "state.json").write_text(json.dumps(existing))
    (release / "model-spec.env").write_text("existing")
    (state_root / "active").symlink_to(release.relative_to(state_root))
    active_before = (state_root / "active").resolve()
    now = datetime(2026, 7, 30, 10, tzinfo=JST)
    assert updater.promote(
        state_root=state_root,
        prediction_date="2026-07-30",
        bundles=bundles,
        jobs=jobs,
        base_model=base,
        first_start=now - timedelta(seconds=1),
        now=now,
    ) == "first_race_boundary_passed"
    assert (state_root / "active").resolve() == active_before
    assert json.loads((active_before / "state.json").read_text()) == existing
