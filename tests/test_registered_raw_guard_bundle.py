import json
from pathlib import Path

import joblib
import pytest

from boatrace_ai.runtime.raw_guard_bundle import build_registered_raw_guard_bundle
from boatrace_ai.runtime.raw_guard_shadow_policy import MIN_RAW_EV, registration


MODEL = "odds_path_observed_closing_return_schedule_quota_triple_head_v21"


def _source(path: Path) -> None:
    path.write_text(json.dumps({
        "model": MODEL,
        "deployment_configuration": {
            "trained_through_date": "2026-07-31",
            "real_betting_enabled": False,
            "selected_policy": {"name": "no_bet", "no_bet": True},
            "candidate_policy": {
                "name": "base",
                "ev_threshold": 1.5,
                "max_estimated_ev": None,
                "max_odds": 40.0,
                "max_tickets_per_race": 1,
                "min_model_market_ratio": 1.2,
                "staking_mode": "kelly_100",
                "v18_ticket_control": {
                    "learned_daily_ticket_limit": 13,
                    "schedule_quota_rounding": "ceil",
                    "schedule_quota_opportunity": None,
                    "result_or_payout_fields_used": False,
                },
            },
            "triple_head_calibration": {"outer_holdout_used": False},
        },
    }), encoding="utf-8")


def _audit(path: Path, *, raw_ev: float = MIN_RAW_EV) -> None:
    path.write_text(json.dumps({
        "comparison_role": "fixed_policy_strict_prior_fold_replay",
        "fixed_policy": {"min_raw_ev": raw_ev},
        "information_boundary": {
            "outer_holdout_used_to_fit_or_select_policy": False,
        },
        "chronological_bankroll": {"race_days": 8, "evaluated_races": 1242},
    }), encoding="utf-8")


def test_builder_freezes_raw_guard_registration(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    audit = tmp_path / "audit.json"
    output = tmp_path / "raw-guard.joblib"
    _source(source)
    _audit(audit)

    result = build_registered_raw_guard_bundle(source, audit, output)
    deployment = joblib.load(output)["deployment"]

    assert result["registration"] == registration()
    assert deployment["candidate_policy"]["min_raw_ev"] == MIN_RAW_EV
    assert deployment["real_betting_enabled"] is False


def test_builder_rejects_different_replay_policy(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    audit = tmp_path / "audit.json"
    _source(source)
    _audit(audit, raw_ev=1.0)
    with pytest.raises(ValueError, match="audit does not match"):
        build_registered_raw_guard_bundle(source, audit, tmp_path / "out.joblib")
