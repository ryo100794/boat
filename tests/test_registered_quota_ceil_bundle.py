import json
from pathlib import Path

import joblib
import pytest

from boatrace_ai.runtime.quota_ceil_bundle import build_registered_quota_ceil_bundle
from boatrace_ai.runtime.quota_ceil_shadow_policy import registration


MODEL = "odds_path_observed_closing_return_schedule_quota_triple_head_v21"


def _candidate_policy(*, rounding: str = "ceil") -> dict:
    return {
        "name": "ev1.50_odds40_r1_ratio1.20_kelly_100",
        "ev_threshold": 1.5,
        "max_estimated_ev": None,
        "max_odds": 40.0,
        "max_tickets_per_race": 1,
        "min_model_market_ratio": 1.2,
        "staking_mode": "kelly_100",
        "v18_ticket_control": {
            "method": "strict_prior_daily_ticket_lower_quantile",
            "learned_daily_ticket_limit": 13,
            "schedule_quota_rounding": rounding,
            "schedule_quota_opportunity": None,
            "stake_granularity_yen": 100,
            "result_or_payout_fields_used": False,
        },
    }


def _source(
    path: Path,
    *,
    trained_through: str = "2026-07-31",
    rounding: str = "ceil",
) -> None:
    path.write_text(
        json.dumps(
            {
                "model": MODEL,
                "calibrator_strategy": MODEL,
                "deployment_configuration": {
                    "trained_through_date": trained_through,
                    "real_betting_enabled": False,
                    "selected_policy": {"name": "no_bet", "no_bet": True},
                    "candidate_policy": _candidate_policy(rounding=rounding),
                    "triple_head_calibration": {"outer_holdout_used": False},
                },
            }
        ),
        encoding="utf-8",
    )


def test_builder_freezes_quota_ceil_registration(tmp_path: Path) -> None:
    source = tmp_path / "job-00009906.json"
    output = tmp_path / "quota-ceil.joblib"
    _source(source)

    result = build_registered_quota_ceil_bundle(source, output)
    deployment = joblib.load(output)["deployment"]

    assert result["trained_through_date"] == "2026-07-31"
    assert deployment["source_evaluation_job_id"] == 9_906
    assert deployment["real_betting_enabled"] is False
    assert deployment["outer_result_or_payout_used"] is False
    assert deployment["prospective_policy_registration"] == registration()


def test_builder_rejects_nonregistered_rounding(tmp_path: Path) -> None:
    source = tmp_path / "job-00009906.json"
    _source(source, rounding="floor")

    with pytest.raises(ValueError, match="ticket control"):
        build_registered_quota_ceil_bundle(source, tmp_path / "quota-ceil.joblib")


def test_builder_rejects_same_day_or_future_training(tmp_path: Path) -> None:
    source = tmp_path / "job-00009906.json"
    _source(source, trained_through="2026-08-02")

    with pytest.raises(ValueError, match="training boundary"):
        build_registered_quota_ceil_bundle(source, tmp_path / "quota-ceil.joblib")
