from pathlib import Path

from boatrace_ai.runtime.prospective_market_promotion import (
    prospective_promotion_payload,
    write_prospective_candidate,
)


def _source(*, return_yen: int = 36_000, without_largest_roi: float = 1.1):
    daily = [
        {
            "race_date": f"2026-08-{day:02d}",
            "stake_yen": 1_000,
            "return_yen": return_yen // 30,
            "profit_yen": return_yen // 30 - 1_000,
        }
        for day in range(1, 31)
    ]
    return {
        "model": "protected_mlp",
        "source_model": "/models/model.joblib",
        "source_model_sha256": "a" * 64,
        "through_date": "2026-08-30",
        "promotion_gate": {
            "calibration_pass": True,
            "market_confidence_pass": True,
            "no_lookahead_pass": True,
        },
        "deployment_configuration": {
            "role": "next_day_refit_not_evaluation",
            "trained_through_date": "2026-08-30",
            "training_races": 10_000,
            "calibrator": {"converged": True},
            "calibrator_strategy": "newton_residual",
        },
        "prospective_normalized_ev_walk_forward": {
            "policy": {"name": "registered_v2", "staking_mode": "normalized_010"},
            "evaluation_days": 30,
            "evaluated_races": 4_000,
            "tickets": 300,
            "hit_tickets": 20,
            "stake_yen": 30_000,
            "return_yen": return_yen,
            "effective_hit_count": 20.0,
            "largest_hit_return_share": 0.1,
            "roi_without_largest_hit": without_largest_roi,
            "daily": daily,
        },
    }


def test_builds_strict_prospective_production_candidate() -> None:
    result = prospective_promotion_payload(_source(), bootstrap_samples=500)

    assert result is not None
    assert result["promotion_eligible"] is True
    assert result["promotion_gate"]["pass"] is True
    assert result["roi"] == 1.2
    assert result["deployment_configuration"]["selected_policy"]["name"] == (
        "registered_v2"
    )
    assert result["prospective_promotion_confidence"]["roi_ci95_lower"] > 1.0


def test_largest_hit_dependence_keeps_policy_shadow_only() -> None:
    result = prospective_promotion_payload(
        _source(without_largest_roi=0.8),
        bootstrap_samples=500,
    )

    assert result is not None
    assert result["promotion_eligible"] is False
    assert result["promotion_gate"]["fold_stability_pass"] is False
    assert result["deployment_configuration"]["selected_policy"]["no_bet"] is True


def test_largest_hit_return_share_above_formal_limit_blocks_promotion() -> None:
    source = _source()
    source["prospective_normalized_ev_walk_forward"][
        "largest_hit_return_share"
    ] = 0.150001

    result = prospective_promotion_payload(source, bootstrap_samples=500)

    assert result is not None
    assert result["promotion_eligible"] is False
    assert result["promotion_gate"]["largest_hit_return_share_pass"] is False


def test_writes_companion_candidate_only_when_prospective_days_exist(
    tmp_path: Path,
) -> None:
    source = tmp_path / "job-00000001.json"
    import json

    source.write_text(json.dumps(_source()), encoding="utf-8")
    written = write_prospective_candidate(source, output_dir=tmp_path / "derived")

    assert written is not None
    assert Path(written).is_file()


def test_uses_trend_point_formal_track_and_requires_its_complete_gate() -> None:
    source = _source()
    track = source.pop("prospective_normalized_ev_walk_forward")
    track["promotion_gate"] = {"pass": True, "no_lookahead_pass": True}
    source["trend_point_empirical_lcb_walk_forward"] = track

    eligible = prospective_promotion_payload(source, bootstrap_samples=500)

    assert eligible is not None
    assert eligible["promotion_eligible"] is True
    assert eligible["prospective_source_track"] == (
        "trend_point_empirical_lcb_walk_forward"
    )
    track["promotion_gate"]["pass"] = False
    rejected = prospective_promotion_payload(source, bootstrap_samples=500)
    assert rejected is not None
    assert rejected["promotion_eligible"] is False
    assert rejected["promotion_gate"]["source_formal_gate_pass"] is False


def test_raw_trend_point_track_is_not_a_promotion_source() -> None:
    source = _source()
    track = source.pop("prospective_normalized_ev_walk_forward")
    source["trend_point_market_offset_kelly_walk_forward"] = track

    assert prospective_promotion_payload(source, bootstrap_samples=500) is None


def test_missing_largest_hit_share_fails_closed() -> None:
    source = _source()
    del source["prospective_normalized_ev_walk_forward"][
        "largest_hit_return_share"
    ]

    result = prospective_promotion_payload(source, bootstrap_samples=500)

    assert result is not None
    assert result["promotion_eligible"] is False
    assert result["promotion_gate"]["largest_hit_return_share_pass"] is False
