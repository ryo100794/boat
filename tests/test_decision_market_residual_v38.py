from __future__ import annotations

from datetime import date, timedelta

import joblib
import pytest

from boatrace_ai.listwise import decision_market_residual_v38 as subject


def _race(day: date, index: int) -> dict:
    race_date = day.isoformat()
    return {
        "race_id": f"{race_date}-01-{index:02d}",
        "race_date": race_date,
        "jcd": "01",
        "rno": index % 12 + 1,
        "actual_combination": "1-2-3",
        "actual_payout_yen": 1000,
        "odds": {"1-2-3": 10.0, "1-3-2": 20.0},
        "model_probabilities": {"1-2-3": 0.55, "1-3-2": 0.45},
        "market_probabilities": {"1-2-3": 0.60, "1-3-2": 0.40},
        "lane_context": {str(lane): {} for lane in range(1, 7)},
        "snapshot_id": index,
        "captured_at": f"{race_date}T10:00:00+09:00",
        "odds_deadline_at": f"{race_date}T10:01:00+09:00",
        "input_snapshot_age_seconds": 60.0,
        "official_closing_odds": {"1-2-3": 1.0},
        "official_closing_market_probabilities": {"1-2-3": 1.0},
        "closing_odds": {"1-2-3": 2.0},
    }


def test_decision_projection_structurally_excludes_closing_fields() -> None:
    race = _race(date(2026, 1, 1), 1)
    projected = subject.decision_time_race(race)
    changed = {
        **race,
        "official_closing_odds": {"1-2-3": 999.0},
        "official_closing_market_probabilities": {"1-2-3": 0.001},
        "closing_odds": {"1-2-3": 888.0},
    }

    assert subject.decision_time_race(changed) == projected
    assert not any(
        key.startswith(subject.FORBIDDEN_SOURCE_PREFIXES)
        for key in projected
    )
    assert projected["market_probability_source"] == "decision_snapshot_odds"
    assert projected["decision_time_boundary_check"] is True


def test_decision_projection_rejects_post_deadline_and_stale_snapshots() -> None:
    race = _race(date(2026, 1, 1), 1)
    post_deadline = {
        **race,
        "captured_at": "2026-01-01T10:02:00+09:00",
        "input_snapshot_age_seconds": 0.0,
    }
    with pytest.raises(ValueError, match="deadline boundary"):
        subject.decision_time_race(post_deadline)

    with pytest.raises(ValueError, match="allowed range"):
        subject.decision_time_race({
            **race,
            "input_snapshot_age_seconds": 66.0,
        })


def test_training_gate_requires_both_days_and_races() -> None:
    start = date(2026, 1, 1)
    races = [_race(start + timedelta(days=index), index) for index in range(5)]
    result = subject.fit_decision_time_market_residual(
        races,
        calibration_through="2026-01-05",
        minimum_training_days=5,
        minimum_training_races=6,
        num_threads=1,
    )

    assert result["status"] == "insufficient_training_history"
    assert result["ready_reasons"] == ["training_races_below_minimum"]
    assert "artifact" not in result


def test_ready_training_keeps_outer_dates_strictly_after_cutoff(monkeypatch) -> None:
    start = date(2026, 1, 1)
    races = [_race(start + timedelta(days=index), index) for index in range(7)]
    observed = {}

    def fake_fit(calibration, evaluation, *, num_threads):
        observed["calibration"] = calibration
        observed["evaluation"] = evaluation
        observed["num_threads"] = num_threads
        return {
            "market_is_exact_nested_null": True,
            "inner_fit_through": "2026-01-03",
            "inner_validation_from": "2026-01-04",
            "selected_tree_preset": "compact",
            "selected_shrinkage": 0.25,
            "candidates": [],
            "artifact": {"booster_sha256": "a" * 64},
            "metrics": {"evaluated_races": 2},
        }

    monkeypatch.setattr(
        subject, "fit_temporal_nonlinear_market_residual", fake_fit
    )
    result = subject.fit_decision_time_market_residual(
        races,
        calibration_through="2026-01-05",
        minimum_training_days=5,
        minimum_training_races=5,
        num_threads=2,
    )

    assert result["status"] == "ready"
    assert result["training_through"] == "2026-01-05"
    assert result["evaluation_from"] == "2026-01-06"
    assert result["official_closing_fields_used"] is False
    assert result["decision_time_boundary_all_passed"] is True
    assert result["decision_time_boundary_violations"] == 0
    assert result["maximum_input_snapshot_age_seconds"] == 60.0
    assert observed["num_threads"] == 2
    assert all(row["race_date"] <= "2026-01-05" for row in observed["calibration"])
    assert all(row["race_date"] > "2026-01-05" for row in observed["evaluation"])
    assert not any(
        key.startswith(subject.FORBIDDEN_SOURCE_PREFIXES)
        for row in observed["calibration"]
        for key in row
    )


def test_scored_cache_training_records_source_hash_and_insufficient_status(
    tmp_path,
) -> None:
    start = date(2026, 1, 1)
    cache = tmp_path / "decision.races.joblib"
    joblib.dump(
        {
            "contract": {
                "version": 14,
                "from_date": "2026-01-01",
                "through_date": "2026-01-05",
            },
            "races": [
                _race(start + timedelta(days=index), index)
                for index in range(5)
            ],
        },
        cache,
    )

    result = subject.train_from_scored_cache(
        cache,
        calibration_through="2026-01-05",
        minimum_training_days=5,
        minimum_training_races=6,
        num_threads=1,
    )

    assert result["status"] == "completed"
    assert result["training_status"] == "insufficient_training_history"
    assert result["decision"] == "insufficient_data"
    assert result["promotion_eligible"] is False
    assert len(result["source_scored_cache_sha256"]) == 64
    assert result["source_cache_contract"]["version"] == 14


def test_challenger_gate_requires_weeklong_market_improvement() -> None:
    payload = {
        "training_status": "ready",
        "official_closing_fields_used": False,
        "market_is_exact_nested_null": True,
        "selected_shrinkage": 0.25,
        "artifact": {"booster_sha256": "a" * 64},
        "holdout_metrics": {
            "evaluated_days": 7,
            "days_better_than_market": 5,
            "log_loss_delta_vs_market": -0.003,
            "trifecta_top5_hit_rate": 0.370,
            "market_trifecta_top5_hit_rate": 0.372,
        },
    }
    assert subject.decision_v38_challenger_eligible(payload) is True

    assert subject.decision_v38_challenger_eligible({
        **payload,
        "holdout_metrics": {
            **payload["holdout_metrics"],
            "evaluated_days": 6,
        },
    }) is False
    assert subject.decision_v38_challenger_eligible({
        **payload,
        "holdout_metrics": {
            **payload["holdout_metrics"],
            "log_loss_delta_vs_market": 0.001,
        },
    }) is False
    assert subject.decision_v38_challenger_eligible({
        **payload,
        "holdout_metrics": {
            **payload["holdout_metrics"],
            "trifecta_top5_hit_rate": 0.360,
        },
    }) is False
