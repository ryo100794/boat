from __future__ import annotations

from pathlib import Path

from boatrace_ai.evaluation_queue import (
    DEFAULT_WORK_TICKETS,
    build_command,
    summarize_result,
)


STRATEGY = "odds_path_role_integrated_registered_band_lcb_v14"


def test_v14_queue_command_is_reproducible(tmp_path: Path) -> None:
    model = tmp_path / "data/models/source.joblib"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"artifact")
    command, _output = build_command({
        "job_id": 14,
        "status": "running",
        "task_type": "market_residual_walk_forward",
        "model_key": "v14-candidate",
        "parameters": {
            "model_input": "data/models/source.joblib",
            "from_date": "2026-07-18",
            "through_date": "2026-08-01",
            "daily_budget_yen": 10_000,
            "min_calibration_days": 2,
            "calibrator_strategy": STRATEGY,
            "v12_closing_fallback_policy": "v11",
        },
    }, app_root=tmp_path, python=tmp_path / ".venv/bin/python", db="postgresql://test")

    assert command[command.index("--calibrator-strategy") + 1] == STRATEGY
    assert command[command.index("--daily-budget-yen") + 1] == "10000"
    assert command[command.index("--through-date") + 1] == "2026-08-01"


def test_v14_summary_preserves_standard_and_calibration_metrics() -> None:
    calibration = {
        "evaluation_days": 5,
        "candidate_count": 300,
        "adjusted_predicted_hits": 18.0,
        "observed_hits": 20,
        "day_bootstrap_observed_to_adjusted_predicted_ratio_lower_95": 1.01,
        "candidate_binary_brier_score": 0.01,
        "candidate_binary_log_loss": 0.05,
        "inconsistent_t300_snapshot_races": 0,
    }
    summary = summarize_result({
        "model": STRATEGY,
        "roi": 1.1,
        "profit_yen": 1_000,
        "roi_without_largest_hit": 1.02,
        "profit_without_largest_hit_yen": 200,
        "daily_cluster_bootstrap_roi_lower_95": 1.01,
        "selected_candidate_calibration": calibration,
        "prospective_role_integrated_v14_walk_forward": {
            "roi": 1.1,
            "profit_yen": 1_000,
            "roi_without_largest_hit": 1.02,
            "profit_without_largest_hit_yen": 200,
            "daily_cluster_bootstrap_roi_lower_95": 1.01,
            "promotion_eligible": True,
            "selected_candidate_calibration": calibration,
        },
    })

    assert summary["roi"] == 1.1
    assert summary["profit_without_largest_hit_yen"] == 200
    assert summary["v14_calibration_candidate_count"] == 300
    assert summary["prospective_v14_roi_without_largest_hit"] == 1.02
    assert summary["prospective_v14_selected_candidate_calibration"] == calibration


def test_v14_and_weekend_pilot_work_tickets_are_seeded() -> None:
    v14_ticket = next(
        row for row in DEFAULT_WORK_TICKETS
        if row[0] == "MODEL-REGISTERED-BAND-LCB-V14-001"
    )
    pilot = next(
        row for row in DEFAULT_WORK_TICKETS
        if row[0] == "MODEL-WEEKEND-PILOT-20260801"
    )

    assert "[0.5,1.0)" in v14_ticket[3]
    assert "最大払戻除外ROI>1" in v14_ticket[4]
    acceptance = pilot[4]
    assert "2026-07-30" in pilot[3]
    assert "2026-07-31" in pilot[3]
    assert "2026-08-01" in pilot[3]
    assert "2026-08-02" in pilot[3]
    assert "最大払戻除外ROI>1" in acceptance
    assert "日次bootstrap下限>1" in acceptance
    assert "データ欠損0" in acceptance
    assert "2,000円/日" in acceptance
    assert "10,000円" in acceptance


def test_old_v13_result_is_forced_non_promotable_in_summary() -> None:
    summary = summarize_result({
        "model": "odds_path_role_integrated_edge_conditional_lcb_v13",
        "status": "completed",
        "promotion_eligible": True,
        "roi": 2.0,
    })

    assert summary["status"] == "research_invalid_deprecated"
    assert summary["research_invalid"] is True
    assert summary["deprecated"] is True
    assert summary["promotion_eligible"] is False
