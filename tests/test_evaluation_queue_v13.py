from __future__ import annotations

from pathlib import Path

from boatrace_ai.evaluation_queue import (
    DEFAULT_WORK_TICKETS,
    build_command,
    summarize_result,
)


STRATEGY = "odds_path_role_integrated_edge_conditional_lcb_v13"


def _job(parameters: dict) -> dict:
    return {
        "job_id": 13,
        "status": "running",
        "task_type": "market_residual_walk_forward",
        "model_key": "v13-candidate",
        "parameters": parameters,
    }


def test_v13_market_residual_command_uses_existing_queue_contract(
    tmp_path: Path,
) -> None:
    model = tmp_path / "data/models/source.joblib"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"artifact")

    command, _output = build_command(
        _job({
            "model_input": "data/models/source.joblib",
            "from_date": "2026-07-18",
            "through_date": "2026-07-29",
            "daily_budget_yen": 10_000,
            "min_calibration_days": 2,
            "calibrator_strategy": STRATEGY,
            "v12_closing_fallback_policy": "v11",
        }),
        app_root=tmp_path,
        python=tmp_path / ".venv/bin/python",
        db="postgresql://test",
    )

    assert command[command.index("--calibrator-strategy") + 1] == STRATEGY
    assert command[command.index("--v12-closing-fallback-policy") + 1] == "v11"
    assert command[command.index("--daily-budget-yen") + 1] == "10000"
    assert command[command.index("--from-date") + 1] == "2026-07-18"
    assert command[command.index("--through-date") + 1] == "2026-07-29"


def test_v13_result_summary_preserves_roi_and_conditional_calibration() -> None:
    calibration = {
        "evaluation_days": 5,
        "coverage_days": 5,
        "daily_lower_bound_coverage": 0.8,
        "candidate_count": 42,
        "raw_expected_hits": 4.5,
        "adjusted_expected_hits": 2.8,
        "observed_hits": 3,
        "raw_overprediction_hits": 1.5,
        "adjusted_overprediction_hits": 0.0,
        "overprediction_reduction_hits": 1.5,
        "relative_overprediction_reduction": 1.0,
        "observed_hits_to_adjusted_predicted_hits_ratio": 3 / 2.8,
        "missing_t300_races": 0,
    }
    divergence = {
        "definition": "log(model_probability / normalized_T300_market_probability)",
        "bands": [{
            "divergence_band": "d_ge_100",
            "unique_races_in_band": 20,
            "tickets": 90,
            "sum_predicted_probability": 6.0,
            "hits": 4,
            "observed_hits_to_predicted_hits_ratio": 2 / 3,
            "actual_payout_roi": 1.2,
        }],
    }
    summary = summarize_result({
        "model": STRATEGY,
        "roi": 1.08,
        "roi_without_largest_hit": 1.01,
        "daily_cluster_bootstrap_roi_lower_95": 0.97,
        "edge_conditional_probability_calibration": calibration,
        "strict_prior_divergence_bands": divergence,
        "prospective_role_integrated_v13_walk_forward": {
            "roi": 1.04,
            "roi_without_largest_hit": 1.0,
            "daily_cluster_bootstrap_roi_lower_95": 0.92,
            "conditional_calibration": calibration,
            "strict_prior_divergence_bands": divergence,
        },
    })

    assert summary["roi"] == 1.08
    assert summary["roi_without_largest_hit"] == 1.01
    assert summary["daily_cluster_bootstrap_roi_lower_95"] == 0.97
    assert summary["edge_conditional_candidate_count"] == 42
    assert summary["edge_conditional_daily_lower_bound_coverage"] == 0.8
    assert summary["strict_prior_divergence_bands"] == divergence
    assert summary["prospective_v13_roi"] == 1.04
    assert summary["prospective_v13_conditional_calibration"] == calibration


def test_v13_work_ticket_is_reproducibly_seeded_with_performance_gates() -> None:
    ticket = next(
        row
        for row in DEFAULT_WORK_TICKETS
        if row[0] == "MODEL-EDGE-CONDITIONAL-LCB-V13-001"
    )
    acceptance = ticket[4]

    assert "同一評価窓" in acceptance
    assert "ROI" in acceptance
    assert "最大払戻除外ROI" in acceptance
    assert "bootstrap片側下限" in acceptance
    assert "条件付きcoverage" in acceptance
    assert "expected-vs-hit" in acceptance
