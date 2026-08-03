import json

import pytest

from boatrace_ai.evaluation_queue import summarize_result
from boatrace_ai.joint_bankroll_evaluation import _joint_value_audit
from boatrace_ai.web.dashboard import model_performance_audit_snapshot


def test_joint_value_audit_exposes_covariance_and_independence_bias() -> None:
    audit = _joint_value_audit({
        "version": "joint_market_value_evaluator_v0",
        "parameter_draws": 20,
        "inner_aggregation": "portfolio_path_weighted_lower_tail_mean",
        "inner_tail_fraction": 0.1,
        "inner_effective_samples_min": 64.0,
        "inner_tail_effective_samples_min": 6.4,
        "outer_alpha": 0.05,
        "outer_quantile_method": "inverted_cdf",
        "marginal_contributions_computed": True,
        "moments_by_draw": [
            {"tickets": {"1-2-3": {
                "probability_multiplier_covariance": -0.72,
                "joint_expected_edge": 0.08,
                "ordinary_hit_independence_approximation_edge": 0.80,
            }}},
            {"tickets": {"1-3-2": {
                "probability_multiplier_covariance": -0.18,
                "joint_expected_edge": 0.02,
                "ordinary_hit_independence_approximation_edge": 0.20,
            }}},
        ],
    })

    assert audit["recorded"] is True
    assert audit["shared_probability_price_scenarios"] is True
    assert audit["portfolio_path_aggregation"] is True
    assert audit["complete_vector_repricing"] is True
    assert audit["probability_multiplier_covariance_mean"] == pytest.approx(-0.45)
    assert audit["negative_covariance_fraction"] == 1.0
    assert audit["independence_approximation_overstatement_mean"] == pytest.approx(0.45)


def test_joint_audit_survives_queue_summary() -> None:
    summary = summarize_result({
        "model": "joint_bankroll_strict_walk_forward_v4",
        "configuration": {"buy_margin": 0.05},
        "joint_value_audit": {
            "recorded": True,
            "audited_portfolios": 3,
            "moment_observations": 60,
            "shared_probability_price_scenarios": True,
            "portfolio_path_aggregation": True,
            "complete_vector_repricing": True,
            "parameter_draws_min": 20,
            "inner_effective_samples_min": 64.0,
            "probability_multiplier_covariance_mean": -0.12,
            "negative_covariance_fraction": 0.75,
            "independence_approximation_overstatement_mean": 0.12,
        },
        "settlement_audit": {
            "integer_yen_accounting": True,
            "self_impact_repricing": True,
            "full_refund_terminal_states": ["cancelled"],
            "partial_refund_supported": True,
            "special_payout_addition_supported": True,
            "rounding": "integer_pool_floor_per_face_unit",
        },
        "daily": [],
    })

    assert summary["joint_audit_recorded"] is True
    assert summary["joint_covariance_mean"] == -0.12
    assert summary["joint_independence_overstatement_mean"] == 0.12
    assert summary["settlement_integer_yen"] is True
    assert summary["settlement_self_impact_repricing"] is True
    assert summary["settlement_refund_supported"] is True


def test_server_audit_snapshot_contains_numeric_rows_without_javascript() -> None:
    section, rows = model_performance_audit_snapshot({
        "generated_at": "2026-08-03T00:00:00+00:00",
        "jobs": [{
            "job_id": 12001,
            "name": "joint_bankroll_strict_walk_forward_v4",
            "status": "完了",
            "evaluation_days": 30,
            "evaluated_races": 4_200,
            "joint_purchase_value_minimum": 0.0612,
            "joint_purchase_safety_margin": 0.02,
            "roi": 1.08,
            "daily_cluster_bootstrap_roi_lower_95": 1.01,
            "joint_audit_recorded": True,
            "joint_shared_scenarios": True,
            "joint_portfolio_path_aggregation": True,
            "joint_complete_vector_repricing": True,
            "joint_parameter_draws_min": 20,
            "joint_inner_ess_min": 64.0,
            "joint_covariance_mean": -0.03125,
            "joint_independence_overstatement_mean": 0.03125,
            "settlement_integer_yen": True,
            "settlement_self_impact_repricing": True,
            "settlement_refund_supported": True,
            "settlement_special_payout_supported": True,
            "day_venue_roi_lower_95": 0.99,
            "venue_meeting_roi_lower_95": 0.97,
            "bootstrap_condition_id": "abc123",
        }],
    })

    assert len(rows) == 1
    assert "joint_bankroll_strict_walk_forward_v4" in section
    assert "0.0612 &gt; 0.0200" in section
    assert "Cov -0.031250" in section
    assert "共同経路 / portfolio ES / 完全vector再価格" in section
    assert "整数円 / 自己投票 / 返還 / 特別払戻" in section
    encoded = section.split(
        '<script id="joint-audit-data" type="application/json">', 1
    )[1].split("</script>", 1)[0]
    assert json.loads(encoded)["rows"][0]["job_id"] == 12001
