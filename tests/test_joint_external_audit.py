import json

import pytest

from boatrace_ai.evaluation_queue import summarize_result
from boatrace_ai.joint_bankroll_evaluation import _joint_value_audit
from boatrace_ai.web.dashboard import model_performance_audit_snapshot


def test_joint_value_audit_exposes_covariance_and_independence_bias() -> None:
    audit = _joint_value_audit({
        "version": "joint_market_value_evaluator_v0",
        "parameter_draws": 20,
        "minimum_outer_draws": 20,
        "inner_scenario_count_s_definition": "future_joint_market_paths_per_outer_parameter_draw",
        "inner_scenario_count_s_min": 64,
        "inner_scenario_count_s_max": 64,
        "inner_aggregation": "portfolio_path_weighted_lower_tail_mean",
        "inner_tail_fraction": 0.1,
        "inner_effective_samples_min": 64.0,
        "inner_effective_samples_mean": 64.0,
        "inner_effective_samples_max": 64.0,
        "inner_tail_effective_samples_min": 6.4,
        "inner_tail_effective_samples_mean": 6.4,
        "inner_tail_effective_samples_max": 6.4,
        "minimum_inner_tail_effective_samples": 5.0,
        "inner_tail_support_for_purchase": True,
        "outer_alpha": 0.05,
        "outer_quantile_method": "inverted_cdf",
        "marginal_contributions_computed": True,
        "moments_by_draw": [
            {"tickets": {"1-2-3": {
                "expected_probability_times_multiplier": 1.08,
                "independence_probability_times_multiplier": 1.80,
                "probability_multiplier_covariance": -0.72,
                "independence_approximation_bias": 0.72,
                "joint_expected_edge": 0.08,
                "ordinary_hit_independence_approximation_edge": 0.80,
            }}},
            {"tickets": {"1-3-2": {
                "expected_probability_times_multiplier": 1.02,
                "independence_probability_times_multiplier": 1.20,
                "probability_multiplier_covariance": -0.18,
                "independence_approximation_bias": 0.18,
                "joint_expected_edge": 0.02,
                "ordinary_hit_independence_approximation_edge": 0.20,
            }}},
        ],
    })

    assert audit["recorded"] is True
    assert audit["shared_probability_price_scenarios"] is True
    assert audit["outer_sample_count_r"] == 20
    assert audit["minimum_outer_draws"] == 20
    assert audit["outer_tail_observations"] == 1
    assert audit["minimum_outer_tail_observations_for_promotion"] == 5
    assert audit["outer_tail_support_for_promotion"] is False
    assert audit["inner_scenario_count_s_min"] == 64
    assert audit["inner_scenario_count_s_max"] == 64
    assert audit["inner_effective_samples_min"] == 64.0
    assert audit["inner_effective_samples_mean"] == 64.0
    assert audit["inner_effective_samples_max"] == 64.0
    assert audit["inner_tail_effective_samples_min"] == 6.4
    assert audit["inner_tail_effective_samples_mean"] == 6.4
    assert audit["inner_tail_effective_samples_max"] == 6.4
    assert audit["inner_tail_support_for_purchase"] is True
    assert audit["portfolio_path_aggregation"] is True
    assert audit["complete_vector_repricing"] is True
    assert audit["expected_probability_times_multiplier_mean"] == pytest.approx(1.05)
    assert audit["independence_probability_times_multiplier_mean"] == pytest.approx(1.50)
    assert audit["joint_expected_edge_mean"] == pytest.approx(0.05)
    assert audit["product_identity_residual_max_abs"] <= 1e-12
    assert audit["product_identity_consistent"] is True
    assert audit["probability_multiplier_covariance_mean"] == pytest.approx(-0.45)
    assert audit["negative_covariance_fraction"] == 1.0
    assert audit["independence_approximation_bias_mean"] == pytest.approx(0.45)
    assert audit["independence_approximation_bias_min"] == pytest.approx(0.18)
    assert audit["independence_approximation_bias_max"] == pytest.approx(0.72)
    assert audit["positive_independence_bias_fraction"] == 1.0
    assert audit["independence_approximation_overstatement_mean"] == pytest.approx(0.45)


def test_outer_r_requires_five_lower_tail_observations_for_promotion() -> None:
    audit = _joint_value_audit({
        "version": "joint_market_value_evaluator_v0",
        "parameter_draws": 100,
        "minimum_outer_draws": 20,
        "outer_alpha": 0.05,
        "inner_aggregation": "portfolio_path_weighted_lower_tail_mean",
        "marginal_contributions_computed": True,
    })

    assert audit["outer_sample_count_r"] == 100
    assert audit["outer_tail_observations"] == 5
    assert audit["outer_tail_support_for_promotion"] is True


def test_joint_audit_survives_queue_summary() -> None:
    summary = summarize_result({
        "model": "joint_bankroll_strict_walk_forward_v4",
        "configuration": {
            "buy_margin": 0.05,
            "outer_draws": 100,
            "search_outer_draws": 20,
            "search_validation_draw_sets_disjoint": True,
        },
        "joint_value_audit": {
            "recorded": True,
            "audited_portfolios": 3,
            "moment_observations": 60,
            "shared_probability_price_scenarios": True,
            "portfolio_path_aggregation": True,
            "complete_vector_repricing": True,
            "parameter_draws_min": 20,
            "outer_sample_count_r_definition": "number_of_outer_model_or_parameter_uncertainty_draws",
            "outer_sample_count_r_min": 20,
            "outer_sample_count_r_max": 20,
            "minimum_outer_draws_max": 20,
            "outer_alpha_min": 0.05,
            "outer_alpha_max": 0.05,
            "outer_tail_observations_min": 1,
            "outer_tail_observations_max": 1,
            "minimum_outer_tail_observations_for_promotion": 5,
            "outer_tail_support_for_promotion": False,
            "inner_scenario_count_s_definition": "future_joint_market_paths_per_outer_parameter_draw",
            "inner_scenario_count_s_min": 64,
            "inner_scenario_count_s_max": 64,
            "inner_effective_samples_min": 64.0,
            "inner_effective_samples_mean": 64.0,
            "inner_effective_samples_max": 64.0,
            "inner_tail_effective_samples_min": 6.4,
            "inner_tail_effective_samples_mean": 6.4,
            "inner_tail_effective_samples_max": 6.4,
            "minimum_inner_tail_effective_samples_max": 5.0,
            "inner_tail_support_for_promotion": True,
            "expected_probability_times_multiplier_mean": 1.05,
            "independence_probability_times_multiplier_mean": 1.17,
            "joint_expected_edge_mean": 0.05,
            "product_identity_residual_mean": 0.0,
            "product_identity_residual_max_abs": 0.0,
            "product_identity_consistent": True,
            "probability_multiplier_covariance_mean": -0.12,
            "negative_covariance_fraction": 0.75,
            "independence_approximation_bias_mean": 0.12,
            "independence_approximation_bias_min": 0.02,
            "independence_approximation_bias_max": 0.31,
            "positive_independence_bias_fraction": 0.75,
            "independence_approximation_overstatement_mean": 0.12,
        },
        "evaluation_protocol_id": "protocol123",
        "evaluation_protocol": {
            "version": "joint_evaluation_protocol_v2",
            "evaluation_time_t": {
                "definition": "purchase_decision_timestamp",
                "source_field": "decision_at_else_odds_deadline_at",
                "earliest": "2026-07-01T10:00:00+09:00",
                "latest": "2026-07-01T15:00:00+09:00",
            },
            "odds_snapshot_age": {
                "definition": "evaluation_time_t_minus_odds_snapshot_captured_at",
                "minimum": 4.0,
                "mean": 12.5,
                "maximum": 31.0,
            },
            "population": {
                "venues": ["01"],
                "wager_types": ["trifecta"],
                "popularity_bands_at_t": ["favorite_share_ge_025"],
            },
        },
        "calibration_ledger": {
            "version": "joint_edge_calibration_ledger_v1",
            "candidate_portfolios": 713,
            "authorized_portfolios": 9,
            "stake_yen": 71_300,
            "return_yen": 64_000,
            "profit_yen": -7_300,
            "roi": 64_000 / 71_300,
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

    assert summary["evaluation_protocol_id"] == "protocol123"
    assert summary["evaluation_time_t_source"] == (
        "decision_at_else_odds_deadline_at"
    )
    assert summary["evaluation_snapshot_age_seconds_mean"] == 12.5
    assert summary["evaluation_wager_types"] == ["trifecta"]
    assert summary["joint_audit_recorded"] is True
    assert summary["joint_expected_pi_d_mean"] == 1.05
    assert summary["joint_independent_pi_times_d_mean"] == 1.17
    assert summary["joint_expected_edge_mean"] == 0.05
    assert summary["joint_product_identity_residual_max_abs"] == 0.0
    assert summary["joint_product_identity_consistent"] is True
    assert summary["joint_covariance_mean"] == -0.12
    assert summary["joint_independence_bias_mean"] == 0.12
    assert summary["joint_independence_bias_min"] == 0.02
    assert summary["joint_independence_bias_max"] == 0.31
    assert summary["joint_positive_independence_bias_fraction"] == 0.75
    assert summary["joint_independence_overstatement_mean"] == 0.12
    assert summary["joint_outer_sample_count_r_min"] == 20
    assert summary["joint_search_outer_sample_count_r_requested"] == 20
    assert summary["joint_validation_outer_sample_count_r_requested"] == 100
    assert summary["joint_search_validation_draw_sets_disjoint"] is True
    assert summary["joint_outer_sample_count_r_max"] == 20
    assert summary["joint_outer_tail_observations_min"] == 1
    assert summary["joint_outer_tail_required"] == 5
    assert summary["joint_outer_tail_support"] is False
    assert summary["joint_inner_s_min"] == 64
    assert summary["joint_inner_s_max"] == 64
    assert summary["joint_inner_ess_min"] == 64.0
    assert summary["joint_inner_ess_mean"] == 64.0
    assert summary["joint_inner_ess_max"] == 64.0
    assert summary["joint_inner_tail_ess_min"] == 6.4
    assert summary["joint_inner_tail_ess_mean"] == 6.4
    assert summary["joint_inner_tail_ess_max"] == 6.4
    assert summary["joint_inner_tail_ess_required"] == 5.0
    assert summary["joint_inner_tail_support"] is True
    assert summary["settlement_integer_yen"] is True
    assert summary["settlement_self_impact_repricing"] is True
    assert summary["settlement_refund_supported"] is True
    assert summary["joint_calibration_candidate_portfolios"] == 713
    assert summary["joint_calibration_roi"] == pytest.approx(64_000 / 71_300)


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
            "joint_outer_sample_count_r_min": 20,
            "joint_outer_sample_count_r_max": 20,
            "joint_outer_alpha_min": 0.05,
            "joint_outer_tail_observations_min": 1,
            "joint_outer_tail_required": 5,
            "joint_outer_tail_support": False,
            "joint_search_outer_sample_count_r_requested": 20,
            "joint_validation_outer_sample_count_r_requested": 100,
            "joint_search_validation_draw_sets_disjoint": True,
            "joint_inner_s_min": 64,
            "joint_inner_s_max": 64,
            "joint_inner_ess_min": 64.0,
            "joint_inner_ess_mean": 64.0,
            "joint_inner_ess_max": 64.0,
            "joint_inner_tail_ess_min": 6.4,
            "joint_inner_tail_ess_mean": 6.4,
            "joint_inner_tail_ess_max": 6.4,
            "joint_inner_tail_ess_required": 5.0,
            "joint_inner_tail_support": True,
            "joint_expected_pi_d_mean": 1.08,
            "joint_independent_pi_times_d_mean": 1.11125,
            "joint_expected_edge_mean": 0.08,
            "joint_product_identity_residual_mean": 0.0,
            "joint_product_identity_residual_max_abs": 0.0,
            "joint_product_identity_consistent": True,
            "joint_covariance_mean": -0.03125,
            "joint_independence_bias_mean": 0.03125,
            "joint_independence_bias_min": 0.002,
            "joint_independence_bias_max": 0.142,
            "joint_positive_independence_bias_fraction": 1.0,
            "joint_independence_overstatement_mean": 0.03125,
            "settlement_integer_yen": True,
            "settlement_self_impact_repricing": True,
            "settlement_refund_supported": True,
            "settlement_special_payout_supported": True,
            "day_venue_roi_lower_95": 0.99,
            "venue_meeting_roi_lower_95": 0.97,
            "evaluation_protocol_id": "protocol123",
            "evaluation_time_t_earliest": "2026-07-01T10:00:00+09:00",
            "evaluation_time_t_latest": "2026-07-30T16:00:00+09:00",
            "evaluation_snapshot_age_seconds_min": 4.0,
            "evaluation_snapshot_age_seconds_mean": 12.5,
            "evaluation_snapshot_age_seconds_max": 31.0,
            "bootstrap_condition_id": "abc123",
        }],
    })

    assert len(rows) == 1
    assert "joint_bankroll_strict_walk_forward_v4" in section
    assert "0.0612 &gt; 0.0200" in section
    assert (
        "結合期待倍率 E[piD] 1.080000 / "
        "独立近似倍率 E[pi]E[D] 1.111250"
    ) in section
    assert "恒等式 合格 (max残差 0.000000000)" in section
    assert "確率・倍率共分散 Cov -0.031250 (減額)" in section
    assert (
        "独立近似バイアス 0.031250 (過大評価) "
        "[0.002000..0.142000]"
    ) in section
    assert (
        "探索R 20 / 検証R 100 / 非重複 合格 / "
        "R 20..20 / 下側 1/5 不足 / α 0.05"
    ) in section
    assert (
        "S 64..64 / 下側ESS 6.40..6.40..6.40 / "
        "必要 5.00 合格 / ESS 64.00..64.00..64.00"
    ) in section
    assert "共同経路 / portfolio ES / 完全vector再価格" in section
    assert "整数円 / 自己投票 / 返還 / 特別払戻" in section
    encoded = section.split(
        '<script id="joint-audit-data" type="application/json">', 1
    )[1].split("</script>", 1)[0]
    payload = json.loads(encoded)
    assert payload["rows"][0]["job_id"] == 12001
    assert payload["joint_rows"][0]["joint_covariance_mean"] == -0.03125
    assert payload["models"][0]["name"] == (
        "joint_bankroll_strict_walk_forward_v4"
    )
    assert "モデル実測値・サーバ描画" in section
    assert "3連単LL" in section
    assert "ROI / LCB" in section
    assert "評価プロトコルID / t" in section
    assert "snapshot age" in section
    assert "age 12.5s [4.0..31.0]" in section
    assert "再標本化条件ID" in section
    assert "protocol123" in section

