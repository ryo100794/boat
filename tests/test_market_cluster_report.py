import json

from boatrace_ai.web.dashboard import _model_track_summaries, _quality_gates


def _formal_market_result() -> dict:
    return {
        "evaluated_races": 600,
        "calibrated_trifecta_log_loss": 3.72,
        "trifecta_top5_hit_rate": 0.33,
        "promotion_eligible": False,
        "market_comparison": {
            "confidence_pass": False,
            "race_level_confidence_pass": True,
            "day_cluster_confidence_pass": False,
            "log_loss_difference_calibrated_minus_market": {
                "observations": 600,
                "mean_difference": -0.03,
                "ci95_lower": -0.05,
                "ci95_upper": -0.01,
                "probability_less_than_zero": 0.999,
            },
            "top5_hit_difference_calibrated_minus_market": {
                "observations": 600,
                "mean_difference": 0.01,
                "ci95_lower": 0.0,
                "ci95_upper": 0.02,
            },
            "day_cluster_log_loss_difference_calibrated_minus_market": {
                "observations": 600,
                "clusters": 5,
                "mean_difference": -0.03,
                "ci95_lower": -0.07,
                "ci95_upper": 0.01,
            },
            "day_cluster_top5_hit_difference_calibrated_minus_market": {
                "observations": 600,
                "clusters": 5,
                "mean_difference": 0.01,
                "ci95_lower": -0.01,
                "ci95_upper": 0.03,
            },
        },
    }


def test_model_report_exposes_day_cluster_confidence(tmp_path) -> None:
    path = tmp_path / "listwise_market_residual_shadow.json"
    path.write_text(json.dumps(_formal_market_result()), encoding="utf-8")

    tracks = {
        row["id"]: row
        for row in _model_track_summaries(tmp_path, [], {"jobs": []})
    }
    residual = tracks["market_residual_shadow"]

    assert residual["market_comparison_days"] == 5
    assert residual["market_race_confidence_pass"] is True
    assert residual["market_day_confidence_pass"] is False
    assert residual["market_day_log_loss_delta_ci95_upper"] == 0.01
    assert residual["market_confidence_pass"] is False


def test_quality_gate_evidence_includes_day_cluster_interval(tmp_path) -> None:
    path = tmp_path / "listwise_market_residual_shadow.json"
    path.write_text(json.dumps(_formal_market_result()), encoding="utf-8")

    gate = next(
        row
        for row in _quality_gates(tmp_path, {"jobs": []})
        if row["target"] == "M6 T-5市場信頼区間"
    )

    assert gate["status"] == "未達"
    assert "day cluster 5日" in gate["evidence"]
    assert "[-0.07000, +0.01000]" in gate["evidence"]
