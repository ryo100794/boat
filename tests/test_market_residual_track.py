import json

from boatrace_ai.web.dashboard import _model_track_summaries


def test_model_report_exposes_newton_market_residual_shadow(tmp_path) -> None:
    (tmp_path / "listwise_market_residual_shadow.json").write_text(
        json.dumps(
            {
                "status": "waiting_for_clean_evaluation_day",
                "calibrator_strategy": "newton_residual",
                "available_races": 270,
                "available_days": 2,
                "evaluated_races": 0,
                "promotion_eligible": False,
            }
        ),
        encoding="utf-8",
    )

    (tmp_path / "listwise_market_residual_intraday_bootstrap.json").write_text(
        json.dumps(
            {
                "evaluation_date": "2026-07-22",
                "point_metrics": {
                    "calibrated_trifecta_log_loss": 3.78665,
                    "calibrated_trifecta_top5_hit_rate": 0.31858,
                },
                "log_loss_difference_calibrated_minus_market": {
                    "observations": 113,
                    "mean_difference": -0.02843,
                    "ci95_lower": -0.05858,
                    "ci95_upper": 0.00212,
                    "probability_less_than_zero": 0.966,
                },
            }
        ),
        encoding="utf-8",
    )

    tracks = {
        row["id"]: row
        for row in _model_track_summaries(tmp_path, [], {"jobs": []})
    }
    residual = tracks["market_residual_shadow"]

    assert residual["status"] == "データ待ち"
    assert residual["include_odds"] is True
    assert residual["eligible_races"] == 0
    assert "Newton法" in residual["training"]
    assert residual["promotion_eligible"] is False
    assert residual["market_comparison_races"] == 113
    assert residual["entry_log_loss"] is None
    assert residual["trifecta_log_loss"] == 3.78665
    assert residual["trifecta_top5_hit_rate"] == 0.31858
    assert residual["market_log_loss_delta"] == -0.02843
    assert residual["market_log_loss_delta_ci95_upper"] == 0.00212
    assert residual["market_improvement_probability"] == 0.966
