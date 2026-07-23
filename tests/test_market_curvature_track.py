import json

from boatrace_ai.web.dashboard import _model_track_summaries


def test_model_report_records_rejected_market_curvature_probe(tmp_path) -> None:
    (tmp_path / "listwise_market_curvature_intraday_probe.json").write_text(
        json.dumps(
            {
                "status": "rejected_no_incremental_value",
                "evaluated_races": 113,
                "calibrated_trifecta_log_loss": 3.787799,
                "trifecta_top5_hit_rate": 0.318584,
                "promotion_eligible": False,
                "market_comparison": {
                    "confidence_pass": False,
                    "log_loss_difference_calibrated_minus_market": {
                        "observations": 113,
                        "mean_difference": -0.027282,
                        "ci95_lower": -0.059683,
                        "ci95_upper": 0.007380,
                        "probability_less_than_zero": 0.94325,
                    },
                    "top5_hit_difference_calibrated_minus_market": {
                        "observations": 113,
                        "mean_difference": 0.008850,
                        "ci95_lower": 0.0,
                        "ci95_upper": 0.026549,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    tracks = {
        row["id"]: row
        for row in _model_track_summaries(tmp_path, [], {"jobs": []})
    }
    curvature = tracks["market_curvature_probe"]

    assert curvature["role"] == "開発診断のみ・増分なしで棄却"
    assert "乖離符号付き二乗" in curvature["training"]
    assert curvature["eligible_races"] == 113
    assert curvature["entry_log_loss"] is None
    assert curvature["trifecta_log_loss"] == 3.787799
    assert curvature["trifecta_top5_hit_rate"] == 0.318584
    assert curvature["market_log_loss_delta_ci95_upper"] == 0.007380
    assert curvature["promotion_eligible"] is False
