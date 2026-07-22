import json

from boatrace_ai.web.dashboard import _model_track_summaries


def test_model_report_records_rejected_market_momentum_probe(tmp_path) -> None:
    (tmp_path / "listwise_market_momentum_intraday_probe.json").write_text(
        json.dumps(
            {
                "status": "rejected_no_incremental_value",
                "evaluated_races": 113,
                "calibrated_trifecta_log_loss": 3.789103,
                "trifecta_top5_hit_rate": 0.309735,
                "promotion_eligible": False,
                "market_comparison": {
                    "confidence_pass": False,
                    "log_loss_difference_calibrated_minus_market": {
                        "observations": 113,
                        "mean_difference": -0.025978,
                        "ci95_lower": -0.052923,
                        "ci95_upper": 0.002577,
                        "probability_less_than_zero": 0.9641,
                    },
                    "top5_hit_difference_calibrated_minus_market": {
                        "observations": 113,
                        "mean_difference": 0.0,
                        "ci95_lower": -0.026549,
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
    momentum = tracks["market_momentum_probe"]

    assert momentum["role"] == "開発診断のみ・増分なしで棄却"
    assert momentum["eligible_races"] == 113
    assert momentum["entry_log_loss"] == 3.789103
    assert momentum["trifecta_top5_hit_rate"] == 0.309735
    assert momentum["market_log_loss_delta_ci95_upper"] == 0.002577
    assert momentum["promotion_eligible"] is False
