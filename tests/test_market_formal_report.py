import json

from boatrace_ai.web.dashboard import _model_track_summaries


def _comparison(*, observations: int, mean: float, upper: float) -> dict:
    return {
        "confidence_pass": upper <= 0,
        "log_loss_difference_calibrated_minus_market": {
            "observations": observations,
            "mean_difference": mean,
            "ci95_lower": mean - 0.01,
            "ci95_upper": upper,
            "probability_less_than_zero": 0.99,
        },
        "top5_hit_difference_calibrated_minus_market": {
            "observations": observations,
            "mean_difference": 0.01,
            "ci95_lower": 0.0,
            "ci95_upper": 0.02,
        },
    }


def test_model_report_prefers_formal_fold_confidence_over_intraday_probe(
    tmp_path,
) -> None:
    (tmp_path / "listwise_market_residual_shadow.json").write_text(
        json.dumps(
            {
                "evaluated_races": 120,
                "calibrated_trifecta_log_loss": 3.70,
                "trifecta_top5_hit_rate": 0.33,
                "market_comparison": _comparison(
                    observations=120,
                    mean=-0.04,
                    upper=-0.01,
                ),
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
                    "calibrated_trifecta_log_loss": 4.20,
                    "calibrated_trifecta_top5_hit_rate": 0.20,
                },
                **_comparison(observations=10, mean=0.20, upper=0.30),
            }
        ),
        encoding="utf-8",
    )

    tracks = {
        row["id"]: row
        for row in _model_track_summaries(tmp_path, [], {"jobs": []})
    }
    residual = tracks["market_residual_shadow"]

    assert residual["entry_log_loss"] == 3.70
    assert residual["trifecta_top5_hit_rate"] == 0.33
    assert residual["market_comparison_races"] == 120
    assert residual["market_log_loss_delta"] == -0.04
    assert residual["market_log_loss_delta_ci95_upper"] == -0.01
    assert residual["market_confidence_pass"] is True
