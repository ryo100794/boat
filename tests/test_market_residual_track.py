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
