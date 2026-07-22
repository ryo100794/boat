import pytest

from boatrace_ai.listwise.market_edge_diagnostics import (
    edge_records,
    summarize_edge_records,
)


def _blend(model, market, *, model_weight, temperature):
    del market, model_weight, temperature
    return model


def test_edge_diagnostics_separates_all_and_top5_without_selecting_policy() -> None:
    race = {
        "race_id": "2026-07-22-01-01",
        "race_date": "2026-07-22",
        "actual_combination": "1-2-3",
        "actual_payout_yen": 800,
        "model_probabilities": {
            "1-2-3": 0.6,
            "2-1-3": 0.3,
            "3-1-2": 0.1,
        },
        "market_probabilities": {
            "1-2-3": 0.5,
            "2-1-3": 0.3,
            "3-1-2": 0.2,
        },
        "odds": {"1-2-3": 2.0, "2-1-3": 3.0, "3-1-2": 12.0},
    }
    records = edge_records(
        [race],
        calibrator={"model_weight": 1.0, "temperature": 1.0},
        probability_blender=_blend,
    )
    report = summarize_edge_records(records)

    assert len(records) == 3
    assert report["evaluation_days"] == 1
    assert report["evaluation_races"] == 1
    high = next(row for row in report["all_tickets"] if row["bin"] == "gte_1.20")
    assert high["tickets"] == 2
    assert high["hits"] == 1
    assert high["mean_predicted_ev"] == pytest.approx(1.2)
    assert high["realized_roi"] == pytest.approx(4.0)
    assert sum(row["tickets"] for row in report["top5_tickets"]) == 3


def test_edge_diagnostics_uses_forecast_closing_odds_when_available() -> None:
    race = {
        "race_id": "2026-07-22-01-01",
        "race_date": "2026-07-22",
        "actual_combination": "1-2-3",
        "actual_payout_yen": 500,
        "model_probabilities": {"1-2-3": 1.0},
        "market_probabilities": {"1-2-3": 1.0},
        "odds": {"1-2-3": 8.0},
        "estimated_final_odds": {"1-2-3": 5.0},
    }
    records = edge_records(
        [race],
        calibrator={"model_weight": 1.0, "temperature": 1.0},
        probability_blender=_blend,
    )

    assert records[0]["forecast_odds"] == 5.0
    assert records[0]["expected_value"] == 5.0
