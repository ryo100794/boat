from boatrace_ai.listwise.market_diagnostics import calibrator_stability_rows


def test_candidate_rows_reports_day_level_market_regret() -> None:
    races = [
        {
            "race_date": "2026-07-20",
            "actual_combination": "1-2-3",
            "model_probabilities": {"1-2-3": 0.7, "1-3-2": 0.3},
            "market_probabilities": {"1-2-3": 0.6, "1-3-2": 0.4},
        },
        {
            "race_date": "2026-07-21",
            "actual_combination": "1-3-2",
            "model_probabilities": {"1-2-3": 0.4, "1-3-2": 0.6},
            "market_probabilities": {"1-2-3": 0.5, "1-3-2": 0.5},
        },
    ]

    rows = calibrator_stability_rows(races)

    assert len(rows) == 15
    assert all(row["races"] == 2 for row in rows)
    assert all(row["days"] == 2 for row in rows)
    assert all(len(row["daily"]) == 2 for row in rows)
    assert all("market_regret" in day for row in rows for day in row["daily"])
