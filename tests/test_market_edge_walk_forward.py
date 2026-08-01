from boatrace_ai.fast_math import TRIFECTA_COMBINATIONS
from boatrace_ai.listwise.market_edge_diagnostics import (
    summarize_edge_stability_grid,
    walk_forward_edge_diagnostics,
)


def _race(day: str) -> dict:
    combinations = ["-".join(str(lane) for lane in item) for item in TRIFECTA_COMBINATIONS]
    probability = 1.0 / len(combinations)
    odds = {combination: 90.0 + index for index, combination in enumerate(combinations)}
    return {
        "race_id": f"{day}-01-01",
        "race_date": day,
        "actual_combination": combinations[0],
        "actual_payout_yen": 9000,
        "model_probabilities": {combination: probability for combination in combinations},
        "market_probabilities": {combination: probability for combination in combinations},
        "odds": odds,
        "closing_odds": {combination: value * 0.95 for combination, value in odds.items()},
    }


def test_walk_forward_edge_diagnostics_scores_only_later_days() -> None:
    report = walk_forward_edge_diagnostics(
        [_race("2026-07-20"), _race("2026-07-21"), _race("2026-07-22")]
    )

    assert report["evaluation_days"] == 2
    assert report["evaluation_races"] == 2
    assert [fold["evaluation_date"] for fold in report["folds"]] == [
        "2026-07-21",
        "2026-07-22",
    ]
    assert report["folds"][0]["training_dates"] == ["2026-07-20"]
    assert sum(row["tickets"] for row in report["all_tickets"]) == 240
    assert sum(row["tickets"] for row in report["top5_tickets"]) == 10



def test_stability_grid_reports_daily_portfolio_risk_and_concentration() -> None:
    records = []
    for day, hit in (("2026-07-20", True), ("2026-07-21", False)):
        records.extend(
            [
                {
                    "race_date": day,
                    "race_id": f"{day}-01-01",
                    "combination": "1-2-3",
                    "probability_rank": 1,
                    "probability": 0.30,
                    "forecast_odds": 10.0,
                    "expected_value": 3.0,
                    "ev_bin": "gte_1.20",
                    "hit": hit,
                    "return_yen": 500 if hit else 0,
                },
                {
                    "race_date": day,
                    "race_id": f"{day}-01-01",
                    "combination": "1-3-2",
                    "probability_rank": 2,
                    "probability": 0.20,
                    "forecast_odds": 10.0,
                    "expected_value": 2.0,
                    "ev_bin": "gte_1.20",
                    "hit": False,
                    "return_yen": 0,
                },
            ]
        )

    report = summarize_edge_stability_grid(records)
    cell = report["cells"][0]

    assert cell["rank_group"] == "top5"
    assert cell["odds_band"] == "lt_20"
    assert cell["tickets"] == 4
    assert cell["hits"] == 1
    assert cell["hit_days"] == 1
    assert cell["expected_hits"] == 1.0
    assert cell["mean_daily_no_hit_probability"] == 0.5
    assert cell["profitable_day_fraction"] == 0.5
    assert cell["realized_roi"] == 1.25
    assert cell["roi_without_largest_hit"] == 0.0
    assert cell["hit_return_hhi"] == 1.0


def test_stability_grid_includes_rank_group_boundaries() -> None:
    records = [
        {
            "race_date": "2026-07-20",
            "race_id": f"r{rank}",
            "combination": "1-2-3",
            "probability_rank": rank,
            "probability": 0.01,
            "forecast_odds": 30.0,
            "expected_value": 0.30,
            "ev_bin": "lt_0.80",
            "hit": False,
            "return_yen": 0,
        }
        for rank in (5, 6, 20, 21)
    ]

    report = summarize_edge_stability_grid(records)
    tickets_by_group = {
        cell["rank_group"]: cell["tickets"] for cell in report["cells"]
    }

    assert tickets_by_group == {"top5": 1, "6-20": 2, "21+": 1}


def test_walk_forward_edge_diagnostics_can_keep_real_t5_prices() -> None:
    report = walk_forward_edge_diagnostics(
        [_race("2026-07-20"), _race("2026-07-21"), _race("2026-07-22")],
        forecast_closing=False,
    )

    assert report["price_basis"] == "real_t5"
    assert report["evaluation_races"] == 2
    assert sum(row["tickets"] for row in report["all_tickets"]) == 240
