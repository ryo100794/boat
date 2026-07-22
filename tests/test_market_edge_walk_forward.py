from boatrace_ai.fast_math import TRIFECTA_COMBINATIONS
from boatrace_ai.listwise.market_edge_diagnostics import (
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



def test_walk_forward_edge_diagnostics_can_keep_real_t5_prices() -> None:
    report = walk_forward_edge_diagnostics(
        [_race("2026-07-20"), _race("2026-07-21"), _race("2026-07-22")],
        forecast_closing=False,
    )

    assert report["price_basis"] == "real_t5"
    assert report["evaluation_races"] == 2
    assert sum(row["tickets"] for row in report["all_tickets"]) == 240
