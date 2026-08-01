from copy import deepcopy

from boatrace_ai.listwise.stable_cell_policy import (
    evaluate_walk_forward_stable_cells,
)


def _record(day: int, index: int, *, hit: bool, odds: float = 3.0) -> dict:
    return {
        "race_date": f"2026-01-{day:02d}",
        "race_id": f"2026-01-{day:02d}-01-{index:02d}",
        "combination": f"1-2-{index + 3}",
        "probability_rank": index + 1,
        "probability": 0.5,
        "forecast_odds": odds,
        "expected_value": 1.02,
        "ev_bin": "1.00_1.05",
        "hit": hit,
        "return_yen": 300 if hit else 0,
    }


def _records() -> list[dict]:
    rows = []
    for day in range(1, 7):
        rows.extend([_record(day, 0, hit=True), _record(day, 1, hit=False)])
    return rows


THRESHOLDS = {
    "minimum_days": 5,
    "minimum_tickets": 10,
    "minimum_hit_days": 5,
    "minimum_expected_hits": 5,
    "maximum_mean_daily_no_hit_probability": 0.6,
    "minimum_profitable_day_fraction": 1.0,
    "minimum_roi_without_largest_hit": 1.0,
    "maximum_hit_return_hhi": 0.25,
}


def test_walk_forward_selects_cell_only_after_five_prior_days() -> None:
    result = evaluate_walk_forward_stable_cells(_records(), thresholds=THRESHOLDS)

    assert [row["tickets"] for row in result["daily"]] == [0, 0, 0, 0, 0, 2]
    assert result["days_with_bets"] == 1
    assert result["stake_yen"] == 200
    assert result["return_yen"] == 300
    assert result["roi"] == 1.5
    assert result["promotion_eligible"] is False


def test_current_outcome_cannot_change_prior_cell_selection() -> None:
    records = _records()
    changed = deepcopy(records)
    for row in changed:
        if row["race_date"] == "2026-01-06":
            row["hit"] = False
            row["return_yen"] = 0

    baseline = evaluate_walk_forward_stable_cells(records, thresholds=THRESHOLDS)
    counterfactual = evaluate_walk_forward_stable_cells(
        changed, thresholds=THRESHOLDS
    )

    assert baseline["daily"][-1]["selected_cells"] == counterfactual["daily"][-1][
        "selected_cells"
    ]
    assert baseline["daily"][-1]["return_yen"] != counterfactual["daily"][-1][
        "return_yen"
    ]


def test_daily_budget_caps_purchased_tickets() -> None:
    result = evaluate_walk_forward_stable_cells(
        _records(),
        thresholds=THRESHOLDS,
        daily_budget_yen=100,
    )

    assert result["daily"][-1]["candidate_tickets"] == 2
    assert result["daily"][-1]["tickets"] == 1
    assert result["stake_yen"] == 100
