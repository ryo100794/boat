from boatrace_ai.adaptive_allocation import allocate_adaptive_day


def _candidates(count: int) -> list[dict]:
    return [
        {
            "race_id": f"race-{index}",
            "race_date": "2026-07-20",
            "combination": "1-2-3",
            "probability": 0.055,
            "estimated_odds": 20.0,
            "estimated_ev": 1.10,
            "actual_payout_yen": 2_000,
            "hit": False,
        }
        for index in range(count)
    ]


def _allocate(mode: str) -> dict:
    candidates = _candidates(30)
    return allocate_adaptive_day(
        "2026-07-20",
        candidates,
        {item["race_id"] for item in candidates},
        daily_budget_yen=10_000,
        fractional_kelly=0.25,
        max_daily_exposure_fraction=0.30,
        min_daily_exposure_fraction=0.10,
        race_cap_fraction=0.05,
        ticket_cap_fraction=0.02,
        max_daily_tickets=30,
        allocation_mode=mode,
        stake_granularity_yen=100,
        min_stake_yen=100,
    )


def test_normalized_kelly_distributes_rounding_units_to_top_candidates() -> None:
    result = _allocate("normalized_kelly")

    assert result["stake_yen"] == 1_000
    assert result["tickets"] == 10
    assert all(row["stake_yen"] == 100 for row in result["selected_sample"])


def test_kelly_floor_does_not_force_minimum_exposure() -> None:
    result = _allocate("kelly_floor")

    assert result["stake_yen"] == 0
    assert result["tickets"] == 0

