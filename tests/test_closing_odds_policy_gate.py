from itertools import permutations

from boatrace_ai.listwise.market_calibration import (
    closing_odds_training_ready,
    evaluate_closing_odds_quantiles,
    verifiable_closing_odds_races,
)


COMBINATIONS = tuple(
    "-".join(map(str, values)) for values in permutations(range(1, 7), 3)
)


def _race(day: int, index: int, *, verified: bool = True) -> dict:
    odds = {
        combination: 10.0 + combination_index
        for combination_index, combination in enumerate(COMBINATIONS)
    }
    scale = 1.0 + 0.01 * ((index % 3) - 1)
    return {
        "race_id": f"2026-07-{day:02d}-{index}",
        "race_date": f"2026-07-{day:02d}",
        "odds": odds,
        "closing_odds": {
            combination: value * scale for combination, value in odds.items()
        },
        "closing_source_changed": verified,
        "closing_odds_changed": verified and scale != 1.0,
        "closing_snapshot_age_seconds": 10.0,
    }


def test_unverifiable_repeated_snapshot_is_not_a_closing_teacher() -> None:
    repeated = _race(20, 0, verified=False)
    changed = _race(20, 1, verified=True)

    assert verifiable_closing_odds_races([repeated, changed]) == [changed]


def test_policy_gate_requires_seven_days_and_500_verified_races() -> None:
    too_few = [_race(20 + index % 7, index) for index in range(499)]
    enough = [_race(20 + index % 7, index) for index in range(500)]

    assert closing_odds_training_ready(too_few) is False
    assert closing_odds_training_ready(enough) is True


def test_quantile_research_metrics_use_daily_walk_forward_only() -> None:
    races = [_race(day, day) for day in (20, 21, 22)]
    result = evaluate_closing_odds_quantiles(races)

    assert result["status"] == "evaluated"
    assert result["evaluation_method"] == "expanding_daily_walk_forward"
    assert result["evaluation_days"] == 2
    assert result["evaluation_races"] == 2
    assert result["closing_odds_log_mae"] is not None
    assert result["teacher"].startswith("last_preclose_odds")


def test_quantile_research_reports_insufficient_independent_days() -> None:
    result = evaluate_closing_odds_quantiles([_race(20, 1)])

    assert result == {
        "status": "insufficient_independent_snapshots",
        "teacher": "last_preclose_odds_with_verified_source_or_value_change",
        "eligible_races": 1,
        "eligible_days": 1,
        "minimum_evaluation_days": 2,
    }
