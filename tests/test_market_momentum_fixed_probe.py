from itertools import permutations

from scripts.analyze_market_momentum import evaluate_momentum_candidate


COMBINATIONS = tuple(
    "-".join(map(str, values)) for values in permutations(range(1, 7), 3)
)


def _race(race_date: str, index: int) -> dict:
    actual = COMBINATIONS[index % len(COMBINATIONS)]
    uniform = {
        combination: 1.0 / len(COMBINATIONS)
        for combination in COMBINATIONS
    }
    earlier = {
        combination: 0.8 / (len(COMBINATIONS) - 1)
        for combination in COMBINATIONS
    }
    earlier[actual] = 0.2
    return {
        "race_id": f"{race_date}-{index}",
        "race_date": race_date,
        "actual_combination": actual,
        "model_probabilities": uniform,
        "market_probabilities": uniform,
        "earlier_market_probabilities": earlier,
        "momentum_scale": 0.5,
    }


def test_fixed_regularization_allows_one_prior_day_without_holdout_selection() -> None:
    races = [
        _race(race_date, index)
        for race_date in ("2026-07-21", "2026-07-22")
        for index in range(8)
    ]

    result = evaluate_momentum_candidate(
        races,
        evaluation_date="2026-07-22",
        fixed_regularization=1.0,
    )

    assert result["calibration_dates"] == ["2026-07-21"]
    assert result["fixed_regularization"] == 1.0
    assert result["momentum_newton_residual"]["selection"][
        "validation_design"
    ] == "fixed regularization; no holdout selection"
