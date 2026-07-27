import copy
import math
from itertools import permutations

import pytest

from boatrace_ai.listwise.closing_odds_quantile import (
    walk_forward_closing_odds_quantiles,
)


COMBINATIONS = tuple(
    "-".join(map(str, values)) for values in permutations(range(1, 7), 3)
)


def _race(race_date: str, residuals: list[float]) -> dict:
    odds = {
        combination: 10.0 + index
        for index, combination in enumerate(COMBINATIONS)
    }
    return {
        "race_date": race_date,
        "race_id": f"{race_date}-1",
        "odds": odds,
        "closing_odds": {
            combination: odds[combination] * math.exp(residuals[index])
            for index, combination in enumerate(COMBINATIONS)
        },
    }


def _residual_grid(scale: float = 1.0, shift: float = 0.0) -> list[float]:
    return [shift + scale * ((index - 59.5) / 59.5) for index in range(120)]


def _fold(result: dict, race_date: str) -> dict:
    return next(
        fold for fold in result["folds"] if fold["evaluation_date"] == race_date
    )


@pytest.mark.parametrize("adaptive_rate", [-0.01, 1.01, math.nan])
def test_adaptive_rate_must_be_finite_and_bounded(adaptive_rate: float) -> None:
    with pytest.raises(ValueError, match="adaptive_rate"):
        walk_forward_closing_odds_quantiles(
            [_race("2026-07-20", _residual_grid())],
            adaptive_rate=adaptive_rate,
        )


def test_fold_alpha_uses_only_previous_fold_coverage() -> None:
    races = [
        _race("2026-07-20", _residual_grid()),
        _race("2026-07-21", _residual_grid(0.2, 4.0)),
        _race("2026-07-22", _residual_grid(0.2, 4.0)),
        _race("2026-07-23", _residual_grid()),
    ]
    result = walk_forward_closing_odds_quantiles(
        races, regularization=0.0, adaptive_rate=0.5
    )
    first, second = result["folds"][:2]

    assert first["alpha_before"] == pytest.approx(0.20)
    assert second["alpha_before"] == pytest.approx(first["alpha_after"])
    assert first["observed_coverage"] == pytest.approx(
        first["metrics"]["closing_odds_interval_coverage"]
    )
    assert result["adaptive_conformal_method"] == (
        "online_adaptive_conformal_miscoverage_control"
    )
    assert result["target_coverage"] == pytest.approx(0.80)


def test_low_coverage_widens_next_interval_and_high_coverage_recovers_alpha() -> None:
    races = [
        _race("2026-07-20", _residual_grid()),
        _race("2026-07-21", _residual_grid(0.1, 4.0)),
        _race("2026-07-22", _residual_grid(0.1, 0.0)),
        _race("2026-07-23", _residual_grid(0.1, 0.0)),
    ]
    result = walk_forward_closing_odds_quantiles(
        races, regularization=0.0, adaptive_rate=0.5
    )
    low_coverage_fold, recovery_fold = result["folds"][:2]

    assert low_coverage_fold["observed_coverage"] < 0.80
    assert low_coverage_fold["alpha_after"] < low_coverage_fold["alpha_before"]
    assert recovery_fold["interval_quantile_levels"][0] < (
        low_coverage_fold["interval_quantile_levels"][0]
    )
    assert recovery_fold["interval_quantile_levels"][2] > (
        low_coverage_fold["interval_quantile_levels"][2]
    )
    assert recovery_fold["metrics"]["closing_odds_interval_mean_log_width"] > (
        low_coverage_fold["metrics"]["closing_odds_interval_mean_log_width"]
    )
    assert recovery_fold["observed_coverage"] > 0.80
    assert recovery_fold["alpha_after"] > recovery_fold["alpha_before"]


def test_future_target_change_cannot_change_prior_fold_or_alpha() -> None:
    races = [
        _race("2026-07-20", _residual_grid()),
        _race("2026-07-21", _residual_grid(0.5)),
        _race("2026-07-22", _residual_grid(0.4)),
        _race("2026-07-23", _residual_grid(0.3)),
    ]
    changed = copy.deepcopy(races)
    changed[-1] = _race("2026-07-23", _residual_grid(0.01, 8.0))

    baseline = walk_forward_closing_odds_quantiles(
        races, regularization=0.0, adaptive_rate=0.5
    )
    mutated = walk_forward_closing_odds_quantiles(
        changed, regularization=0.0, adaptive_rate=0.5
    )

    assert _fold(baseline, "2026-07-22") == _fold(mutated, "2026-07-22")


def test_adaptive_quantile_levels_remain_ordered_at_alpha_bounds() -> None:
    races = [
        _race("2026-07-20", _residual_grid()),
        _race("2026-07-21", _residual_grid(0.1, 5.0)),
        _race("2026-07-22", _residual_grid()),
        _race("2026-07-23", _residual_grid()),
    ]

    result = walk_forward_closing_odds_quantiles(
        races, regularization=0.0, adaptive_rate=1.0
    )

    for fold in result["folds"]:
        lower, median, upper = fold["interval_quantile_levels"]
        assert 0.0 < lower <= median <= upper < 1.0
        assert 0.02 <= fold["alpha_before"] <= 0.40
        assert 0.02 <= fold["alpha_after"] <= 0.40
