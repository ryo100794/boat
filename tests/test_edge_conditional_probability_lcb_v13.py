from __future__ import annotations

from copy import deepcopy

import pytest

from boatrace_ai.listwise.edge_conditional_probability_lcb_v13 import (
    METHOD,
    artifact_fingerprint,
    fit_edge_conditional_probability_lcb,
    probability_lower_bound_details,
)
from boatrace_ai.listwise.strict_prior_divergence_diagnostics import (
    aggregate_strict_prior_divergence_band_metrics,
    strict_prior_divergence_band_metrics,
)


COMBINATIONS = [
    f"{first}{second}{third}"
    for first in range(1, 7)
    for second in range(1, 7)
    if second != first
    for third in range(1, 7)
    if third not in (first, second)
]


def _race(day: int, *, actual: str = "456", payout: int = 2_000) -> dict:
    odds = {
        combination: 80.0 + index
        for index, combination in enumerate(COMBINATIONS)
    }
    probabilities = {combination: 0.8 / 119 for combination in COMBINATIONS}
    probabilities["123"] = 0.2
    return {
        "race_id": f"202607{day:02d}0101",
        "race_date": f"2026-07-{day:02d}",
        "jcd": 1,
        "rno": 1,
        "model_probabilities": probabilities,
        "actual_combination": actual,
        "actual_payout_yen": payout,
        "odds_checkpoints": {
            "t300": {
                "target_offset_seconds": 300,
                "captured_age_seconds": 301,
                "odds": odds,
            },
            "t120": {
                "target_offset_seconds": 120,
                "captured_age_seconds": 121,
                "odds": {key: value * 9 for key, value in odds.items()},
            },
        },
    }


def test_fit_is_daily_clustered_hierarchical_and_deterministic() -> None:
    races = [_race(day) for day in range(1, 9)]
    artifact = fit_edge_conditional_probability_lcb(
        races, bootstrap_samples=200, seed=17
    )
    repeated = fit_edge_conditional_probability_lcb(
        races, bootstrap_samples=200, seed=17
    )

    assert artifact["ready"] is True
    assert artifact["method"] == METHOD
    assert artifact["bootstrap_unit"] == "whole_race_day"
    assert artifact["trained_through_date"] == "2026-07-08"
    assert artifact["uses_payout"] is False
    assert artifact_fingerprint(artifact) == artifact_fingerprint(repeated)
    assert artifact["conditional_cells"]
    assert any(
        row["resolution"] == "conditional_cell_shrunk_to_rank"
        and 0.0 < row["shrinkage_weight"] < 1.0
        for row in artifact["conditional_cells"].values()
    )
    for cell in artifact["conditional_cells"].values():
        low = min(cell["raw_lower_factor"], cell["parent_factor"])
        high = max(cell["raw_lower_factor"], cell["parent_factor"])
        assert low <= cell["factor"] <= high


def test_fit_ignores_payout_and_future_t120_but_uses_prior_results_as_labels() -> None:
    races = [_race(day) for day in range(1, 9)]
    changed = deepcopy(races)
    for race in changed:
        race["actual_payout_yen"] = 999_999
        race["odds_checkpoints"]["t120"]["odds"] = {
            key: 1.01 for key in COMBINATIONS
        }

    original = fit_edge_conditional_probability_lcb(
        races, bootstrap_samples=200, seed=19
    )
    altered = fit_edge_conditional_probability_lcb(
        changed, bootstrap_samples=200, seed=19
    )

    assert artifact_fingerprint(original) == artifact_fingerprint(altered)


def test_probability_lookup_requires_complete_t300_and_never_increases_probability() -> None:
    races = [_race(day) for day in range(1, 9)]
    artifact = fit_edge_conditional_probability_lcb(
        races, bootstrap_samples=200, seed=23
    )
    detail = probability_lower_bound_details(races[-1], "123", artifact)
    assert detail["probability"] <= detail["raw_probability"]
    assert detail["divergence_band"] == "d_ge_100"
    assert detail["cell_key"].startswith("top2|")

    missing = deepcopy(races[-1])
    del missing["odds_checkpoints"]["t300"]
    no_market = probability_lower_bound_details(missing, "123", artifact)
    assert no_market["probability"] == 0.0
    assert no_market["resolution"] == "missing_complete_t300_market"


def test_divergence_band_metrics_are_settlement_only_and_aggregate_roi() -> None:
    race = _race(1, actual="123", payout=2_000)
    metric = strict_prior_divergence_band_metrics([race])
    high = next(
        row for row in metric["bands"]
        if row["divergence_band"] == "d_ge_100"
    )

    assert metric["closing_odds_used_as_feature"] is False
    assert metric["result_and_payout_usage"] == "settlement_diagnostic_only"
    assert high["races"] == 1
    assert high["tickets"] >= 1
    assert high["sum_predicted_probability"] >= 0.2
    assert high["hits"] == 1
    assert high["hit_to_expected_ratio"] == pytest.approx(
        high["hits"] / high["sum_predicted_probability"]
    )
    assert high["actual_payout_roi"] == pytest.approx(
        high["return_yen"] / high["stake_yen"]
    )

    aggregate = aggregate_strict_prior_divergence_band_metrics([metric, metric])
    aggregate_high = next(
        row for row in aggregate["bands"]
        if row["divergence_band"] == "d_ge_100"
    )
    assert aggregate_high["races"] == 2
    assert aggregate_high["tickets"] == high["tickets"] * 2
    assert aggregate_high["return_yen"] == high["return_yen"] * 2
