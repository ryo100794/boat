from __future__ import annotations

from copy import deepcopy
from itertools import permutations

import pytest

from boatrace_ai.listwise.edge_conditional_probability_lcb_v14 import (
    METHOD,
    artifact_fingerprint,
    fit_edge_conditional_probability_lcb_v14,
    probability_lower_bound_details_v14,
    t300_snapshot_consistency,
)


COMBINATIONS = tuple("".join(map(str, values)) for values in permutations(range(1, 7), 3))
TARGET = "123"


def _race(day: int, *, actual: str = TARGET) -> dict:
    odds = {combination: 120.0 for combination in COMBINATIONS}
    probability = 2.0 / 120.0
    remainder = (1.0 - probability) / 119.0
    probabilities = {
        combination: probability if combination == TARGET else remainder
        for combination in COMBINATIONS
    }
    snapshot_id = day * 100
    return {
        "race_id": f"202607{day:02d}0101",
        "race_date": f"2026-07-{day:02d}",
        "jcd": 1,
        "rno": 1,
        "model_probabilities": probabilities,
        "market_probabilities": {
            combination: 1.0 / 120.0 for combination in COMBINATIONS
        },
        "actual_combination": actual,
        "actual_payout_yen": 5_000,
        "snapshot_id": snapshot_id,
        "odds": odds,
        "odds_checkpoints": {
            "t300": {
                "target_offset_seconds": 300,
                "captured_age_seconds": 300,
                "snapshot_id": snapshot_id,
                "odds": odds,
            },
            "t120": {
                "target_offset_seconds": 120,
                "captured_age_seconds": 120,
                "snapshot_id": snapshot_id + 1,
                "odds": {key: value * 7 for key, value in odds.items()},
            },
        },
    }


def test_zero_hit_bootstrap_factor_is_exactly_zero_without_pseudo_counts() -> None:
    races = [_race(day, actual="456") for day in range(1, 9)]
    artifact = fit_edge_conditional_probability_lcb_v14(
        races, bootstrap_samples=300, seed=41
    )

    assert artifact["ready"] is True
    assert artifact["method"] == METHOD
    assert artifact["optimistic_pseudo_counts"] is False
    assert artifact["double_shrinkage"] is False
    target_cell = next(
        row
        for key, row in artifact["conditional_cells"].items()
        if key.endswith("d_050_100") and row["sum_predicted_probability"] > 0.0
    )
    assert target_cell["observed_hits"] == 0.0
    assert target_cell["lower_observed_to_predicted_ratio"] == 0.0
    assert target_cell["factor"] == 0.0


def test_every_usable_child_factor_is_at_most_its_rank_parent() -> None:
    races = [
        _race(day, actual=TARGET if day in {1, 3, 5, 7} else "456")
        for day in range(1, 9)
    ]
    artifact = fit_edge_conditional_probability_lcb_v14(
        races, bootstrap_samples=300, seed=43
    )

    assert artifact["global_all_ticket_factor_used"] is False
    for cell in artifact["conditional_cells"].values():
        assert cell["factor"] <= cell["parent_factor"] + 1e-15
        assert cell["resolution"] in {
            "cell_min_parent_and_cell_lower",
            "sparse_or_missing_parent_no_bet",
        }


def test_fit_has_strict_input_boundary_and_ignores_payout_and_future_checkpoint() -> None:
    races = [_race(day) for day in range(1, 9)]
    changed = deepcopy(races)
    for race in changed:
        race["actual_payout_yen"] = 999_999
        race["odds_checkpoints"]["t120"]["odds"] = {
            combination: 1.01 for combination in COMBINATIONS
        }

    original = fit_edge_conditional_probability_lcb_v14(
        races, bootstrap_samples=300, seed=47
    )
    altered = fit_edge_conditional_probability_lcb_v14(
        changed, bootstrap_samples=300, seed=47
    )

    assert original["trained_through_date"] == "2026-07-08"
    assert artifact_fingerprint(original) == artifact_fingerprint(altered)
    assert original["uses_payout"] is False
    assert original["decision_checkpoint"] == "t300"


def test_snapshot_mismatch_is_detected_and_rejected() -> None:
    race = _race(1)
    mismatch = deepcopy(race)
    mismatch["odds_checkpoints"]["t300"]["snapshot_id"] = 999
    consistency = t300_snapshot_consistency(mismatch)

    assert consistency == {
        "consistent": False,
        "reason": "t300_snapshot_id_mismatch",
        "v8_snapshot_id": race["snapshot_id"],
        "checkpoint_snapshot_id": 999,
    }

    artifact = fit_edge_conditional_probability_lcb_v14(
        [_race(day) for day in range(1, 9)], bootstrap_samples=300, seed=53
    )
    detail = probability_lower_bound_details_v14(mismatch, TARGET, artifact)
    assert detail["probability"] == 0.0
    assert detail["resolution"] == "inconsistent_t300_snapshot"


def test_registered_band_is_fixed_and_outside_band_is_no_bet() -> None:
    races = [_race(day) for day in range(1, 9)]
    artifact = fit_edge_conditional_probability_lcb_v14(
        races, bootstrap_samples=300, seed=59
    )
    inside = probability_lower_bound_details_v14(races[-1], TARGET, artifact)
    outside = probability_lower_bound_details_v14(races[-1], "456", artifact)

    assert 0.5 <= inside["log_model_market_divergence"] < 1.0
    assert inside["probability"] <= inside["raw_probability"]
    assert outside["probability"] == 0.0
    assert outside["resolution"] == "outside_registered_divergence_band"
