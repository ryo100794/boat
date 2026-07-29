from __future__ import annotations

import math
from copy import deepcopy
from itertools import permutations

import pytest

from boatrace_ai.listwise import odds_path_selection_conformal_v10 as v10
from boatrace_ai.listwise.edge_conditional_probability_lcb_v13 import (
    probability_lower_bound_details,
)
from boatrace_ai.listwise.strict_prior_t300_divergence_passthrough_v16 import (
    METHOD,
    fit_strict_prior_t300_divergence_passthrough_v16,
)


COMBINATIONS = tuple(
    "".join(map(str, values)) for values in permutations(range(1, 7), 3)
)
TARGET = "123"


def _race(divergence: float) -> dict:
    market_probability = 1.0 / 120.0
    target_probability = market_probability * math.exp(divergence)
    remainder = (1.0 - target_probability) / 119.0
    probabilities = {
        combination: target_probability if combination == TARGET else remainder
        for combination in COMBINATIONS
    }
    odds = {combination: 120.0 for combination in COMBINATIONS}
    return {
        "race_id": "202607300101",
        "race_date": "2026-07-30",
        "jcd": 1,
        "rno": 1,
        "model_probabilities": probabilities,
        "market_probabilities": {
            combination: market_probability for combination in COMBINATIONS
        },
        "actual_combination": TARGET,
        "actual_payout_yen": 99_999,
        "snapshot_id": 300,
        "odds": odds,
        "odds_checkpoints": {
            "t300": {
                "target_offset_seconds": 300,
                "captured_age_seconds": 300,
                "snapshot_id": 300,
                "odds": odds,
            }
        },
    }


def test_fixed_band_passes_raw_probability_and_enforces_boundaries() -> None:
    artifact = fit_strict_prior_t300_divergence_passthrough_v16([])

    lower = _race(0.5)
    inside = _race(0.75)
    upper = _race(1.0)
    below = _race(0.499)

    lower_detail = probability_lower_bound_details(lower, TARGET, artifact)
    inside_detail = probability_lower_bound_details(inside, TARGET, artifact)
    upper_detail = probability_lower_bound_details(upper, TARGET, artifact)
    below_detail = probability_lower_bound_details(below, TARGET, artifact)

    assert lower_detail["probability"] == pytest.approx(
        lower["model_probabilities"][TARGET]
    )
    assert inside_detail["probability"] == pytest.approx(
        inside["model_probabilities"][TARGET]
    )
    assert inside_detail["factor"] == 1.0
    assert upper_detail["probability"] == 0.0
    assert below_detail["probability"] == 0.0


def test_artifact_is_result_and_payout_invariant() -> None:
    original = [_race(0.75)]
    changed = deepcopy(original)
    changed[0]["actual_combination"] = "654"
    changed[0]["actual_payout_yen"] = 9_999_999

    first = fit_strict_prior_t300_divergence_passthrough_v16(original)
    second = fit_strict_prior_t300_divergence_passthrough_v16(changed)

    assert first == second
    assert first["artifact_method"] == METHOD
    assert first["uses_result"] is False
    assert first["uses_payout"] is False
    assert first["fit_parameters_from_outcomes"] is False
    assert all(node["factor"] == 1.0 for node in first["rank_nodes"].values())
    assert all(
        cell["factor"] == 1.0
        for cell in first["conditional_cells"].values()
    )


def test_incomplete_t300_is_no_bet() -> None:
    race = _race(0.75)
    race["odds_checkpoints"] = {}
    artifact = fit_strict_prior_t300_divergence_passthrough_v16([])

    detail = probability_lower_bound_details(race, TARGET, artifact)

    assert detail["probability"] == 0.0
    assert detail["resolution"] == "inconsistent_t300_snapshot"


def test_purchase_decision_is_independent_of_result_and_payout(monkeypatch) -> None:
    race = _race(0.75)
    changed = deepcopy(race)
    changed["actual_combination"] = "654"
    changed["actual_payout_yen"] = 9_999_999
    artifact = fit_strict_prior_t300_divergence_passthrough_v16([])
    forecast = {combination: 100.0 for combination in COMBINATIONS}
    envelope = {"ready": True, "haircut": 0.9}
    calls = []
    original_allocator = v10.allocate_discrete_log_day

    def capture_allocator(*args, **kwargs):
        calls.append({
            "candidates": [dict(candidate) for candidate in args[1]],
            "settlements": dict(kwargs["settlements"]),
        })
        return original_allocator(*args, **kwargs)

    monkeypatch.setattr(v10, "allocate_discrete_log_day", capture_allocator)
    first, _ = v10._simulate_selection_conformal_policy(
        [race],
        closing_forecasts={race["race_id"]: forecast},
        probability_lcb=artifact,
        daily_budget_yen=10_000,
        selection_conformal=envelope,
    )
    second, _ = v10._simulate_selection_conformal_policy(
        [changed],
        closing_forecasts={race["race_id"]: forecast},
        probability_lcb=artifact,
        daily_budget_yen=10_000,
        selection_conformal=envelope,
    )

    assert calls[0]["candidates"] == calls[1]["candidates"]
    assert calls[0]["settlements"] != calls[1]["settlements"]
    assert first["tickets"] == second["tickets"]
    assert first["stake_yen"] == second["stake_yen"]
    assert first["daily"][0]["expected_log_growth"] == pytest.approx(
        second["daily"][0]["expected_log_growth"]
    )
