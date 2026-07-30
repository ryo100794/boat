from __future__ import annotations

import json
from typing import Any

from boatrace_ai.listwise import odds_path_role_integrated_v13 as v13
from boatrace_ai.listwise import odds_path_role_integrated_v14 as v14
from boatrace_ai.listwise.edge_conditional_probability_lcb_v14 import METHOD
from boatrace_ai.listwise.odds_path_selection_conformal_v10 import (
    _simulate_selection_conformal_policy,
)

from test_edge_conditional_probability_lcb_v14 import COMBINATIONS, TARGET, _race


class _Estimator:
    pass


def _ready_artifact() -> dict[str, Any]:
    return {
        "ready": True,
        "method": METHOD,
        "trained_through_date": "2026-07-28",
        "rank_nodes": {
            "top2": {"usable": True, "factor": 1.0},
        },
        "conditional_cells": {
            "top2|p_010_020|d_050_100": {
                "usable": True,
                "factor": 1.0,
                "parent_factor": 1.0,
            },
        },
    }


def test_calibration_population_exactly_matches_preallocation_candidates() -> None:
    race = _race(30)
    closing = {
        str(race["race_id"]): {
            combination: 80.0 for combination in COMBINATIONS
        }
    }
    _bankroll, diagnostic = _simulate_selection_conformal_policy(
        [race],
        closing_forecasts=closing,
        probability_lcb=_ready_artifact(),
        daily_budget_yen=10_000,
        selection_conformal={"ready": True, "haircut": 1.0},
        capture_preallocation_candidates=True,
    )
    population = diagnostic.pop("_preallocation_candidates")
    metrics = v14._selected_population_metrics(
        [race],
        closing_forecasts=closing,
        probability_lcb=_ready_artifact(),
        selected_candidates=population,
    )

    expected_keys = [
        [str(row["race_id"]), str(row["combination"])] for row in population
    ]
    assert metrics["candidate_keys"] == expected_keys
    assert metrics["candidate_count"] == len(population) == 1
    assert metrics["candidate_population_stage"].endswith("before_allocation")
    assert metrics["observed_hits"] == 1


def test_v13_is_research_invalid_and_cannot_promote(monkeypatch) -> None:
    monkeypatch.setattr(v13, "walk_forward_evaluate_v12", lambda *args, **kwargs: {
        "folds": [],
        "prospective_role_integrated_v12_walk_forward": {
            "promotion_gate": {"bankroll_pass": True},
            "promotion_eligible": True,
        },
        "deployment_configuration": {},
    })
    result = v13.walk_forward_evaluate_v13(
        [], daily_budget_yen=10_000, min_calibration_days=2
    )

    assert result["status"] == "research_invalid_deprecated"
    assert result["research_invalid"] is True
    assert result["deprecated"] is True
    assert result["promotion_eligible"] is False
    assert result[v13.PROSPECTIVE_OUTPUT_KEY]["promotion_eligible"] is False
    assert result["promotion_gate"]["valid_probability_lower_bound_pass"] is False


def _calibration(day: str, candidates: int = 60, hits: int = 4) -> dict[str, Any]:
    adjusted = 2.0
    return {
        "evaluation_days": 1,
        "candidate_count": candidates,
        "raw_predicted_hits": 3.0,
        "adjusted_predicted_hits": adjusted,
        "observed_hits": hits,
        "candidate_binary_brier_score": 0.01,
        "candidate_binary_log_loss": 0.05,
        "candidate_population_fingerprint": day,
        "inconsistent_t300_snapshot_races": 0,
        "daily": [{
            "race_date": day,
            "candidate_count": candidates,
            "raw_predicted_hits": 3.0,
            "adjusted_predicted_hits": adjusted,
            "observed_hits": hits,
        }],
    }


def test_v14_promotion_uses_only_post_registration_days_and_serializes(monkeypatch) -> None:
    dates = ["2026-07-25", "2026-07-30", "2026-07-31", "2026-08-01", "2026-08-02", "2026-08-03"]
    folds = [
        {
            "evaluation_date": day,
            "selected_policy": {"name": "v12"},
            "leakage_guard": {"pass": True},
            "closing_model": {
                "model_name": "closing_odds_t300_nonlinear_v12",
                "estimator": _Estimator(),
            },
            "probability_lcb_metrics": {
                "selected_candidate_calibration": _calibration(
                    day, candidates=10_000 if day == "2026-07-25" else 60,
                    hits=1_000 if day == "2026-07-25" else 4,
                ),
                "strict_prior_divergence_bands": {"bands": []},
            },
        }
        for day in dates
    ]
    monkeypatch.setattr(v14, "walk_forward_evaluate_v12", lambda *args, **kwargs: {
        "folds": folds,
        "prospective_role_integrated_v12_walk_forward": {
            "roi": 1.10,
            "profit_yen": 1_000,
            "tickets": 300,
            "hit_tickets": 20,
            "roi_without_largest_hit": 1.02,
            "profit_without_largest_hit_yen": 200,
            "daily_cluster_bootstrap_roi_lower_95": 1.01,
            "promotion_gate": {
                "largest_hit_excluded_roi_pass": True,
                "daily_cluster_bootstrap_roi_lower_pass": True,
            },
        },
        "deployment_configuration": {
            "closing_t300_v12_model": {
                "model_name": "closing_odds_t300_nonlinear_v12",
                "estimator": _Estimator(),
            },
        },
    })

    result = v14.walk_forward_evaluate_v14(
        [], daily_budget_yen=10_000, min_calibration_days=2
    )
    prospective = result[v14.PROSPECTIVE_OUTPUT_KEY]
    historical = result[v14.HISTORICAL_OUTPUT_KEY]

    assert prospective["selected_candidate_calibration"]["evaluation_days"] == 5
    assert prospective["selected_candidate_calibration"]["candidate_count"] == 300
    assert prospective["selected_candidate_calibration"]["observed_hits"] == 20
    assert prospective["promotion_eligible"] is True
    assert historical["promotion_evidence"] is False
    assert historical["selected_candidate_calibration"]["candidate_count"] == 10_000
    assert result["fixed_policy"]["registered_divergence_lower_inclusive"] == 0.5
    assert result["fixed_policy"]["registered_divergence_upper_exclusive"] == 1.0
    assert "closing_model" not in result["folds"][0]
    assert "closing_t300_v12_model" not in result["deployment_configuration"]
    json.dumps(result)


def test_promotion_gate_enforces_days_candidates_hits_and_bootstrap_ratio() -> None:
    passing = v14._promotion_calibration_gate({
        "evaluation_days": 5,
        "candidate_count": 300,
        "observed_hits": 20,
        "day_bootstrap_observed_to_adjusted_predicted_ratio_lower_95": 1.01,
        "inconsistent_t300_snapshot_races": 0,
    })
    failing = v14._promotion_calibration_gate({
        "evaluation_days": 4,
        "candidate_count": 299,
        "observed_hits": 19,
        "day_bootstrap_observed_to_adjusted_predicted_ratio_lower_95": 1.0,
        "inconsistent_t300_snapshot_races": 1,
    })

    assert all(value for key, value in passing.items() if key.endswith("_pass"))
    assert not any(value for key, value in failing.items() if key.endswith("_pass"))


def test_prefiltered_prospective_snapshot_mismatch_still_blocks_promotion(
    monkeypatch,
) -> None:
    dates = ["2026-07-30", "2026-07-31", "2026-08-01", "2026-08-02", "2026-08-03"]
    folds = [{
        "evaluation_date": day,
        "selected_policy": {"name": "v12"},
        "leakage_guard": {"pass": True},
        "probability_lcb_metrics": {
            "selected_candidate_calibration": _calibration(day),
            "strict_prior_divergence_bands": {"bands": []},
        },
    } for day in dates]
    monkeypatch.setattr(v14, "walk_forward_evaluate_v12", lambda *args, **kwargs: {
        "folds": folds,
        "prospective_role_integrated_v12_walk_forward": {
            "promotion_gate": {"bankroll_pass": True},
        },
        "deployment_configuration": {},
    })
    mismatch = _race(30)
    mismatch["odds_checkpoints"]["t300"]["snapshot_id"] = 999

    result = v14.walk_forward_evaluate_v14(
        [mismatch], daily_budget_yen=10_000, min_calibration_days=2
    )
    prospective = result[v14.PROSPECTIVE_OUTPUT_KEY]

    assert prospective["selected_candidate_calibration"][
        "inconsistent_t300_snapshot_races"
    ] == 1
    assert prospective["promotion_gate"]["t300_snapshot_consistency_pass"] is False
    assert prospective["promotion_eligible"] is False


def test_v14_requires_fixed_ten_thousand_yen_daily_bankroll() -> None:
    try:
        v14.walk_forward_evaluate_v14(
            [], daily_budget_yen=9_900, min_calibration_days=2
        )
    except ValueError as exc:
        assert "JPY10000" in str(exc)
    else:
        raise AssertionError("V14 accepted a non-registered daily bankroll")
