from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import pytest

from boatrace_ai.runtime.intraday_t300_shadow import (
    RaceWindow,
    T300Snapshot,
    V21TripleHeadModelAdapter,
    build_adapter,
)


JST = timezone(timedelta(hours=9))
COMBINATIONS = [
    f"{first}-{second}-{third}"
    for first in range(1, 7)
    for second in range(1, 7)
    if second != first
    for third in range(1, 7)
    if third not in (first, second)
]
V21_STRATEGY = (
    "odds_path_observed_closing_return_schedule_quota_triple_head_v21"
)


def calibrator(weight: float) -> dict[str, Any]:
    return {
        "model_weight": weight,
        "temperature": 1.0,
        "converged": True,
        "training_races": 1447,
    }


def deployment() -> dict[str, Any]:
    probability = calibrator(1.0)
    ranking = calibrator(0.8)
    return {
        "calibrator_strategy": V21_STRATEGY,
        "comparison_role": (
            "strict_prior_triple_head_probability_v19_ranking_v18_"
            "purchase_v18_evaluation_only"
        ),
        "deployment_mode": "evaluation_only",
        "real_betting_enabled": False,
        "outer_result_or_payout_used": False,
        "daily_stake_limit_fraction": 1.0,
        "trained_through_date": "2026-07-29",
        "source_evaluation_job_id": 8666,
        "winner_and_logloss_head": "probability_head",
        "trifecta_top5_head": "ranking_head",
        "market_logloss_comparison_head": "probability_head",
        "market_top5_comparison_head": "ranking_head",
        "chronological_bankroll_head": "purchase_head",
        "calibrator": copy.deepcopy(probability),
        "probability_calibrator": probability,
        "ranking_calibrator": ranking,
        "purchase_calibrator": copy.deepcopy(ranking),
        "triple_head_calibration": {
            "architecture": "strict_prior_triple_calibrator_heads_v21",
            "selection_data": (
                "strict_prior_training_and_inner_prequential_folds_only"
            ),
            "outer_holdout_used": False,
            "ranking_purchase_share_v18_selection": True,
            "probability_head": {
                "role": "winner_and_trifecta_logloss",
                "calibrator": copy.deepcopy(probability),
            },
            "ranking_head": {
                "role": "trifecta_top5_ranking",
                "calibrator": copy.deepcopy(ranking),
            },
            "purchase_head": {
                "role": "purchase_policy_and_chronological_bankroll",
                "calibrator": copy.deepcopy(ranking),
            },
        },
        "operational_model": {
            "model_type": "odds_path_observed_closing_return_v4",
            "weights": [0.0] * 11,
            "performance_priors": {"buckets": {}},
        },
        "candidate_policy": {
            "name": "v18-fixed-purchase",
            "ev_threshold": 1.5,
            "max_estimated_ev": None,
            "max_odds": None,
            "max_tickets_per_race": 1,
            "min_model_market_ratio": 1.2,
            "v18_ticket_control": {
                "method": "strict_prior_daily_ticket_lower_quantile",
                "learned_daily_ticket_limit": 26,
                "stake_granularity_yen": 100,
                "result_or_payout_fields_used": False,
            },
        },
        "selected_policy": {"name": "no_bet", "no_bet": True},
    }


def write_artifacts(tmp_path: Path, value: dict[str, Any] | None = None) -> tuple[Path, Path]:
    bundle = tmp_path / "v21.joblib"
    base = tmp_path / "base.joblib"
    joblib.dump({"deployment": value or deployment()}, bundle)
    joblib.dump({"feature_schema_version": 1}, base)
    return bundle, base


def race_snapshot(race_date: str = "2026-07-30") -> tuple[RaceWindow, T300Snapshot]:
    deadline = datetime.fromisoformat(f"{race_date}T12:00:00+09:00")
    race = RaceWindow(f"{race_date.replace('-', '')}-01-01", race_date, "01", 1, deadline)
    captured = race.target_t300_at - timedelta(seconds=5)
    return race, T300Snapshot(
        8666,
        captured,
        captured.isoformat(),
        {},
        {combination: 100.0 for combination in COMBINATIONS},
    )


def runtime_limits() -> dict[str, int]:
    return {
        "schedule_races_elapsed": 10,
        "schedule_races_total": 100,
        "cumulative_ticket_quota": 2,
        "used_tickets": 1,
        "remaining_ticket_quota": 1,
        "gross_stake_yen": 100,
        "realized_cumulative_profit_yen": 0,
        "gross_stake_allowance_yen": 10_000,
        "remaining_gross_stake_allowance_yen": 9_900,
        "allocatable_bankroll_yen": 9_900,
    }


def test_v21_routes_three_heads_at_t300_without_result_or_payout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, base = write_artifacts(tmp_path)
    adapter = V21TripleHeadModelAdapter(
        model_key="v21_daily", bundle_path=bundle, base_model_path=base
    )
    race, snapshot = race_snapshot()
    raw = {
        combination: (0.02 if index == 0 else 0.98 / 119)
        for index, combination in enumerate(COMBINATIONS)
    }
    monkeypatch.setattr(adapter, "_base_probabilities", lambda conn, race: raw)
    monkeypatch.setattr(
        adapter,
        "_runtime_limits",
        lambda conn, race, bankroll_yen: runtime_limits(),
    )

    def attach(rows, operational):
        item = copy.deepcopy(rows[0])
        assert "actual_combination" not in item
        assert "actual_payout_yen" not in item
        item["model_probabilities"] = raw
        item["historical_return_multipliers"] = {
            combination: 1.0 for combination in COMBINATIONS
        }
        return [item]

    monkeypatch.setattr(
        "boatrace_ai.runtime.intraday_t300_shadow.attach_odds_path_model", attach
    )
    monkeypatch.setattr(
        "boatrace_ai.runtime.intraday_t300_shadow.allocate_adaptive_day",
        lambda day, candidates, races, **kwargs: {
            "selected_sample": [
                {**candidates[0], "stake_yen": 100, "hit": False, "return_yen": 0}
            ],
            "allocation_candidate_tickets": len(candidates),
        },
    )

    first = adapter.decide(
        {"result": "1-2-3", "payout": 999_999},
        race,
        snapshot,
        bankroll_yen=10_000,
    )
    second = adapter.decide(
        {"result": "6-5-4", "payout": 0},
        race,
        snapshot,
        bankroll_yen=10_000,
    )

    assert first == second
    assert first.probabilities == pytest.approx(raw)
    assert len(first.selected_candidates) == 1
    diagnostic = first.diagnostics["v21_triple_head"]
    ranking = diagnostic["ranking_probabilities"]
    assert len(ranking) == 120
    assert sum(ranking.values()) == pytest.approx(1.0)
    assert ranking != pytest.approx(first.probabilities)
    assert diagnostic["ranking_top5"] == sorted(
        ranking, key=lambda combination: (-ranking[combination], combination)
    )[:5]
    assert first.selected_candidates[0]["probability"] == pytest.approx(
        ranking[first.selected_candidates[0]["combination"]]
    )
    assert diagnostic["decision_features"] == "t300_or_earlier"
    assert diagnostic["outer_result_used"] is False
    assert diagnostic["outer_payout_used"] is False
    assert diagnostic["real_betting_enabled"] is False
    assert diagnostic["source_evaluation_job_id"] == 8666
    assert all(
        len(diagnostic[key]) == 64
        for key in (
            "probability_output_sha256",
            "ranking_output_sha256",
            "purchase_decisions_sha256",
        )
    )


def test_v21_adapter_registration_and_identity_freeze_inputs(tmp_path: Path) -> None:
    bundle, base = write_artifacts(tmp_path)
    built = build_adapter(
        f"v21_daily:v21_triple_head_t300:{bundle}:{base}"
    )
    original_hash = built.identity.model_hash

    changed = deployment()
    changed["trained_through_date"] = "2026-07-28"
    joblib.dump({"deployment": changed}, bundle)
    changed_identity = build_adapter(
        f"v21_daily:v21_triple_head_t300:{bundle}:{base}"
    ).identity

    assert isinstance(built, V21TripleHeadModelAdapter)
    assert built.identity.strategy_name == "v21_triple_head_t300"
    assert changed_identity.model_key == built.identity.model_key
    assert changed_identity.model_hash != original_hash


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(real_betting_enabled=True), "shadow-only"),
        (
            lambda value: value.update(outer_result_or_payout_used=True),
            "routing or provenance",
        ),
        (
            lambda value: value["triple_head_calibration"].update(
                outer_holdout_used=True
            ),
            "routing or provenance",
        ),
        (
            lambda value: value.update(source_evaluation_job_id=9999),
            "routing or provenance",
        ),
    ],
)
def test_v21_rejects_unsafe_or_noncanonical_artifacts(
    tmp_path: Path, mutation, message: str,
) -> None:
    value = deployment()
    mutation(value)
    bundle, base = write_artifacts(tmp_path, value)
    with pytest.raises(ValueError, match=message):
        V21TripleHeadModelAdapter(
            model_key="v21_daily", bundle_path=bundle, base_model_path=base
        )


def test_v21_rejects_same_day_training_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = deployment()
    value["trained_through_date"] = "2026-07-30"
    bundle, base = write_artifacts(tmp_path, value)
    adapter = V21TripleHeadModelAdapter(
        model_key="v21_daily", bundle_path=bundle, base_model_path=base
    )
    race, snapshot = race_snapshot()

    with pytest.raises(ValueError, match="strictly prior"):
        adapter.decide(object(), race, snapshot, bankroll_yen=10_000)
