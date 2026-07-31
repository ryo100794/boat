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
    V23Top5NarrowModelAdapter,
    build_adapter,
)


COMBINATIONS = [
    f"{first}-{second}-{third}"
    for first in range(1, 7)
    for second in range(1, 7)
    if second != first
    for third in range(1, 7)
    if third not in (first, second)
]


def _calibrator(weight: float) -> dict[str, Any]:
    return {"model_weight": weight, "temperature": 1.0, "converged": True}


def _deployment() -> dict[str, Any]:
    probability = _calibrator(1.0)
    ranking = _calibrator(1.0)
    return {
        "calibrator_strategy": "odds_path_observed_closing_return_schedule_quota_triple_head_v21",
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
            "selection_data": "strict_prior_training_and_inner_prequential_folds_only",
            "outer_holdout_used": False,
            "ranking_purchase_share_v18_selection": True,
            "probability_head": {"role": "winner_and_trifecta_logloss", "calibrator": copy.deepcopy(probability)},
            "ranking_head": {"role": "trifecta_top5_ranking", "calibrator": copy.deepcopy(ranking)},
            "purchase_head": {"role": "purchase_policy_and_chronological_bankroll", "calibrator": copy.deepcopy(ranking)},
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
        "closing_odds_selection": {
            "selected": "momentum",
            "baseline_model": {},
            "momentum_model": {},
        },
    }


def _artifacts(tmp_path: Path) -> tuple[Path, Path]:
    bundle = tmp_path / "v21.joblib"
    base = tmp_path / "base.joblib"
    joblib.dump({"deployment": _deployment()}, bundle)
    joblib.dump({"feature_schema_version": 1}, base)
    return bundle, base


class _Rows:
    def fetchall(self):
        return []


class _Connection:
    def __init__(self, result: str, payout: int) -> None:
        self.result = result
        self.payout = payout

    def execute(self, query, parameters):
        return _Rows()


def test_v23_uses_ranking_and_forecast_only_and_stays_shadow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, base = _artifacts(tmp_path)
    adapter = V23Top5NarrowModelAdapter(
        model_key="v23_daily", bundle_path=bundle, base_model_path=base
    )
    values = {combination: 0.9 / 115 for combination in COMBINATIONS}
    for combination in sorted(COMBINATIONS)[:5]:
        values[combination] = 0.02
    market = {combination: 1.0 / 120 for combination in COMBINATIONS}
    transformed = {
        "model_probabilities": values,
        "odds_path_points": 4,
    }
    monkeypatch.setattr(
        adapter,
        "_v23_model_race",
        lambda conn, race, snapshot: (transformed, market, "ok"),
    )
    forecast = {combination: 200.0 for combination in COMBINATIONS}
    for combination in sorted(COMBINATIONS)[:5]:
        forecast[combination] = 51.0
    monkeypatch.setattr(
        "boatrace_ai.runtime.intraday_t300_shadow.attach_selected_closing_odds",
        lambda races, selection: [{**races[0], "estimated_final_odds": forecast, "closing_odds_forecast_source": "momentum"}],
    )
    start = datetime(2026, 7, 30, 12, 10, tzinfo=timezone(timedelta(hours=9)))
    race = RaceWindow("2026-07-30-01-01", "2026-07-30", "01", 1, start)
    snapshot = T300Snapshot(
        23,
        race.target_t300_at,
        race.target_t300_at.isoformat(),
        {},
        {combination: 100.0 for combination in COMBINATIONS},
    )

    first = adapter.decide(
        _Connection("1-2-3", 999_999), race, snapshot, bankroll_yen=10_000
    )
    second = adapter.decide(
        _Connection("6-5-4", 0), race, snapshot, bankroll_yen=10_000
    )

    assert first == second
    assert len(first.selected_candidates) == 5
    assert sum(row["stake_yen"] for row in first.selected_candidates) == 500
    assert first.no_bet_reason is None
    diagnostic = first.diagnostics["v23_top5_narrow"]
    assert diagnostic["closing_odds_forecast_source"] == "momentum"
    assert diagnostic["decision_features"] == "t300_or_earlier"
    assert diagnostic["outer_result_used"] is False
    assert diagnostic["outer_payout_used"] is False
    assert diagnostic["real_betting_enabled"] is False


def test_v23_is_registered_with_distinct_model_key(tmp_path: Path) -> None:
    bundle, base = _artifacts(tmp_path)
    adapter = build_adapter(
        f"v23_daily:v23_top5_narrow_t300:{bundle}:{base}"
    )
    assert isinstance(adapter, V23Top5NarrowModelAdapter)
    assert adapter.identity.model_key == "v23_daily"
    assert adapter.identity.strategy_name == "v23_top5_narrow_t300"


def test_prewarm_runs_once_per_adapter_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, base = _artifacts(tmp_path)
    adapter = V23Top5NarrowModelAdapter(
        model_key="v23_daily", bundle_path=bundle, base_model_path=base
    )
    calls = {"state": 0, "rows": 0}

    def state(conn, *, race_date):
        calls["state"] += 1
        return {"race_date": race_date}

    def rows(conn, *, race_date, feature_schema_version):
        calls["rows"] += 1
        return {}

    monkeypatch.setattr(
        "boatrace_ai.runtime.intraday_t300_shadow.historical_state", state
    )
    monkeypatch.setattr(
        "boatrace_ai.runtime.intraday_t300_shadow.load_date_races", rows
    )

    adapter.prewarm(object(), "2026-08-01")
    adapter.prewarm(object(), "2026-08-01")

    assert calls == {"state": 1, "rows": 1}
