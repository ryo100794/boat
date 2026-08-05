from __future__ import annotations

from datetime import date, timedelta

from boatrace_ai.listwise import mature_stacked_value as subject


def _race(day: date, index: int) -> dict:
    race_date = day.isoformat()
    return {
        "race_id": f"{race_date}-{index}",
        "race_date": race_date,
        "actual_combination": "1-2-3",
        "actual_payout_yen": 300,
        "odds": {"1-2-3": 3.0, "1-3-2": 4.0},
        "model_probabilities": {"1-2-3": 0.6, "1-3-2": 0.4},
        "market_probabilities": {"1-2-3": 0.6, "1-3-2": 0.4},
        "snapshot_id": index,
    }


class _NoPurchaseArtifact:
    ready = True
    trained_through_date = "2026-06-29"

    def predict(self, raw_ev, probability_rank=None, forecast_odds=None):
        return {"purchase_lcb95_available": False}

    def as_dict(self):
        return {
            "ready": True,
            "ready_reasons": [],
            "trained_through_date": self.trained_through_date,
            "training_days": 120,
            "candidate_days": 120,
            "tickets": 2400,
            "context_ready_cells": 2,
            "context_cells": 12,
            "cells": [],
        }


def test_mature_value_keeps_60_120_and_outer_periods_disjoint(
    monkeypatch,
) -> None:
    observed = {}

    def fake_fit(model_training, evaluation, *, num_threads):
        observed["training"] = model_training
        assert evaluation == []
        assert num_threads == 2
        return {
            "base_training_through": "2026-02-17",
            "stack_validation_from": "2026-02-18",
            "selected_stack": "market",
            "selected_weights": {
                "market": 1.0,
                "linear": 0.0,
                "nonlinear": 0.0,
            },
            "component_selection": {},
            "artifact": {"artifact_sha256": "a" * 64},
        }

    def fake_contextual(records, **kwargs):
        observed["records"] = list(records)
        observed["contextual_kwargs"] = kwargs
        return _NoPurchaseArtifact()

    monkeypatch.setattr(
        subject, "fit_temporal_stacked_market_residual", fake_fit
    )
    monkeypatch.setattr(
        subject,
        "_score",
        lambda races, artifact: [dict(race) for race in races],
    )
    monkeypatch.setattr(
        subject,
        "stacked_metrics",
        lambda races, artifact: {"evaluated_races": len(races)},
    )
    monkeypatch.setattr(
        subject,
        "fit_contextual_empirical_ev_calibration",
        fake_contextual,
    )
    start = date(2026, 1, 1)
    calibration = [
        _race(start + timedelta(days=day), day * 10 + race)
        for day in range(180)
        for race in range(10)
    ]
    evaluation = [
        _race(start + timedelta(days=180 + day), 5000 + day * 10 + race)
        for day in range(5)
        for race in range(10)
    ]

    result = subject.evaluate_mature_stacked_value(
        calibration,
        evaluation,
        daily_budget_yen=10_000,
        num_threads=2,
    )

    assert len(observed["training"]) == 600
    assert len(observed["records"]) == 2400
    assert observed["contextual_kwargs"]["min_days"] == 120
    assert observed["contextual_kwargs"]["prediction_date"] == "2026-06-30"
    assert result["model_training_days"] == 60
    assert result["value_calibration_days"] == 120
    assert result["evaluation_from"] == "2026-06-30"
    assert result["calibration_ledger_candidates"] == 2400
    assert result["evaluation_ledger_candidates"] == 100
    assert result["outer_period_used_for_selection"] is False
    assert result["purchase_max_rank"] == 20
    assert result["evidence_role"] == (
        "retrospective_research_only_candidate_universe_search"
    )
    assert result["promotion_eligible"] is False
    assert result["real_betting_enabled"] is False


def test_mature_value_requires_180_prior_days() -> None:
    start = date(2026, 1, 1)
    races = [_race(start + timedelta(days=day), day) for day in range(179)]

    result = subject.evaluate_mature_stacked_value(
        races,
        [],
        daily_budget_yen=10_000,
    )

    assert result["status"] == "insufficient_nested_days"
    assert result["required_days"] == 180
    assert result["real_betting_enabled"] is False
