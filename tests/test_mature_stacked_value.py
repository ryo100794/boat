from __future__ import annotations

from datetime import date, timedelta

import pytest

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
            "training_days": 60,
            "candidate_days": 60,
            "tickets": 2400,
            "context_ready_cells": 2,
            "context_cells": 12,
            "cells": [],
        }


def test_mature_value_keeps_60_60_60_and_outer_periods_disjoint(
    monkeypatch,
) -> None:
    observed = {}

    def fake_fit(model_training, evaluation, *, num_threads):
        observed.setdefault("training_calls", []).append(list(model_training))
        assert evaluation == []
        assert num_threads == 2
        return {
            "base_training_through": "2026-02-17",
            "stack_validation_from": "2026-02-18",
            "raw_selected_stack": "linear",
            "stack_selection_gate": {
                "status": "fallback_market",
                "fallback_reasons": ["validation_top5_below_market"],
            },
            "selected_stack": "market",
            "selected_weights": {
                "market": 1.0,
                "linear": 0.0,
                "nonlinear": 0.0,
            },
            "component_selection": {},
            "artifact": {
                "selected_stack": "market",
                "weights": {
                    "market": 1.0,
                    "linear": 0.0,
                    "nonlinear": 0.0,
                },
                "artifact_sha256": "a" * 64,
            },
        }

    def fake_select(probability, stack_selection):
        observed["stack_selection"] = list(stack_selection)
        return dict(probability["artifact"]), {
            "policy_id": subject.VALUE_ALIGNED_STACK_POLICY_ID,
            "status": "not_available",
            "outer_period_used": False,
            "search_validation_draw_sets_disjoint": True,
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
        "select_value_aligned_stack",
        fake_select,
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

    assert len(observed["training_calls"]) == 2
    assert len(observed["training_calls"][0]) == 600
    assert len(observed["training_calls"][1]) == 1200
    assert len(observed["stack_selection"]) == 600
    assert len(observed["records"]) == 1200
    assert observed["contextual_kwargs"]["min_days"] == 60
    assert observed["contextual_kwargs"]["min_candidate_days"] == 40
    assert observed["contextual_kwargs"]["min_rank_days"] == 45
    assert observed["contextual_kwargs"]["min_cell_days"] == 30
    assert observed["contextual_kwargs"]["prediction_date"] == "2026-06-30"
    assert result["model_training_days"] == 60
    assert result["value_stack_selection_days"] == 60
    assert result["value_calibration_days"] == 60
    assert result["stack_selection_calibration_disjoint"] is True
    assert result["search_validation_draw_sets_disjoint"] is True
    assert result["evaluation_from"] == "2026-06-30"
    assert result["calibration_ledger_candidates"] == 1200
    assert result["evaluation_ledger_candidates"] == 100
    assert result["outer_period_used_for_selection"] is False
    assert result["purchase_max_rank"] == 20
    assert result["purchase_max_tickets_per_race"] == 1
    assert result["formal_roi_gate"] == {
        "cluster_unit": "complete_race_date",
        "lower_quantile": 0.05,
        "quantile_method": "inverted_cdf",
        "condition": "day_block_roi_lower_quantile_strictly_above_one",
    }
    assert result["bankroll"]["roi_lower_quantile"] == 0.05
    assert result["bankroll"]["roi_quantile_method"] == "inverted_cdf"
    assert result["probability_selection"]["raw_selected_stack"] == "linear"
    assert result["probability_selection"]["selected_stack"] == "market"
    assert result["probability_selection"]["stack_selection_gate"] == {
        "status": "fallback_market",
        "fallback_reasons": ["validation_top5_below_market"],
    }
    assert result["value_aligned_stack_selection"]["status"] == "not_available"
    aligned = result["value_aligned_stack_selection"]
    assert aligned["outer_period_used"] is False
    assert aligned["probability_component_refit_after_selection"] is True
    assert aligned["selected_stack_fixed_before_refit"] is True
    assert aligned["refit_excludes_empirical_gate_calibration"] is True
    assert aligned["refit_training_days"] == 120
    assert aligned["refit_training_races"] == 1200
    assert result["probability_refit"]["training_days"] == 120
    assert result["probability_refit"]["training_races"] == 1200
    assert result["probability_refit"]["selected_stack_fixed_before_refit"] is True
    assert result["probability_refit"]["empirical_gate_calibration_used"] is False
    assert "value-aligned stack reweighting" in result["validation_design"]
    assert result["evidence_role"] == (
        "retrospective_research_only_candidate_universe_search"
    )
    assert result["promotion_eligible"] is False
    assert result["real_betting_enabled"] is False


def test_context_value_audit_uses_predeclared_outer_cells(
    monkeypatch,
) -> None:
    monkeypatch.setattr(subject, "CONTEXT_AUDIT_BOOTSTRAP_SAMPLES", 100)
    calibration = [
        {
            "race_date": "2026-06-01",
            "probability_rank": 1,
            "forecast_odds": 10.0,
            "raw_estimated_ev": 1.1,
            "gross_return_per_yen": 2.0,
        },
        {
            "race_date": "2026-06-02",
            "probability_rank": 4,
            "forecast_odds": 15.0,
            "raw_estimated_ev": 0.9,
            "gross_return_per_yen": 0.0,
        },
        {
            "race_date": "2026-06-02",
            "probability_rank": 6,
            "forecast_odds": 25.0,
            "raw_estimated_ev": 1.2,
            "gross_return_per_yen": 3.0,
        },
    ]
    evaluation = [
        {
            "race_date": "2026-07-01",
            "probability_rank": 2,
            "forecast_odds": 12.0,
            "raw_estimated_ev": 1.05,
            "gross_return_per_yen": 1.5,
        },
    ]

    audit = subject.context_value_audit(calibration, evaluation)

    assert audit["evaluation_used_for_context_definition"] is False
    assert audit["bootstrap_cluster_unit"] == "race_date"
    calibration_cells = {
        (row["rank_group"], row["odds_band"]): row
        for row in audit["calibration"]
    }
    top5 = calibration_cells[("top5", "<20")]
    assert top5["candidates"] == 2
    assert top5["candidate_days"] == 2
    assert top5["mean_predicted_raw_ev"] == 1.0
    assert top5["realized_roi"] == 1.0
    assert top5["roi_excluding_largest_hit"] == 0.0
    rank_6_20 = calibration_cells[("6-20", "20-50")]
    assert rank_6_20["candidates"] == 1
    assert rank_6_20["realized_roi"] == 3.0
    evaluation_cells = {
        (row["rank_group"], row["odds_band"]): row
        for row in audit["evaluation"]
    }
    assert evaluation_cells[("top5", "<20")]["realized_roi"] == 1.5
    assert evaluation_cells[("6-20", ">=101")]["realized_roi"] is None


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

def test_value_aligned_stack_selects_highest_supported_lcb(
    monkeypatch,
) -> None:
    probability = {
        "selected_stack": "linear",
        "artifact": {
            "model": "stacked_market_residual_v42",
            "selected_stack": "linear",
            "weights": {"market": 0.0, "linear": 1.0, "nonlinear": 0.0},
            "artifact_sha256": "a" * 64,
        },
        "stack_candidates": [
            {
                "name": "market",
                "weights": {
                    "market": 1.0,
                    "linear": 0.0,
                    "nonlinear": 0.0,
                },
                "metrics": {"trifecta_log_loss": 1.00},
            },
            {
                "name": "linear",
                "weights": {
                    "market": 0.0,
                    "linear": 1.0,
                    "nonlinear": 0.0,
                },
                "metrics": {"trifecta_log_loss": 0.99},
            },
            {
                "name": "nonlinear",
                "weights": {
                    "market": 0.0,
                    "linear": 0.0,
                    "nonlinear": 1.0,
                },
                "metrics": {"trifecta_log_loss": 0.98},
            },
        ],
    }
    metrics = {
        "market": (0.70, 0.80, 0.60),
        "linear": (0.90, 0.95, 0.61),
        "nonlinear": (1.10, 1.20, 0.59),
    }

    def fake_metrics(races, artifact):
        lcb, roi, top5 = metrics[artifact["selected_stack"]]
        return {
            "tickets": 600,
            "hit_tickets": 20,
            "candidate_days": 100,
            "mean_selected_raw_ev": 1.0,
            "roi": roi,
            "roi_lcb95": lcb,
            "probability_roi_above_one": 0.5,
            "roi_excluding_largest_hit": roi - 0.1,
            "trifecta_top5_hit_rate": top5,
            "market_trifecta_top5_hit_rate": 0.60,
        }

    monkeypatch.setattr(subject, "_value_alignment_metrics", fake_metrics)

    artifact, audit = subject.select_value_aligned_stack(
        probability,
        [{"race_date": "2026-01-01"}],
    )

    assert audit["selected_stack"] == "linear"
    assert audit["status"] == "selected"
    assert audit["outer_period_used"] is False
    assert audit["candidate_family_size"] == 3
    assert audit["shortlisted_candidates"] == 3
    assert audit["selection_lower_quantile"] == 0.01
    assert audit["selection_quantile_method"] == "inverted_cdf"
    assert audit["familywise_candidate_cap"] == 5
    assert audit["minimum_candidate_days"] == 50
    nonlinear = next(
        row for row in audit["candidates"] if row["name"] == "nonlinear"
    )
    assert nonlinear["roi_lcb95"] == 1.10
    assert nonlinear["top5_noninferior_to_market"] is False
    assert nonlinear["eligible"] is False
    assert artifact["selected_stack"] == "linear"
    assert artifact["pre_value_alignment_stack"] == "linear"
    assert artifact["value_alignment_policy_id"] == (
        "top20_max_raw_ev_familywise_q01_top5_noninferiority_disjoint_v2"
    )
    assert audit["stack_selection_shared_with_empirical_gate_training"] is False
    assert audit["search_validation_draw_sets_disjoint"] is True
    assert len(artifact["artifact_sha256"]) == 64

def test_value_alignment_metrics_use_one_ticket_per_race_and_yen_payout(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        subject,
        "_score",
        lambda races, artifact: [dict(race) for race in races],
    )
    monkeypatch.setattr(
        subject,
        "VALUE_ALIGNED_STACK_BOOTSTRAP_SAMPLES",
        100,
    )
    first = _race(date(2026, 1, 1), 1)
    second = _race(date(2026, 1, 2), 2)

    metrics = subject._value_alignment_metrics(
        [first, second],
        {"selected_stack": "test"},
    )

    assert metrics["tickets"] == 2
    assert metrics["hit_tickets"] == 2
    assert metrics["candidate_days"] == 2
    assert metrics["mean_selected_raw_ev"] == pytest.approx(1.8)
    assert metrics["roi"] == 3.0
    assert metrics["roi_lcb95"] == 3.0
    assert metrics["roi_lower_quantile"] == 0.01
    assert metrics["roi_quantile_method"] == "inverted_cdf"
    assert metrics["roi_excluding_largest_hit"] == 1.5
    assert metrics["trifecta_top5_hit_rate"] == 1.0
    assert metrics["market_trifecta_top5_hit_rate"] == 1.0
