from __future__ import annotations

from copy import deepcopy

import pytest

from boatrace_ai.listwise import empirical_policy_replay as replay


def _race(race_date: str) -> dict:
    return {
        "race_id": f"{race_date}-01-01",
        "race_date": race_date,
        "actual_combination": "1-2-3",
    }


def test_replay_rejects_source_edge_mismatch(monkeypatch) -> None:
    races = [_race("2026-01-01")]
    source = {
        "folds": [
            {
                "evaluation_date": "2026-01-01",
                "evaluation_races": 1,
                "operational_model": {"model": "test"},
                "purchase_calibrator": {"model_weight": 1.0},
                "closing_odds_selection": None,
                "closing_odds_policy_fallback_reason": "test",
            }
        ],
        "edge_diagnostics": {"expected": True},
    }
    monkeypatch.setattr(replay, "attach_odds_path_model", lambda rows, model: rows)
    monkeypatch.setattr(
        replay,
        "_attach_t5_policy_fallback",
        lambda rows, *, reason: deepcopy(rows),
    )
    monkeypatch.setattr(
        replay,
        "apply_prequential_closing_odds_policy_inputs",
        lambda rows, inputs: rows,
    )
    monkeypatch.setattr(replay, "edge_records", lambda *args, **kwargs: [])
    monkeypatch.setattr(replay, "summarize_edge_records", lambda rows: {})
    monkeypatch.setattr(
        replay,
        "_fit_prior_empirical_ev_artifact",
        lambda *args, **kwargs: type(
            "Artifact",
            (),
            {
                "ready": False,
                "trained_through_date": None,
                "training_days": 0,
                "tickets": 0,
                "candidate_days": 0,
                "ready_reasons": ("prior",),
            },
        )(),
    )
    monkeypatch.setattr(
        replay,
        "simulate_empirical_lcb_policy",
        lambda *args, **kwargs: {"daily": []},
    )
    monkeypatch.setattr(replay, "policy_edge_records", lambda *args, **kwargs: [])

    with pytest.raises(ValueError, match="does not match source edge diagnostics"):
        replay.replay_bandwise_empirical_policy(source, {"races": races})


def test_replay_requires_exact_holdout_population() -> None:
    source = {
        "folds": [{"evaluation_date": "2026-01-01", "evaluation_races": 2}],
        "edge_diagnostics": {},
    }

    with pytest.raises(ValueError, match="holdout race count mismatch"):
        replay.replay_bandwise_empirical_policy(
            source,
            {"races": [_race("2026-01-01")]},
        )
