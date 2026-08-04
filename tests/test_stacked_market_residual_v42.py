from __future__ import annotations

from datetime import date, timedelta

import pytest

from boatrace_ai.listwise import stacked_market_residual_v42 as subject


def _race(day: date, winner: str) -> dict:
    return {
        "race_id": f"{day.isoformat()}-{winner}",
        "race_date": day.isoformat(),
        "actual_combination": winner,
        "market_probabilities": {"1-2-3": 0.6, "1-3-2": 0.4},
    }


def test_market_stack_is_exact_market_distribution() -> None:
    market = {"1-2-3": 0.6, "1-3-2": 0.4}
    result = subject._blend(
        (market, {"1-2-3": 0.9, "1-3-2": 0.1}, market),
        (1.0, 0.0, 0.0),
    )
    assert result == pytest.approx(market)


def test_stack_weight_uses_separate_prior_block_and_refits(monkeypatch) -> None:
    start = date(2026, 1, 1)
    calibration = [
        _race(start + timedelta(days=index), "1-2-3") for index in range(10)
    ]
    evaluation = [_race(start + timedelta(days=10), "1-2-3")]
    calls: list[list[str]] = []

    def fake_components(races, *, num_threads):
        calls.append([race["race_date"] for race in races])
        return (
            {"artifact": {"regularization": 0.1, "kind": "linear"}},
            {
                "artifact": {"kind": "nonlinear"},
                "selected_shrinkage": 0.5,
                "selected_context_variant": "core",
                "selected_tree_preset": "tiny",
            },
        )

    monkeypatch.setattr(subject, "_fit_components", fake_components)
    monkeypatch.setattr(
        subject,
        "direct_context_probabilities",
        lambda race, artifact: {"1-2-3": 0.9, "1-3-2": 0.1},
    )
    monkeypatch.setattr(
        subject,
        "nonlinear_residual_probabilities",
        lambda race, artifact, *, shrinkage: {"1-2-3": 0.5, "1-3-2": 0.5},
    )
    result = subject.fit_temporal_stacked_market_residual(
        calibration,
        evaluation,
        stack_candidates=(
            {"name": "market", "market": 1.0, "linear": 0.0, "nonlinear": 0.0},
            {"name": "linear", "market": 0.0, "linear": 1.0, "nonlinear": 0.0},
        ),
        num_threads=1,
    )

    assert result["selected_stack"] == "linear"
    assert result["outer_period_used_for_selection"] is False
    assert result["base_training_through"] < result["stack_validation_from"]
    assert result["stack_validation_from"] < evaluation[0]["race_date"]
    assert len(calls[0]) == 8
    assert len(calls[1]) == 10
    assert result["artifact"]["artifact_sha256"]
