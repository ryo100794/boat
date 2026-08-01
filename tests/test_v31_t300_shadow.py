from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from boatrace_ai.runtime.intraday_t300_shadow import RaceWindow, T300Snapshot
from boatrace_ai.runtime.v31_uncertainty_adjusted_shadow import (
    V31UncertaintyAdjustedTop5ModelAdapter,
)


COMBINATIONS = [
    f"{first}-{second}-{third}"
    for first in range(1, 7)
    for second in range(1, 7)
    if second != first
    for third in range(1, 7)
    if third not in (first, second)
]


def distribution(top: list[str], top_value: float) -> dict[str, float]:
    remainder = (1.0 - len(top) * top_value) / (120 - len(top))
    values = {combination: remainder for combination in COMBINATIONS}
    values.update({combination: top_value for combination in top})
    return values


def test_v31_keeps_ranking_order_and_ev_probability_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = object.__new__(V31UncertaintyAdjustedTop5ModelAdapter)
    adapter._bundle = {"trained_through_date": "2026-07-31"}
    adapter._closing_v12_model = {
        "ready": True,
        "trained_through_date": "2026-07-31",
        "lower_quantile_model": {"quantile": 0.1},
    }
    adapter._probability_calibrator = {"head": "probability"}
    adapter._ranking_calibrator = {"head": "ranking"}
    ranked = sorted(COMBINATIONS)[:5]
    probability_output = distribution(ranked, 0.015)
    ranking_output = distribution(ranked, 0.02)
    market = distribution([], 0.0)
    transformed = {"odds_path_points": 4}
    monkeypatch.setattr(
        adapter, "_v23_model_race",
        lambda conn, race, snapshot: (transformed, market, "ok"),
    )
    monkeypatch.setattr(
        adapter, "_blend_head",
        lambda transformed, market, calibrator: (
            probability_output
            if calibrator["head"] == "probability"
            else ranking_output
        ),
    )
    monkeypatch.setattr(
        adapter, "_capital_limits",
        lambda conn, race, bankroll_yen: {
            "gross_stake_yen": 0,
            "realized_cumulative_profit_yen": 0,
            "gross_stake_allowance_yen": 10_000,
            "remaining_gross_stake_allowance_yen": 10_000,
            "allocatable_bankroll_yen": 10_000,
        },
    )
    lower = {combination: 10.0 for combination in COMBINATIONS}
    for combination in ranked:
        lower[combination] = 70.0
    monkeypatch.setattr(
        "boatrace_ai.runtime.v31_uncertainty_adjusted_shadow."
        "forecast_closing_odds_t300_nonlinear_v12",
        lambda transformed, model, prediction_date: {
            "ready": True,
            "future_checkpoint_offsets_used": [],
            "lower_final_odds": lower,
        },
    )
    start = datetime(2026, 8, 2, 12, 10, tzinfo=timezone(timedelta(hours=9)))
    race = RaceWindow("2026-08-02-01-01", "2026-08-02", "01", 1, start)
    snapshot = T300Snapshot(
        24, race.target_t300_at, race.target_t300_at.isoformat(), {},
        {combination: 100.0 for combination in COMBINATIONS},
    )

    decision = adapter.decide(object(), race, snapshot, bankroll_yen=10_000)

    assert [row["combination"] for row in decision.selected_candidates] == ranked
    for row in decision.selected_candidates:
        combination = row["combination"]
        assert row["probability"] == decision.probabilities[combination]
        assert row["probability"] == probability_output[combination]
        assert row["ranking_score"] == ranking_output[combination]
        assert row["estimated_ev"] == pytest.approx(
            probability_output[combination] * lower[combination]
        )
    assert decision.closing_lower_odds == lower
    diagnostic = decision.diagnostics["v31_uncertainty_adjusted_top5"]
    assert diagnostic["ranking_head_usage"] == "top5_order_only"
    assert diagnostic["probability_head_usage"] == "ticket_probability_and_ev"
    assert diagnostic["real_betting_enabled"] is False


def test_v31_rejects_future_closing_artifact() -> None:
    adapter = object.__new__(V31UncertaintyAdjustedTop5ModelAdapter)
    adapter._bundle = {"trained_through_date": "2026-07-31"}
    adapter._closing_v12_model = {"trained_through_date": "2026-08-02"}
    race = RaceWindow(
        "2026-08-02-01-01", "2026-08-02", "01", 1,
        datetime(2026, 8, 2, 12, 10, tzinfo=timezone(timedelta(hours=9))),
    )
    snapshot = T300Snapshot(24, race.target_t300_at, None, {}, {})

    with pytest.raises(ValueError, match="closing artifact"):
        adapter.decide(object(), race, snapshot, bankroll_yen=10_000)
