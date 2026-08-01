from __future__ import annotations

import json
from pathlib import Path

from boatrace_ai.listwise.direct_context_market_residual_v25 import FEATURE_DIMENSION
from boatrace_ai.listwise.market_calibration import (
    load_v25_probability_artifact,
)
from boatrace_ai.listwise.v25_top1_narrow_policy_v33 import (
    POLICY,
    simulate_v25_top1_narrow_v33,
)


COMBINATIONS = [
    f"{first}-{second}-{third}"
    for first in range(1, 7)
    for second in range(1, 7)
    if second != first
    for third in range(1, 7)
    if third not in (first, second)
]


def _race(*, top_odds: float) -> dict:
    top = "1-2-3"
    remainder = 0.9 / 119
    probabilities = {key: remainder for key in COMBINATIONS}
    probabilities[top] = 0.1
    odds = {key: 50.0 for key in COMBINATIONS}
    odds[top] = top_odds
    return {
        "race_id": "2026-08-02-01-01",
        "race_date": "2026-08-02",
        "jcd": "01",
        "rno": 1,
        "model_probabilities": dict(probabilities),
        "market_probabilities": dict(probabilities),
        "estimated_final_odds": odds,
        "actual_combination": top,
        "actual_payout_yen": 1_000,
        "captured_at": "2026-08-02T10:00:00+09:00",
        "odds_deadline_at": "2026-08-02T10:05:00+09:00",
    }


def _artifact() -> dict:
    return {"coefficients": [0.0] * FEATURE_DIMENSION}


def test_v33_selects_only_top1_inside_fixed_ev_band() -> None:
    result = simulate_v25_top1_narrow_v33(
        [_race(top_odds=9.5)],
        probability_artifact=_artifact(),
    )
    assert result["tickets"] == 1
    assert result["hit_tickets"] == 1
    assert result["stake_yen"] == 100
    assert result["return_yen"] == 1_000
    assert result["policy"] == POLICY
    assert result["real_betting_enabled"] is False


def test_v33_rejects_below_and_above_registered_ev_band() -> None:
    below = simulate_v25_top1_narrow_v33(
        [_race(top_odds=9.4)],
        probability_artifact=_artifact(),
    )
    above = simulate_v25_top1_narrow_v33(
        [_race(top_odds=10.1)],
        probability_artifact=_artifact(),
    )
    assert below["tickets"] == 0
    assert above["tickets"] == 0


def test_v25_artifact_loader_validates_and_audits_source(tmp_path: Path) -> None:
    path = tmp_path / "v25.json"
    payload = {
        "temporal_residual_diagnostic": {
            "calibration_through": "2026-07-17",
            "evaluation_from": "2026-07-18",
            "evaluation_through": "2026-07-31",
            "direct_context_market_residual_v25": {
                "inner_fit_through": "2026-07-10",
                "artifact": {
                    "coefficients": [0.0] * FEATURE_DIMENSION,
                    "training_races": 1234,
                },
            },
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    artifact, audit = load_v25_probability_artifact(path)

    assert artifact["training_races"] == 1234
    assert audit["calibration_through"] == "2026-07-17"
    assert audit["feature_dimension"] == FEATURE_DIMENSION
    assert len(audit["sha256"]) == 64
