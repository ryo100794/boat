from __future__ import annotations

from typing import Any, Mapping

from .direct_context_market_residual_v25 import direct_context_probabilities
from .flat_policy import simulate_chronological_flat_policy


MODEL_NAME = "v25_top1_narrow_ev_v33"
POLICY_NAME = "registered_v25_top1_ev095_100_forecast_flat100_v33"
REGISTERED_AFTER = "2026-08-01"
POLICY: dict[str, Any] = {
    "name": POLICY_NAME,
    "max_model_rank": 1,
    "min_odds": None,
    "max_odds": 80.0,
    "ev_threshold": 0.95,
    "max_estimated_ev": 1.0,
    "min_model_market_ratio": 0.0,
    "stake_per_ticket_yen": 100,
}


def _identity_probabilities(
    model: Mapping[str, float],
    _market: Mapping[str, float],
    **_kwargs: Any,
) -> dict[str, float]:
    total = sum(float(value) for value in model.values())
    return {str(key): float(value) / total for key, value in model.items()}

def simulate_v25_top1_narrow_v33(
    races: list[dict[str, Any]],
    *,
    probability_artifact: Mapping[str, Any],
    initial_bankroll_yen: int = 10_000,
) -> dict[str, Any]:
    """Replay V33 using only decision-time context and forecast closing odds."""
    transformed = [
        {
            **race,
            "model_probabilities": direct_context_probabilities(
                race, probability_artifact
            ),
        }
        for race in races
    ]
    result = simulate_chronological_flat_policy(
        transformed,
        calibrator={"model_weight": 1.0, "temperature": 1.0},
        policy=POLICY,
        probability_blender=_identity_probabilities,
        initial_bankroll_yen=initial_bankroll_yen,
    )
    return {
        **result,
        "model": MODEL_NAME,
        "policy": dict(POLICY),
        "probability_head": "direct_context_market_residual_v25",
        "odds_head": "strict_prior_point_forecast",
        "real_betting_enabled": False,
    }
