from __future__ import annotations

from typing import Any, Iterable

from .odds_path_conservative_v7 import (
    _walk_forward_evaluate_conservative_ev,
)
from .odds_path_probability_v8 import (
    attach_odds_path_probability_v8,
    fit_odds_path_probability_v8,
)


MODEL_NAME = "odds_path_market_offset_crossfit_conservative_ev_v8"
STRATEGY_NAME = "odds_path_market_offset_crossfit_conservative_ev"
REGISTERED_AFTER = "2026-07-29"
PROSPECTIVE_OUTPUT_KEY = (
    "prospective_market_offset_crossfit_conservative_ev_v8_walk_forward"
)


def walk_forward_evaluate_v8(
    races: list[dict[str, Any]],
    *,
    daily_budget_yen: int,
    min_calibration_days: int,
    evaluation_dates: Iterable[str] | None = None,
) -> dict[str, Any]:
    return _walk_forward_evaluate_conservative_ev(
        races,
        daily_budget_yen=daily_budget_yen,
        min_calibration_days=min_calibration_days,
        evaluation_dates=evaluation_dates,
        model_name=MODEL_NAME,
        strategy_name=STRATEGY_NAME,
        registered_after=REGISTERED_AFTER,
        prospective_output_key=PROSPECTIVE_OUTPUT_KEY,
        comparison_role=(
            "real_t5_market_offset_crossfit_q20_fixed_safe_ev_shadow"
        ),
        prospective_comparison_role=(
            "pre_registered_strict_outer_day_v8_shadow"
        ),
        probability_fit=fit_odds_path_probability_v8,
        probability_attach=attach_odds_path_probability_v8,
        deployment_waiting_status="shadow_only_until_v8_promotion_gate",
    )
