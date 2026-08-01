from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ..listwise.closing_odds_t300_nonlinear_v12 import (
    MODEL_NAME as V12_CLOSING_MODEL_NAME,
    forecast_closing_odds_t300_nonlinear_v12,
)
from .intraday_t300_shadow import (
    RaceWindow,
    ShadowDecision,
    T300Snapshot,
    V23Top5NarrowModelAdapter,
    _no_bet,
    _payload_hash,
    main as shadow_main,
    register_adapter,
)
from .uncertainty_adjusted_top5_policy import (
    POLICY_NAME,
    REGISTERED_AFTER,
    STAKE_YEN,
    select_uncertainty_adjusted_top5_candidates,
)


class V32UncertaintyAdjustedTop5ModelAdapter(V23Top5NarrowModelAdapter):
    """Rank and price with separate heads, using the conformal odds lower bound."""

    strategy_name = "v32_uncertainty_adjusted_top5_t300"
    artifact_label = "V32"

    def __init__(
        self, *, model_key: str, bundle_path: Path, base_model_path: Path
    ) -> None:
        super().__init__(
            model_key=model_key,
            bundle_path=bundle_path,
            base_model_path=base_model_path,
        )
        self._closing_v12_model = self._component(
            "closing_t300_v12_model", "closing_v12_model", "closing_model"
        )
        if (
            not self._closing_v12_model.get("ready")
            or str(self._closing_v12_model.get("model_name"))
            != V12_CLOSING_MODEL_NAME
        ):
            raise ValueError("V32 requires the ready V12 T300 closing model")

    def decide(
        self, conn: Any, race: RaceWindow, snapshot: T300Snapshot, *, bankroll_yen: int
    ) -> ShadowDecision:
        trained = str(self._bundle.get("trained_through_date") or "")
        closing_trained = str(
            self._closing_v12_model.get("trained_through_date") or ""
        )
        if trained and trained >= race.race_date:
            raise ValueError("V32 artifacts are not strictly prior to race date")
        if closing_trained and closing_trained >= race.race_date:
            raise ValueError("V32 closing artifact is not strictly prior to race date")
        prepared = self._v23_model_race(conn, race, snapshot)
        if prepared is None:
            return _no_bet("invalid_v32_t300_market_features")
        transformed, market, earlier_reason = prepared
        probability_output = self._blend_head(
            transformed, market, self._probability_calibrator
        )
        ranking_output = self._blend_head(
            transformed, market, self._ranking_calibrator
        )
        if not probability_output or not ranking_output:
            return _no_bet("invalid_v32_probability_or_ranking_head_output")
        forecast = forecast_closing_odds_t300_nonlinear_v12(
            transformed, self._closing_v12_model, prediction_date=race.race_date
        )
        if forecast.get("future_checkpoint_offsets_used"):
            raise ValueError("V32 forecast used a post-T300 checkpoint")
        if not forecast.get("ready"):
            return _no_bet(str(forecast.get("reason") or "v32_closing_not_ready"))
        closing_lower_odds = {
            str(key): float(value)
            for key, value in (forecast.get("lower_final_odds") or {}).items()
        }
        if (
            len(closing_lower_odds) != 120
            or set(closing_lower_odds) != set(probability_output)
            or set(closing_lower_odds) != set(ranking_output)
        ):
            return _no_bet("invalid_v32_closing_lower_output")
        limits = self._capital_limits(conn, race, bankroll_yen=bankroll_yen)
        selected = select_uncertainty_adjusted_top5_candidates(
            ranking_output,
            probability_output,
            closing_lower_odds,
            race_id=race.race_id,
            race_date=race.race_date,
            jcd=race.jcd,
            rno=race.rno,
            snapshot_id=snapshot.snapshot_id,
            captured_at=snapshot.captured_at.isoformat(),
            available_capital_yen=limits["allocatable_bankroll_yen"],
        )
        diagnostics = {
            "v32_uncertainty_adjusted_top5": {
                **limits,
                "status": "selected" if selected else "no_bet",
                "policy_name": POLICY_NAME,
                "registered_after": REGISTERED_AFTER,
                "checkpoint": "t300",
                "source_snapshot_id": snapshot.snapshot_id,
                "source_evaluation_job_id": 10139,
                "ranking_top5": sorted(
                    ranking_output,
                    key=lambda combination: (-ranking_output[combination], combination),
                )[:5],
                "ranking_output_sha256": _payload_hash(ranking_output),
                "probability_output_sha256": _payload_hash(probability_output),
                "ranking_head_usage": "top5_order_only",
                "probability_head_usage": "ticket_probability_and_ev",
                "closing_odds_forecast_source": "v12_t300_conformal_lower",
                "closing_lower_quantile": self._closing_v12_model.get(
                    "lower_quantile_model", {}
                ).get("quantile"),
                "earlier_market_status": earlier_reason,
                "odds_path_points": int(transformed.get("odds_path_points") or 0),
                "decision_features": "t300_or_earlier",
                "outer_result_used": False,
                "outer_payout_used": False,
                "settlement_fields_used_for_capital_only": True,
                "real_betting_enabled": False,
            }
        }
        if limits["allocatable_bankroll_yen"] < STAKE_YEN:
            reason = "v32_daily_capital_exhausted"
        else:
            reason = None if selected else "v32_no_top5_conservative_ev_candidate"
        return ShadowDecision(
            probability_output, closing_lower_odds, selected, reason, diagnostics
        )


def register_v32_adapter() -> None:
    register_adapter(
        V32UncertaintyAdjustedTop5ModelAdapter.strategy_name,
        lambda key, bundle, base: V32UncertaintyAdjustedTop5ModelAdapter(
            model_key=key, bundle_path=bundle, base_model_path=base
        ),
    )


def main() -> int:
    register_v32_adapter()
    return shadow_main()


if __name__ == "__main__":
    raise SystemExit(main())
