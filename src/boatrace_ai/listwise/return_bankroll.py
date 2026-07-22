from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from ..adaptive_allocation import allocate_adaptive_day, append_day_result, zero_totals
from ..fast_math import TRIFECTA_COMBINATIONS
from ..roi_attribution import (
    merge_roi_attribution,
    new_roi_attribution,
    summarize_fold_signal_stability,
    summarize_roi_attribution,
)
from .direct_bankroll import (
    COMBINATION_INDEX,
    COMBINATION_LABELS,
    direct_candidates,
    standard_direct_policy,
)
from .return_calibrator import (
    fit_expected_return_calibrator,
    predict_expected_returns,
)


COMBINATION_LANES = np.asarray(TRIFECTA_COMBINATIONS, dtype=np.int64) - 1


def _settle_candidate_days(
    candidates_by_day: dict[str, list[dict[str, Any]]],
    evaluated_by_day: dict[str, set[str]],
    race_dates: list[str],
    policy: dict[str, Any],
) -> dict[str, Any]:
    totals = zero_totals()
    daily = []
    state = (0, 0, 0)
    fold_count = min(5, len(race_dates))
    fold_attributions = [new_roi_attribution() for _ in range(fold_count)]
    for day_index, race_date in enumerate(race_dates):
        fold_index = min(
            fold_count - 1,
            day_index * fold_count // len(race_dates),
        )
        result = allocate_adaptive_day(
            race_date,
            candidates_by_day.get(race_date, []),
            evaluated_by_day.get(race_date, set()),
            daily_budget_yen=int(policy["daily_budget_yen"]),
            fractional_kelly=float(policy["fractional_kelly"]),
            max_daily_exposure_fraction=float(
                policy["max_daily_exposure_fraction"]
            ),
            min_daily_exposure_fraction=float(
                policy["min_daily_exposure_fraction"]
            ),
            race_cap_fraction=float(policy["race_cap_fraction"]),
            ticket_cap_fraction=float(policy["ticket_cap_fraction"]),
            max_daily_tickets=int(policy["max_daily_tickets"]),
            allocation_mode=str(policy["allocation_mode"]),
            stake_granularity_yen=int(policy["stake_granularity_yen"]),
            min_stake_yen=int(policy["min_stake_yen"]),
            roi_attribution=fold_attributions[fold_index],
        )
        state = append_day_result(
            daily,
            totals,
            result,
            cumulative_profit=state[0],
            peak_profit=state[1],
            max_drawdown=state[2],
        )

    attribution = new_roi_attribution()
    for fold_attribution in fold_attributions:
        merge_roi_attribution(attribution, fold_attribution)
    attribution_summary = summarize_roi_attribution(attribution)
    attribution_summary["fold_stability"] = summarize_fold_signal_stability(
        [
            summarize_roi_attribution(fold_attribution)
            for fold_attribution in fold_attributions
        ]
    )
    stake_yen = int(totals["stake_yen"])
    return_yen = int(totals["return_yen"])
    return {
        "policy": policy,
        "evaluation_days": len(daily),
        "evaluated_races": int(totals["evaluated_races"]),
        "candidate_tickets": int(totals["candidate_tickets"]),
        "selected_tickets": int(totals["tickets"]),
        "races_bet": int(totals["races_bet"]),
        "hit_tickets": int(totals["hit_tickets"]),
        "stake_yen": stake_yen,
        "return_yen": return_yen,
        "profit_yen": return_yen - stake_yen,
        "roi": return_yen / stake_yen if stake_yen else 0.0,
        "winning_days": int(totals["winning_days"]),
        "losing_days": int(totals["losing_days"]),
        "max_drawdown_yen": int(state[2]),
        "ticket_roi_attribution": attribution_summary,
        "daily": daily,
    }


def simulate_expected_return_calibrated_bankroll(
    probabilities: np.ndarray,
    *,
    race_keys: list[tuple[str, str, str, int]],
    payouts: dict[str, dict[str, Any]],
    market_reference_probabilities: np.ndarray,
    calibration_probabilities: np.ndarray,
    calibration_market_reference_probabilities: np.ndarray,
    calibration_race_keys: list[tuple[str, str, str, int]],
    policy: dict[str, Any] | None = None,
    regularization: float = 0.01,
    max_iterations: int = 20,
    batch_races: int = 500,
) -> dict[str, Any]:
    values = np.asarray(probabilities, dtype=np.float64)
    market_values = np.asarray(market_reference_probabilities, dtype=np.float64)
    if values.shape != (len(race_keys), len(COMBINATION_LABELS)):
        raise ValueError("probability matrix and race keys must align")
    if market_values.shape != values.shape:
        raise ValueError("market reference probability matrix must align")
    calibration_values = np.asarray(calibration_probabilities, dtype=np.float64)
    calibration_market_values = np.asarray(
        calibration_market_reference_probabilities,
        dtype=np.float64,
    )
    if calibration_values.shape != (
        len(calibration_race_keys),
        len(COMBINATION_LABELS),
    ):
        raise ValueError("calibration probabilities and race keys must align")
    if calibration_market_values.shape != calibration_values.shape:
        raise ValueError("calibration market probabilities must align")

    calibrator = fit_expected_return_calibrator(
        calibration_values,
        calibration_market_values,
        calibration_race_keys,
        payouts,
        COMBINATION_LANES,
        COMBINATION_INDEX,
        regularization=regularization,
        max_iterations=max_iterations,
        batch_races=batch_races,
    )
    expected_returns = predict_expected_returns(
        calibrator,
        values,
        market_values,
        race_keys,
        COMBINATION_LANES,
        batch_races=batch_races,
    )
    selected_policy = dict(policy or standard_direct_policy())
    selected_policy.update(
        {
            "payout_estimator": (
                "pre-evaluation all-ticket expected-return Poisson calibration"
            ),
            "market_reference": "fixed baseline probability",
            "expected_return_regularization": float(regularization),
            "expected_return_training_samples": int(calibrator.training_samples),
            "selection": (
                "fixed pre-evaluation return calibration; no evaluation-period tuning"
            ),
        }
    )
    candidates_by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    evaluated_by_day: dict[str, set[str]] = defaultdict(set)
    max_expected_return = 0.0
    for row_index, race_key in enumerate(race_keys):
        race_id, race_date, _jcd, _rno = race_key
        actual = payouts.get(race_id)
        if actual is None:
            continue
        evaluated_by_day[race_date].add(race_id)
        payout_model = {}
        for combination_index, combination in enumerate(COMBINATION_LABELS):
            probability = max(1e-15, float(values[row_index, combination_index]))
            expected_return = float(expected_returns[row_index, combination_index])
            max_expected_return = max(max_expected_return, expected_return)
            estimated_odds = min(2_000.0, max(1.1, expected_return / probability))
            payout_model[combination] = {
                "estimated_odds": estimated_odds,
                "estimated_payout_yen": estimated_odds * 100.0,
                "history_count": float(calibrator.training_samples),
                "odds_source": "expected_return_poisson",
            }
        candidates_by_day[race_date].extend(
            direct_candidates(
                values[row_index],
                race_key=race_key,
                actual=actual,
                payout_model=payout_model,
                ev_threshold=float(selected_policy["ev_threshold"]),
            )
        )

    result = _settle_candidate_days(
        candidates_by_day,
        evaluated_by_day,
        sorted({str(row[1]) for row in race_keys}),
        selected_policy,
    )
    result["return_calibrator"] = {
        "method": "Poisson log-link damped Newton",
        "regularization": float(calibrator.regularization),
        "training_samples": int(calibrator.training_samples),
        "iterations": int(calibrator.iterations),
        "converged": bool(calibrator.converged),
        "objective": float(calibrator.objective),
        "gradient_norm": float(calibrator.gradient_norm),
        "max_expected_return": float(max_expected_return),
    }
    return result
