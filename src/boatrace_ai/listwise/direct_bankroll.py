from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from ..adaptive_allocation import (
    allocate_adaptive_day,
    append_day_result,
    zero_totals,
)
from ..bankroll_backtest import _build_payout_model
from ..fast_math import TRIFECTA_COMBINATIONS
from ..roi_attribution import (
    merge_roi_attribution,
    new_roi_attribution,
    summarize_fold_signal_stability,
    summarize_roi_attribution,
)


COMBINATION_LABELS = tuple(
    "-".join(str(lane) for lane in combination)
    for combination in TRIFECTA_COMBINATIONS
)


def _finite_quantile(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        raise ValueError("bootstrap produced no finite ratio samples")
    lower, upper = np.quantile(finite, (0.025, 0.975))
    return float(lower), float(upper)


def _roi(return_yen: float, stake_yen: float) -> float:
    return float(return_yen / stake_yen) if stake_yen > 0.0 else 0.0


def bootstrap_daily_bankroll(
    daily: list[dict[str, Any]],
    *,
    baseline_daily: list[dict[str, Any]] | None = None,
    samples: int = 20_000,
    seed: int = 20260727,
    chunk_size: int = 1_000,
) -> dict[str, Any]:
    if not daily:
        raise ValueError("daily bankroll rows must not be empty")
    if samples < 100:
        raise ValueError("samples must be at least 100")
    dates = [str(row["race_date"]) for row in daily]
    if len(set(dates)) != len(dates):
        raise ValueError("daily bankroll rows must contain unique dates")

    stakes = np.asarray([row["stake_yen"] for row in daily], dtype=np.float64)
    returns = np.asarray([row["return_yen"] for row in daily], dtype=np.float64)
    if not np.all(np.isfinite(stakes)) or not np.all(np.isfinite(returns)):
        raise ValueError("daily stake and return values must be finite")
    if np.any(stakes < 0.0) or np.any(returns < 0.0):
        raise ValueError("daily stake and return values must be non-negative")

    baseline_stakes = None
    baseline_returns = None
    if baseline_daily is not None:
        baseline_by_date = {str(row["race_date"]): row for row in baseline_daily}
        if set(baseline_by_date) != set(dates):
            raise ValueError("candidate and baseline daily dates must match")
        baseline_stakes = np.asarray(
            [baseline_by_date[date]["stake_yen"] for date in dates],
            dtype=np.float64,
        )
        baseline_returns = np.asarray(
            [baseline_by_date[date]["return_yen"] for date in dates],
            dtype=np.float64,
        )
        if (
            not np.all(np.isfinite(baseline_stakes))
            or not np.all(np.isfinite(baseline_returns))
            or np.any(baseline_stakes < 0.0)
            or np.any(baseline_returns < 0.0)
        ):
            raise ValueError(
                "baseline stake and return values must be finite and non-negative"
            )

    rng = np.random.default_rng(seed)
    boot_profit = np.empty(samples, dtype=np.float64)
    boot_roi = np.empty(samples, dtype=np.float64)
    boot_profit_delta = (
        np.empty(samples, dtype=np.float64) if baseline_stakes is not None else None
    )
    boot_roi_delta = (
        np.empty(samples, dtype=np.float64) if baseline_stakes is not None else None
    )
    step = max(1, int(chunk_size))
    for start in range(0, samples, step):
        stop = min(samples, start + step)
        indices = rng.integers(0, len(dates), size=(stop - start, len(dates)))
        sampled_stakes = stakes[indices].sum(axis=1)
        sampled_returns = returns[indices].sum(axis=1)
        boot_profit[start:stop] = sampled_returns - sampled_stakes
        candidate_roi = np.divide(
            sampled_returns,
            sampled_stakes,
            out=np.zeros_like(sampled_returns),
            where=sampled_stakes > 0.0,
        )
        boot_roi[start:stop] = candidate_roi
        if baseline_stakes is not None and baseline_returns is not None:
            sampled_baseline_stakes = baseline_stakes[indices].sum(axis=1)
            sampled_baseline_returns = baseline_returns[indices].sum(axis=1)
            boot_profit_delta[start:stop] = (
                sampled_returns
                - sampled_stakes
                - sampled_baseline_returns
                + sampled_baseline_stakes
            )
            baseline_roi = np.divide(
                sampled_baseline_returns,
                sampled_baseline_stakes,
                out=np.zeros_like(sampled_baseline_returns),
                where=sampled_baseline_stakes > 0.0,
            )
            boot_roi_delta[start:stop] = candidate_roi - baseline_roi

    profit_lower, profit_upper = _finite_quantile(boot_profit)
    roi_lower, roi_upper = _finite_quantile(boot_roi)
    result = {
        "days": len(dates),
        "samples": int(samples),
        "profit_yen": float(returns.sum() - stakes.sum()),
        "profit_ci95_lower_yen": float(profit_lower),
        "profit_ci95_upper_yen": float(profit_upper),
        "roi": _roi(returns.sum(), stakes.sum()),
        "roi_ci95_lower": float(roi_lower),
        "roi_ci95_upper": float(roi_upper),
        "probability_profit_above_zero": float(np.mean(boot_profit > 0.0)),
        "probability_roi_above_one": float(
            np.mean(boot_roi[np.isfinite(boot_roi)] > 1.0)
        ),
    }
    if baseline_stakes is not None and baseline_returns is not None:
        profit_delta_lower, profit_delta_upper = _finite_quantile(
            boot_profit_delta
        )
        roi_delta_lower, roi_delta_upper = _finite_quantile(boot_roi_delta)
        result.update(
            {
                "baseline_profit_yen": float(
                    baseline_returns.sum() - baseline_stakes.sum()
                ),
                "baseline_roi": _roi(
                    baseline_returns.sum(), baseline_stakes.sum()
                ),
                "profit_delta_yen": float(
                    returns.sum()
                    - stakes.sum()
                    - baseline_returns.sum()
                    + baseline_stakes.sum()
                ),
                "profit_delta_ci95_lower_yen": float(profit_delta_lower),
                "profit_delta_ci95_upper_yen": float(profit_delta_upper),
                "roi_delta": (
                    _roi(returns.sum(), stakes.sum())
                    - _roi(baseline_returns.sum(), baseline_stakes.sum())
                ),
                "roi_delta_ci95_lower": float(roi_delta_lower),
                "roi_delta_ci95_upper": float(roi_delta_upper),
                "probability_profit_delta_above_zero": float(
                    np.mean(boot_profit_delta > 0.0)
                ),
                "probability_roi_delta_above_zero": float(
                    np.mean(boot_roi_delta[np.isfinite(boot_roi_delta)] > 0.0)
                ),
            }
        )
    return result


def standard_direct_policy() -> dict[str, Any]:
    return {
        "daily_budget_yen": 10_000,
        "ev_threshold": 1.20,
        "payout_prior_weight": 30.0,
        "fractional_kelly": 0.25,
        "max_daily_exposure_fraction": 0.60,
        "min_daily_exposure_fraction": 0.40,
        "race_cap_fraction": 0.10,
        "ticket_cap_fraction": 0.03,
        "max_daily_tickets": 30,
        "allocation_mode": "normalized_kelly",
        "stake_granularity_yen": 100,
        "min_stake_yen": 100,
        "bet_type": "3連単",
        "include_odds": False,
        "payout_estimator": (
            "training-period trifecta payout mean with global prior"
        ),
        "selection": "fixed standard_365d_v2 policy; no holdout tuning",
    }


def direct_candidates(
    probabilities: np.ndarray,
    *,
    race_key: tuple[str, str, str, int],
    actual: dict[str, Any],
    payout_model: dict[str, dict[str, float]],
    ev_threshold: float,
) -> list[dict[str, Any]]:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (len(COMBINATION_LABELS),):
        raise ValueError("probabilities must contain all 120 trifecta combinations")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("probabilities must have positive mass")
    values = values / total
    race_id, race_date, jcd, rno = race_key
    candidates = []
    for combination, probability in zip(COMBINATION_LABELS, values):
        estimate = payout_model.get(combination)
        if not estimate:
            continue
        estimated_odds = float(estimate["estimated_odds"])
        estimated_ev = float(probability) * estimated_odds
        if estimated_ev < float(ev_threshold):
            continue
        candidates.append(
            {
                "race_id": race_id,
                "race_date": race_date,
                "jcd": jcd,
                "rno": int(rno),
                "combination": combination,
                "probability": float(probability),
                "estimated_odds": estimated_odds,
                "estimated_payout_yen": float(
                    estimate["estimated_payout_yen"]
                ),
                "estimated_ev": estimated_ev,
                "payout_history_count": int(estimate["history_count"]),
                "odds_source": "payout_model",
                "actual_combination": str(actual["combination"]),
                "actual_payout_yen": int(actual["payout_yen"]),
                "hit": combination == str(actual["combination"]),
            }
        )
    return candidates


def simulate_direct_bankroll(
    probabilities: np.ndarray,
    *,
    race_keys: list[tuple[str, str, str, int]],
    payouts: dict[str, dict[str, Any]],
    training_races: set[str],
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (len(race_keys), len(COMBINATION_LABELS)):
        raise ValueError("probability matrix and race keys must align")
    selected_policy = dict(policy or standard_direct_policy())
    payout_model = _build_payout_model(
        payouts,
        train_races=training_races,
        prior_weight=float(selected_policy["payout_prior_weight"]),
    )
    candidates_by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    evaluated_by_day: dict[str, set[str]] = defaultdict(set)
    for row_index, race_key in enumerate(race_keys):
        race_id, race_date, _jcd, _rno = race_key
        actual = payouts.get(race_id)
        if actual is None:
            continue
        evaluated_by_day[race_date].add(race_id)
        candidates_by_day[race_date].extend(
            direct_candidates(
                values[row_index],
                race_key=race_key,
                actual=actual,
                payout_model=payout_model,
                ev_threshold=float(selected_policy["ev_threshold"]),
            )
        )

    totals = zero_totals()
    daily = []
    state = (0, 0, 0)
    race_dates = sorted({str(row[1]) for row in race_keys})
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
            daily_budget_yen=int(selected_policy["daily_budget_yen"]),
            fractional_kelly=float(selected_policy["fractional_kelly"]),
            max_daily_exposure_fraction=float(
                selected_policy["max_daily_exposure_fraction"]
            ),
            min_daily_exposure_fraction=float(
                selected_policy["min_daily_exposure_fraction"]
            ),
            race_cap_fraction=float(selected_policy["race_cap_fraction"]),
            ticket_cap_fraction=float(selected_policy["ticket_cap_fraction"]),
            max_daily_tickets=int(selected_policy["max_daily_tickets"]),
            allocation_mode=str(selected_policy["allocation_mode"]),
            stake_granularity_yen=int(
                selected_policy["stake_granularity_yen"]
            ),
            min_stake_yen=int(selected_policy["min_stake_yen"]),
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
    stake_yen = int(totals["stake_yen"])
    return_yen = int(totals["return_yen"])
    attribution = new_roi_attribution()
    for fold_attribution in fold_attributions:
        merge_roi_attribution(attribution, fold_attribution)
    attribution_summary = summarize_roi_attribution(attribution)
    fold_summaries = [
        summarize_roi_attribution(fold_attribution)
        for fold_attribution in fold_attributions
    ]
    attribution_summary["fold_stability"] = summarize_fold_signal_stability(
        fold_summaries
    )
    return {
        "policy": selected_policy,
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
