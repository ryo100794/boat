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
from .return_policy import calibration_policy_split
from .payout_estimator import (
    ConditionalPayoutStatistics,
    fit_conditional_payout_statistics,
    predict_conditional_odds,
)


COMBINATION_LABELS = tuple(
    "-".join(str(lane) for lane in combination)
    for combination in TRIFECTA_COMBINATIONS
)
COMBINATION_INDEX = {
    combination: index for index, combination in enumerate(COMBINATION_LABELS)
}


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


def _winner_samples(
    probabilities: np.ndarray,
    race_keys: list[tuple[str, str, str, int]],
    payouts: dict[str, dict[str, Any]],
) -> tuple[list[float], list[str], list[tuple[str, str, str, int]], list[float]]:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (len(race_keys), len(COMBINATION_LABELS)):
        raise ValueError("winner sample probabilities and race keys must align")
    winner_probabilities = []
    winner_combinations = []
    winner_keys = []
    winner_payouts = []
    for row_index, race_key in enumerate(race_keys):
        actual = payouts.get(str(race_key[0]))
        if actual is None:
            continue
        combination = str(actual["combination"])
        combination_index = COMBINATION_INDEX.get(combination)
        if combination_index is None:
            continue
        winner_probabilities.append(float(values[row_index, combination_index]))
        winner_combinations.append(combination)
        winner_keys.append(race_key)
        winner_payouts.append(float(actual["payout_yen"]))
    return winner_probabilities, winner_combinations, winner_keys, winner_payouts


def _settle_policy_diagnostic(
    candidates_by_day: dict[str, list[dict[str, Any]]],
    evaluated_by_day: dict[str, set[str]],
    race_dates: list[str],
    policy: dict[str, Any],
) -> dict[str, Any]:
    totals = zero_totals()
    daily = []
    state = (0, 0, 0)
    for race_date in race_dates:
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
    return {
        "tickets": int(totals["tickets"]),
        "selected_races": int(totals["races_bet"]),
        "hits": int(totals["hit_tickets"]),
        "stake_yen": stake_yen,
        "return_yen": return_yen,
        "profit_yen": return_yen - stake_yen,
        "roi": return_yen / stake_yen if stake_yen else 0.0,
        "winning_days": int(totals["winning_days"]),
        "losing_days": int(totals["losing_days"]),
        "max_drawdown_yen": int(state[2]),
    }


def _select_conditional_payout_policy(
    calibration_probabilities: np.ndarray,
    calibration_market_probabilities: np.ndarray,
    calibration_race_keys: list[tuple[str, str, str, int]],
    payouts: dict[str, dict[str, Any]],
    *,
    selection_days: int,
    base_policy: dict[str, Any],
    fallback_ridge: float,
    ridge_candidates: tuple[float, ...],
    correction_candidates: tuple[float, ...],
    threshold_candidates: tuple[float, ...],
    minimum_tickets: int,
    minimum_hits: int,
    minimum_winning_days: int,
    minimum_roi: float,
) -> tuple[float, float, float, str, list[dict[str, Any]], dict[str, Any] | None]:
    split = calibration_policy_split(
        calibration_race_keys, selection_days=selection_days
    )
    fallback_threshold = float(base_policy["ev_threshold"])
    if split is None:
        return fallback_ridge, 0.0, fallback_threshold, "fallback_fixed_policy", [], None
    statistics = ConditionalPayoutStatistics.empty()
    statistics.update(
        *_winner_samples(
            calibration_market_probabilities[:split],
            calibration_race_keys[:split],
            payouts,
        )
    )
    selection_probabilities = calibration_probabilities[split:]
    selection_market = calibration_market_probabilities[split:]
    selection_keys = calibration_race_keys[split:]
    race_dates = sorted({str(row[1]) for row in selection_keys})
    flat_combinations = list(COMBINATION_LABELS) * len(selection_keys)
    flat_keys = [key for key in selection_keys for _ in COMBINATION_LABELS]
    threshold_values = tuple(
        sorted({fallback_threshold, *(float(value) for value in threshold_candidates)})
    )
    candidate_floor = min(threshold_values)
    diagnostics = []
    for ridge_value in sorted(
        {float(fallback_ridge), *(float(value) for value in ridge_candidates)}
    ):
        model = fit_conditional_payout_statistics(statistics, ridge=ridge_value)
        for correction_factor in sorted(
            {0.0, *(float(value) for value in correction_candidates)}
        ):
            estimated_odds = predict_conditional_odds(
                model,
                selection_market.reshape(-1),
                flat_combinations,
                flat_keys,
                mean_correction_factor=correction_factor,
            ).reshape(len(selection_keys), len(COMBINATION_LABELS))
            candidates_by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
            evaluated_by_day: dict[str, set[str]] = defaultdict(set)
            for row_index, race_key in enumerate(selection_keys):
                race_id, race_date, _jcd, _rno = race_key
                actual = payouts.get(str(race_id))
                if actual is None:
                    continue
                evaluated_by_day[race_date].add(str(race_id))
                payout_model = {
                    combination: {
                        "estimated_odds": float(estimated_odds[row_index, index]),
                        "estimated_payout_yen": float(
                            estimated_odds[row_index, index] * 100.0
                        ),
                        "history_count": float(model.training_samples),
                        "odds_source": "conditional_payout_pre_evaluation",
                    }
                    for index, combination in enumerate(COMBINATION_LABELS)
                }
                candidates_by_day[race_date].extend(
                    direct_candidates(
                        selection_probabilities[row_index],
                        race_key=race_key,
                        actual=actual,
                        payout_model=payout_model,
                        ev_threshold=candidate_floor,
                    )
                )
            for threshold in threshold_values:
                policy = dict(base_policy)
                policy["ev_threshold"] = threshold
                filtered = {
                    race_date: [
                        row
                        for row in rows
                        if float(row["estimated_ev"]) >= threshold
                    ]
                    for race_date, rows in candidates_by_day.items()
                }
                settled = _settle_policy_diagnostic(
                    filtered, evaluated_by_day, race_dates, policy
                )
                diagnostics.append(
                    {
                        "ridge": ridge_value,
                        "mean_correction_factor": correction_factor,
                        "ev_threshold": threshold,
                        **settled,
                    }
                )
    eligible = [
        row
        for row in diagnostics
        if int(row["tickets"]) >= minimum_tickets
        and int(row["hits"]) >= minimum_hits
        and int(row["winning_days"]) >= minimum_winning_days
        and float(row["roi"]) >= minimum_roi
        and int(row["profit_yen"]) > 0
    ]
    if eligible:
        selected = max(
            eligible,
            key=lambda row: (
                int(row["tickets"]),
                int(row["winning_days"]),
                int(row["hits"]),
                -float(row["ev_threshold"]),
                -float(row["mean_correction_factor"]),
                float(row["ridge"]),
            ),
        )
        source = "pre_evaluation_adaptive_selection"
    else:
        selected = {
            "ridge": float(fallback_ridge),
            "mean_correction_factor": 0.0,
            "ev_threshold": fallback_threshold,
        }
        source = "fallback_fixed_policy"
    period = {
        "fit_from": str(calibration_race_keys[0][1]),
        "fit_through": str(calibration_race_keys[split - 1][1]),
        "selection_from": str(calibration_race_keys[split][1]),
        "selection_through": str(calibration_race_keys[-1][1]),
        "fit_races": split,
        "selection_races": len(calibration_race_keys) - split,
    }
    return (
        float(selected["ridge"]),
        float(selected["mean_correction_factor"]),
        float(selected["ev_threshold"]),
        source,
        diagnostics,
        period,
    )


def simulate_conditional_payout_walk_forward(
    probabilities: np.ndarray,
    *,
    race_keys: list[tuple[str, str, str, int]],
    payouts: dict[str, dict[str, Any]],
    calibration_probabilities: np.ndarray,
    calibration_race_keys: list[tuple[str, str, str, int]],
    market_reference_probabilities: np.ndarray | None = None,
    calibration_market_reference_probabilities: np.ndarray | None = None,
    policy: dict[str, Any] | None = None,
    ridge: float = 10.0,
    ridge_candidates: tuple[float, ...] = (1.0, 10.0, 100.0),
    mean_correction_candidates: tuple[float, ...] = (0.0, 0.5, 1.0),
    threshold_candidates: tuple[float, ...] = (1.05, 1.10, 1.20),
    policy_selection_days: int = 30,
    minimum_selection_tickets: int = 100,
    minimum_selection_hits: int = 10,
    minimum_selection_winning_days: int = 8,
    minimum_selection_roi: float = 1.05,
) -> dict[str, Any]:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (len(race_keys), len(COMBINATION_LABELS)):
        raise ValueError("probability matrix and race keys must align")
    market_values = np.asarray(
        probabilities
        if market_reference_probabilities is None
        else market_reference_probabilities,
        dtype=np.float64,
    )
    if market_values.shape != values.shape:
        raise ValueError("market reference probabilities and race keys must align")
    calibration_values = np.asarray(calibration_probabilities, dtype=np.float64)
    calibration_market_values = np.asarray(
        calibration_values
        if calibration_market_reference_probabilities is None
        else calibration_market_reference_probabilities,
        dtype=np.float64,
    )
    if calibration_market_values.shape != calibration_values.shape:
        raise ValueError("calibration market probabilities and race keys must align")
    independent_market_reference = market_reference_probabilities is not None
    selected_policy = dict(policy or standard_direct_policy())
    (
        selected_ridge,
        selected_mean_correction,
        selected_threshold,
        selection_source,
        selection_diagnostics,
        selection_period,
    ) = _select_conditional_payout_policy(
        calibration_values,
        calibration_market_values,
        calibration_race_keys,
        payouts,
        selection_days=policy_selection_days,
        base_policy=selected_policy,
        fallback_ridge=ridge,
        ridge_candidates=ridge_candidates,
        correction_candidates=mean_correction_candidates,
        threshold_candidates=threshold_candidates,
        minimum_tickets=minimum_selection_tickets,
        minimum_hits=minimum_selection_hits,
        minimum_winning_days=minimum_selection_winning_days,
        minimum_roi=minimum_selection_roi,
    )
    selected_policy["ev_threshold"] = selected_threshold
    selected_policy.update(
        {
            "payout_estimator": (
                "daily walk-forward log-payout ridge using fixed baseline market "
                "reference, finish lanes, venue, and race number"
                if independent_market_reference
                else "daily walk-forward log-payout ridge using model probability, "
                "finish lanes, venue, and race number"
            ),
            "market_reference": (
                "fixed baseline probability"
                if independent_market_reference
                else "candidate probability"
            ),
            "payout_point_estimate": (
                "conditional lognormal interpolation selected before evaluation"
            ),
            "conditional_payout_ridge": float(selected_ridge),
            "mean_correction_factor": float(selected_mean_correction),
            "selection": (
                "pre-evaluation adaptive two-stage payout policy selection; "
                "no evaluation-period tuning"
            ),
        }
    )
    statistics = ConditionalPayoutStatistics.empty()
    statistics.update(
        *_winner_samples(
            calibration_market_values,
            calibration_race_keys,
            payouts,
        )
    )
    initial_samples = statistics.samples
    dates = sorted({str(row[1]) for row in race_keys})
    row_indices_by_date = {
        race_date: [
            index for index, row in enumerate(race_keys) if str(row[1]) == race_date
        ]
        for race_date in dates
    }
    totals = zero_totals()
    daily = []
    state = (0, 0, 0)
    fold_count = min(5, len(dates))
    fold_attributions = [new_roi_attribution() for _ in range(fold_count)]
    combination_rows = list(COMBINATION_LABELS)
    ev_thresholds = (0.80, 0.90, 1.00, 1.05, 1.10, 1.20)
    ev_counts = {f"{threshold:.2f}": 0 for threshold in ev_thresholds}
    max_estimated_ev = 0.0
    residual_variances = []
    for day_index, race_date in enumerate(dates):
        model = fit_conditional_payout_statistics(statistics, ridge=selected_ridge)
        residual_variances.append(float(model.residual_variance))
        row_indices = row_indices_by_date[race_date]
        day_keys = [race_keys[index] for index in row_indices]
        day_probabilities = values[row_indices]
        day_market_probabilities = market_values[row_indices]
        flat_combinations = combination_rows * len(row_indices)
        flat_keys = [race_key for race_key in day_keys for _ in COMBINATION_LABELS]
        estimated_odds = predict_conditional_odds(
            model,
            day_market_probabilities.reshape(-1),
            flat_combinations,
            flat_keys,
            mean_correction_factor=selected_mean_correction,
        ).reshape(len(row_indices), len(COMBINATION_LABELS))
        estimated_ev = day_probabilities * estimated_odds
        max_estimated_ev = max(max_estimated_ev, float(estimated_ev.max()))
        for threshold in ev_thresholds:
            ev_counts[f"{threshold:.2f}"] += int(np.sum(estimated_ev >= threshold))
        candidates = []
        evaluated_races = set()
        for local_index, race_key in enumerate(day_keys):
            actual = payouts.get(str(race_key[0]))
            if actual is None:
                continue
            evaluated_races.add(str(race_key[0]))
            payout_model = {
                combination: {
                    "estimated_odds": float(
                        estimated_odds[local_index, combo_index]
                    ),
                    "estimated_payout_yen": float(
                        estimated_odds[local_index, combo_index] * 100.0
                    ),
                    "history_count": float(model.training_samples),
                    "odds_source": "conditional_payout_walk_forward",
                }
                for combo_index, combination in enumerate(COMBINATION_LABELS)
            }
            candidates.extend(
                direct_candidates(
                    day_probabilities[local_index],
                    race_key=race_key,
                    actual=actual,
                    payout_model=payout_model,
                    ev_threshold=float(selected_policy["ev_threshold"]),
                )
            )
        fold_index = min(
            fold_count - 1,
            day_index * fold_count // len(dates),
        )
        result = allocate_adaptive_day(
            race_date,
            candidates,
            evaluated_races,
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
            stake_granularity_yen=int(selected_policy["stake_granularity_yen"]),
            min_stake_yen=int(selected_policy["min_stake_yen"]),
            roi_attribution=fold_attributions[fold_index],
        )
        result["payout_training_samples"] = int(model.training_samples)
        state = append_day_result(
            daily,
            totals,
            result,
            cumulative_profit=state[0],
            peak_profit=state[1],
            max_drawdown=state[2],
        )
        statistics.update(
            *_winner_samples(day_market_probabilities, day_keys, payouts)
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
        "payout_training_samples_initial": int(initial_samples),
        "payout_training_samples_final": int(statistics.samples),
        "policy_selection": {
            "source": selection_source,
            "selection_days": int(policy_selection_days),
            "minimum_tickets": int(minimum_selection_tickets),
            "minimum_hits": int(minimum_selection_hits),
            "minimum_winning_days": int(minimum_selection_winning_days),
            "minimum_roi": float(minimum_selection_roi),
            "selected_ridge": float(selected_ridge),
            "selected_mean_correction_factor": float(selected_mean_correction),
            "selected_ev_threshold": float(selected_threshold),
            "period": selection_period,
            "diagnostics": selection_diagnostics,
        },
        "payout_diagnostics": {
            "candidate_combinations": int(len(race_keys) * len(COMBINATION_LABELS)),
            "max_estimated_ev": max_estimated_ev,
            "estimated_ev_at_least": ev_counts,
            "residual_variance_initial": (
                residual_variances[0] if residual_variances else None
            ),
            "residual_variance_final": (
                residual_variances[-1] if residual_variances else None
            ),
        },
        "ticket_roi_attribution": attribution_summary,
        "daily": daily,
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
                "odds_source": str(estimate.get("odds_source") or "payout_model"),
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
