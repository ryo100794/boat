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
    FEATURE_COUNT as PAYOUT_FEATURE_COUNT,
    FEATURE_SCHEMA as PAYOUT_FEATURE_SCHEMA,
    ConditionalPayoutStatistics,
    ConditionalPayoutTailCalibrator,
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
PAYOUT_TAIL_SCHEMA = "conditional_payout_tail_probability_bins_v1"


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


def _validate_nondecreasing_race_dates(
    race_keys: list[tuple[str, str, str, int]],
    *,
    name: str,
) -> None:
    dates = [str(row[1]) for row in race_keys]
    if dates != sorted(dates):
        raise ValueError(f"{name} dates must be non-decreasing")


def _update_tail_for_winners(
    tail_calibrator: ConditionalPayoutTailCalibrator,
    market_probabilities: np.ndarray,
    raw_estimated_odds: np.ndarray,
    race_keys: list[tuple[str, str, str, int]],
    payouts: dict[str, dict[str, Any]],
) -> None:
    winner_probabilities = []
    winner_raw_odds = []
    winner_actual_odds = []
    for row_index, race_key in enumerate(race_keys):
        actual = payouts.get(str(race_key[0]))
        if actual is None:
            continue
        combination_index = COMBINATION_INDEX.get(str(actual["combination"]))
        if combination_index is None:
            continue
        winner_probabilities.append(
            float(market_probabilities[row_index, combination_index])
        )
        winner_raw_odds.append(
            float(raw_estimated_odds[row_index, combination_index])
        )
        winner_actual_odds.append(float(actual["payout_yen"]) / 100.0)
    if winner_probabilities:
        tail_calibrator.update(
            winner_probabilities,
            winner_raw_odds,
            winner_actual_odds,
        )


def _selection_walk_forward_for_ridge(
    calibration_probabilities: np.ndarray,
    calibration_market_probabilities: np.ndarray,
    calibration_race_keys: list[tuple[str, str, str, int]],
    payouts: dict[str, dict[str, Any]],
    *,
    split: int,
    ridge: float,
    threshold_values: tuple[float, ...],
    candidate_floor: float,
    base_policy: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    ConditionalPayoutStatistics,
    ConditionalPayoutTailCalibrator,
]:
    statistics = ConditionalPayoutStatistics.empty()
    statistics.update(
        *_winner_samples(
            calibration_market_probabilities[:split],
            calibration_race_keys[:split],
            payouts,
        )
    )
    tail_calibrator = ConditionalPayoutTailCalibrator.empty()
    tail_initial = tail_calibrator.diagnostics()
    selection_keys = calibration_race_keys[split:]
    selection_probabilities = calibration_probabilities[split:]
    selection_market = calibration_market_probabilities[split:]
    race_dates = sorted({str(row[1]) for row in selection_keys})
    runs = {
        threshold: {"totals": zero_totals(), "daily": [], "state": (0, 0, 0)}
        for threshold in threshold_values
    }
    for race_date in race_dates:
        row_indices = [
            index for index, row in enumerate(selection_keys)
            if str(row[1]) == race_date
        ]
        day_keys = [selection_keys[index] for index in row_indices]
        day_probabilities = selection_probabilities[row_indices]
        day_market = selection_market[row_indices]
        model = fit_conditional_payout_statistics(statistics, ridge=ridge)
        flat_combinations = list(COMBINATION_LABELS) * len(day_keys)
        flat_keys = [key for key in day_keys for _ in COMBINATION_LABELS]
        raw_odds = predict_conditional_odds(
            model,
            day_market.reshape(-1),
            flat_combinations,
            flat_keys,
            mean_correction_factor=0.0,
        ).reshape(len(day_keys), len(COMBINATION_LABELS))
        calibrated_odds = tail_calibrator.calibrate(
            day_market.reshape(-1), raw_odds.reshape(-1)
        ).reshape(raw_odds.shape)
        candidates = []
        evaluated_races = set()
        for local_index, race_key in enumerate(day_keys):
            actual = payouts.get(str(race_key[0]))
            if actual is None:
                continue
            evaluated_races.add(str(race_key[0]))
            payout_model = {
                combination: {
                    "estimated_odds": float(calibrated_odds[local_index, index]),
                    "raw_estimated_odds": float(raw_odds[local_index, index]),
                    "estimated_payout_yen": float(
                        calibrated_odds[local_index, index] * 100.0
                    ),
                    "history_count": float(model.training_samples),
                    "odds_source": "conditional_payout_pre_evaluation_tail_calibrated",
                }
                for index, combination in enumerate(COMBINATION_LABELS)
            }
            candidates.extend(
                direct_candidates(
                    day_probabilities[local_index],
                    race_key=race_key,
                    actual=actual,
                    payout_model=payout_model,
                    ev_threshold=candidate_floor,
                )
            )
        candidates.sort(key=lambda row: (str(row["race_id"]), row["combination"]))
        for threshold, run in runs.items():
            policy = dict(base_policy)
            policy["ev_threshold"] = threshold
            result = allocate_adaptive_day(
                race_date,
                [
                    row
                    for row in candidates
                    if float(row["estimated_ev"]) >= threshold
                ],
                evaluated_races,
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
            run["state"] = append_day_result(
                run["daily"],
                run["totals"],
                result,
                cumulative_profit=run["state"][0],
                peak_profit=run["state"][1],
                max_drawdown=run["state"][2],
            )
        _update_tail_for_winners(
            tail_calibrator, day_market, raw_odds, day_keys, payouts
        )
        statistics.update(*_winner_samples(day_market, day_keys, payouts))

    tail_final = tail_calibrator.diagnostics()
    diagnostics = []
    for threshold, run in runs.items():
        totals = run["totals"]
        stake_yen = int(totals["stake_yen"])
        return_yen = int(totals["return_yen"])
        diagnostics.append(
            {
                "ridge": float(ridge),
                "mean_correction_factor": 0.0,
                "ev_threshold": float(threshold),
                "tickets": int(totals["tickets"]),
                "selected_races": int(totals["races_bet"]),
                "hits": int(totals["hit_tickets"]),
                "stake_yen": stake_yen,
                "return_yen": return_yen,
                "profit_yen": return_yen - stake_yen,
                "roi": return_yen / stake_yen if stake_yen else 0.0,
                "winning_days": int(totals["winning_days"]),
                "losing_days": int(totals["losing_days"]),
                "max_drawdown_yen": int(run["state"][2]),
                "tail_schema": PAYOUT_TAIL_SCHEMA,
                "tail_initial": tail_initial,
                "tail_final": tail_final,
            }
        )
    return diagnostics, statistics, tail_calibrator


def _select_conditional_payout_policy_state(
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
) -> tuple[
    float,
    float,
    float,
    str,
    list[dict[str, Any]],
    dict[str, Any] | None,
    ConditionalPayoutStatistics,
    ConditionalPayoutTailCalibrator,
]:
    _validate_nondecreasing_race_dates(
        calibration_race_keys, name="calibration_race_keys"
    )
    calibration_values = np.asarray(calibration_probabilities, dtype=np.float64)
    calibration_market = np.asarray(calibration_market_probabilities, dtype=np.float64)
    expected_shape = (len(calibration_race_keys), len(COMBINATION_LABELS))
    if calibration_values.shape != expected_shape:
        raise ValueError("calibration probabilities and race keys must align")
    if calibration_market.shape != expected_shape:
        raise ValueError("calibration market probabilities and race keys must align")
    _ = correction_candidates
    split = calibration_policy_split(
        calibration_race_keys, selection_days=selection_days
    )
    fallback_threshold = float(base_policy["ev_threshold"])
    if split is None:
        statistics = ConditionalPayoutStatistics.empty()
        statistics.update(
            *_winner_samples(calibration_market, calibration_race_keys, payouts)
        )
        return (
            float(fallback_ridge), 0.0, fallback_threshold,
            "fallback_fixed_policy", [], None, statistics,
            ConditionalPayoutTailCalibrator.empty(),
        )

    threshold_values = tuple(
        sorted({fallback_threshold, *(float(value) for value in threshold_candidates)})
    )
    diagnostics = []
    states = {}
    for ridge_value in sorted(
        {float(fallback_ridge), *(float(value) for value in ridge_candidates)}
    ):
        ridge_diagnostics, statistics, tail_calibrator = (
            _selection_walk_forward_for_ridge(
                calibration_values,
                calibration_market,
                calibration_race_keys,
                payouts,
                split=split,
                ridge=ridge_value,
                threshold_values=threshold_values,
                candidate_floor=min(threshold_values),
                base_policy=base_policy,
            )
        )
        diagnostics.extend(ridge_diagnostics)
        states[ridge_value] = (statistics, tail_calibrator)
    eligible = [
        row for row in diagnostics
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
                int(row["tickets"]), int(row["winning_days"]),
                int(row["hits"]), -float(row["ev_threshold"]),
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
    selected_ridge = float(selected["ridge"])
    statistics, tail_calibrator = states[selected_ridge]
    period = {
        "fit_from": str(calibration_race_keys[0][1]),
        "fit_through": str(calibration_race_keys[split - 1][1]),
        "selection_from": str(calibration_race_keys[split][1]),
        "selection_through": str(calibration_race_keys[-1][1]),
        "fit_races": split,
        "selection_races": len(calibration_race_keys) - split,
        "tail_schema": PAYOUT_TAIL_SCHEMA,
        "tail_initial_samples": 0,
        "tail_final_samples": int(tail_calibrator.samples),
    }
    return (
        selected_ridge, 0.0, float(selected["ev_threshold"]), source,
        diagnostics, period, statistics, tail_calibrator,
    )


def _select_conditional_payout_policy(
    calibration_probabilities: np.ndarray,
    calibration_market_probabilities: np.ndarray,
    calibration_race_keys: list[tuple[str, str, str, int]],
    payouts: dict[str, dict[str, Any]],
    **kwargs: Any,
) -> tuple[
    float,
    float,
    float,
    str,
    list[dict[str, Any]],
    dict[str, Any] | None,
]:
    selected = _select_conditional_payout_policy_state(
        calibration_probabilities,
        calibration_market_probabilities,
        calibration_race_keys,
        payouts,
        **kwargs,
    )
    return selected[:6]


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
    _validate_nondecreasing_race_dates(race_keys, name="race_keys")
    _validate_nondecreasing_race_dates(
        calibration_race_keys, name="calibration_race_keys"
    )
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
        statistics,
        tail_calibrator,
    ) = _select_conditional_payout_policy_state(
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
            "payout_feature_schema": PAYOUT_FEATURE_SCHEMA,
            "payout_feature_count": PAYOUT_FEATURE_COUNT,
            "market_reference": (
                "fixed baseline probability"
                if independent_market_reference
                else "candidate probability"
            ),
            "payout_point_estimate": (
                "raw conditional log-payout ridge followed by conservative "
                "probability-bin tail calibration"
            ),
            "payout_tail_schema": PAYOUT_TAIL_SCHEMA,
            "conditional_payout_ridge": float(selected_ridge),
            "mean_correction_factor": float(selected_mean_correction),
            "selection": (
                "pre-evaluation adaptive two-stage payout policy selection; "
                "no evaluation-period tuning"
            ),
        }
    )
    initial_samples = statistics.samples
    tail_initial = tail_calibrator.diagnostics()
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
    max_raw_estimated_ev = 0.0
    residual_variances = []
    for day_index, race_date in enumerate(dates):
        day_tail_initial = tail_calibrator.diagnostics()
        model = fit_conditional_payout_statistics(statistics, ridge=selected_ridge)
        residual_variances.append(float(model.residual_variance))
        row_indices = row_indices_by_date[race_date]
        day_keys = [race_keys[index] for index in row_indices]
        day_probabilities = values[row_indices]
        day_market_probabilities = market_values[row_indices]
        flat_combinations = combination_rows * len(row_indices)
        flat_keys = [race_key for race_key in day_keys for _ in COMBINATION_LABELS]
        raw_estimated_odds = predict_conditional_odds(
            model,
            day_market_probabilities.reshape(-1),
            flat_combinations,
            flat_keys,
            mean_correction_factor=0.0,
        ).reshape(len(row_indices), len(COMBINATION_LABELS))
        estimated_odds = tail_calibrator.calibrate(
            day_market_probabilities.reshape(-1),
            raw_estimated_odds.reshape(-1),
        ).reshape(raw_estimated_odds.shape)
        raw_estimated_ev = day_probabilities * raw_estimated_odds
        estimated_ev = day_probabilities * estimated_odds
        max_raw_estimated_ev = max(
            max_raw_estimated_ev, float(raw_estimated_ev.max())
        )
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
                    "raw_estimated_odds": float(
                        raw_estimated_odds[local_index, combo_index]
                    ),
                    "estimated_payout_yen": float(
                        estimated_odds[local_index, combo_index] * 100.0
                    ),
                    "history_count": float(model.training_samples),
                    "odds_source": "conditional_payout_walk_forward_tail_calibrated",
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
        candidates.sort(
            key=lambda row: (str(row["race_id"]), row["combination"])
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
        result["tail_calibration_samples_initial"] = int(
            day_tail_initial["samples"]
        )
        result["tail_calibration_bin_counts_initial"] = list(
            day_tail_initial["bin_counts"]
        )
        result["tail_calibration_bin_factors_initial"] = list(
            day_tail_initial["bin_factors"]
        )
        result["max_raw_estimated_odds"] = float(raw_estimated_odds.max())
        result["max_calibrated_estimated_odds"] = float(estimated_odds.max())
        _update_tail_for_winners(
            tail_calibrator,
            day_market_probabilities,
            raw_estimated_odds,
            day_keys,
            payouts,
        )
        day_tail_final = tail_calibrator.diagnostics()
        result["tail_calibration_samples_final"] = int(
            day_tail_final["samples"]
        )
        result["tail_calibration_bin_counts_final"] = list(
            day_tail_final["bin_counts"]
        )
        result["tail_calibration_bin_factors_final"] = list(
            day_tail_final["bin_factors"]
        )
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
        "tail_calibration_samples_initial": int(tail_initial["samples"]),
        "tail_calibration_samples_final": int(tail_calibrator.samples),
        "policy_selection": {
            "source": selection_source,
            "selection_days": int(policy_selection_days),
            "minimum_tickets": int(minimum_selection_tickets),
            "minimum_hits": int(minimum_selection_hits),
            "minimum_winning_days": int(minimum_selection_winning_days),
            "minimum_roi": float(minimum_selection_roi),
            "selected_ridge": float(selected_ridge),
            "selected_mean_correction_factor": float(selected_mean_correction),
            "tail_schema": PAYOUT_TAIL_SCHEMA,
            "tail_initial": tail_initial,
            "tail_final": tail_calibrator.diagnostics(),
            "selected_ev_threshold": float(selected_threshold),
            "period": selection_period,
            "diagnostics": selection_diagnostics,
        },
        "payout_diagnostics": {
            "feature_schema": PAYOUT_FEATURE_SCHEMA,
            "feature_count": PAYOUT_FEATURE_COUNT,
            "candidate_combinations": int(len(race_keys) * len(COMBINATION_LABELS)),
            "tail_schema": PAYOUT_TAIL_SCHEMA,
            "tail_initial": tail_initial,
            "tail_final": tail_calibrator.diagnostics(),
            "max_raw_estimated_ev": max_raw_estimated_ev,
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
                "raw_estimated_odds": float(
                    estimate.get("raw_estimated_odds", estimated_odds)
                ),
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
