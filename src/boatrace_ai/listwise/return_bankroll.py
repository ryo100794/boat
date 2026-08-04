from __future__ import annotations

from collections import defaultdict
from typing import Any, MutableMapping

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
    SELECTION_BOOTSTRAP_SAMPLES,
    _validate_nondecreasing_race_dates,
    _validate_pre_evaluation_period,
    bootstrap_daily_bankroll,
    COMBINATION_LABELS,
    direct_candidates,
    standard_direct_policy,
)
from .return_calibrator import (
    CombinationReturnCalibrator,
    ExpectedReturnCalibrator,
    calibrate_combination_returns,
    expected_return_poisson_loss,
    fit_combination_return_calibrator,
    fit_expected_return_calibrator,
    predict_expected_returns,
)
from .return_policy import (
    DEFAULT_THRESHOLD_CANDIDATES,
    calibration_policy_split,
    select_policy_threshold,
)


COMBINATION_LANES = np.asarray(TRIFECTA_COMBINATIONS, dtype=np.int64) - 1
EXPECTED_RETURN_STATE_SCHEMA = "expected_return_next_day_inference_v1"
EXPECTED_RETURN_STATE_KEYS = frozenset(
    {
        "state_schema",
        "state_role",
        "trained_through",
        "valid_for_dates_after",
        "contains_evaluation_outcomes",
        "holdout_replay_state",
        "return_calibrator",
        "combination_calibrator",
        "policy",
    }
)


def validate_expected_return_inference_state(
    state: dict[str, Any],
) -> None:
    missing = EXPECTED_RETURN_STATE_KEYS - set(state)
    if missing:
        raise ValueError(
            "expected return inference state is incomplete: "
            + ", ".join(sorted(missing))
        )
    if state["state_schema"] != EXPECTED_RETURN_STATE_SCHEMA:
        raise ValueError("expected return inference state schema is invalid")
    if state["state_role"] != "next_day_inference_after_evaluation":
        raise ValueError("expected return state is not for next-day inference")
    if state["contains_evaluation_outcomes"] is not True:
        raise ValueError("expected return state must include completed evaluation")
    if state["holdout_replay_state"] is not False:
        raise ValueError("expected return state must not replay its holdout")
    if state["valid_for_dates_after"] != state["trained_through"]:
        raise ValueError("expected return state validity boundary is invalid")
    if not isinstance(state["return_calibrator"], ExpectedReturnCalibrator):
        raise ValueError("expected return state calibrator is invalid")
    if not isinstance(
        state["combination_calibrator"], CombinationReturnCalibrator
    ):
        raise ValueError("expected return state combination calibrator is invalid")
    if not isinstance(state["policy"], dict):
        raise ValueError("expected return state policy is invalid")


def predict_expected_returns_from_state(
    state: dict[str, Any],
    candidate_probabilities: np.ndarray,
    market_probabilities: np.ndarray,
    race_keys: list[tuple[str, str, str, int]],
    *,
    batch_races: int = 500,
) -> np.ndarray:
    validate_expected_return_inference_state(state)
    if any(
        str(race_key[1]) <= str(state["valid_for_dates_after"])
        for race_key in race_keys
    ):
        raise ValueError(
            "expected return next-day state cannot score its training period"
        )
    predicted = predict_expected_returns(
        state["return_calibrator"],
        candidate_probabilities,
        market_probabilities,
        race_keys,
        COMBINATION_LANES,
        batch_races=batch_races,
    )
    return calibrate_combination_returns(
        state["combination_calibrator"], predicted
    )


def _combination_calibration_summary(calibrator: Any) -> dict[str, Any]:
    return {
        "method": "daily-bootstrap one-sided 95% lower bound",
        "training_races": int(calibrator.training_races),
        "training_days": int(calibrator.training_days),
        "bootstrap_samples": int(calibrator.bootstrap_samples),
        "factor_min": float(calibrator.factors.min()),
        "factor_median": float(np.median(calibrator.factors)),
        "factor_max": float(calibrator.factors.max()),
        "zero_factor_combinations": int(
            np.count_nonzero(calibrator.factors == 0.0)
        ),
        "below_legacy_floor_combinations": int(
            np.count_nonzero(calibrator.factors < 0.25)
        ),
        "factors": {
            combination: float(calibrator.factors[index])
            for index, combination in enumerate(COMBINATION_LABELS)
        },
        "point_ratios": {
            combination: float(calibrator.point_ratios[index])
            for index, combination in enumerate(COMBINATION_LABELS)
        },
        "lower_bounds": {
            combination: float(calibrator.lower_bounds[index])
            for index, combination in enumerate(COMBINATION_LABELS)
        },
    }


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
    largest_hit_return_yen = max(
        (int(row.get("largest_hit_return_yen") or 0) for row in daily),
        default=0,
    )
    hit_return_square_sum = sum(
        float(row.get("hit_return_square_sum_yen2") or 0.0)
        for row in daily
    )
    effective_hit_count = (
        return_yen * return_yen / hit_return_square_sum
        if return_yen > 0 and hit_return_square_sum > 0
        else 0.0
    )
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
        "largest_hit_return_yen": largest_hit_return_yen,
        "roi_without_largest_hit": (
            (return_yen - largest_hit_return_yen) / stake_yen
            if stake_yen else 0.0
        ),
        "largest_hit_return_share": (
            largest_hit_return_yen / return_yen if return_yen else None
        ),
        "effective_hit_count": effective_hit_count,
        "winning_days": int(totals["winning_days"]),
        "losing_days": int(totals["losing_days"]),
        "max_drawdown_yen": int(state[2]),
        "ticket_roi_attribution": attribution_summary,
        "daily": daily,
    }


def _adaptive_threshold_diagnostics(
    probabilities: np.ndarray,
    expected_returns: np.ndarray,
    race_keys: list[tuple[str, str, str, int]],
    payouts: dict[str, dict[str, Any]],
    thresholds: tuple[float, ...],
    base_policy: dict[str, Any],
    training_samples: int,
) -> list[dict[str, Any]]:
    candidate_floor = min(float(base_policy["ev_threshold"]), *(float(value) for value in thresholds))
    candidates_by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    evaluated_by_day: dict[str, set[str]] = defaultdict(set)
    for row_index, race_key in enumerate(race_keys):
        race_id, race_date, _jcd, _rno = race_key
        actual = payouts.get(race_id)
        if actual is None:
            continue
        evaluated_by_day[race_date].add(race_id)
        payout_model = {}
        for combination_index, combination in enumerate(COMBINATION_LABELS):
            probability = max(1e-15, float(probabilities[row_index, combination_index]))
            expected_return = float(expected_returns[row_index, combination_index])
            estimated_odds = min(2_000.0, max(1.1, expected_return / probability))
            payout_model[combination] = {
                "estimated_odds": estimated_odds,
                "estimated_payout_yen": estimated_odds * 100.0,
                "history_count": float(training_samples),
                "odds_source": "expected_return_poisson",
            }
        candidates_by_day[race_date].extend(
            direct_candidates(
                probabilities[row_index],
                race_key=race_key,
                actual=actual,
                payout_model=payout_model,
                ev_threshold=candidate_floor,
            )
        )

    race_dates = sorted({str(row[1]) for row in race_keys})
    diagnostics = []
    for threshold in thresholds:
        threshold_value = float(threshold)
        threshold_policy = dict(base_policy)
        threshold_policy["ev_threshold"] = threshold_value
        filtered = {
            race_date: [
                row
                for row in rows
                if float(row["estimated_ev"]) >= threshold_value
            ]
            for race_date, rows in candidates_by_day.items()
        }
        settled = _settle_candidate_days(
            filtered, evaluated_by_day, race_dates, threshold_policy
        )
        confidence = bootstrap_daily_bankroll(
            settled["daily"],
            samples=SELECTION_BOOTSTRAP_SAMPLES,
            seed=20260730,
        )
        diagnostics.append(
            {
                "ev_threshold": threshold_value,
                "tickets": int(settled["selected_tickets"]),
                "selected_races": int(settled["races_bet"]),
                "hits": int(settled["hit_tickets"]),
                "stake_yen": int(settled["stake_yen"]),
                "return_yen": int(settled["return_yen"]),
                "profit_yen": int(settled["profit_yen"]),
                "roi": float(settled["roi"]),
                "roi_without_largest_hit": float(
                    settled["roi_without_largest_hit"]
                ),
                "effective_hit_count": float(
                    settled["effective_hit_count"]
                ),
                "selection_roi_ci95_lower": float(
                    confidence["roi_ci95_lower"]
                ),
                "selection_probability_roi_above_one": float(
                    confidence["probability_roi_above_one"]
                ),
                "selection_profit_ci95_lower_yen": float(
                    confidence["profit_ci95_lower_yen"]
                ),
                "max_drawdown_yen": int(settled["max_drawdown_yen"]),
                "winning_days": int(settled["winning_days"]),
                "losing_days": int(settled["losing_days"]),
            }
        )
    return diagnostics


def _select_return_regularization(
    candidate_probabilities: np.ndarray,
    market_probabilities: np.ndarray,
    race_keys: list[tuple[str, str, str, int]],
    payouts: dict[str, dict[str, Any]],
    *,
    fit_stop: int,
    validation_days: int,
    candidates: tuple[float, ...],
    fallback: float,
    max_iterations: int,
    batch_races: int,
) -> tuple[float, list[dict[str, Any]], dict[str, Any] | None]:
    validation_split = calibration_policy_split(
        race_keys[:fit_stop], selection_days=validation_days
    )
    candidate_values = tuple(
        sorted({float(fallback), *(float(value) for value in candidates)})
    )
    if validation_split is None:
        return float(fallback), [], None
    diagnostics = []
    for regularization in candidate_values:
        model = fit_expected_return_calibrator(
            candidate_probabilities[:validation_split],
            market_probabilities[:validation_split],
            race_keys[:validation_split],
            payouts,
            COMBINATION_LANES,
            COMBINATION_INDEX,
            regularization=regularization,
            max_iterations=max_iterations,
            batch_races=batch_races,
        )
        predicted = predict_expected_returns(
            model,
            candidate_probabilities[validation_split:fit_stop],
            market_probabilities[validation_split:fit_stop],
            race_keys[validation_split:fit_stop],
            COMBINATION_LANES,
            batch_races=batch_races,
        )
        loss = expected_return_poisson_loss(
            predicted,
            race_keys[validation_split:fit_stop],
            payouts,
            COMBINATION_INDEX,
        )
        diagnostics.append(
            {
                "regularization": regularization,
                "poisson_loss": loss,
                "iterations": int(model.iterations),
                "converged": bool(model.converged),
                "gradient_norm": float(model.gradient_norm),
            }
        )
    selected = min(
        diagnostics,
        key=lambda row: (float(row["poisson_loss"]), -float(row["regularization"])),
    )
    period = {
        "fit_from": str(race_keys[0][1]),
        "fit_through": str(race_keys[validation_split - 1][1]),
        "validation_from": str(race_keys[validation_split][1]),
        "validation_through": str(race_keys[fit_stop - 1][1]),
        "fit_races": validation_split,
        "validation_races": fit_stop - validation_split,
    }
    return float(selected["regularization"]), diagnostics, period


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
    regularization_candidates: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0),
    regularization_validation_days: int = 15,
    max_iterations: int = 20,
    batch_races: int = 500,
    policy_selection_days: int = 30,
    threshold_candidates: tuple[float, ...] = DEFAULT_THRESHOLD_CANDIDATES,
    minimum_selection_tickets: int = 100,
    minimum_selection_roi: float = 1.05,
    minimum_selection_hits: int = 10,
    minimum_selection_winning_days: int = 8,
    state_output: MutableMapping[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_nondecreasing_race_dates(race_keys, name="race_keys")
    _validate_nondecreasing_race_dates(
        calibration_race_keys, name="calibration_race_keys"
    )
    _validate_pre_evaluation_period(calibration_race_keys, race_keys)
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

    base_policy = dict(policy or standard_direct_policy())
    fixed_threshold = float(base_policy["ev_threshold"])
    split = calibration_policy_split(
        calibration_race_keys,
        selection_days=policy_selection_days,
    )
    selected_regularization = float(regularization)
    regularization_diagnostics: list[dict[str, Any]] = []
    regularization_period = None
    if split is not None:
        (
            selected_regularization,
            regularization_diagnostics,
            regularization_period,
        ) = _select_return_regularization(
            calibration_values,
            calibration_market_values,
            calibration_race_keys,
            payouts,
            fit_stop=split,
            validation_days=regularization_validation_days,
            candidates=regularization_candidates,
            fallback=regularization,
            max_iterations=max_iterations,
            batch_races=batch_races,
        )
    policy_diagnostics: list[dict[str, Any]] = []
    selected_threshold = fixed_threshold
    selection_source = "fallback_fixed_threshold"
    selection_period = None
    selection_combination_calibration = None
    if split is not None:
        selection_calibrator = fit_expected_return_calibrator(
            calibration_values[:split],
            calibration_market_values[:split],
            calibration_race_keys[:split],
            payouts,
            COMBINATION_LANES,
            COMBINATION_INDEX,
            regularization=selected_regularization,
            max_iterations=max_iterations,
            batch_races=batch_races,
        )
        selection_fit_returns = predict_expected_returns(
            selection_calibrator,
            calibration_values[:split],
            calibration_market_values[:split],
            calibration_race_keys[:split],
            COMBINATION_LANES,
            batch_races=batch_races,
        )
        selection_combination_calibrator = fit_combination_return_calibrator(
            selection_fit_returns,
            calibration_race_keys[:split],
            payouts,
            COMBINATION_INDEX,
        )
        selection_combination_calibration = _combination_calibration_summary(
            selection_combination_calibrator
        )
        selection_returns = calibrate_combination_returns(
            selection_combination_calibrator,
            predict_expected_returns(
                selection_calibrator,
                calibration_values[split:],
                calibration_market_values[split:],
                calibration_race_keys[split:],
                COMBINATION_LANES,
                batch_races=batch_races,
            ),
        )
        policy_diagnostics = _adaptive_threshold_diagnostics(
            calibration_values[split:],
            selection_returns,
            calibration_race_keys[split:],
            payouts,
            threshold_candidates,
            base_policy,
            selection_calibrator.training_samples,
        )
        selected_threshold, selection_source = select_policy_threshold(
            policy_diagnostics,
            fallback=fixed_threshold,
            minimum_tickets=minimum_selection_tickets,
            minimum_roi=minimum_selection_roi,
            minimum_hits=minimum_selection_hits,
            minimum_winning_days=minimum_selection_winning_days,
        )
        selection_period = {
            "fit_from": str(calibration_race_keys[0][1]),
            "fit_through": str(calibration_race_keys[split - 1][1]),
            "selection_from": str(calibration_race_keys[split][1]),
            "selection_through": str(calibration_race_keys[-1][1]),
            "fit_races": split,
            "selection_races": len(calibration_race_keys) - split,
        }

    calibrator = fit_expected_return_calibrator(
        calibration_values,
        calibration_market_values,
        calibration_race_keys,
        payouts,
        COMBINATION_LANES,
        COMBINATION_INDEX,
        regularization=selected_regularization,
        max_iterations=max_iterations,
        batch_races=batch_races,
    )
    calibration_returns = predict_expected_returns(
        calibrator,
        calibration_values,
        calibration_market_values,
        calibration_race_keys,
        COMBINATION_LANES,
        batch_races=batch_races,
    )
    combination_calibrator = fit_combination_return_calibrator(
        calibration_returns,
        calibration_race_keys,
        payouts,
        COMBINATION_INDEX,
    )
    expected_returns = calibrate_combination_returns(
        combination_calibrator,
        predict_expected_returns(
            calibrator,
            values,
            market_values,
            race_keys,
            COMBINATION_LANES,
            batch_races=batch_races,
        ),
    )
    selected_policy = base_policy
    selected_policy["ev_threshold"] = float(selected_threshold)
    selected_policy.update(
        {
            "payout_estimator": (
                "pre-evaluation Poisson expected-return plus conservative "
                "combination calibration"
            ),
            "market_reference": "fixed baseline probability",
            "expected_return_regularization": float(selected_regularization),
            "expected_return_training_samples": int(calibrator.training_samples),
            "selection": (
                "pre-evaluation temporal threshold selection; no evaluation-period tuning"
            ),
        }
    )
    candidate_floor = min(float(selected_threshold), fixed_threshold)
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
                ev_threshold=candidate_floor,
            )
        )

    selected_candidates_by_day = {
        race_date: [
            row
            for row in rows
            if float(row["estimated_ev"]) >= float(selected_threshold)
        ]
        for race_date, rows in candidates_by_day.items()
    }
    result = _settle_candidate_days(
        selected_candidates_by_day,
        evaluated_by_day,
        sorted({str(row[1]) for row in race_keys}),
        selected_policy,
    )
    if not np.isclose(selected_threshold, fixed_threshold):
        fixed_policy = dict(selected_policy)
        fixed_policy["ev_threshold"] = fixed_threshold
        fixed_policy["selection"] = (
            "fixed threshold comparison; no evaluation-period tuning"
        )
        fixed_candidates_by_day = {
            race_date: [
                row
                for row in rows
                if float(row["estimated_ev"]) >= fixed_threshold
            ]
            for race_date, rows in candidates_by_day.items()
        }
        result["fixed_threshold_comparison"] = _settle_candidate_days(
            fixed_candidates_by_day,
            evaluated_by_day,
            sorted({str(row[1]) for row in race_keys}),
            fixed_policy,
        )
    result["regularization_selection"] = {
        "source": (
            "pre_evaluation_poisson_validation"
            if regularization_diagnostics
            else "fallback_fixed_regularization"
        ),
        "fallback_regularization": float(regularization),
        "selected_regularization": float(selected_regularization),
        "validation_days": int(regularization_validation_days),
        "period": regularization_period,
        "diagnostics": regularization_diagnostics,
    }
    result["return_calibrator"] = {
        "method": "Poisson log-link damped Newton",
        "regularization": float(calibrator.regularization),
        "training_samples": int(calibrator.training_samples),
        "iterations": int(calibrator.iterations),
        "converged": bool(calibrator.converged),
        "objective": float(calibrator.objective),
        "gradient_norm": float(calibrator.gradient_norm),
        "max_expected_return": float(max_expected_return),
        "combination_calibration": _combination_calibration_summary(
            combination_calibrator
        ),
    }
    result["policy_selection"] = {
        "source": selection_source,
        "fixed_fallback_threshold": fixed_threshold,
        "selected_ev_threshold": float(selected_threshold),
        "selection_days": int(policy_selection_days),
        "minimum_tickets": int(minimum_selection_tickets),
        "minimum_roi": float(minimum_selection_roi),
        "minimum_hits": int(minimum_selection_hits),
        "minimum_winning_days": int(minimum_selection_winning_days),
        "period": selection_period,
        "combination_calibration": selection_combination_calibration,
        "threshold_diagnostics": policy_diagnostics,
    }
    if state_output is not None:
        combined_probabilities = np.concatenate(
            (calibration_values, values), axis=0
        )
        combined_market = np.concatenate(
            (calibration_market_values, market_values), axis=0
        )
        combined_keys = [*calibration_race_keys, *race_keys]
        deployment_calibrator = fit_expected_return_calibrator(
            combined_probabilities,
            combined_market,
            combined_keys,
            payouts,
            COMBINATION_LANES,
            COMBINATION_INDEX,
            regularization=selected_regularization,
            max_iterations=max_iterations,
            batch_races=batch_races,
        )
        deployment_returns = predict_expected_returns(
            deployment_calibrator,
            combined_probabilities,
            combined_market,
            combined_keys,
            COMBINATION_LANES,
            batch_races=batch_races,
        )
        deployment_combination_calibrator = (
            fit_combination_return_calibrator(
                deployment_returns,
                combined_keys,
                payouts,
                COMBINATION_INDEX,
            )
        )
        trained_through = (
            str(combined_keys[-1][1]) if combined_keys else ""
        )
        state_output.clear()
        state_output.update(
            {
                "state_schema": EXPECTED_RETURN_STATE_SCHEMA,
                "state_role": "next_day_inference_after_evaluation",
                "trained_through": trained_through,
                "valid_for_dates_after": trained_through,
                "contains_evaluation_outcomes": True,
                "holdout_replay_state": False,
                "return_calibrator": deployment_calibrator,
                "combination_calibrator": (
                    deployment_combination_calibrator
                ),
                "policy": dict(selected_policy),
            }
        )
        validate_expected_return_inference_state(dict(state_output))
    return result
