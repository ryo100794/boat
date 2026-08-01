from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping

import numpy as np

from .v25_top1_narrow_policy_v33 import simulate_v25_top1_narrow_v33


MODEL_NAME = "v25_top1_t5_safety_v35"
T5_MIN_AGE_SECONDS = 240.0
T5_MAX_AGE_SECONDS = 420.0
ODDS_BUCKET_UPPER_BOUNDS = (10.0, 20.0, 40.0, 80.0, math.inf)


def _complete_odds(value: Any) -> dict[str, float] | None:
    if not isinstance(value, Mapping) or len(value) != 120:
        return None
    try:
        odds = {str(key): float(item) for key, item in value.items()}
    except (TypeError, ValueError, OverflowError):
        return None
    if len(odds) != 120 or any(
        not math.isfinite(item) or item <= 0.0 for item in odds.values()
    ):
        return None
    return odds


def _t5_checkpoint(
    race: Mapping[str, Any],
) -> tuple[dict[str, float], Mapping[str, Any]] | None:
    checkpoints = race.get("odds_checkpoints")
    if not isinstance(checkpoints, Mapping):
        return None
    checkpoint = checkpoints.get("300", checkpoints.get(300))
    if not isinstance(checkpoint, Mapping):
        return None
    odds = _complete_odds(checkpoint.get("odds"))
    try:
        age = float(checkpoint["captured_age_seconds"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if odds is None or not T5_MIN_AGE_SECONDS <= age <= T5_MAX_AGE_SECONDS:
        return None
    return odds, checkpoint


def _bucket_index(odds: float) -> int:
    return next(
        index
        for index, upper in enumerate(ODDS_BUCKET_UPPER_BOUNDS)
        if odds <= upper
    )


def fit_t5_safety_factors(
    races: list[dict[str, Any]],
    *,
    lower_quantile: float = 0.10,
    minimum_bucket_tickets: int = 500,
) -> dict[str, Any]:
    """Fit closing/T-5 safety factors without using outcomes or payouts."""
    if not 0.01 <= lower_quantile <= 0.49:
        raise ValueError("lower_quantile must be between 0.01 and 0.49")
    if minimum_bucket_tickets < 1:
        raise ValueError("minimum_bucket_tickets must be positive")
    ratios: list[float] = []
    by_bucket: dict[int, list[float]] = defaultdict(list)
    training_races = 0
    training_dates: set[str] = set()
    for race in races:
        checkpoint = _t5_checkpoint(race)
        closing = _complete_odds(
            race.get("official_closing_odds") or race.get("closing_odds")
        )
        if checkpoint is None or closing is None:
            continue
        current, _metadata = checkpoint
        if set(current) != set(closing):
            continue
        training_races += 1
        training_dates.add(str(race.get("race_date") or ""))
        for combination, current_odds in current.items():
            ratio = float(closing[combination]) / current_odds
            if not math.isfinite(ratio) or ratio <= 0.0:
                continue
            ratios.append(ratio)
            by_bucket[_bucket_index(current_odds)].append(ratio)
    if not ratios:
        raise ValueError(
            "T-5 safety calibration requires paired official closing odds"
        )
    global_factor = float(
        np.quantile(np.asarray(ratios, dtype=np.float64), lower_quantile)
    )
    factors: list[float] = []
    bucket_tickets: list[int] = []
    for index in range(len(ODDS_BUCKET_UPPER_BOUNDS)):
        values = by_bucket[index]
        bucket_tickets.append(len(values))
        factor = (
            float(
                np.quantile(
                    np.asarray(values, dtype=np.float64), lower_quantile
                )
            )
            if len(values) >= minimum_bucket_tickets
            else global_factor
        )
        factors.append(min(1.0, max(0.25, factor)))
    return {
        "model": "strict_prior_t5_closing_ratio_lower_quantile",
        "teacher": "official_closing_odds_divided_by_real_t5_odds",
        "uses_outcome_teacher": False,
        "uses_payout_teacher": False,
        "lower_quantile": float(lower_quantile),
        "minimum_bucket_tickets": int(minimum_bucket_tickets),
        "odds_bucket_upper_bounds": [
            None if math.isinf(value) else value
            for value in ODDS_BUCKET_UPPER_BOUNDS
        ],
        "global_factor": min(1.0, max(0.25, global_factor)),
        "bucket_factors": factors,
        "bucket_tickets": bucket_tickets,
        "training_races": training_races,
        "training_tickets": len(ratios),
        "training_dates": sorted(value for value in training_dates if value),
    }


def attach_t5_safety_odds(
    races: list[dict[str, Any]], factor_model: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], int]:
    factors = [float(value) for value in factor_model["bucket_factors"]]
    if len(factors) != len(ODDS_BUCKET_UPPER_BOUNDS):
        raise ValueError("T-5 safety factor bucket count mismatch")
    transformed: list[dict[str, Any]] = []
    excluded = 0
    for race in races:
        checkpoint = _t5_checkpoint(race)
        if checkpoint is None:
            excluded += 1
            continue
        odds, metadata = checkpoint
        item = dict(race)
        item["estimated_final_odds"] = {
            combination: min(
                999.9,
                max(1.0, value * factors[_bucket_index(value)]),
            )
            for combination, value in odds.items()
        }
        item["odds"] = odds
        item["captured_at"] = metadata.get("captured_at")
        item["closing_odds_forecast_target"] = (
            "strict_prior_t5_ratio_lower_quantile"
        )
        item["closing_odds_safety_factors"] = list(factors)
        transformed.append(item)
    return transformed, excluded


def walk_forward_v25_t5_safety_v35(
    races: list[dict[str, Any]],
    *,
    probability_artifact: Mapping[str, Any],
    evaluation_from: str,
    minimum_training_days: int = 2,
    lower_quantile: float = 0.10,
    minimum_bucket_tickets: int = 500,
    initial_bankroll_yen: int = 10_000,
) -> dict[str, Any]:
    if minimum_training_days < 1:
        raise ValueError("minimum_training_days must be positive")
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for race in races:
        by_day[str(race["race_date"])].append(race)
    transformed: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    for evaluation_date in sorted(day for day in by_day if day >= evaluation_from):
        training_dates = sorted(day for day in by_day if day < evaluation_date)
        if len(training_dates) < minimum_training_days:
            folds.append(
                {
                    "evaluation_date": evaluation_date,
                    "status": "no_bet_insufficient_strict_prior_days",
                    "training_days": len(training_dates),
                    "evaluation_races": len(by_day[evaluation_date]),
                }
            )
            continue
        factor_model = fit_t5_safety_factors(
            [race for day in training_dates for race in by_day[day]],
            lower_quantile=lower_quantile,
            minimum_bucket_tickets=minimum_bucket_tickets,
        )
        holdout, excluded = attach_t5_safety_odds(
            by_day[evaluation_date], factor_model
        )
        transformed.extend(holdout)
        folds.append(
            {
                "evaluation_date": evaluation_date,
                "status": "strict_prior_t5_safety_forecast",
                "training_days": len(training_dates),
                "training_max_date": training_dates[-1],
                "evaluation_races": len(by_day[evaluation_date]),
                "eligible_races": len(holdout),
                "excluded_without_real_t5": excluded,
                "factor_model": factor_model,
            }
        )
    bankroll = simulate_v25_top1_narrow_v33(
        transformed,
        probability_artifact=probability_artifact,
        initial_bankroll_yen=initial_bankroll_yen,
    )
    return {
        **bankroll,
        "model": MODEL_NAME,
        "odds_head": "strict_prior_real_t5_ratio_lower_quantile",
        "evaluation_from": evaluation_from,
        "minimum_training_days": minimum_training_days,
        "lower_quantile": lower_quantile,
        "folds": folds,
        "promotion_evidence": False,
        "status": "retrospective_diagnostic_only",
        "real_betting_enabled": False,
    }
