from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from typing import Any

from boatrace_ai.listwise.market_offset_calibration import (
    fit_market_offset_calibration,
)


DEFAULT_REGULARIZATION = 1.0
DEFAULT_CANDIDATES = (0.01, 0.1, 1.0, 10.0)
LOG_EPSILON = 1e-300


def _iso_date(value: object, name: str) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError as exc:
        raise ValueError(f"{name} must start with an ISO date") from exc


def _candidate_values(candidates: Iterable[float]) -> tuple[float, ...]:
    values: set[float] = set()
    for candidate in candidates:
        try:
            value = float(candidate)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "regularization candidates must be finite and positive"
            ) from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                "regularization candidates must be finite and positive"
            )
        values.add(value)
    if not values:
        raise ValueError("at least one regularization candidate is required")
    return tuple(sorted(values))


def _empty_candidate_metrics(
    candidates: tuple[float, ...],
    *,
    training_days: int,
    training_races: int,
    validation_races: int,
    reason: str,
) -> list[dict[str, object]]:
    return [
        {
            "regularization": regularization,
            "validation_log_loss": None,
            "converged": False,
            "fitted": False,
            "eligible": False,
            "training_days": training_days,
            "training_races": training_races,
            "validation_races": validation_races,
            "fallback_reason": reason,
        }
        for regularization in candidates
    ]


def _result(
    *,
    selected_regularization: float,
    validation_date: str | None,
    training_through: str | None,
    candidate_metrics: list[dict[str, object]],
    fallback_reason: str | None,
    prediction_date: str,
    prior_dates: tuple[str, ...],
    prior_records: int,
    excluded_dates: tuple[str, ...],
    excluded_records: int,
) -> dict[str, Any]:
    return {
        "selected_regularization": selected_regularization,
        "validation_date": validation_date,
        "training_through": training_through,
        "candidates": candidate_metrics,
        "fallback_reason": fallback_reason,
        "audit": {
            "prediction_date": prediction_date,
            "strictly_prior_dates": list(prior_dates),
            "strictly_prior_records": prior_records,
            "excluded_non_past_dates": list(excluded_dates),
            "excluded_non_past_records": excluded_records,
            "validation_design": (
                "latest complete strictly-prior day is validation; only "
                "earlier complete days are training"
            ),
            "teacher": "one_hot_actual_combination",
            "uses_profit_or_payout_teacher": False,
        },
    }


def select_market_offset_regularization(
    records: Iterable[Mapping[str, Any]],
    prediction_date: object,
    candidates: Iterable[float] = DEFAULT_CANDIDATES,
    min_inner_training_days: int = 3,
    min_inner_training_races: int = 300,
) -> dict[str, Any]:
    """Select market-offset regularization on one strict prior-day holdout.

    The outer prediction day and all later records are removed before any
    feature or teacher field is read. The latest remaining day is held out in
    full; every candidate is fitted using only earlier complete days.
    """
    target = _iso_date(prediction_date, "prediction_date")
    candidate_values = _candidate_values(candidates)
    if min_inner_training_days < 1:
        raise ValueError("min_inner_training_days must be positive")
    if min_inner_training_races < 1:
        raise ValueError("min_inner_training_races must be positive")

    by_day: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    excluded_dates: set[str] = set()
    excluded_records = 0
    for record in records:
        race_date = _iso_date(record.get("race_date"), "race_date")
        if race_date >= target:
            excluded_dates.add(race_date)
            excluded_records += 1
            continue
        by_day[race_date].append(record)

    prior_dates = tuple(sorted(by_day))
    prior_records = sum(len(by_day[day]) for day in prior_dates)
    if not prior_dates:
        reason = "no_strictly_prior_validation_day"
        return _result(
            selected_regularization=DEFAULT_REGULARIZATION,
            validation_date=None,
            training_through=None,
            candidate_metrics=_empty_candidate_metrics(
                candidate_values,
                training_days=0,
                training_races=0,
                validation_races=0,
                reason=reason,
            ),
            fallback_reason=reason,
            prediction_date=target,
            prior_dates=prior_dates,
            prior_records=prior_records,
            excluded_dates=tuple(sorted(excluded_dates)),
            excluded_records=excluded_records,
        )

    validation_date = prior_dates[-1]
    training_dates = prior_dates[:-1]
    training_records = [
        record for day in training_dates for record in by_day[day]
    ]
    validation_records = list(by_day[validation_date])
    training_through = training_dates[-1] if training_dates else None

    shortage_reason = None
    if len(training_dates) < min_inner_training_days:
        shortage_reason = "insufficient_inner_training_days"
    elif len(training_records) < min_inner_training_races:
        shortage_reason = "insufficient_inner_training_races"
    if shortage_reason is not None:
        return _result(
            selected_regularization=DEFAULT_REGULARIZATION,
            validation_date=validation_date,
            training_through=training_through,
            candidate_metrics=_empty_candidate_metrics(
                candidate_values,
                training_days=len(training_dates),
                training_races=len(training_records),
                validation_races=len(validation_records),
                reason=shortage_reason,
            ),
            fallback_reason=shortage_reason,
            prediction_date=target,
            prior_dates=prior_dates,
            prior_records=prior_records,
            excluded_dates=tuple(sorted(excluded_dates)),
            excluded_records=excluded_records,
        )

    metrics: list[dict[str, object]] = []
    for regularization in candidate_values:
        artifact = fit_market_offset_calibration(
            training_records,
            prediction_date=validation_date,
            regularization=regularization,
            min_training_races=min_inner_training_races,
        )
        eligible = bool(artifact.fitted and artifact.converged)
        validation_log_loss = None
        reason = artifact.fallback_reason
        if eligible:
            losses: list[float] = []
            for record in validation_records:
                prediction = artifact.predict(
                    record["model_probabilities"],
                    record["market_probabilities"],
                    record["forecast_odds"],
                    prediction_date=validation_date,
                )
                actual = str(record["actual_combination"])
                try:
                    probability = float(prediction.probabilities[actual])
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(
                        f"actual combination {actual!r} is missing from prediction"
                    ) from exc
                if not math.isfinite(probability) or probability < 0.0:
                    raise ValueError(
                        "validation probability must be finite and non-negative"
                    )
                losses.append(-math.log(max(LOG_EPSILON, probability)))
            validation_log_loss = math.fsum(sorted(losses)) / len(losses)
        elif reason is None:
            reason = "candidate_did_not_converge"
        metrics.append(
            {
                "regularization": regularization,
                "validation_log_loss": validation_log_loss,
                "converged": bool(artifact.converged),
                "fitted": bool(artifact.fitted),
                "eligible": eligible,
                "training_days": len(artifact.training_dates),
                "training_races": int(artifact.training_races),
                "validation_races": len(validation_records),
                "fallback_reason": reason,
            }
        )

    eligible_metrics = [metric for metric in metrics if metric["eligible"]]
    if eligible_metrics:
        selected = min(
            eligible_metrics,
            key=lambda metric: (
                float(metric["validation_log_loss"]),
                -float(metric["regularization"]),
            ),
        )
        selected_regularization = float(selected["regularization"])
        fallback_reason = None
    else:
        selected_regularization = DEFAULT_REGULARIZATION
        fallback_reason = "no_converged_candidates"

    return _result(
        selected_regularization=selected_regularization,
        validation_date=validation_date,
        training_through=training_through,
        candidate_metrics=metrics,
        fallback_reason=fallback_reason,
        prediction_date=target,
        prior_dates=prior_dates,
        prior_records=prior_records,
        excluded_dates=tuple(sorted(excluded_dates)),
        excluded_records=excluded_records,
    )
