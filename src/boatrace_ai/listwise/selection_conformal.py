from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable, Iterable

import numpy as np

from .odds_path_conservative_v7 import (
    MAX_TICKETS_PER_RACE,
    MIN_CLOSING_TRAINING_DAYS,
    MIN_CLOSING_TRAINING_RACES,
    SAFE_EV_THRESHOLD,
    _closing_teachers,
    _crossfit_probability_rows,
    _paired_race,
    fit_closing_log_ratio_q20_model,
    fit_probability_lcb,
    forecast_closing_q20,
)
from .edge_conditional_probability_lcb_v13 import probability_lower_bound_details


TARGET_COVERAGE = 0.80
MIN_TRAINING_DAYS = 4
MIN_TRAINING_CANDIDATES = 30
METHOD = "selected_top2_daily_cluster_finite_sample_lower_conformal_v2"


def selected_safe_ev_candidates(
    races: Iterable[dict[str, Any]],
    *,
    closing_forecasts: dict[str, dict[str, float]],
    probability_lcb: dict[str, Any],
) -> list[dict[str, Any]]:
    """Apply the registered pre-haircut selection rule without settlement data."""
    if not probability_lcb.get("ready"):
        return []
    selected: list[dict[str, Any]] = []
    for race in races:
        race_id = str(race["race_id"])
        closing = closing_forecasts.get(race_id) or {}
        probabilities = race.get("model_probabilities") or {}
        if len(closing) != 120 or len(probabilities) != 120:
            continue
        candidates = []
        for combination, predicted_closing in closing.items():
            lcb_detail = probability_lower_bound_details(
                race, str(combination), probability_lcb
            )
            probability = float(lcb_detail["probability"])
            safe_ev = probability * float(predicted_closing)
            if safe_ev < SAFE_EV_THRESHOLD:
                continue
            candidates.append({
                "race_id": race_id,
                "race_date": str(race["race_date"]),
                "combination": str(combination),
                "probability": probability,
                "predicted_closing": float(predicted_closing),
                "raw_safe_ev": safe_ev,
                "probability_lcb_detail": lcb_detail,
            })
        candidates.sort(
            key=lambda row: (
                -float(row["raw_safe_ev"]),
                -float(row["probability"]),
                str(row["combination"]),
            )
        )
        selected.extend(candidates[:MAX_TICKETS_PER_RACE])
    return selected


def _finite_sample_lower_rank(
    values: np.ndarray, *, target_coverage: float
) -> tuple[float, int, float]:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    alpha = 1.0 - target_coverage
    rank = max(
        1,
        int(math.floor((len(ordered) + 1) * alpha + 1e-12)),
    )
    rank = min(rank, len(ordered))
    return float(ordered[rank - 1]), rank, (
        (len(ordered) + 1 - rank) / (len(ordered) + 1)
    )


def fit_selection_conformal_haircut(
    observations: Iterable[dict[str, Any]],
    *,
    evaluation_date: str,
    target_coverage: float = TARGET_COVERAGE,
    minimum_training_days: int = MIN_TRAINING_DAYS,
    minimum_training_candidates: int = MIN_TRAINING_CANDIDATES,
) -> dict[str, Any]:
    """Fit a finite-sample lower conformal ratio using strictly prior days."""
    if not 0.0 < target_coverage < 1.0:
        raise ValueError("target_coverage must be between zero and one")
    rows = []
    for item in observations:
        date = str(item["race_date"])
        ratio = float(item["closing_ratio"])
        if date >= evaluation_date:
            raise ValueError("selection conformal observations must be prior-day only")
        if math.isfinite(ratio) and ratio > 0.0:
            rows.append({**item, "race_date": date, "closing_ratio": ratio})
    dates = sorted({row["race_date"] for row in rows})
    base = {
        "method": METHOD,
        "target_coverage": target_coverage,
        "evaluation_date": evaluation_date,
        "training_days": len(dates),
        "training_candidates": len(rows),
        "training_dates": dates,
        "trained_through_date": dates[-1] if dates else None,
        "minimum_training_days": minimum_training_days,
        "minimum_training_candidates": minimum_training_candidates,
    }
    if len(dates) < minimum_training_days or len(rows) < minimum_training_candidates:
        return {
            **base,
            "ready": False,
            "haircut": None,
            "finite_sample_unit": "prior_day",
            "finite_sample_rank": None,
            "finite_sample_coverage": None,
            "daily_lower_ratios": {},
            "daily_candidate_counts": {},
            "reason": "insufficient_prior_selected_candidates",
        }
    ratios = np.asarray(
        [float(row["closing_ratio"]) for row in rows], dtype=np.float64
    )
    by_day: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_day[row["race_date"]].append(float(row["closing_ratio"]))
    daily_lower: dict[str, float] = {}
    daily_counts: dict[str, int] = {}
    for date in dates:
        day_values = np.asarray(by_day[date], dtype=np.float64)
        daily_counts[date] = len(day_values)
        daily_lower[date] = _finite_sample_lower_rank(
            day_values, target_coverage=target_coverage
        )[0]
    raw_haircut, rank, finite_coverage = _finite_sample_lower_rank(
        np.asarray(list(daily_lower.values()), dtype=np.float64),
        target_coverage=target_coverage,
    )
    return {
        **base,
        "ready": True,
        "haircut": min(1.0, raw_haircut),
        "uncapped_haircut": raw_haircut,
        "finite_sample_unit": "prior_day",
        "finite_sample_rank": rank,
        "finite_sample_coverage": finite_coverage,
        "daily_lower_ratios": daily_lower,
        "daily_candidate_counts": daily_counts,
        "daily_lower_ratio_median": float(np.median(list(daily_lower.values()))),
        "ratio_mean": float(np.mean(ratios)),
        "ratio_p10": float(np.quantile(ratios, 0.10)),
        "ratio_p20": float(np.quantile(ratios, 0.20)),
        "ratio_median": float(np.median(ratios)),
        "ratio_min": float(np.min(ratios)),
        "ratio_max": float(np.max(ratios)),
        "reason": None,
    }


def selection_coverage_metrics(
    races: Iterable[dict[str, Any]],
    selected: Iterable[dict[str, Any]],
    *,
    haircut: float | None,
) -> dict[str, Any]:
    race_by_id = {str(race["race_id"]): race for race in races}
    selected = list(selected)
    ratios: list[float] = []
    for candidate in selected:
        race = race_by_id.get(str(candidate["race_id"]))
        closing = (race or {}).get("closing_odds") or {}
        actual = closing.get(str(candidate["combination"]))
        predicted = float(
            candidate.get("raw_predicted_closing_odds")
            or candidate["predicted_closing"]
        )
        if (
            actual is None
            or predicted <= 0.0
            or not math.isfinite(float(actual))
            or float(actual) <= 0.0
        ):
            continue
        ratio = float(actual) / predicted
        if math.isfinite(ratio) and ratio > 0.0:
            ratios.append(ratio)
    raw_covered = sum(ratio >= 1.0 for ratio in ratios)
    guarded_covered = (
        sum(ratio >= float(haircut) for ratio in ratios)
        if haircut is not None
        else 0
    )
    denominator = len(selected)
    missing = denominator - len(ratios)
    return {
        "selection_evaluation_candidates": denominator,
        "selection_observed_closing_candidates": len(ratios),
        "selection_closing_missing_candidates": missing,
        "selection_raw_covered_candidates": raw_covered,
        "selection_guarded_covered_candidates": guarded_covered,
        "selection_raw_closing_coverage": (
            raw_covered / denominator if denominator else None
        ),
        "selection_guarded_closing_coverage": (
            guarded_covered / denominator
            if denominator and haircut is not None
            else None
        ),
        "selection_closing_complete": denominator > 0 and missing == 0,
        "selection_closing_ratio_mean": (
            float(np.mean(ratios)) if ratios else None
        ),
        "selection_closing_ratio_p10": (
            float(np.quantile(ratios, 0.10)) if ratios else None
        ),
        "selection_closing_ratio_median": (
            float(np.median(ratios)) if ratios else None
        ),
        "selection_closing_ratios": ratios,
    }


def build_prequential_selection_conformal(
    races: list[dict[str, Any]],
    *,
    min_calibration_days: int,
    probability_fit: Callable[[list[dict[str, Any]]], dict[str, Any]],
    probability_attach: Callable[
        [list[dict[str, Any]], dict[str, Any]], list[dict[str, Any]]
    ],
) -> dict[str, Any]:
    """Build selected-candidate residuals with a strict outer-day boundary."""
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for race in races:
        by_day[str(race["race_date"])].append(race)
    dates = sorted(by_day)
    all_crossfit_rows = _crossfit_probability_rows(
        races,
        probability_fit=probability_fit,
        probability_attach=probability_attach,
    )
    observations: list[dict[str, Any]] = []
    artifacts: dict[str, dict[str, Any]] = {}
    for index, evaluation_date in enumerate(dates):
        artifacts[evaluation_date] = fit_selection_conformal_haircut(
            observations,
            evaluation_date=evaluation_date,
        )
        if index < min_calibration_days:
            continue
        training = [
            race for date in dates[:index] for race in by_day[date]
        ]
        teachers = _closing_teachers(training)
        teacher_days = {str(race["race_date"]) for race in teachers}
        if (
            len(teacher_days) < MIN_CLOSING_TRAINING_DAYS
            or len(teachers) < MIN_CLOSING_TRAINING_RACES
        ):
            continue
        probability_model = probability_fit(training)
        transformed = probability_attach(
            by_day[evaluation_date], probability_model
        )
        prior_crossfit = [
            row
            for row in all_crossfit_rows
            if str(row["race_date"]) < evaluation_date
        ]
        probability_lcb = fit_probability_lcb(prior_crossfit)
        if not probability_lcb.get("ready"):
            continue
        closing_model = fit_closing_log_ratio_q20_model(teachers)
        forecasts = {
            str(race["race_id"]): forecast_closing_q20(race, closing_model)
            for race in transformed
        }
        selected = selected_safe_ev_candidates(
            transformed,
            closing_forecasts=forecasts,
            probability_lcb=probability_lcb,
        )
        race_by_id = {str(race["race_id"]): race for race in transformed}
        for candidate in selected:
            race = race_by_id[candidate["race_id"]]
            paired = _paired_race(race)
            actual = (race.get("closing_odds") or {}).get(
                candidate["combination"]
            )
            if paired is None or actual is None:
                continue
            ratio = float(actual) / float(candidate["predicted_closing"])
            if math.isfinite(ratio) and ratio > 0.0:
                observations.append({
                    "race_date": evaluation_date,
                    "race_id": candidate["race_id"],
                    "combination": candidate["combination"],
                    "closing_ratio": ratio,
                })
    future_date = "9999-12-31"
    return {
        "artifacts_by_date": artifacts,
        "deployment_artifact": fit_selection_conformal_haircut(
            observations, evaluation_date=future_date
        ),
        "observations": observations,
    }
