#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib

from boatrace_ai.db import connection
from boatrace_ai.features import latest_trifecta_odds_before_deadline
from boatrace_ai.listwise.market_calibration import (
    normalized_market_probabilities,
    snapshot_age_seconds,
)
from boatrace_ai.listwise.market_momentum import (
    fit_momentum_newton,
    momentum_probabilities,
    momentum_probability_metrics,
    select_momentum_regularization_prequential,
)
from boatrace_ai.listwise.market_residual import (
    fit_log_pool_newton,
    log_pool_probabilities,
    residual_probability_metrics,
    select_regularization_prequential,
)
from boatrace_ai.listwise.paired_bootstrap import paired_mean_bootstrap


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a low-dimensional T-10 to T-5 odds-momentum residual "
            "on a forward-only day holdout."
        )
    )
    parser.add_argument("cache", type=Path)
    parser.add_argument("--db", required=True)
    parser.add_argument("--evaluation-date")
    parser.add_argument("--fixed-regularization", type=float)
    parser.add_argument("--earlier-decision-lead-minutes", type=int, default=10)
    parser.add_argument("--max-snapshot-age-seconds", type=float, default=60.0)
    parser.add_argument("--output", type=Path)
    return parser


def attach_earlier_market_probabilities(
    conn,
    races: list[dict[str, Any]],
    *,
    earlier_decision_lead_minutes: int,
    max_snapshot_age_seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    augmented = []
    skipped = Counter()
    by_day = Counter()
    for race in races:
        snapshot = latest_trifecta_odds_before_deadline(
            conn,
            str(race["race_id"]),
            min_combinations=120,
            decision_lead_minutes=earlier_decision_lead_minutes,
        )
        if snapshot is None or len(snapshot.get("odds") or {}) != 120:
            skipped["missing_earlier_snapshot"] += 1
            continue
        age = snapshot_age_seconds(snapshot)
        if age is None or age < 0.0 or age > max_snapshot_age_seconds:
            skipped["stale_earlier_snapshot"] += 1
            continue
        earlier = normalized_market_probabilities(snapshot["odds"])
        try:
            current_captured = datetime.fromisoformat(str(race["captured_at"]))
            earlier_captured = datetime.fromisoformat(str(snapshot["captured_at"]))
        except (KeyError, TypeError, ValueError):
            skipped["invalid_capture_interval"] += 1
            continue
        if current_captured.tzinfo is None:
            current_captured = current_captured.replace(tzinfo=timezone.utc)
        if earlier_captured.tzinfo is None:
            earlier_captured = earlier_captured.replace(tzinfo=timezone.utc)
        gap_seconds = (
            current_captured
            - earlier_captured.astimezone(current_captured.tzinfo)
        ).total_seconds()
        if gap_seconds <= 0.0 or gap_seconds > 900.0:
            skipped["invalid_capture_interval"] += 1
            continue
        if set(earlier) != set(race["market_probabilities"]):
            skipped["combination_mismatch"] += 1
            continue
        item = dict(race)
        item["earlier_market_probabilities"] = earlier
        item["earlier_snapshot_id"] = snapshot.get("snapshot_id")
        item["earlier_captured_at"] = snapshot.get("captured_at")
        item["earlier_odds_deadline_at"] = snapshot.get("odds_deadline_at")
        item["earlier_snapshot_age_seconds"] = age
        item["momentum_interval_seconds"] = gap_seconds
        item["momentum_scale"] = 300.0 / gap_seconds
        augmented.append(item)
        by_day[str(item["race_date"])] += 1
    return augmented, {
        "input_races": len(races),
        "eligible_momentum_races": len(augmented),
        "earlier_decision_lead_minutes": earlier_decision_lead_minutes,
        "max_snapshot_age_seconds": max_snapshot_age_seconds,
        "eligible_by_day": dict(sorted(by_day.items())),
        "skipped": dict(sorted(skipped.items())),
    }


def paired_model_comparison(
    races: list[dict[str, Any]],
    *,
    left_probability: Callable[[dict[str, Any]], dict[str, float]],
    right_probability: Callable[[dict[str, Any]], dict[str, float]],
    label: str,
) -> dict[str, Any]:
    loss_differences = []
    top5_differences = []
    for race in races:
        left = left_probability(race)
        right = right_probability(race)
        actual = str(race["actual_combination"])
        loss_differences.append(
            -math.log(max(1e-12, float(left.get(actual, 0.0))))
            + math.log(max(1e-12, float(right.get(actual, 0.0))))
        )
        left_top5 = sorted(left, key=left.get, reverse=True)[:5]
        right_top5 = sorted(right, key=right.get, reverse=True)[:5]
        top5_differences.append(
            float(actual in left_top5) - float(actual in right_top5)
        )
    return {
        "comparison": label,
        "log_loss_difference": paired_mean_bootstrap(
            loss_differences,
            samples=20_000,
            seed=20260724,
        ),
        "top5_hit_difference": paired_mean_bootstrap(
            top5_differences,
            samples=20_000,
            seed=20260725,
        ),
    }


def evaluate_momentum_candidate(
    races: list[dict[str, Any]],
    *,
    evaluation_date: str | None = None,
    fixed_regularization: float | None = None,
) -> dict[str, Any]:
    dates = sorted({str(race["race_date"]) for race in races})
    selected_evaluation_date = evaluation_date or (dates[-1] if dates else None)
    calibration = [
        race
        for race in races
        if selected_evaluation_date is not None
        and str(race["race_date"]) < selected_evaluation_date
    ]
    holdout = [
        race
        for race in races
        if str(race["race_date"]) == selected_evaluation_date
    ]
    calibration_dates = sorted({str(race["race_date"]) for race in calibration})
    if len(calibration_dates) < 2 and fixed_regularization is None:
        raise ValueError(
            "at least two complete calibration days are required unless a "
            "regularization is fixed before evaluation"
        )
    if not holdout:
        raise ValueError("evaluation date has no eligible momentum races")

    if fixed_regularization is None:
        baseline_selection = select_regularization_prequential(calibration)
        momentum_selection = select_momentum_regularization_prequential(calibration)
    else:
        baseline_selection = {
            "validation_design": "fixed regularization; no holdout selection",
            "dates": calibration_dates,
            "selected_regularization": fixed_regularization,
            "final_calibrator": fit_log_pool_newton(
                calibration,
                regularization=fixed_regularization,
            ),
            "candidates": [],
        }
        momentum_selection = {
            "validation_design": "fixed regularization; no holdout selection",
            "dates": calibration_dates,
            "selected_regularization": fixed_regularization,
            "final_calibrator": fit_momentum_newton(
                calibration,
                regularization=fixed_regularization,
            ),
            "candidates": [],
        }
    baseline_calibrator = baseline_selection["final_calibrator"]
    momentum_calibrator = momentum_selection["final_calibrator"]
    baseline_metrics = residual_probability_metrics(holdout, baseline_calibrator)
    momentum_metrics = momentum_probability_metrics(holdout, momentum_calibrator)

    def baseline_probability(race: dict[str, Any]) -> dict[str, float]:
        return log_pool_probabilities(
            race["model_probabilities"],
            race["market_probabilities"],
            model_coefficient=float(baseline_calibrator["model_coefficient"]),
            market_coefficient=float(baseline_calibrator["market_coefficient"]),
        )

    return {
        "comparison_role": (
            "development-only; calibration and regularization use earlier full "
            "days, evaluation date is untouched"
        ),
        "calibration_dates": calibration_dates,
        "evaluation_date": selected_evaluation_date,
        "calibration_races": len(calibration),
        "evaluation_races": len(holdout),
        "fixed_regularization": fixed_regularization,
        "baseline_newton_residual": {
            "selection": baseline_selection,
            "metrics": baseline_metrics,
        },
        "momentum_newton_residual": {
            "selection": momentum_selection,
            "metrics": momentum_metrics,
        },
        "momentum_vs_baseline": paired_model_comparison(
            holdout,
            left_probability=lambda race: momentum_probabilities(
                race, momentum_calibrator
            ),
            right_probability=baseline_probability,
            label="momentum_minus_same-subset_newton-residual",
        ),
        "momentum_vs_market": paired_model_comparison(
            holdout,
            left_probability=lambda race: momentum_probabilities(
                race, momentum_calibrator
            ),
            right_probability=lambda race: race["market_probabilities"],
            label="momentum_minus_t5-market",
        ),
    }


def main() -> int:
    args = build_parser().parse_args()
    cached = joblib.load(args.cache)
    races = cached.get("races") if isinstance(cached, dict) else None
    if not isinstance(races, list):
        raise ValueError("cache does not contain a races list")
    with connection(args.db) as conn:
        augmented, dataset = attach_earlier_market_probabilities(
            conn,
            races,
            earlier_decision_lead_minutes=args.earlier_decision_lead_minutes,
            max_snapshot_age_seconds=args.max_snapshot_age_seconds,
        )
    evaluation = evaluate_momentum_candidate(
        augmented,
        evaluation_date=args.evaluation_date,
        fixed_regularization=args.fixed_regularization,
    )
    incremental = evaluation["momentum_vs_baseline"]
    versus_market = evaluation["momentum_vs_market"]
    incremental_loss = incremental["log_loss_difference"]
    incremental_top5 = incremental["top5_hit_difference"]
    incremental_pass = bool(
        float(incremental_loss["ci95_upper"]) <= 0.0
        and float(incremental_top5["ci95_lower"]) >= 0.0
    )
    market_loss = versus_market["log_loss_difference"]
    market_top5 = versus_market["top5_hit_difference"]
    market_confidence_pass = bool(
        float(market_loss["ci95_upper"]) <= 0.0
        and float(market_top5["ci95_lower"]) >= 0.0
    )
    momentum_metrics = evaluation["momentum_newton_residual"]["metrics"]
    payload = {
        "status": (
            "candidate_requires_new_day_confirmation"
            if incremental_pass
            else "rejected_no_incremental_value"
        ),
        "promotion_eligible": False,
        "source_cache": str(args.cache),
        "dataset": dataset,
        "evaluated_races": momentum_metrics["evaluated_races"],
        "calibrated_trifecta_log_loss": momentum_metrics["trifecta_log_loss"],
        "trifecta_top5_hit_rate": momentum_metrics["trifecta_top5_hit_rate"],
        "incremental_confidence_pass": incremental_pass,
        "market_comparison": {
            "comparison_role": "paired development holdout; momentum minus T-5 market",
            "evaluation_races": momentum_metrics["evaluated_races"],
            "log_loss_difference_calibrated_minus_market": market_loss,
            "top5_hit_difference_calibrated_minus_market": market_top5,
            "confidence_pass": market_confidence_pass,
        },
        **evaluation,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
