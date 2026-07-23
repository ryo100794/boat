#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import joblib

from boatrace_ai.listwise.market_rank_residual import (
    rank_residual_metrics,
    rank_residual_probabilities,
    select_rank_regularization_prequential,
)
from boatrace_ai.listwise.market_residual import (
    log_pool_probabilities,
    residual_probability_metrics,
    select_regularization_prequential,
)
from boatrace_ai.listwise.paired_bootstrap import paired_mean_bootstrap


def paired_comparison(
    races: list[dict[str, Any]],
    *,
    left: Callable[[dict[str, Any]], dict[str, float]],
    right: Callable[[dict[str, Any]], dict[str, float]],
    label: str,
) -> dict[str, Any]:
    loss_differences = []
    top5_differences = []
    for race in races:
        left_probabilities = left(race)
        right_probabilities = right(race)
        actual = str(race["actual_combination"])
        loss_differences.append(
            -math.log(max(1e-12, left_probabilities.get(actual, 0.0)))
            + math.log(max(1e-12, right_probabilities.get(actual, 0.0)))
        )
        left_top5 = sorted(
            left_probabilities, key=left_probabilities.get, reverse=True
        )[:5]
        right_top5 = sorted(
            right_probabilities, key=right_probabilities.get, reverse=True
        )[:5]
        top5_differences.append(
            float(actual in left_top5) - float(actual in right_top5)
        )
    return {
        "comparison": label,
        "log_loss_difference": paired_mean_bootstrap(
            loss_differences, samples=20_000, seed=20260729
        ),
        "top5_hit_difference": paired_mean_bootstrap(
            top5_differences, samples=20_000, seed=20260730
        ),
    }


def evaluate(
    races: list[dict[str, Any]], *, evaluation_date: str | None = None
) -> dict[str, Any]:
    dates = sorted({str(race["race_date"]) for race in races})
    selected_date = evaluation_date or dates[-1]
    calibration = [
        race for race in races if str(race["race_date"]) < selected_date
    ]
    holdout = [
        race for race in races if str(race["race_date"]) == selected_date
    ]
    calibration_dates = sorted({str(race["race_date"]) for race in calibration})
    if len(calibration_dates) < 2:
        raise ValueError("at least two earlier full days are required")
    if not holdout:
        raise ValueError("evaluation date has no eligible races")

    baseline_selection = select_regularization_prequential(calibration)
    rank_selection = select_rank_regularization_prequential(calibration)
    baseline = baseline_selection["final_calibrator"]
    rank_calibrator = rank_selection["final_calibrator"]

    def baseline_probabilities(race: dict[str, Any]) -> dict[str, float]:
        return log_pool_probabilities(
            race["model_probabilities"],
            race["market_probabilities"],
            model_coefficient=float(baseline["model_coefficient"]),
            market_coefficient=float(baseline["market_coefficient"]),
        )

    def candidate_probabilities(race: dict[str, Any]) -> dict[str, float]:
        return rank_residual_probabilities(
            race["model_probabilities"],
            race["market_probabilities"],
            coefficients=rank_calibrator["coefficients"],
        )

    return {
        "status": "development_holdout_only",
        "promotion_eligible": False,
        "feature": "market_rank_bucket_bias_and_model_market_residual",
        "parameter_count": 6,
        "calibration_dates": calibration_dates,
        "evaluation_date": selected_date,
        "calibration_races": len(calibration),
        "evaluation_races": len(holdout),
        "baseline": {
            "selection": baseline_selection,
            "metrics": residual_probability_metrics(holdout, baseline),
        },
        "rank_residual": {
            "selection": rank_selection,
            "metrics": rank_residual_metrics(holdout, rank_calibrator),
        },
        "rank_residual_vs_baseline": paired_comparison(
            holdout,
            left=candidate_probabilities,
            right=baseline_probabilities,
            label="rank-residual-minus-global-newton-residual",
        ),
        "rank_residual_vs_market": paired_comparison(
            holdout,
            left=candidate_probabilities,
            right=lambda race: race["market_probabilities"],
            label="rank-residual-minus-t5-market",
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a six-parameter market-rank residual model."
    )
    parser.add_argument("cache", type=Path)
    parser.add_argument("--evaluation-date")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cached = joblib.load(args.cache)
    races = cached.get("races") if isinstance(cached, dict) else None
    if not isinstance(races, list):
        raise ValueError("cache does not contain a races list")
    result = evaluate(races, evaluation_date=args.evaluation_date)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        temporary.replace(args.output)
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
