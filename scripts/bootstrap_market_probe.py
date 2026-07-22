#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib

from boatrace_ai.listwise.market_calibration import blend_probabilities
from boatrace_ai.listwise.market_residual import log_pool_probabilities
from boatrace_ai.listwise.paired_bootstrap import paired_mean_bootstrap


EPSILON = 1e-12


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paired bootstrap of a market calibration probe versus market odds."
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20_000)
    return parser


def _calibrated_probabilities(
    race: dict,
    *,
    strategy: str,
    calibrator: dict,
) -> dict[str, float]:
    if strategy == "newton_residual":
        return log_pool_probabilities(
            race["model_probabilities"],
            race["market_probabilities"],
            model_coefficient=float(calibrator["model_coefficient"]),
            market_coefficient=float(calibrator["market_coefficient"]),
        )
    if strategy == "grid":
        return blend_probabilities(
            race["model_probabilities"],
            race["market_probabilities"],
            model_weight=float(calibrator["model_weight"]),
            temperature=float(calibrator["temperature"]),
        )
    raise ValueError(f"unsupported calibrator strategy: {strategy}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cache = joblib.load(args.cache)
    races = cache.get("races") if isinstance(cache, dict) else None
    if not isinstance(races, list):
        raise ValueError("cache has no races list")
    result = json.loads(args.result.read_text(encoding="utf-8"))
    folds = result.get("folds") or []
    if len(folds) != 1:
        raise ValueError("probe result must contain exactly one evaluation fold")
    fold = folds[0]
    evaluation_date = str(fold["evaluation_date"])
    calibrator = fold["calibrator"]
    strategy = str(result["calibrator_strategy"])
    evaluation = [
        race for race in races if str(race["race_date"]) == evaluation_date
    ]
    loss_differences = []
    top5_differences = []
    for race in evaluation:
        actual = str(race["actual_combination"])
        market = race["market_probabilities"]
        calibrated = _calibrated_probabilities(
            race,
            strategy=strategy,
            calibrator=calibrator,
        )
        market_loss = -math.log(max(EPSILON, float(market.get(actual, 0.0))))
        calibrated_loss = -math.log(
            max(EPSILON, float(calibrated.get(actual, 0.0)))
        )
        loss_differences.append(calibrated_loss - market_loss)
        market_top5 = sorted(market, key=market.get, reverse=True)[:5]
        calibrated_top5 = sorted(
            calibrated, key=calibrated.get, reverse=True
        )[:5]
        top5_differences.append(
            float(actual in calibrated_top5) - float(actual in market_top5)
        )
    payload = {
        "comparison_role": "paired race-level bootstrap; negative loss difference is better",
        "cache": str(args.cache),
        "result": str(args.result),
        "evaluation_date": evaluation_date,
        "calibrator_strategy": strategy,
        "calibrator": calibrator,
        "point_metrics": fold.get("probability_metrics") or {},
        "log_loss_difference_calibrated_minus_market": paired_mean_bootstrap(
            loss_differences,
            samples=args.samples,
            seed=20260722,
        ),
        "top5_hit_difference_calibrated_minus_market": paired_mean_bootstrap(
            top5_differences,
            samples=args.samples,
            seed=20260723,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
