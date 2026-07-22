#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib

from scripts.analyze_market_momentum import evaluate_momentum_candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one clipped nonlinear model-market disagreement feature "
            "with forward-only regularization selection."
        )
    )
    parser.add_argument("cache", type=Path)
    parser.add_argument("--evaluation-date")
    parser.add_argument("--disagreement-clip", type=float, default=4.0)
    parser.add_argument("--output", type=Path)
    return parser


def attach_disagreement_curvature(
    races: list[dict[str, Any]],
    *,
    disagreement_clip: float,
) -> list[dict[str, Any]]:
    if disagreement_clip <= 0.0 or not math.isfinite(disagreement_clip):
        raise ValueError("disagreement clip must be positive and finite")
    transformed = []
    for race in races:
        model = race["model_probabilities"]
        market = race["market_probabilities"]
        combinations = sorted(set(model) & set(market))
        if len(combinations) != 120:
            continue
        log_pseudo = {}
        for combination in combinations:
            disagreement = math.log(
                max(1e-12, float(model[combination]))
            ) - math.log(max(1e-12, float(market[combination])))
            clipped = min(
                disagreement_clip,
                max(-disagreement_clip, disagreement),
            )
            curvature = clipped * abs(clipped)
            log_pseudo[combination] = math.log(
                max(1e-12, float(market[combination]))
            ) - curvature
        maximum = max(log_pseudo.values())
        values = {
            combination: math.exp(value - maximum)
            for combination, value in log_pseudo.items()
        }
        total = sum(values.values())
        item = dict(race)
        item["earlier_market_probabilities"] = {
            combination: value / total
            for combination, value in values.items()
        }
        item["momentum_scale"] = 1.0
        transformed.append(item)
    return transformed


def standardized_payload(
    evaluation: dict[str, Any],
    *,
    source_cache: Path,
    disagreement_clip: float,
) -> dict[str, Any]:
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
    market_pass = bool(
        float(market_loss["ci95_upper"]) <= 0.0
        and float(market_top5["ci95_lower"]) >= 0.0
    )
    metrics = evaluation["momentum_newton_residual"]["metrics"]
    return {
        "status": (
            "candidate_requires_new_day_confirmation"
            if incremental_pass
            else "rejected_no_incremental_value"
        ),
        "promotion_eligible": False,
        "source_cache": str(source_cache),
        "feature": "clipped_log_model_market_disagreement_times_absolute",
        "disagreement_clip": disagreement_clip,
        "evaluated_races": metrics["evaluated_races"],
        "calibrated_trifecta_log_loss": metrics["trifecta_log_loss"],
        "trifecta_top5_hit_rate": metrics["trifecta_top5_hit_rate"],
        "incremental_confidence_pass": incremental_pass,
        "market_comparison": {
            "comparison_role": (
                "paired development holdout; disagreement curvature minus T-5 market"
            ),
            "evaluation_races": metrics["evaluated_races"],
            "log_loss_difference_calibrated_minus_market": market_loss,
            "top5_hit_difference_calibrated_minus_market": market_top5,
            "confidence_pass": market_pass,
        },
        **evaluation,
    }


def main() -> int:
    args = build_parser().parse_args()
    cached = joblib.load(args.cache)
    races = cached.get("races") if isinstance(cached, dict) else None
    if not isinstance(races, list):
        raise ValueError("cache does not contain a races list")
    transformed = attach_disagreement_curvature(
        races,
        disagreement_clip=args.disagreement_clip,
    )
    evaluation = evaluate_momentum_candidate(
        transformed,
        evaluation_date=args.evaluation_date,
    )
    payload = standardized_payload(
        evaluation,
        source_cache=args.cache,
        disagreement_clip=args.disagreement_clip,
    )
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
