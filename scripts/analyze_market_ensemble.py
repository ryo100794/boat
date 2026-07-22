#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from boatrace_ai.listwise.market_ensemble import (
    align_scored_races,
    fit_log_pool_newton,
    probability_metrics,
    select_source_subset_prequential,
)


def _cache_argument(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("cache must use NAME=PATH")
    return name, Path(path)


def _load_races(path: Path) -> list[dict]:
    payload = joblib.load(path)
    races = payload.get("races") if isinstance(payload, dict) else None
    if not isinstance(races, list):
        raise ValueError(f"cache has no races list: {path}")
    return races


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Forward-select a multi-model market log-pool ensemble."
    )
    parser.add_argument(
        "--cache",
        action="append",
        type=_cache_argument,
        required=True,
        help="Scored cache in NAME=PATH form; repeat for each model.",
    )
    parser.add_argument("--evaluation-date", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cache_paths = dict(args.cache)
    if len(cache_paths) != len(args.cache):
        raise ValueError("cache source names must be unique")
    aligned = align_scored_races(
        {name: _load_races(path) for name, path in cache_paths.items()}
    )
    calibration = [
        race for race in aligned if str(race["race_date"]) < args.evaluation_date
    ]
    evaluation = [
        race for race in aligned if str(race["race_date"]) == args.evaluation_date
    ]
    if not evaluation:
        raise ValueError("evaluation date has no common scored races")
    subset_selection = select_source_subset_prequential(
        calibration,
        available_sources=tuple(cache_paths),
    )
    selected = subset_selection["selected"]
    source_names = tuple(selected["source_names"])
    calibrator = selected["final_calibrator"]
    evaluation_metrics = probability_metrics(
        evaluation,
        source_names=source_names,
        calibrator=calibrator,
    )
    market_calibrator = {
        "coefficients": {
            "market": 1.0,
            **{name: 0.0 for name in source_names},
        }
    }
    pure_metrics = {
        "market": probability_metrics(
            evaluation,
            source_names=source_names,
            calibrator=market_calibrator,
        )
    }
    for name in cache_paths:
        pure_metrics[name] = probability_metrics(
            evaluation,
            source_names=(name,),
            calibrator={"coefficients": {"market": 0.0, name: 1.0}},
        )
    payload = {
        "comparison_role": (
            "source subset and regularization selected before untouched evaluation date"
        ),
        "cache_paths": {name: str(path) for name, path in cache_paths.items()},
        "common_races": len(aligned),
        "calibration_dates": sorted(
            {str(race["race_date"]) for race in calibration}
        ),
        "calibration_races": len(calibration),
        "evaluation_date": args.evaluation_date,
        "evaluation_races": len(evaluation),
        "evaluation_not_used_for_selection": True,
        "selected_sources": list(source_names),
        "selected_regularization": selected["selected_regularization"],
        "prequential_log_loss": selected["prequential_log_loss"],
        "prequential_top5_hit_rate": selected["prequential_top5_hit_rate"],
        "calibrator": calibrator,
        "evaluation_metrics": evaluation_metrics,
        "pure_evaluation_metrics": pure_metrics,
        "selection_candidates": [
            {
                "source_names": row["source_names"],
                "selected_regularization": row["selected_regularization"],
                "prequential_log_loss": row["prequential_log_loss"],
                "prequential_top5_hit_rate": row["prequential_top5_hit_rate"],
            }
            for row in subset_selection["candidates"]
        ],
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
