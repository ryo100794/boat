from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib

from boatrace_ai.listwise.closing_odds_multihorizon_v11 import (
    CHECKPOINT_OFFSETS_SECONDS,
    closing_odds_multihorizon_v11_metrics,
    fit_closing_odds_multihorizon_v11,
)


def _load_races(path: Path) -> list[dict[str, Any]]:
    payload = joblib.load(path)
    races = payload.get("races") if isinstance(payload, dict) else payload
    if not isinstance(races, list):
        raise ValueError("cache must contain a race list")
    return races


def evaluate(cache: Path, *, first_evaluation_date: str | None) -> dict[str, Any]:
    races = _load_races(cache)
    dates = sorted({str(race["race_date"]) for race in races})
    folds = []
    for evaluation_date in dates:
        if first_evaluation_date and evaluation_date < first_evaluation_date:
            continue
        training = [
            race for race in races if str(race["race_date"]) < evaluation_date
        ]
        holdout = [
            race for race in races if str(race["race_date"]) == evaluation_date
        ]
        model = fit_closing_odds_multihorizon_v11(
            training,
            prediction_date=evaluation_date,
        )
        metrics = {
            f"t{offset}": closing_odds_multihorizon_v11_metrics(
                holdout,
                model,
                as_of_offset_seconds=offset,
            )
            for offset in CHECKPOINT_OFFSETS_SECONDS
        }
        folds.append(
            {
                "evaluation_date": evaluation_date,
                "training_days": len(
                    {str(race["race_date"]) for race in training}
                ),
                "training_races": len(training),
                "evaluation_races": len(holdout),
                "model_ready": bool(model.get("ready")),
                "trained_through_date": model.get("trained_through_date"),
                "horizon_selection": model.get("horizon_selection"),
                "metrics": metrics,
                "leakage_guard": model.get("leakage_guard"),
            }
        )
    return {
        "model": "closing_odds_multihorizon_v11",
        "validation": "strict_prior_outer_day",
        "cache": str(cache),
        "available_dates": dates,
        "available_races": len(races),
        "folds": folds,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate closing-odds V11 on strict prior-day folds."
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-evaluation-date")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = evaluate(
        args.cache,
        first_evaluation_date=args.first_evaluation_date,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
