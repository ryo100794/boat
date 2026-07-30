from __future__ import annotations

import argparse
from datetime import date
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib

from boatrace_ai.listwise.closing_odds_t300_nonlinear_v12 import (
    MODEL_NAME,
    closing_odds_t300_nonlinear_v12_metrics,
    fit_closing_odds_t300_nonlinear_v12,
)


VALIDATION_NAME = "strict_prior_outer_day"


def _iso_date(value: object, name: str) -> str:
    text = str(value).strip()
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO date") from exc


def _race_sort_key(race: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(race["race_date"]),
        str(race.get("race_id") or ""),
        str(race.get("jcd") or race.get("venue_code") or ""),
        str(race.get("rno") or race.get("race_no") or ""),
    )


def _load_races(path: Path) -> list[dict[str, Any]]:
    payload = joblib.load(path)
    races = payload.get("races") if isinstance(payload, dict) else payload
    if not isinstance(races, list):
        raise ValueError("scored cache must contain a race list")
    result: list[dict[str, Any]] = []
    for index, race in enumerate(races):
        if not isinstance(race, dict):
            raise ValueError(f"race at index {index} must be a mapping")
        item = dict(race)
        item["race_date"] = _iso_date(item.get("race_date"), "race_date")
        result.append(item)
    result.sort(key=_race_sort_key)
    return result


def _finite_metric(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _weighted_aggregate(folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    weighted_baseline = 0.0
    weighted_selected = 0.0
    weighted_coverage = 0.0
    metric_tickets = 0
    coverage_tickets = 0
    adopted_tickets = 0
    adopted_folds = 0
    evaluated_folds = 0
    engines_by_fold: dict[str, int] = {}
    engines_by_evaluation_tickets: dict[str, int] = {}
    for fold in folds:
        metrics = fold.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        try:
            tickets = int(metrics.get("evaluation_tickets") or 0)
        except (TypeError, ValueError, OverflowError):
            tickets = 0
        tickets = max(0, tickets)
        baseline = _finite_metric(metrics.get("baseline_current_log_mae"))
        selected = _finite_metric(metrics.get("selected_point_log_mae"))
        coverage = _finite_metric(metrics.get("lower_bound_coverage"))
        engine = str(fold.get("engine") or "unknown")
        engines_by_fold[engine] = engines_by_fold.get(engine, 0) + 1
        engines_by_evaluation_tickets[engine] = (
            engines_by_evaluation_tickets.get(engine, 0) + tickets
        )
        if baseline is not None and selected is not None and tickets > 0:
            weighted_baseline += baseline * tickets
            weighted_selected += selected * tickets
            metric_tickets += tickets
            evaluated_folds += 1
            if bool(fold.get("challenger_adopted")):
                adopted_tickets += tickets
        if coverage is not None and tickets > 0:
            weighted_coverage += coverage * tickets
            coverage_tickets += tickets
        adopted_folds += int(bool(fold.get("challenger_adopted")))
    baseline_mae = (
        weighted_baseline / metric_tickets if metric_tickets else None
    )
    selected_mae = (
        weighted_selected / metric_tickets if metric_tickets else None
    )
    improvement = (
        1.0 - selected_mae / baseline_mae
        if selected_mae is not None
        and baseline_mae is not None
        and baseline_mae > 0.0
        else None
    )
    return {
        "weighting": "evaluation_tickets",
        "folds": len(folds),
        "evaluated_folds": evaluated_folds,
        "evaluation_tickets": metric_tickets,
        "baseline_current_log_mae": baseline_mae,
        "selected_point_log_mae": selected_mae,
        "selected_relative_mae_improvement": improvement,
        "lower_bound_coverage": (
            weighted_coverage / coverage_tickets if coverage_tickets else None
        ),
        "coverage_evaluation_tickets": coverage_tickets,
        "adopted_folds": adopted_folds,
        "adopted_fold_rate": adopted_folds / len(folds) if folds else None,
        "adopted_evaluation_tickets": adopted_tickets,
        "adopted_evaluation_ticket_rate": (
            adopted_tickets / metric_tickets if metric_tickets else None
        ),
        "engines_by_fold": dict(sorted(engines_by_fold.items())),
        "engines_by_evaluation_tickets": dict(
            sorted(engines_by_evaluation_tickets.items())
        ),
    }


def evaluate(
    cache: Path, *, first_evaluation_date: str | None
) -> dict[str, Any]:
    races = _load_races(cache)
    first_date = (
        _iso_date(first_evaluation_date, "first_evaluation_date")
        if first_evaluation_date is not None
        else None
    )
    dates = sorted({str(race["race_date"]) for race in races})
    folds: list[dict[str, Any]] = []
    for evaluation_date in dates:
        if first_date is not None and evaluation_date < first_date:
            continue
        training = [
            race for race in races if str(race["race_date"]) < evaluation_date
        ]
        holdout = [
            race for race in races if str(race["race_date"]) == evaluation_date
        ]
        model = fit_closing_odds_t300_nonlinear_v12(
            training,
            prediction_date=evaluation_date,
        )
        trained_through = model.get("trained_through_date")
        strict_boundary = bool(
            trained_through is None or str(trained_through) < evaluation_date
        )
        if not strict_boundary:
            raise ValueError(
                f"V12 fold leaked non-prior data into {evaluation_date}"
            )
        metrics = closing_odds_t300_nonlinear_v12_metrics(holdout, model)
        folds.append(
            {
                "evaluation_date": evaluation_date,
                "training_days": len(
                    {str(race["race_date"]) for race in training}
                ),
                "training_races": len(training),
                "evaluation_races": len(holdout),
                "model_ready": bool(model.get("ready")),
                "trained_through_date": trained_through,
                "strict_prior_boundary": strict_boundary,
                "challenger_adopted": bool(model.get("challenger_adopted")),
                "selected_mode": model.get("selected_mode"),
                "engine": model.get("actual_engine"),
                "selection_reason": model.get("selection_reason"),
                "strict_prior_baseline_current_mae": model.get(
                    "strict_prior_baseline_current_mae"
                ),
                "strict_prior_challenger_mae": model.get(
                    "strict_prior_challenger_mae"
                ),
                "strict_prior_relative_mae_improvement": model.get(
                    "strict_prior_relative_mae_improvement"
                ),
                "metrics": metrics,
            }
        )
    return {
        "model": MODEL_NAME,
        "validation": VALIDATION_NAME,
        "cache": str(cache),
        "first_evaluation_date": first_date,
        "available_dates": dates,
        "available_races": len(races),
        "daily": folds,
        "aggregate": _weighted_aggregate(folds),
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate closing-odds V12 on strict prior-day folds."
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-evaluation-date")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate(
        args.cache,
        first_evaluation_date=args.first_evaluation_date,
    )
    _atomic_write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
