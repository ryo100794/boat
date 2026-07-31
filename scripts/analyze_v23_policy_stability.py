#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib

from boatrace_ai.bankroll_bootstrap import bootstrap_daily_roi
from boatrace_ai.listwise.flat_policy import simulate_chronological_flat_policy
from boatrace_ai.listwise.market_calibration import blend_probabilities


DEFAULT_DISCOVERY_DATES = (
    "2026-07-22",
    "2026-07-23",
    "2026-07-24",
    "2026-07-25",
    "2026-07-26",
    "2026-07-27",
)
DEFAULT_VALIDATION_DATES = ("2026-07-29", "2026-07-30")


def policy_grid() -> list[dict[str, Any]]:
    policies = []
    for max_rank in (3, 4, 5):
        for minimum_ev in (0.98, 1.00, 1.02):
            for maximum_ev in (1.03, 1.05, 1.08):
                if minimum_ev >= maximum_ev:
                    continue
                for maximum_odds in (20.0, 40.0, None):
                    policies.append({
                        "name": (
                            f"v23_r{max_rank}_ev{minimum_ev:.2f}_"
                            f"{maximum_ev:.2f}_maxodds{maximum_odds or 'none'}"
                        ),
                        "max_model_rank": max_rank,
                        "min_odds": None,
                        "max_odds": maximum_odds,
                        "ev_threshold": minimum_ev,
                        "max_estimated_ev": maximum_ev,
                        "min_model_market_ratio": 0.0,
                        "stake_per_ticket_yen": 100,
                    })
    return policies


def _merge_daily(results: list[dict[str, Any]]) -> dict[str, Any]:
    daily = [row for result in results for row in result["daily"]]
    stake = sum(int(row["stake_yen"]) for row in daily)
    returned = sum(int(row["return_yen"]) for row in daily)
    bootstrap = bootstrap_daily_roi(daily, samples=20_000, seed=20260731)
    leave_one_out_roi = []
    for omitted in range(len(daily)):
        kept = [row for index, row in enumerate(daily) if index != omitted]
        kept_stake = sum(int(row["stake_yen"]) for row in kept)
        kept_return = sum(int(row["return_yen"]) for row in kept)
        if kept_stake:
            leave_one_out_roi.append(kept_return / kept_stake)
    return {
        "days": len(daily),
        "tickets": sum(int(row["tickets"]) for row in daily),
        "stake_yen": stake,
        "return_yen": returned,
        "profit_yen": returned - stake,
        "roi": returned / stake if stake else 0.0,
        "winning_days": sum(int(row["profit_yen"] > 0) for row in daily),
        "worst_day_profit_yen": min((int(row["profit_yen"]) for row in daily), default=0),
        "leave_one_day_out_min_roi": min(leave_one_out_roi, default=0.0),
        "bootstrap_roi_ci95_lower": bootstrap["roi_ci95_lower"],
        "bootstrap_probability_roi_above_one": bootstrap["probability_roi_above_one"],
        "daily": daily,
    }


def _evaluate(
    races_by_date: dict[str, list[dict[str, Any]]],
    calibrators: dict[str, dict[str, float]],
    dates: tuple[str, ...],
    policy: dict[str, Any],
) -> dict[str, Any]:
    results = []
    for race_date in dates:
        results.append(
            simulate_chronological_flat_policy(
                races_by_date.get(race_date, []),
                calibrator=calibrators[race_date],
                policy=policy,
                probability_blender=blend_probabilities,
            )
        )
    return _merge_daily(results)


def _score(metrics: dict[str, Any]) -> tuple[float, float, float, int]:
    if metrics["tickets"] < 100 or metrics["winning_days"] < 3:
        return (-math.inf, -math.inf, -math.inf, 0)
    return (
        float(metrics["bootstrap_roi_ci95_lower"]),
        float(metrics["leave_one_day_out_min_roi"]),
        float(metrics["roi"]),
        -int(metrics["tickets"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select V23 policy on discovery days and report untouched validation days"
    )
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    cache_path = Path(evaluation["scored_cache"])
    cache = joblib.load(cache_path)
    races_by_date: dict[str, list[dict[str, Any]]] = {}
    for race in cache["races"]:
        races_by_date.setdefault(str(race["race_date"]), []).append(race)
    calibrators = {
        str(fold["evaluation_date"]): fold["calibrator"]
        for fold in evaluation["folds"]
    }

    rows = []
    for policy in policy_grid():
        discovery = _evaluate(
            races_by_date, calibrators, DEFAULT_DISCOVERY_DATES, policy
        )
        rows.append({"policy": policy, "discovery": discovery})
    rows.sort(key=lambda row: _score(row["discovery"]), reverse=True)
    selected = rows[0]
    validation = _evaluate(
        races_by_date,
        calibrators,
        DEFAULT_VALIDATION_DATES,
        selected["policy"],
    )
    payload = {
        "status": "diagnostic_only_not_promotion_evidence",
        "selection_boundary": {
            "discovery_dates": list(DEFAULT_DISCOVERY_DATES),
            "validation_dates": list(DEFAULT_VALIDATION_DATES),
            "validation_used_for_selection": False,
        },
        "selected_policy": selected["policy"],
        "discovery": selected["discovery"],
        "validation": validation,
        "top_discovery_candidates": rows[:10],
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
