from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib

from .joint_market_value import TRIFECTA_OUTCOMES
from .joint_scenario_model import evaluate_joint_scenario_walk_forward
from .terminal_probability_oof import (
    build_terminal_probability_oof_artifact,
    joint_observations_from_terminal_oof,
)


EVALUATION_VERSION = "joint_scenario_strict_walk_forward_v1"


def _load_scored_races(path: Path) -> list[dict[str, Any]]:
    payload = joblib.load(path)
    if not isinstance(payload, Mapping):
        raise ValueError("scored cache root must be a mapping")
    races = payload.get("races")
    if not isinstance(races, list) or not races:
        raise ValueError("scored cache races must be a non-empty list")
    if not all(isinstance(row, dict) for row in races):
        raise ValueError("every scored cache race must be a mapping")
    return races


def _eligible_terminal_rows(
    races: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible = []
    reasons: Counter[str] = Counter()
    excluded_by_day: Counter[str] = Counter()
    for race in races:
        if "official_closing_odds" not in race:
            reason = "missing_official_closing_odds"
            reasons[reason] += 1
            excluded_by_day[str(race.get("race_date") or "unknown")] += 1
            continue
        eligible.append(race)
    return eligible, {
        "input_races": len(races),
        "eligible_races": len(eligible),
        "excluded_races": len(races) - len(eligible),
        "exclusion_reasons": dict(sorted(reasons.items())),
        "excluded_by_day": dict(sorted(excluded_by_day.items())),
    }


def _with_deltas(metrics: Mapping[str, float]) -> dict[str, float]:
    result = {str(key): float(value) for key, value in metrics.items()}
    result["generated_log_loss_delta_vs_decision_model"] = (
        result["generated_log_loss"] - result["decision_model_log_loss"]
    )
    result["generated_log_loss_delta_vs_decision_market"] = (
        result["generated_log_loss"] - result["decision_market_log_loss"]
    )
    result["generated_brier_delta_vs_decision_model"] = (
        result["generated_brier"] - result["decision_model_brier"]
    )
    result["generated_top5_delta_vs_decision_model"] = (
        result["generated_top5"] - result["decision_model_top5"]
    )
    result["closing_cross_entropy_delta_vs_decision_market"] = (
        result["closing_cross_entropy"]
        - result["decision_market_cross_entropy"]
    )
    result["closing_total_variation_delta_vs_decision_market"] = (
        result["closing_total_variation"]
        - result["decision_market_total_variation"]
    )
    return result


def run_joint_scenario_evaluation(
    scored_cache: Path,
    *,
    terminal_min_training_days: int,
    joint_min_training_days: int,
    scenarios_per_race: int,
    rank: int,
    pooling_strength: float,
    seed: int,
    expected_outcomes: Sequence[str] = TRIFECTA_OUTCOMES,
    learn_residual_scales: bool = False,
) -> dict[str, Any]:
    races = _load_scored_races(scored_cache)
    eligible, coverage = _eligible_terminal_rows(races)
    terminal = build_terminal_probability_oof_artifact(
        eligible,
        minimum_training_days=terminal_min_training_days,
        expected_outcomes=expected_outcomes,
    )
    observations = joint_observations_from_terminal_oof(eligible, terminal)
    actual = {
        str(race["race_id"]): str(race["actual_combination"])
        for race in eligible
    }
    joint = evaluate_joint_scenario_walk_forward(
        observations,
        actual,
        minimum_training_days=joint_min_training_days,
        scenarios_per_race=scenarios_per_race,
        rank=rank,
        pooling_strength=pooling_strength,
        seed=seed,
        learn_residual_scales=learn_residual_scales,
    )
    joint["metrics"] = _with_deltas(joint["metrics"])
    for day in joint["days"]:
        day["metrics"] = _with_deltas(day["metrics"])
    return {
        "model": EVALUATION_VERSION,
        "status": "diagnostic_complete_not_policy_connected",
        "promotion_eligible": False,
        "deployment_eligible": False,
        "scored_cache": str(scored_cache),
        "coverage": coverage,
        "terminal_probability_oof": {
            "version": terminal["version"],
            "artifact_contract_sha256": terminal["artifact_contract_sha256"],
            "predicted_races": terminal["predicted_races"],
            "prediction_dates": terminal["prediction_dates"],
            "strict_oof_metrics": terminal["strict_oof_metrics"],
            "deployment_eligible": terminal["deployment_eligible"],
        },
        "joint_walk_forward": joint,
        "metrics": {
            "evaluated_days": joint["evaluated_days"],
            "evaluated_races": joint["evaluated_races"],
            "evaluation_from": joint["evaluation_from"],
            "evaluation_through": joint["evaluation_through"],
            **joint["metrics"],
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--terminal-min-training-days", type=int, default=5)
    parser.add_argument("--joint-min-training-days", type=int, default=3)
    parser.add_argument("--scenarios-per-race", type=int, default=64)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--pooling-strength", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=33036)
    parser.add_argument("--learn-residual-scales", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_joint_scenario_evaluation(
        args.scored_cache,
        terminal_min_training_days=args.terminal_min_training_days,
        joint_min_training_days=args.joint_min_training_days,
        scenarios_per_race=args.scenarios_per_race,
        rank=args.rank,
        pooling_strength=args.pooling_strength,
        seed=args.seed,
        learn_residual_scales=args.learn_residual_scales,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(result["metrics"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
