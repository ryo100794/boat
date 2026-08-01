from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import joblib

from ..bankroll_bootstrap import bootstrap_daily_roi
from ..chronological_bankroll import summarize_chronological_bankroll_days
from .empirical_policy_replay import _reconstruct_policy_races
from .market_calibration import bankroll_reliability_metrics, simulate_policy


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_scored_cache(path: Path) -> dict[str, Any]:
    value = joblib.load(path)
    if not isinstance(value, dict) or not isinstance(value.get("races"), list):
        raise ValueError("scored cache must contain a races list")
    return value


def _validate_fixed_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "ev_threshold",
        "max_tickets_per_race",
        "min_model_market_ratio",
        "staking_mode",
    )
    missing = [key for key in required if key not in policy]
    if missing:
        raise ValueError(f"fixed policy is missing {missing[0]}")
    control = policy.get("v18_ticket_control")
    if not isinstance(control, Mapping):
        raise ValueError("fixed policy requires v18_ticket_control")
    if control.get("result_or_payout_fields_used") is not False:
        raise ValueError("fixed policy must exclude result and payout fields")
    if int(control.get("learned_daily_ticket_limit") or 0) <= 0:
        raise ValueError("fixed policy daily ticket limit must be positive")
    if control.get("schedule_quota_rounding") not in {"floor", "ceil"}:
        raise ValueError("fixed policy schedule quota rounding is invalid")
    return copy.deepcopy(dict(policy))


def replay_fixed_policy(
    evaluation_result: Mapping[str, Any],
    scored_cache: Mapping[str, Any],
    fixed_policy: Mapping[str, Any],
    *,
    daily_budget_yen: int = 10_000,
) -> dict[str, Any]:
    if daily_budget_yen <= 0:
        raise ValueError("daily_budget_yen must be positive")
    policy = _validate_fixed_policy(fixed_policy)
    races = scored_cache.get("races")
    if not isinstance(races, list) or not all(isinstance(row, dict) for row in races):
        raise ValueError("scored cache must contain object races")
    folds = evaluation_result.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError("evaluation result must contain folds")

    by_day: dict[str, list[dict[str, Any]]] = {}
    for race in races:
        by_day.setdefault(str(race["race_date"]), []).append(race)

    daily_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for fold_number, fold in enumerate(folds, start=1):
        if not isinstance(fold, dict):
            raise ValueError("evaluation folds must be objects")
        evaluation_date = str(fold.get("evaluation_date") or "")
        holdout = by_day.get(evaluation_date, [])
        if len(holdout) != int(fold.get("evaluation_races") or 0):
            raise ValueError(f"{evaluation_date}: holdout race count mismatch")
        calibrator = fold.get("purchase_calibrator")
        if not isinstance(calibrator, dict):
            raise ValueError(f"{evaluation_date}: purchase calibrator is missing")
        policy_races = _reconstruct_policy_races(races, holdout, fold)
        simulated = simulate_policy(
            policy_races,
            calibrator=calibrator,
            policy=policy,
            daily_budget_yen=daily_budget_yen,
            include_chronological=True,
        )
        chronological = simulated["chronological_bankroll"]
        fold_daily = chronological.get("daily") or []
        if len(fold_daily) != 1 or fold_daily[0].get("race_date") != evaluation_date:
            raise ValueError(f"{evaluation_date}: chronological daily result mismatch")
        daily_rows.extend(fold_daily)
        fold_rows.append(
            {
                "fold": fold_number,
                "evaluation_date": evaluation_date,
                "calibration_dates": list(fold.get("calibration_dates") or []),
                "evaluation_races": len(policy_races),
                "operational_model_training_races": int(
                    (fold.get("operational_model") or {}).get("training_races") or 0
                ),
                "closing_odds_model_trained_through_date": fold.get(
                    "closing_odds_model_trained_through_date"
                ),
                "bankroll": {
                    key: value
                    for key, value in chronological.items()
                    if key != "daily"
                },
            }
        )

    aggregate = summarize_chronological_bankroll_days(daily_rows)
    reliability = bankroll_reliability_metrics(
        daily_rows, evaluated_races=int(aggregate["evaluated_races"])
    )
    bootstrap = bootstrap_daily_roi(daily_rows)
    aggregate.update(
        {
            **reliability,
            "profitable_day_fraction": (
                aggregate["winning_days"] / aggregate["race_days"]
                if aggregate["race_days"]
                else None
            ),
            "normalized_drawdown": (
                aggregate["max_drawdown_yen"] / aggregate["stake_yen"]
                if aggregate["stake_yen"]
                else None
            ),
            "daily_cluster_bootstrap_roi_lower_95": bootstrap["roi_ci95_lower"],
            "bootstrap_probability_roi_above_one": bootstrap[
                "probability_roi_above_one"
            ],
        }
    )
    return {
        "comparison_role": "fixed_policy_strict_prior_fold_replay",
        "evaluation_model": evaluation_result.get("model"),
        "evaluation_calibrator_strategy": evaluation_result.get(
            "calibrator_strategy"
        ),
        "fixed_policy": policy,
        "daily_budget_yen": daily_budget_yen,
        "information_boundary": {
            "fold_model_and_calibrator": "strict_prior_from_evaluation_result",
            "closing_odds_policy": "strict_prior_reconstructed_per_fold",
            "fixed_policy_result_or_payout_fields_used": False,
            "outer_holdout_used_to_fit_or_select_policy": False,
        },
        "folds": fold_rows,
        "chronological_bankroll": aggregate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay one frozen purchase policy across strict-prior folds."
    )
    parser.add_argument("--evaluation-result", type=Path, required=True)
    parser.add_argument("--policy-source-result", type=Path, required=True)
    parser.add_argument("--scored-cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--daily-budget-yen", type=int, default=10_000)
    args = parser.parse_args()

    evaluation = _load_json(args.evaluation_result)
    policy_source = _load_json(args.policy_source_result)
    deployment = policy_source.get("deployment_configuration")
    if not isinstance(deployment, dict):
        raise ValueError("policy source deployment configuration is missing")
    candidate_policy = deployment.get("candidate_policy")
    if not isinstance(candidate_policy, dict):
        raise ValueError("policy source candidate policy is missing")
    cache_path = args.scored_cache or Path(str(evaluation.get("scored_cache") or ""))
    if not str(cache_path):
        raise ValueError("scored cache path is missing")
    result = replay_fixed_policy(
        evaluation,
        _load_scored_cache(cache_path),
        candidate_policy,
        daily_budget_yen=args.daily_budget_yen,
    )
    result.update(
        {
            "evaluation_result": str(args.evaluation_result),
            "evaluation_result_sha256": hashlib.sha256(
                args.evaluation_result.read_bytes()
            ).hexdigest(),
            "policy_source_result": str(args.policy_source_result),
            "policy_source_result_sha256": hashlib.sha256(
                args.policy_source_result.read_bytes()
            ).hexdigest(),
            "scored_cache": str(cache_path),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
