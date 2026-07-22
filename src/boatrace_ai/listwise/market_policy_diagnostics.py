from __future__ import annotations

from collections import defaultdict
from typing import Any

from .flat_policy import select_flat_policy, simulate_flat_policy
from .market_calibration import (
    blend_probabilities,
    select_policy,
    simulate_policy,
    summarize_flat_candidates,
    summarize_policy_candidates,
)
from .market_residual import fit_log_pool_newton


def forward_policy_diagnostics(
    races: list[dict[str, Any]],
    *,
    regularization: float,
    daily_budget_yen: int = 10_000,
) -> dict[str, Any]:
    """Select betting rules on past days and score them only on the next day."""
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for race in races:
        by_day[str(race["race_date"])].append(race)
    dates = sorted(by_day)
    if len(dates) < 2:
        raise ValueError("at least two dates are required")

    folds = []
    for index in range(1, len(dates)):
        training_dates = dates[:index]
        evaluation_date = dates[index]
        training = [
            race for race_date in training_dates for race in by_day[race_date]
        ]
        holdout = by_day[evaluation_date]
        calibrator = fit_log_pool_newton(
            training, regularization=regularization
        )
        policy, policy_grid = select_policy(
            training,
            calibrator=calibrator,
            daily_budget_yen=daily_budget_yen,
        )
        result = simulate_policy(
            holdout,
            calibrator=calibrator,
            policy=policy,
            daily_budget_yen=daily_budget_yen,
        )
        flat_policy, flat_grid = select_flat_policy(
            training,
            calibrator=calibrator,
            probability_blender=blend_probabilities,
        )
        flat_result = simulate_flat_policy(
            holdout,
            calibrator=calibrator,
            policy=flat_policy,
            probability_blender=blend_probabilities,
        )
        folds.append(
            {
                "training_dates": training_dates,
                "evaluation_date": evaluation_date,
                "training_races": len(training),
                "evaluation_races": len(holdout),
                "calibrator": calibrator,
                "selected_policy": policy,
                "training_policy_diagnostics": summarize_policy_candidates(
                    policy_grid
                ),
                "evaluation": {
                    key: value for key, value in result.items() if key != "daily"
                },
                "evaluation_daily": result["daily"],
                "selected_flat_policy": flat_policy,
                "training_flat_diagnostics": summarize_flat_candidates(flat_grid),
                "flat_evaluation": {
                    key: value
                    for key, value in flat_result.items()
                    if key != "daily"
                },
                "flat_evaluation_daily": flat_result["daily"],
            }
        )

    adaptive_stake = sum(int(fold["evaluation"]["stake_yen"]) for fold in folds)
    adaptive_return = sum(int(fold["evaluation"]["return_yen"]) for fold in folds)
    flat_stake = sum(int(fold["flat_evaluation"]["stake_yen"]) for fold in folds)
    flat_return = sum(int(fold["flat_evaluation"]["return_yen"]) for fold in folds)
    return {
        "validation_design": (
            "Each policy and calibrator is fit on strictly earlier full days and "
            "evaluated on the immediately following untouched day"
        ),
        "regularization": regularization,
        "daily_budget_yen": daily_budget_yen,
        "dates": dates,
        "folds": folds,
        "adaptive": {
            "evaluation_days": len(folds),
            "evaluation_races": sum(
                int(fold["evaluation_races"]) for fold in folds
            ),
            "tickets": sum(int(fold["evaluation"]["tickets"]) for fold in folds),
            "hit_tickets": sum(
                int(fold["evaluation"]["hit_tickets"]) for fold in folds
            ),
            "stake_yen": adaptive_stake,
            "return_yen": adaptive_return,
            "profit_yen": adaptive_return - adaptive_stake,
            "roi": adaptive_return / adaptive_stake if adaptive_stake else 0.0,
            "winning_days": sum(
                int(fold["evaluation"]["profit_yen"] > 0) for fold in folds
            ),
        },
        "flat": {
            "evaluation_days": len(folds),
            "evaluation_races": sum(
                int(fold["evaluation_races"]) for fold in folds
            ),
            "tickets": sum(
                int(fold["flat_evaluation"]["tickets"]) for fold in folds
            ),
            "hit_tickets": sum(
                int(fold["flat_evaluation"]["hit_tickets"]) for fold in folds
            ),
            "stake_yen": flat_stake,
            "return_yen": flat_return,
            "profit_yen": flat_return - flat_stake,
            "roi": flat_return / flat_stake if flat_stake else 0.0,
            "winning_days": sum(
                int(fold["flat_evaluation"]["profit_yen"] > 0) for fold in folds
            ),
        },
    }
