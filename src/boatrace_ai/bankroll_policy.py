from __future__ import annotations

import math
from typing import Any, Callable


NO_BET_POLICY = {"name": "no_bet", "no_bet": True}


def race_confidence(lane_probabilities: dict[int, float]) -> dict[str, float]:
    probabilities = sorted(
        (max(0.0, float(value)) for value in lane_probabilities.values()),
        reverse=True,
    )
    total = sum(probabilities)
    if total <= 0.0:
        return {
            "race_top_lane_probability": 0.0,
            "race_top_lane_margin": 0.0,
            "race_normalized_entropy": 1.0,
        }
    normalized = [value / total for value in probabilities]
    entropy = -sum(value * math.log(value) for value in normalized if value > 0.0)
    entropy_scale = math.log(max(2, len(normalized)))
    return {
        "race_top_lane_probability": normalized[0],
        "race_top_lane_margin": (
            normalized[0] - normalized[1] if len(normalized) > 1 else normalized[0]
        ),
        "race_normalized_entropy": entropy / entropy_scale,
    }


def filter_candidates(
    candidates: list[dict[str, Any]],
    policy: dict[str, Any],
) -> list[dict[str, Any]]:
    if policy.get("no_bet"):
        return []
    result = []
    for item in candidates:
        if not _at_least(item, "probability", policy.get("min_ticket_probability")):
            continue
        if not _at_most(item, "estimated_odds", policy.get("max_estimated_odds")):
            continue
        if not _at_least(
            item,
            "race_top_lane_probability",
            policy.get("min_race_top_lane_probability"),
        ):
            continue
        if not _at_least(
            item,
            "race_top_lane_margin",
            policy.get("min_race_top_lane_margin"),
        ):
            continue
        if not _at_most(
            item,
            "race_normalized_entropy",
            policy.get("max_race_normalized_entropy"),
        ):
            continue
        result.append(item)
    return result


def default_policy_grid() -> list[dict[str, Any]]:
    return [
        {"name": "baseline"},
        {"name": "ticket_p010", "min_ticket_probability": 0.010},
        {"name": "ticket_p015", "min_ticket_probability": 0.015},
        {"name": "ticket_p020", "min_ticket_probability": 0.020},
        {"name": "odds_cap_80", "max_estimated_odds": 80.0},
        {"name": "odds_cap_60", "max_estimated_odds": 60.0},
        {"name": "odds_cap_40", "max_estimated_odds": 40.0},
        {"name": "top_p028", "min_race_top_lane_probability": 0.28},
        {"name": "top_p032", "min_race_top_lane_probability": 0.32},
        {"name": "top_p036", "min_race_top_lane_probability": 0.36},
        {"name": "margin_004", "min_race_top_lane_margin": 0.04},
        {"name": "margin_008", "min_race_top_lane_margin": 0.08},
        {"name": "entropy_095", "max_race_normalized_entropy": 0.95},
        {"name": "entropy_090", "max_race_normalized_entropy": 0.90},
        {
            "name": "p010_odds60",
            "min_ticket_probability": 0.010,
            "max_estimated_odds": 60.0,
        },
        {
            "name": "top032_odds60",
            "min_race_top_lane_probability": 0.32,
            "max_estimated_odds": 60.0,
        },
        {
            "name": "entropy095_odds60",
            "max_race_normalized_entropy": 0.95,
            "max_estimated_odds": 60.0,
        },
        NO_BET_POLICY,
    ]


def split_calibration_dates(
    dates: set[str],
    *,
    calibration_fraction: float,
) -> tuple[list[str], list[str]]:
    ordered = sorted(dates)
    if len(ordered) < 2:
        return [], ordered
    count = max(1, min(len(ordered) - 1, math.ceil(len(ordered) * calibration_fraction)))
    return ordered[:count], ordered[count:]


def select_temporal_policy(
    calibration_dates: list[str],
    candidates_by_day: dict[str, list[dict[str, Any]]],
    evaluated_by_day: dict[str, set[str]],
    *,
    allocate_day: Callable[..., dict[str, Any]],
    allocation_kwargs: dict[str, Any],
    policies: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evaluations = []
    for policy in policies or default_policy_grid():
        stake_yen = 0
        return_yen = 0
        tickets = 0
        cumulative_profit = 0
        peak_profit = 0
        max_drawdown_yen = 0
        for race_date in calibration_dates:
            filtered = filter_candidates(candidates_by_day.get(race_date, []), policy)
            row = allocate_day(
                race_date,
                filtered,
                evaluated_by_day.get(race_date, set()),
                **allocation_kwargs,
            )
            stake_yen += int(row["stake_yen"])
            return_yen += int(row["return_yen"])
            tickets += int(row["tickets"])
            cumulative_profit += int(row["profit_yen"])
            peak_profit = max(peak_profit, cumulative_profit)
            max_drawdown_yen = max(max_drawdown_yen, peak_profit - cumulative_profit)
        profit_yen = return_yen - stake_yen
        evaluations.append(
            {
                "policy": dict(policy),
                "calibration_days": len(calibration_dates),
                "tickets": tickets,
                "stake_yen": stake_yen,
                "return_yen": return_yen,
                "profit_yen": profit_yen,
                "roi": return_yen / stake_yen if stake_yen else 0.0,
                "max_drawdown_yen": max_drawdown_yen,
            }
        )
    best = max(
        evaluations,
        key=lambda row: (
            int(row["profit_yen"]),
            -int(row["max_drawdown_yen"]),
            -int(row["stake_yen"]),
        ),
    )
    return dict(best["policy"]), evaluations


def _at_least(item: dict[str, Any], key: str, threshold: Any) -> bool:
    if threshold is None:
        return True
    return float(item.get(key, 0.0)) >= float(threshold)


def _at_most(item: dict[str, Any], key: str, threshold: Any) -> bool:
    if threshold is None:
        return True
    return float(item.get(key, math.inf)) <= float(threshold)
