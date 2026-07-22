from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable, Iterable


STAKE_YEN = 100


def default_flat_policy_grid() -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = [{"name": "no_bet", "no_bet": True}]
    for max_rank in (1, 2, 3, 5):
        for min_odds in (None, 5.0, 10.0):
            for max_odds in (20.0, 40.0, 80.0, None):
                if min_odds is not None and max_odds is not None and min_odds >= max_odds:
                    continue
                for ev_threshold in (0.80, 0.90, 1.00, 1.05, 1.10, 1.20):
                    for min_ratio in (1.00, 1.05, 1.10):
                        policies.append(
                            {
                                "name": (
                                    f"flat_r{max_rank}_min{_label(min_odds)}_"
                                    f"max{_label(max_odds)}_ev{ev_threshold:.2f}_"
                                    f"ratio{min_ratio:.2f}"
                                ),
                                "max_model_rank": max_rank,
                                "min_odds": min_odds,
                                "max_odds": max_odds,
                                "ev_threshold": ev_threshold,
                                "min_model_market_ratio": min_ratio,
                                "stake_per_ticket_yen": STAKE_YEN,
                            }
                        )
    return policies


def _label(value: float | None) -> str:
    return "none" if value is None else str(int(value))


def simulate_flat_policy(
    races: list[dict[str, Any]],
    *,
    calibrator: dict[str, float],
    policy: dict[str, Any],
    probability_blender: Callable[..., dict[str, float]],
) -> dict[str, Any]:
    by_day: dict[str, dict[str, int]] = defaultdict(
        lambda: {"evaluated_races": 0, "tickets": 0, "hits": 0, "stake_yen": 0, "return_yen": 0}
    )
    for race in races:
        day = by_day[str(race["race_date"])]
        day["evaluated_races"] += 1
        if policy.get("no_bet"):
            continue
        probabilities = probability_blender(
            race["model_probabilities"],
            race["market_probabilities"],
            model_weight=float(calibrator["model_weight"]),
            temperature=float(calibrator["temperature"]),
        )
        ranked = sorted(probabilities, key=probabilities.get, reverse=True)
        selected = []
        for combination in ranked[: int(policy["max_model_rank"])]:
            probability = float(probabilities[combination])
            market_probability = float(race["market_probabilities"][combination])
            odds = float(race["odds"][combination])
            if policy.get("min_odds") is not None and odds < float(policy["min_odds"]):
                continue
            if policy.get("max_odds") is not None and odds > float(policy["max_odds"]):
                continue
            if probability * odds < float(policy["ev_threshold"]):
                continue
            if probability / max(1e-15, market_probability) < float(
                policy["min_model_market_ratio"]
            ):
                continue
            selected.append(combination)
        day["tickets"] += len(selected)
        day["stake_yen"] += len(selected) * STAKE_YEN
        actual = str(race["actual_combination"])
        if actual in selected:
            day["hits"] += 1
            day["return_yen"] += int(race["actual_payout_yen"])

    daily = []
    cumulative_profit = peak_profit = max_drawdown = 0
    for race_date in sorted(by_day):
        row = {"race_date": race_date, **by_day[race_date]}
        row["profit_yen"] = row["return_yen"] - row["stake_yen"]
        row["roi"] = (
            row["return_yen"] / row["stake_yen"] if row["stake_yen"] else None
        )
        cumulative_profit += row["profit_yen"]
        peak_profit = max(peak_profit, cumulative_profit)
        max_drawdown = max(max_drawdown, peak_profit - cumulative_profit)
        row["cumulative_profit_yen"] = cumulative_profit
        daily.append(row)
    stake = sum(int(row["stake_yen"]) for row in daily)
    returned = sum(int(row["return_yen"]) for row in daily)
    return {
        "evaluated_races": len(races),
        "evaluation_days": len(daily),
        "tickets": sum(int(row["tickets"]) for row in daily),
        "hit_tickets": sum(int(row["hits"]) for row in daily),
        "stake_yen": stake,
        "return_yen": returned,
        "profit_yen": returned - stake,
        "roi": returned / stake if stake else 0.0,
        "winning_days": sum(int(row["profit_yen"] > 0) for row in daily),
        "max_drawdown_yen": max_drawdown,
        "daily": daily,
    }


def select_flat_policy(
    races: list[dict[str, Any]],
    *,
    calibrator: dict[str, float],
    probability_blender: Callable[..., dict[str, float]],
    policies: Iterable[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    minimum_tickets = max(50, math.ceil(len(races) * 0.20))
    rows = []
    for policy in policies or default_flat_policy_grid():
        result = simulate_flat_policy(
            races,
            calibrator=calibrator,
            policy=policy,
            probability_blender=probability_blender,
        )
        minimum_winning_days = math.ceil(max(1, result["evaluation_days"]) * 0.60)
        eligible = bool(
            policy.get("no_bet")
            or (
                result["tickets"] >= minimum_tickets
                and result["profit_yen"] > 0
                and result["roi"] >= 1.05
                and result["winning_days"] >= minimum_winning_days
                and result["max_drawdown_yen"] <= result["stake_yen"] * 0.50
            )
        )
        rows.append(
            {
                "policy": dict(policy),
                "eligible": eligible,
                **{key: value for key, value in result.items() if key != "daily"},
            }
        )
    eligible_rows = [row for row in rows if row["eligible"]]
    selected = max(
        eligible_rows,
        key=lambda row: (
            int(row["profit_yen"]) - int(row["max_drawdown_yen"]),
            float(row["roi"]),
            int(row["tickets"]),
        ),
    )
    return dict(selected["policy"]), rows


def summarize_flat_candidates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in rows if not row["policy"].get("no_bet")]
    funded = [row for row in candidates if int(row["tickets"]) > 0]

    def compact(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "policy": row["policy"],
            "eligible": bool(row["eligible"]),
            "tickets": int(row["tickets"]),
            "hit_tickets": int(row["hit_tickets"]),
            "stake_yen": int(row["stake_yen"]),
            "return_yen": int(row["return_yen"]),
            "profit_yen": int(row["profit_yen"]),
            "roi": float(row["roi"]),
            "winning_days": int(row["winning_days"]),
            "max_drawdown_yen": int(row["max_drawdown_yen"]),
        }

    return {
        "candidate_count": len(candidates),
        "funded_candidate_count": len(funded),
        "eligible_candidate_count": sum(bool(row["eligible"]) for row in candidates),
        "minimum_ticket_rule": "max(50, ceil(calibration_races * 0.20))",
        "best_profit": compact(
            max(funded, key=lambda row: (row["profit_yen"], row["roi"]), default=None)
        ),
        "best_roi_with_50_tickets": compact(
            max(
                (row for row in funded if int(row["tickets"]) >= 50),
                key=lambda row: (row["roi"], row["profit_yen"]),
                default=None,
            )
        ),
    }
