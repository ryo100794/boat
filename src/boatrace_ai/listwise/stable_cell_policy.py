from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping

from .market_edge_diagnostics import (
    ODDS_BANDS,
    RANK_GROUPS,
    STAKE_YEN,
    _bounded_group,
    summarize_edge_stability_grid,
)


DEFAULT_STABILITY_THRESHOLDS: dict[str, float] = {
    "minimum_days": 5,
    "minimum_tickets": 200,
    "minimum_hit_days": 4,
    "minimum_expected_hits": 10.0,
    "maximum_mean_daily_no_hit_probability": 0.35,
    "minimum_profitable_day_fraction": 0.50,
    "minimum_roi_without_largest_hit": 1.0,
    "maximum_hit_return_hhi": 0.15,
}


def _cell_key(record: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        _bounded_group(float(record["probability_rank"]), RANK_GROUPS),
        _bounded_group(float(record["forecast_odds"]), ODDS_BANDS),
        str(record["ev_bin"]),
    )


def _eligible(cell: Mapping[str, Any], thresholds: Mapping[str, float]) -> bool:
    hhi = cell.get("hit_return_hhi")
    robust_roi = cell.get("roi_without_largest_hit")
    no_hit = cell.get("mean_daily_no_hit_probability")
    profitable = cell.get("profitable_day_fraction")
    return bool(
        int(cell["days"]) >= int(thresholds["minimum_days"])
        and int(cell["tickets"]) >= int(thresholds["minimum_tickets"])
        and int(cell["hit_days"]) >= int(thresholds["minimum_hit_days"])
        and float(cell["expected_hits"])
        >= float(thresholds["minimum_expected_hits"])
        and no_hit is not None
        and float(no_hit)
        <= float(thresholds["maximum_mean_daily_no_hit_probability"])
        and profitable is not None
        and float(profitable)
        >= float(thresholds["minimum_profitable_day_fraction"])
        and robust_roi is not None
        and float(robust_roi)
        > float(thresholds["minimum_roi_without_largest_hit"])
        and hhi is not None
        and float(hhi) <= float(thresholds["maximum_hit_return_hhi"])
    )


def _stability_score(cell: Mapping[str, Any]) -> float:
    robust_margin = max(float(cell["roi_without_largest_hit"]) - 1.0, 0.0)
    expected_hits = max(float(cell["expected_hits"]), 0.0)
    no_hit = min(max(float(cell["mean_daily_no_hit_probability"]), 0.0), 1.0)
    hhi = min(max(float(cell["hit_return_hhi"]), 0.0), 1.0)
    return robust_margin * math.sqrt(expected_hits) * (1.0 - no_hit) * (1.0 - hhi)


def select_prior_stable_cells(
    prior_records: list[dict[str, Any]],
    *,
    thresholds: Mapping[str, float] | None = None,
    maximum_cells: int = 1,
) -> list[dict[str, Any]]:
    if maximum_cells < 1:
        raise ValueError("maximum_cells must be positive")
    limits = dict(DEFAULT_STABILITY_THRESHOLDS)
    if thresholds:
        limits.update({key: float(value) for key, value in thresholds.items()})
    unknown = set(limits) - set(DEFAULT_STABILITY_THRESHOLDS)
    if unknown:
        raise ValueError(f"unknown stability thresholds: {sorted(unknown)}")
    cells = summarize_edge_stability_grid(prior_records)["cells"]
    eligible = []
    for cell in cells:
        if not _eligible(cell, limits):
            continue
        compact = {key: value for key, value in cell.items() if key != "daily"}
        compact["stability_score"] = _stability_score(cell)
        compact["cell_key"] = [
            str(cell["rank_group"]),
            str(cell["odds_band"]),
            str(cell["ev_bin"]),
        ]
        eligible.append(compact)
    return sorted(
        eligible,
        key=lambda cell: (
            -float(cell["stability_score"]),
            -int(cell["tickets"]),
            tuple(cell["cell_key"]),
        ),
    )[:maximum_cells]


def evaluate_walk_forward_stable_cells(
    records: list[dict[str, Any]],
    *,
    daily_budget_yen: int = 10_000,
    maximum_cells: int = 1,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    if daily_budget_yen < STAKE_YEN:
        raise ValueError("daily_budget_yen must fund at least one ticket")
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_day[str(record["race_date"])].append(record)

    prior_records: list[dict[str, Any]] = []
    daily = []
    hit_returns: list[int] = []
    ticket_limit = daily_budget_yen // STAKE_YEN
    for race_date in sorted(by_day):
        selected_cells = select_prior_stable_cells(
            prior_records,
            thresholds=thresholds,
            maximum_cells=maximum_cells,
        )
        selected_keys = {tuple(cell["cell_key"]) for cell in selected_cells}
        candidates = [
            record
            for record in by_day[race_date]
            if _cell_key(record) in selected_keys
        ]
        candidates.sort(
            key=lambda record: (
                -float(record["expected_value"]),
                -float(record["probability"]),
                float(record["forecast_odds"]),
                str(record["race_id"]),
                str(record["combination"]),
            )
        )
        purchased = candidates[:ticket_limit]
        stake_yen = len(purchased) * STAKE_YEN
        return_yen = sum(int(record["return_yen"]) for record in purchased)
        day_hit_returns = [
            int(record["return_yen"]) for record in purchased if record["hit"]
        ]
        hit_returns.extend(day_hit_returns)
        daily.append(
            {
                "race_date": race_date,
                "prior_days": len({str(row["race_date"]) for row in prior_records}),
                "selected_cells": selected_cells,
                "candidate_tickets": len(candidates),
                "tickets": len(purchased),
                "hits": len(day_hit_returns),
                "stake_yen": stake_yen,
                "return_yen": return_yen,
                "profit_yen": return_yen - stake_yen,
                "roi": return_yen / stake_yen if stake_yen else None,
            }
        )
        prior_records.extend(by_day[race_date])

    stake_yen = sum(int(row["stake_yen"]) for row in daily)
    return_yen = sum(int(row["return_yen"]) for row in daily)
    largest_hit = max(hit_returns, default=0)
    hhi_denominator = return_yen * return_yen
    limits = dict(DEFAULT_STABILITY_THRESHOLDS)
    if thresholds:
        limits.update({key: float(value) for key, value in thresholds.items()})
    return {
        "comparison_role": "strict_prior_stable_cell_policy_exploratory_diagnostic",
        "validation_design": (
            "Cell eligibility for each date uses only records from earlier dates; "
            "current-date outcomes are appended after purchase simulation."
        ),
        "promotion_eligible": False,
        "promotion_exclusion_reason": (
            "thresholds were designed after inspecting the current outer holdout"
        ),
        "daily_budget_yen": daily_budget_yen,
        "maximum_cells": maximum_cells,
        "thresholds": limits,
        "evaluation_days": len(daily),
        "days_with_bets": sum(int(row["tickets"] > 0) for row in daily),
        "profitable_days": sum(int(row["profit_yen"] > 0) for row in daily),
        "tickets": sum(int(row["tickets"]) for row in daily),
        "hits": len(hit_returns),
        "stake_yen": stake_yen,
        "return_yen": return_yen,
        "profit_yen": return_yen - stake_yen,
        "roi": return_yen / stake_yen if stake_yen else None,
        "largest_hit_return_yen": largest_hit,
        "roi_without_largest_hit": (
            (return_yen - largest_hit) / stake_yen if stake_yen else None
        ),
        "hit_return_hhi": (
            sum(value * value for value in hit_returns) / hhi_denominator
            if hhi_denominator
            else None
        ),
        "daily": daily,
    }
