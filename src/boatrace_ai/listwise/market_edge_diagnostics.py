from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

from .closing_odds import decision_odds


STAKE_YEN = 100
EV_BINS = (
    ("lt_0.80", None, 0.80),
    ("0.80_0.90", 0.80, 0.90),
    ("0.90_1.00", 0.90, 1.00),
    ("1.00_1.05", 1.00, 1.05),
    ("1.05_1.10", 1.05, 1.10),
    ("1.10_1.20", 1.10, 1.20),
    ("gte_1.20", 1.20, None),
)


def _bin_name(expected_value: float) -> str:
    for name, lower, upper in EV_BINS:
        if (lower is None or expected_value >= lower) and (
            upper is None or expected_value < upper
        ):
            return name
    raise ValueError(f"unhandled expected value: {expected_value}")


def edge_records(
    races: list[dict[str, Any]],
    *,
    calibrator: dict[str, float],
    probability_blender: Callable[..., dict[str, float]],
) -> list[dict[str, Any]]:
    records = []
    for race in races:
        probabilities = probability_blender(
            race["model_probabilities"],
            race["market_probabilities"],
            model_weight=float(calibrator["model_weight"]),
            temperature=float(calibrator["temperature"]),
        )
        odds = decision_odds(race)
        ranked = sorted(probabilities, key=probabilities.get, reverse=True)
        ranks = {combination: index + 1 for index, combination in enumerate(ranked)}
        actual = str(race["actual_combination"])
        for combination, probability in probabilities.items():
            price = float(odds[combination])
            expected_value = float(probability) * price
            records.append(
                {
                    "race_date": str(race["race_date"]),
                    "race_id": str(race["race_id"]),
                    "combination": combination,
                    "probability_rank": ranks[combination],
                    "probability": float(probability),
                    "forecast_odds": price,
                    "expected_value": expected_value,
                    "ev_bin": _bin_name(expected_value),
                    "hit": combination == actual,
                    "return_yen": (
                        int(race["actual_payout_yen"])
                        if combination == actual
                        else 0
                    ),
                }
            )
    return records


def _summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["ev_bin"])].append(record)
    rows = []
    for name, lower, upper in EV_BINS:
        values = grouped.get(name, [])
        tickets = len(values)
        stake_yen = tickets * STAKE_YEN
        return_yen = sum(int(value["return_yen"]) for value in values)
        rows.append(
            {
                "bin": name,
                "lower": lower,
                "upper": upper,
                "tickets": tickets,
                "races": len({str(value["race_id"]) for value in values}),
                "hits": sum(int(bool(value["hit"])) for value in values),
                "mean_predicted_ev": (
                    sum(float(value["expected_value"]) for value in values) / tickets
                    if tickets
                    else None
                ),
                "stake_yen": stake_yen,
                "return_yen": return_yen,
                "profit_yen": return_yen - stake_yen,
                "realized_roi": return_yen / stake_yen if stake_yen else None,
            }
        )
    return rows


def summarize_edge_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    dates = sorted({str(record["race_date"]) for record in records})
    return {
        "comparison_role": (
            "untouched-fold fixed-100-yen calibration diagnostic; not policy selection"
        ),
        "evaluation_days": len(dates),
        "evaluation_races": len({str(record["race_id"]) for record in records}),
        "all_tickets": _summarize(records),
        "top5_tickets": _summarize(
            [record for record in records if int(record["probability_rank"]) <= 5]
        ),
    }



RANK_GROUPS = (("top5", 1, 6), ("6-20", 6, 21), ("21+", 21, None))
ODDS_BANDS = (
    ("lt_20", None, 20.0),
    ("20_50", 20.0, 50.0),
    ("50_101", 50.0, 101.0),
    ("gte_101", 101.0, None),
)


def _bounded_group(
    value: float,
    groups: tuple[tuple[str, float | None, float | None], ...],
) -> str:
    for name, lower, upper in groups:
        if (lower is None or value >= lower) and (upper is None or value < upper):
            return name
    raise ValueError(f"unhandled grouped value: {value}")


def _daily_no_hit_probability(values: list[dict[str, Any]]) -> float:
    probability_by_race: dict[str, float] = defaultdict(float)
    for value in values:
        probability_by_race[str(value["race_id"])] += float(value["probability"])
    result = 1.0
    for probability in probability_by_race.values():
        result *= 1.0 - min(max(probability, 0.0), 1.0)
    return result


def summarize_edge_stability_grid(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe retrospective rank, odds, and EV cells without selecting a policy."""
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        rank_group = _bounded_group(
            float(record["probability_rank"]), RANK_GROUPS
        )
        odds_band = _bounded_group(
            float(record["forecast_odds"]), ODDS_BANDS
        )
        grouped[(rank_group, odds_band, str(record["ev_bin"]))].append(record)

    rows = []
    for (rank_group, odds_band, ev_bin), values in sorted(grouped.items()):
        tickets = len(values)
        stake_yen = tickets * STAKE_YEN
        return_yen = sum(int(value["return_yen"]) for value in values)
        hit_returns = [
            int(value["return_yen"]) for value in values if value["hit"]
        ]
        largest_hit = max(hit_returns, default=0)
        return_square_sum = sum(value * value for value in hit_returns)
        by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for value in values:
            by_day[str(value["race_date"])].append(value)
        daily_no_hit = [_daily_no_hit_probability(day) for day in by_day.values()]
        winning_days = sum(
            int(sum(int(row["return_yen"]) for row in day) > len(day) * STAKE_YEN)
            for day in by_day.values()
        )
        rows.append(
            {
                "rank_group": rank_group,
                "odds_band": odds_band,
                "ev_bin": ev_bin,
                "days": len(by_day),
                "tickets": tickets,
                "races": len({str(value["race_id"]) for value in values}),
                "hits": len(hit_returns),
                "hit_days": len(
                    {str(value["race_date"]) for value in values if value["hit"]}
                ),
                "expected_hits": sum(
                    float(value["probability"]) for value in values
                ),
                "mean_daily_no_hit_probability": (
                    sum(daily_no_hit) / len(daily_no_hit) if daily_no_hit else None
                ),
                "winning_days": winning_days,
                "profitable_day_fraction": (
                    winning_days / len(by_day) if by_day else None
                ),
                "stake_yen": stake_yen,
                "return_yen": return_yen,
                "profit_yen": return_yen - stake_yen,
                "realized_roi": return_yen / stake_yen if stake_yen else None,
                "largest_hit_return_yen": largest_hit,
                "roi_without_largest_hit": (
                    (return_yen - largest_hit) / stake_yen if stake_yen else None
                ),
                "hit_return_hhi": (
                    return_square_sum / return_yen**2 if return_yen > 0 else None
                ),
            }
        )
    return {
        "comparison_role": "retrospective_outer_holdout_diagnostic_not_policy_selection",
        "dimensions": ["probability_rank", "forecast_odds", "predicted_ev"],
        "cells": rows,
    }


def walk_forward_edge_diagnostics(
    races: list[dict[str, Any]],
    *,
    min_calibration_days: int = 1,
    forecast_closing: bool = True,
) -> dict[str, Any]:
    from .closing_odds import attach_forecast_closing_odds, fit_closing_odds_model
    from .market_calibration import blend_probabilities
    from .market_residual import (
        fit_fixed_regularization,
        select_regularization_prequential,
    )

    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for race in races:
        by_day[str(race["race_date"])].append(race)
    dates = sorted(by_day)
    records = []
    folds = []
    for index in range(min_calibration_days, len(dates)):
        training_dates = dates[:index]
        training = [race for day in training_dates for race in by_day[day]]
        selection = (
            select_regularization_prequential(training)
            if len(training_dates) >= 2
            else fit_fixed_regularization(training)
        )
        calibrator = dict(selection["final_calibrator"])
        closing_model = fit_closing_odds_model(training)
        raw_holdout = by_day[dates[index]]
        holdout = (
            attach_forecast_closing_odds(raw_holdout, closing_model)
            if forecast_closing
            else raw_holdout
        )
        fold_records = edge_records(
            holdout,
            calibrator=calibrator,
            probability_blender=blend_probabilities,
        )
        records.extend(fold_records)
        folds.append(
            {
                "training_dates": training_dates,
                "evaluation_date": dates[index],
                "training_races": len(training),
                "evaluation_races": len(holdout),
                "calibrator": calibrator,
                "closing_odds_model": closing_model,
            }
        )
    return {
        **summarize_edge_records(records),
        "price_basis": "forecast_closing" if forecast_closing else "real_t5",
        "folds": folds,
    }
