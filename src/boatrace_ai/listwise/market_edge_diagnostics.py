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
