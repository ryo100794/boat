from __future__ import annotations

from collections import defaultdict
from typing import Any

from .market_calibration import BLEND_WEIGHTS, TEMPERATURES, probability_metrics


def calibrator_stability_rows(races: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Measure each calibrator against the market on every available day."""
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for race in races:
        by_day[str(race["race_date"])].append(race)
    if not by_day:
        return []

    rows = []
    for model_weight in BLEND_WEIGHTS:
        for temperature in TEMPERATURES:
            calibrator = {
                "model_weight": float(model_weight),
                "temperature": float(temperature),
            }
            daily = []
            weighted_loss = 0.0
            total_races = 0
            for race_date in sorted(by_day):
                metrics = probability_metrics(
                    by_day[race_date], calibrator=calibrator
                )
                count = int(metrics["evaluated_races"])
                calibrated_loss = float(metrics["calibrated_trifecta_log_loss"])
                market_loss = float(metrics["market_trifecta_log_loss"])
                weighted_loss += calibrated_loss * count
                total_races += count
                daily.append(
                    {
                        "race_date": race_date,
                        "races": count,
                        "calibrated_log_loss": calibrated_loss,
                        "market_log_loss": market_loss,
                        "market_regret": calibrated_loss - market_loss,
                        "calibrated_top5_hit_rate": metrics[
                            "calibrated_trifecta_top5_hit_rate"
                        ],
                        "market_top5_hit_rate": metrics[
                            "market_trifecta_top5_hit_rate"
                        ],
                    }
                )
            regrets = [float(row["market_regret"]) for row in daily]
            rows.append(
                {
                    **calibrator,
                    "races": total_races,
                    "pooled_log_loss": weighted_loss / total_races,
                    "mean_daily_market_regret": sum(regrets) / len(regrets),
                    "worst_daily_market_regret": max(regrets),
                    "days_beating_market": sum(regret <= 0.0 for regret in regrets),
                    "days": len(daily),
                    "daily": daily,
                }
            )
    return rows
