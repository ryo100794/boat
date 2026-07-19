from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..adaptive_allocation import allocate_adaptive_day
from ..bankroll_backtest import (
    _build_payout_model,
    _candidate_tickets,
    _load_trifecta_payouts,
)
from ..db import connection, init_db


def evaluate_prediction_file(
    conn,
    *,
    prediction: dict[str, Any],
    source_path: Path,
) -> dict[str, Any]:
    race_date = str(prediction["race_date"])
    payouts = _load_trifecta_payouts(conn)
    train_races = {
        str(row["race_id"])
        for row in conn.execute(
            "SELECT race_id FROM races WHERE race_date < ?",
            (race_date,),
        ).fetchall()
    }
    payout_model = _build_payout_model(
        payouts,
        train_races=train_races,
        prior_weight=30.0,
    )
    candidates = []
    evaluated_races: set[str] = set()
    for race in prediction.get("predictions") or []:
        race_id = str(race["race_id"])
        actual = payouts.get(race_id)
        lane_rows = race.get("lane_probabilities") or []
        if actual is None or len(lane_rows) != 6:
            continue
        rows = [
            {
                "race_id": race_id,
                "race_date": race_date,
                "jcd": str(race["jcd"]),
                "rno": int(race["rno"]),
                "lane": int(item["lane"]),
                "probability": float(item["probability"]),
            }
            for item in lane_rows
        ]
        evaluated_races.add(race_id)
        candidates.extend(
            _candidate_tickets(
                rows,
                actual=actual,
                payout_model=payout_model,
                ev_threshold=1.20,
            )
        )

    day = allocate_adaptive_day(
        race_date,
        candidates,
        evaluated_races,
        daily_budget_yen=10_000,
        fractional_kelly=0.25,
        max_daily_exposure_fraction=0.60,
        min_daily_exposure_fraction=0.40,
        race_cap_fraction=0.10,
        ticket_cap_fraction=0.03,
        max_daily_tickets=30,
        allocation_mode="normalized_kelly",
        stake_granularity_yen=100,
        min_stake_yen=100,
    )
    day["cumulative_profit_yen"] = int(day["profit_yen"])
    policy = {
        "daily_budget_yen": 10_000,
        "bet_type": "3連単",
        "include_odds": False,
        "ev_threshold": 1.20,
        "payout_prior_weight": 30.0,
        "payout_estimator": "races before target date only",
        "fractional_kelly": 0.25,
        "max_daily_exposure_fraction": 0.60,
        "min_daily_exposure_fraction": 0.40,
        "race_cap_fraction": 0.10,
        "ticket_cap_fraction": 0.03,
        "max_daily_tickets": 30,
        "allocation_mode": "normalized_kelly",
        "stake_granularity_yen": 100,
        "min_stake_yen": 100,
        "selection": "prediction fixed before result; result used only for settlement",
    }
    return {
        "generated_at": prediction.get("generated_at"),
        "model": prediction.get("source_model") or prediction.get("model"),
        "comparison_role": "single_day_listwise_shadow_bankroll",
        "evaluation_scope": f"single_day:{race_date}",
        "source_prediction_file": str(source_path),
        "race_date": race_date,
        "policy": policy,
        "evaluated_races": len(evaluated_races),
        "race_days": 1,
        "candidate_tickets": len(candidates),
        "selected_races": int(day["races_bet"]),
        "tickets": int(day["tickets"]),
        "hit_tickets": int(day["hit_tickets"]),
        "ticket_hit_rate": (
            int(day["hit_tickets"]) / int(day["tickets"])
            if day["tickets"]
            else 0.0
        ),
        "stake_yen": int(day["stake_yen"]),
        "return_yen": int(day["return_yen"]),
        "profit_yen": int(day["profit_yen"]),
        "roi": float(day["roi"] or 0.0),
        "max_drawdown_yen": max(0, -int(day["profit_yen"])),
        "winner_top1_accuracy": prediction.get("winner_top1_accuracy"),
        "trifecta_top5_hit_rate": prediction.get("trifecta_top5_hit_rate"),
        "entry_log_loss": prediction.get("entry_log_loss"),
        "daily": [day],
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One-day bankroll settlement for listwise shadow predictions.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    prediction = json.loads(args.predictions.read_text(encoding="utf-8"))
    init_db(args.db)
    with connection(args.db) as conn:
        result = evaluate_prediction_file(
            conn,
            prediction=prediction,
            source_path=args.predictions,
        )
    write_json_atomic(args.output, result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "race_date",
                    "evaluated_races",
                    "selected_races",
                    "tickets",
                    "hit_tickets",
                    "stake_yen",
                    "return_yen",
                    "profit_yen",
                    "roi",
                )
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
