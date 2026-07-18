from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import bankroll_backtest as base
from .db import connection, init_db
from .features_pastlog_v1 import load_training_examples
from .modeling_pastlog_v1 import FEATURE_SET, make_pipeline, positive_probs


def configure() -> None:
    base.load_training_examples = load_training_examples
    base._make_pipeline = make_pipeline
    base._positive_probs = positive_probs


def bankroll_backtest_pastlog(
    conn,
    *,
    output_path: Path,
    daily_budget_yen: int = 10_000,
    unit_yen: int = 100,
    folds: int = 5,
    min_train_races: int = 500,
    ev_threshold: float = 1.0,
    max_tickets_per_race: int = 5,
    payout_prior_weight: float = 30.0,
) -> dict[str, Any]:
    configure()
    result = base.bankroll_backtest(
        conn,
        output_path=output_path,
        daily_budget_yen=daily_budget_yen,
        unit_yen=unit_yen,
        folds=folds,
        min_train_races=min_train_races,
        include_odds=False,
        ev_threshold=ev_threshold,
        max_tickets_per_race=max_tickets_per_race,
        payout_prior_weight=payout_prior_weight,
    )
    result["feature_set"] = FEATURE_SET
    result["model"] = "win_model_pastlog_v1"
    result["policy"]["model"] = "win_model_pastlog_v1"
    result["policy"]["feature_set"] = FEATURE_SET
    result["policy"]["role"] = "primary_pastlog"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backtest daily bankroll buying with past-log primary model.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--output", default="data/models/bankroll_backtest_pastlog_v1_10000.json")
    parser.add_argument("--daily-budget-yen", type=int, default=10_000)
    parser.add_argument("--unit-yen", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-train-races", type=int, default=500)
    parser.add_argument("--ev-threshold", type=float, default=1.0)
    parser.add_argument("--max-tickets-per-race", type=int, default=5)
    parser.add_argument("--payout-prior-weight", type=float, default=30.0)
    args = parser.parse_args(argv)

    init_db(args.db)
    with connection(args.db) as conn:
        result = bankroll_backtest_pastlog(
            conn,
            output_path=Path(args.output),
            daily_budget_yen=args.daily_budget_yen,
            unit_yen=args.unit_yen,
            folds=args.folds,
            min_train_races=args.min_train_races,
            ev_threshold=args.ev_threshold,
            max_tickets_per_race=args.max_tickets_per_race,
            payout_prior_weight=args.payout_prior_weight,
        )
    print(json.dumps(_console_summary(result), ensure_ascii=False, indent=2), flush=True)
    return 0


def _console_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "daily"} | {"daily_rows": len(result.get("daily", []))}


if __name__ == "__main__":
    raise SystemExit(main())
