from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import bankroll_backtest as base
from .db import connection, init_db


def _make_sparse_ok_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("vectorizer", DictVectorizer(sparse=True)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=400,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backtest daily bankroll buying with sparse-safe solver.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--output", default="data/models/bankroll_backtest_10000.json")
    parser.add_argument("--daily-budget-yen", type=int, default=10_000)
    parser.add_argument("--unit-yen", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-train-races", type=int, default=500)
    parser.add_argument("--include-odds", action="store_true")
    parser.add_argument("--ev-threshold", type=float, default=1.0)
    parser.add_argument("--max-tickets-per-race", type=int, default=5)
    parser.add_argument("--payout-prior-weight", type=float, default=30.0)
    args = parser.parse_args(argv)

    base._make_pipeline = _make_sparse_ok_pipeline
    init_db(args.db)
    with connection(args.db) as conn:
        result = base.bankroll_backtest(
            conn,
            output_path=Path(args.output),
            daily_budget_yen=args.daily_budget_yen,
            unit_yen=args.unit_yen,
            folds=args.folds,
            min_train_races=args.min_train_races,
            include_odds=args.include_odds,
            ev_threshold=args.ev_threshold,
            max_tickets_per_race=args.max_tickets_per_race,
            payout_prior_weight=args.payout_prior_weight,
        )
    print(json.dumps(_console_summary(result), ensure_ascii=False, indent=2))
    return 0


def _console_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"daily"}
    } | {"daily_rows": len(result.get("daily", []))}


if __name__ == "__main__":
    raise SystemExit(main())
