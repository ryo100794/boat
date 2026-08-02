from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..chronological_bankroll import summarize_chronological_bankroll_days
from .four_head_v22_bankroll import _bankroll_stability, _promotion_gate


PREDICTION_METRICS = (
    "winner_log_loss",
    "winner_top1_accuracy",
    "trifecta_log_loss",
    "trifecta_top1_accuracy",
    "trifecta_top5_hit_rate",
    "closing_odds_log_mae",
)


def _weighted_mean(rows: Iterable[tuple[float | None, int]]) -> float | None:
    values = [(float(value), int(weight)) for value, weight in rows if value is not None]
    total = sum(weight for _value, weight in values)
    return sum(value * weight for value, weight in values) / total if total else None


def aggregate_four_head_folds(
    payloads: Iterable[Mapping[str, Any]],
    *,
    source_job_ids: Iterable[int] = (),
) -> dict[str, Any]:
    folds = [dict(payload) for payload in payloads]
    if not folds:
        raise ValueError("at least one four-head fold is required")
    folds.sort(key=lambda row: str((row.get("periods") or {}).get("outer_from") or ""))
    losses = {str(row.get("purchase_loss") or "") for row in folds}
    if len(losses) != 1 or "" in losses:
        raise ValueError("folds must use one explicit purchase loss")

    daily: list[dict[str, Any]] = []
    seen_dates: set[str] = set()
    fold_rows: list[dict[str, Any]] = []
    for payload in folds:
        formal = payload.get("formal_bankroll")
        periods = payload.get("periods")
        if not isinstance(formal, Mapping) or not isinstance(periods, Mapping):
            raise ValueError("fold lacks formal bankroll or period metadata")
        rows = formal.get("daily")
        if not isinstance(rows, list) or not rows:
            raise ValueError("fold lacks formal daily bankroll rows")
        dates = [str(row.get("race_date") or "") for row in rows]
        if any(not value or value in seen_dates for value in dates):
            raise ValueError("fold daily dates are empty or overlap")
        if dates != sorted(dates):
            raise ValueError("fold daily dates are not chronological")
        seen_dates.update(dates)
        daily.extend(dict(row) for row in rows)
        fold_rows.append(
            {
                "outer_from": str(periods.get("outer_from") or ""),
                "outer_through": str(periods.get("outer_through") or ""),
                "evaluation_days": len(rows),
                "evaluated_races": int(formal.get("races") or formal.get("bankroll", {}).get("evaluated_races") or 0),
                "tickets": int(formal.get("tickets") or 0),
                "hit_tickets": int(formal.get("hit_tickets") or 0),
                "roi": float(formal.get("roi") or 0.0),
                "profit_yen": int(formal.get("profit_yen") or 0),
            }
        )

    daily.sort(key=lambda row: str(row["race_date"]))
    bankroll = summarize_chronological_bankroll_days(daily)
    stability = _bankroll_stability(bankroll)
    gate = _promotion_gate(bankroll, stability)
    weighted_prediction: dict[str, float | None] = {}
    for key in PREDICTION_METRICS:
        weighted_prediction[key] = _weighted_mean(
            (
                (row.get("formal_bankroll", {}).get(key), int(row.get("formal_bankroll", {}).get("races") or 0))
                for row in folds
            )
        )

    positive_tickets = sum(
        int((row.get("purchase_value_diagnostics") or {}).get("positive_predicted_tickets") or 0)
        for row in folds
    )
    positive_roi = _weighted_mean(
        (
            (
                (row.get("purchase_value_diagnostics") or {}).get("positive_observed_capped_roi"),
                int((row.get("purchase_value_diagnostics") or {}).get("positive_predicted_tickets") or 0),
            )
            for row in folds
        )
    )
    return {
        "schema_version": 1,
        "model_key": f"{next(iter(losses))}_temporal_aggregate",
        "purchase_loss": next(iter(losses)),
        "source_job_ids": [int(value) for value in source_job_ids],
        "fold_count": len(folds),
        "folds": fold_rows,
        "evaluation_days": stability["evaluation_days"],
        "evaluated_races": int(bankroll["evaluated_races"]),
        "tickets": stability["tickets"],
        "hit_tickets": stability["hit_tickets"],
        "stake_yen": int(bankroll["stake_yen"]),
        "return_yen": int(bankroll["return_yen"]),
        "profit_yen": int(bankroll["profit_yen"]),
        "roi": float(bankroll["roi"]),
        "max_drawdown_yen": int(bankroll["max_drawdown_yen"]),
        **stability,
        **weighted_prediction,
        "purchase_value_positive_predicted_tickets": positive_tickets,
        "purchase_value_positive_observed_capped_roi": positive_roi,
        "promotion_gate": gate,
        "promotion_eligible": all(gate.values()),
        "daily": daily,
        "aggregation_contract": {
            "bankroll": "concatenated_daily_ledgers_then_recomputed",
            "prediction_metrics": "race_count_weighted_fold_metrics",
            "purchase_positive_roi": "positive_ticket_count_weighted_fold_metric",
            "purchase_correlation": "not_composable_from_fold_summaries",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate disjoint four-head folds")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--source-job-id", type=int, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.input]
    result = aggregate_four_head_folds(payloads, source_job_ids=args.source_job_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
