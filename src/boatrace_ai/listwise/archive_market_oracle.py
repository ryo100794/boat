from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib

from ..archive_closing_odds import SOURCE_KEY, ensure_archive_schema
from ..bankroll_bootstrap import bootstrap_daily_roi
from ..db import connection, init_db
from .market_calibration import (
    _validate_artifact_before_period,
    iter_scored_artifact_feature_rows,
    normalized_market_probabilities,
    probability_metrics,
    simulate_policy,
    write_json_atomic,
)


MODEL_NAME = "archive_closing_market_oracle_v1"
EVALUATION_VERSION = 1
PRIMARY_CALIBRATOR = {"model_weight": 0.75, "temperature": 1.0}
PRIMARY_POLICY: dict[str, Any] = {
    "name": "preregistered_closing_oracle_ev105_120_odds80_r3_ratio105_kelly025",
    "ev_threshold": 1.05,
    "max_estimated_ev": 1.20,
    "max_odds": 80.0,
    "max_tickets_per_race": 3,
    "min_model_market_ratio": 1.05,
    "staking_mode": "kelly_025",
}
DIAGNOSTIC_CONFIGS: tuple[tuple[str, dict[str, float], dict[str, Any]], ...] = (
    ("model_only_conservative", {"model_weight": 1.0, "temperature": 1.0}, {
        **PRIMARY_POLICY, "name": "model_only_ev120_odds80_r3", "ev_threshold": 1.20,
        "max_estimated_ev": None,
    }),
    ("blend_075_primary", PRIMARY_CALIBRATOR, PRIMARY_POLICY),
    ("blend_050_conservative", {"model_weight": 0.5, "temperature": 1.0}, {
        **PRIMARY_POLICY, "name": "blend050_ev102_odds40_r1", "ev_threshold": 1.02,
        "max_odds": 40.0, "max_tickets_per_race": 1,
    }),
)


def restrict_probabilities_to_available(
    probabilities: Mapping[str, float], available: Iterable[str]
) -> dict[str, float]:
    keys = tuple(sorted(set(available)))
    if not keys or any(key not in probabilities for key in keys):
        raise ValueError("model probabilities do not cover archive market")
    values = {key: float(probabilities[key]) for key in keys}
    total = sum(values.values())
    if total <= 0.0:
        raise ValueError("available model probability mass must be positive")
    return {key: value / total for key, value in values.items()}


def load_archive_markets(
    conn: Any, *, from_date: str, through_date: str
) -> dict[str, dict[str, Any]]:
    ensure_archive_schema(conn)
    markets: dict[str, dict[str, Any]] = {}
    for row in conn.execute(
        """
        SELECT a.race_id, r.race_date, r.jcd, r.rno, a.odds_count,
               a.verification_status,
               p.combination AS actual_combination,
               p.payout_yen AS actual_payout_yen
        FROM archive_closing_odds_snapshots a
        JOIN races r ON r.race_id = a.race_id
        JOIN payouts p ON p.race_id = a.race_id AND p.bet_type = '3連単'
        WHERE a.source_key = ?
          AND a.verification_status IN (
            'all_market_official_match',
            'winner_only_match_unverified_market'
          )
          AND r.race_date BETWEEN ? AND ?
        ORDER BY r.race_date, r.jcd, r.rno
        """,
        (SOURCE_KEY, from_date, through_date),
    ):
        race_id = str(row["race_id"])
        markets[race_id] = {
            "race_id": race_id,
            "race_date": str(row["race_date"]),
            "jcd": str(row["jcd"]),
            "rno": int(row["rno"]),
            "archive_odds_count": int(row["odds_count"]),
            "archive_verification_status": str(row["verification_status"]),
            "actual_combination": str(row["actual_combination"]),
            "actual_payout_yen": int(row["actual_payout_yen"]),
            "odds": {},
        }
    for row in conn.execute(
        """
        SELECT o.race_id, o.combination, o.odds
        FROM archive_closing_odds o
        JOIN races r ON r.race_id = o.race_id
        WHERE o.source_key = ? AND r.race_date BETWEEN ? AND ?
        ORDER BY o.race_id, o.combination
        """,
        (SOURCE_KEY, from_date, through_date),
    ):
        market = markets.get(str(row["race_id"]))
        if market is not None:
            market["odds"][str(row["combination"])] = float(row["odds"])
    return markets


def score_archive_markets(
    conn: Any,
    *,
    artifact: dict[str, Any],
    from_date: str,
    through_date: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    _validate_artifact_before_period(artifact, from_date=from_date)
    markets = load_archive_markets(
        conn, from_date=from_date, through_date=through_date
    )
    target_ids = set(markets)
    races: list[dict[str, Any]] = []
    skipped_incomplete = skipped_probability = 0
    feature_scored_ids: set[str] = set()
    for feature_rows, probabilities in iter_scored_artifact_feature_rows(
        conn, target_ids=target_ids, artifact=artifact
    ):
        race_id = str(feature_rows[0]["meta"]["race_id"])
        feature_scored_ids.add(race_id)
        market = markets.get(race_id)
        if market is None:
            continue
        odds = market["odds"]
        actual = str(market["actual_combination"])
        if (
            len(odds) != int(market["archive_odds_count"])
            or not 1 <= len(odds) <= 120
            or actual not in odds
        ):
            skipped_incomplete += 1
            continue
        try:
            available_model = restrict_probabilities_to_available(
                probabilities, odds
            )
        except ValueError:
            skipped_probability += 1
            continue
        market_probabilities = normalized_market_probabilities(odds)
        if set(market_probabilities) != set(odds):
            skipped_incomplete += 1
            continue
        races.append({
            **market,
            "model_probabilities": available_model,
            "market_probabilities": market_probabilities,
            "archive_source_key": SOURCE_KEY,
            "archive_market_role": "closing_oracle_research_only",
        })
    races.sort(key=lambda row: (row["race_date"], row["jcd"], row["rno"]))
    return races, {
        "archive_target_races": len(markets),
        "evaluated_races": len(races),
        "partial_market_races": sum(
            int(len(row["odds"]) < 120) for row in races
        ),
        "skipped_no_complete_features": len(target_ids - feature_scored_ids),
        "skipped_incomplete_archive": skipped_incomplete,
        "skipped_probability_mismatch": skipped_probability,
        "fully_official_verified_races": sum(
            int(row["archive_verification_status"] == "all_market_official_match")
            for row in races
        ),
        "winner_verified_secondary_races": sum(
            int(
                row["archive_verification_status"]
                == "winner_only_match_unverified_market"
            )
            for row in races
        ),
    }


def evaluate_archive_oracle(
    races: list[dict[str, Any]], *, daily_budget_yen: int
) -> dict[str, Any]:
    diagnostics = []
    primary = None
    for name, calibrator, policy in DIAGNOSTIC_CONFIGS:
        result = simulate_policy(
            races,
            calibrator=calibrator,
            policy=policy,
            daily_budget_yen=daily_budget_yen,
        )
        item = {
            "name": name,
            "calibrator": dict(calibrator),
            "policy": dict(policy),
            **result,
        }
        diagnostics.append(item)
        if name == "blend_075_primary":
            primary = item
    if primary is None:
        raise AssertionError("primary oracle policy is missing")
    bootstrap = (
        bootstrap_daily_roi(primary["daily"])
        if primary["daily"]
        else {
            "days": 0, "roi": None, "roi_ci95_lower": None,
            "probability_roi_above_one": None,
        }
    )
    research_gate = {
        "minimum_days": int(primary["race_days"]) >= 300,
        "minimum_races": len(races) >= 40_000,
        "minimum_tickets": int(primary["tickets"]) >= 1_000,
        "positive_profit": int(primary["profit_yen"]) > 0,
        "roi_above_one": float(primary["roi"]) > 1.0,
        "roi_lower95_above_one": float(bootstrap.get("roi_ci95_lower") or 0.0) > 1.0,
        "largest_hit_excluded_roi_above_one": float(
            primary.get("roi_without_largest_hit") or 0.0
        ) > 1.0,
        "effective_hit_count": float(primary.get("effective_hit_count") or 0.0) >= 30.0,
    }
    prediction = probability_metrics(races, calibrator=PRIMARY_CALIBRATOR)
    return {
        "model": MODEL_NAME,
        "status": "completed",
        "comparison_role": "unavailable_at_decision_closing_oracle_research_only",
        "market_source_scope": (
            "secondary closing archive; each winning price matches official payout, "
            "losing prices are not independently official-verified"
        ),
        "production_transfer_required": True,
        "promotion_eligible": False,
        "research_gate_pass": all(research_gate.values()),
        "research_gate": research_gate,
        "probability_metrics": prediction,
        "trifecta_log_loss": prediction["calibrated_trifecta_log_loss"],
        "trifecta_top5_hit_rate": prediction[
            "calibrated_trifecta_top5_hit_rate"
        ],
        "winner_log_loss": prediction["calibrated_winner_log_loss"],
        "winner_top1_accuracy": prediction[
            "calibrated_winner_top1_accuracy"
        ],
        "primary": primary,
        "primary_bootstrap": bootstrap,
        "diagnostics": diagnostics,
        "evaluated_races": len(races),
        "roi": primary["roi"],
        "profit_yen": primary["profit_yen"],
        "stake_yen": primary["stake_yen"],
        "return_yen": primary["return_yen"],
        "max_drawdown_yen": primary["max_drawdown_yen"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Leakage-bounded research upper bound using archived closing odds"
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--from-date", required=True)
    parser.add_argument("--through-date", required=True)
    parser.add_argument("--daily-budget-yen", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.from_date > args.through_date:
        raise ValueError("from-date must not be after through-date")
    if args.daily_budget_yen < 100:
        raise ValueError("daily-budget-yen must be at least 100")
    init_db(args.db)
    artifact = joblib.load(args.model)
    with connection(args.db) as conn:
        races, dataset = score_archive_markets(
            conn,
            artifact=artifact,
            from_date=args.from_date,
            through_date=args.through_date,
        )
    result = evaluate_archive_oracle(
        races, daily_budget_yen=args.daily_budget_yen
    )
    result.update({
        "evaluation_version": EVALUATION_VERSION,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "from_date": args.from_date,
        "through_date": args.through_date,
        "source_model": str(args.model),
        "source_model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "dataset": dataset,
    })
    write_json_atomic(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key not in {"diagnostics", "primary"}}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
