from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np

from ..adaptive_allocation import allocate_adaptive_day
from ..bankroll_backtest import _load_trifecta_payouts
from ..db import connection, init_db
from ..feature_tuning import (
    _ensure_sparse_index32,
    iter_race_feature_rows,
    load_complete_race_ids,
    normalize_drop_feature_groups,
    to_hashable,
)
from ..features import latest_trifecta_odds_before_deadline
from ..modeling import trifecta_predictions
from .model import ListwiseLinearModel, stable_softmax


MODEL_NAME = "listwise_newton_market_calibrated_v1"
STAKE_YEN = 100
BLEND_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
TEMPERATURES = (0.75, 1.0, 1.25)
EV_THRESHOLDS = (1.05, 1.10, 1.15, 1.20, 1.30, 1.50)
MAX_ODDS = (20.0, 40.0, 80.0, None)
MAX_TICKETS_PER_RACE = (1, 2, 3, 5)
MIN_MODEL_MARKET_RATIOS = (1.0, 1.05, 1.10, 1.20)
STAKING_MODES = {
    "kelly_025": {
        "fractional_kelly": 0.25,
        "allocation_mode": "kelly_floor",
        "min_daily_exposure_fraction": 0.0,
    },
    "kelly_100": {
        "fractional_kelly": 1.0,
        "allocation_mode": "kelly_floor",
        "min_daily_exposure_fraction": 0.0,
    },
    "normalized_010": {
        "fractional_kelly": 0.25,
        "allocation_mode": "normalized_kelly",
        "min_daily_exposure_fraction": 0.10,
    },
}
EPSILON = 1e-12


def artifact_drop_feature_groups(artifact: dict[str, Any]) -> tuple[str, ...]:
    return normalize_drop_feature_groups(
        artifact.get("drop_feature_groups") or (),
    )


def iter_artifact_feature_rows(
    conn,
    *,
    target_ids: set[str],
    artifact: dict[str, Any],
):
    return iter_race_feature_rows(
        conn,
        include_races=target_ids,
        drop_feature_groups=artifact_drop_feature_groups(artifact),
    )


def normalized_market_probabilities(odds: dict[str, float]) -> dict[str, float]:
    inverse = {
        combination: 1.0 / float(value)
        for combination, value in odds.items()
        if math.isfinite(float(value)) and float(value) > 0.0
    }
    total = sum(inverse.values())
    if not inverse or total <= 0.0:
        return {}
    return {combination: value / total for combination, value in inverse.items()}


def blend_probabilities(
    model: dict[str, float],
    market: dict[str, float],
    *,
    model_weight: float,
    temperature: float,
) -> dict[str, float]:
    if not 0.0 <= model_weight <= 1.0:
        raise ValueError("model_weight must be between zero and one")
    if temperature <= 0.0 or not math.isfinite(temperature):
        raise ValueError("temperature must be positive")
    combinations = sorted(set(model) & set(market))
    if not combinations:
        return {}
    logits = np.asarray(
        [
            (
                model_weight * math.log(max(EPSILON, float(model[combination])))
                + (1.0 - model_weight)
                * math.log(max(EPSILON, float(market[combination])))
            )
            / temperature
            for combination in combinations
        ],
        dtype=np.float64,
    )
    probabilities = stable_softmax(logits)
    return {
        combination: float(probability)
        for combination, probability in zip(combinations, probabilities)
    }


def select_calibrator(races: list[dict[str, Any]]) -> tuple[dict[str, float], list[dict[str, float]]]:
    if not races:
        raise ValueError("calibration requires at least one race")
    rows: list[dict[str, float]] = []
    for model_weight in BLEND_WEIGHTS:
        for temperature in TEMPERATURES:
            losses = []
            top5_hits = 0
            for race in races:
                probabilities = blend_probabilities(
                    race["model_probabilities"],
                    race["market_probabilities"],
                    model_weight=model_weight,
                    temperature=temperature,
                )
                actual = str(race["actual_combination"])
                losses.append(-math.log(max(EPSILON, probabilities.get(actual, 0.0))))
                top5 = sorted(probabilities, key=probabilities.get, reverse=True)[:5]
                top5_hits += int(actual in top5)
            rows.append(
                {
                    "model_weight": model_weight,
                    "temperature": temperature,
                    "trifecta_log_loss": sum(losses) / len(losses),
                    "trifecta_top5_hit_rate": top5_hits / len(losses),
                }
            )
    selected = min(
        rows,
        key=lambda row: (
            row["trifecta_log_loss"],
            -row["trifecta_top5_hit_rate"],
            row["model_weight"],
        ),
    )
    return {
        "model_weight": float(selected["model_weight"]),
        "temperature": float(selected["temperature"]),
    }, rows


def default_policy_grid() -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = [{"name": "no_bet", "no_bet": True}]
    for ev_threshold in EV_THRESHOLDS:
        for max_odds in MAX_ODDS:
            for max_tickets in MAX_TICKETS_PER_RACE:
                for min_ratio in MIN_MODEL_MARKET_RATIOS:
                    for staking_mode in STAKING_MODES:
                        odds_name = "none" if max_odds is None else str(int(max_odds))
                        policies.append(
                            {
                                "name": (
                                    f"ev{ev_threshold:.2f}_odds{odds_name}_"
                                    f"r{max_tickets}_ratio{min_ratio:.2f}_{staking_mode}"
                                ),
                                "ev_threshold": ev_threshold,
                                "max_odds": max_odds,
                                "max_tickets_per_race": max_tickets,
                                "min_model_market_ratio": min_ratio,
                                "staking_mode": staking_mode,
                            }
                        )
    return policies


def simulate_policy(
    races: list[dict[str, Any]],
    *,
    calibrator: dict[str, float],
    policy: dict[str, Any],
    daily_budget_yen: int,
) -> dict[str, Any]:
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    evaluated_by_day: dict[str, set[str]] = defaultdict(set)
    if not policy.get("no_bet"):
        for race in races:
            race_id = str(race["race_id"])
            race_date = str(race["race_date"])
            evaluated_by_day[race_date].add(race_id)
            calibrated = blend_probabilities(
                race["model_probabilities"],
                race["market_probabilities"],
                model_weight=float(calibrator["model_weight"]),
                temperature=float(calibrator["temperature"]),
            )
            candidates = []
            for combination, probability in calibrated.items():
                odds = float(race["odds"][combination])
                market_probability = float(race["market_probabilities"][combination])
                estimated_ev = probability * odds
                if estimated_ev < float(policy["ev_threshold"]):
                    continue
                if policy.get("max_odds") is not None and odds > float(policy["max_odds"]):
                    continue
                ratio = probability / max(EPSILON, market_probability)
                if ratio < float(policy["min_model_market_ratio"]):
                    continue
                candidates.append(
                    {
                        "race_id": race_id,
                        "race_date": race_date,
                        "jcd": race["jcd"],
                        "rno": int(race["rno"]),
                        "combination": combination,
                        "probability": probability,
                        "market_probability": market_probability,
                        "model_probability": float(race["model_probabilities"][combination]),
                        "estimated_odds": odds,
                        "estimated_ev": estimated_ev,
                        "estimated_payout_yen": odds * STAKE_YEN,
                        "payout_history_count": 0,
                        "odds_source": "real_t5",
                        "actual_combination": race["actual_combination"],
                        "actual_payout_yen": int(race["actual_payout_yen"]),
                        "hit": combination == race["actual_combination"],
                        "real_odds_snapshot_id": race.get("snapshot_id"),
                        "real_odds_captured_at": race.get("captured_at"),
                        "real_odds_deadline_at": race.get("odds_deadline_at"),
                        "real_odds_combinations": len(race["odds"]),
                    }
                )
            candidates.sort(
                key=lambda item: (item["estimated_ev"], item["probability"]),
                reverse=True,
            )
            by_day[race_date].extend(
                candidates[: int(policy["max_tickets_per_race"])]
            )
    else:
        for race in races:
            evaluated_by_day[str(race["race_date"])].add(str(race["race_id"]))

    daily = []
    stake_yen = return_yen = tickets = hit_tickets = 0
    cumulative_profit = peak_profit = max_drawdown_yen = 0
    for race_date in sorted(evaluated_by_day):
        staking = STAKING_MODES.get(
            str(policy.get("staking_mode") or "kelly_025"),
            STAKING_MODES["kelly_025"],
        )
        result = allocate_adaptive_day(
            race_date,
            by_day.get(race_date, []),
            evaluated_by_day[race_date],
            daily_budget_yen=daily_budget_yen,
            fractional_kelly=float(staking["fractional_kelly"]),
            max_daily_exposure_fraction=0.30,
            min_daily_exposure_fraction=float(staking["min_daily_exposure_fraction"]),
            race_cap_fraction=0.05,
            ticket_cap_fraction=0.02,
            max_daily_tickets=30,
            allocation_mode=str(staking["allocation_mode"]),
            stake_granularity_yen=STAKE_YEN,
            min_stake_yen=STAKE_YEN,
        )
        cumulative_profit += int(result["profit_yen"])
        peak_profit = max(peak_profit, cumulative_profit)
        max_drawdown_yen = max(max_drawdown_yen, peak_profit - cumulative_profit)
        result["cumulative_profit_yen"] = cumulative_profit
        daily.append(result)
        stake_yen += int(result["stake_yen"])
        return_yen += int(result["return_yen"])
        tickets += int(result["tickets"])
        hit_tickets += int(result["hit_tickets"])
    return {
        "evaluated_races": len(races),
        "race_days": len(daily),
        "tickets": tickets,
        "hit_tickets": hit_tickets,
        "stake_yen": stake_yen,
        "return_yen": return_yen,
        "profit_yen": return_yen - stake_yen,
        "roi": return_yen / stake_yen if stake_yen else 0.0,
        "max_drawdown_yen": max_drawdown_yen,
        "winning_days": sum(int(row["profit_yen"] > 0) for row in daily),
        "daily": daily,
    }


def select_policy(
    races: list[dict[str, Any]],
    *,
    calibrator: dict[str, float],
    daily_budget_yen: int,
    policies: Iterable[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    minimum_tickets = max(10, math.ceil(len(races) * 0.05))
    minimum_stake = minimum_tickets * STAKE_YEN
    for policy in policies or default_policy_grid():
        result = simulate_policy(
            races,
            calibrator=calibrator,
            policy=policy,
            daily_budget_yen=daily_budget_yen,
        )
        eligible = bool(
            policy.get("no_bet")
            or policy_calibration_eligible(
                result,
                minimum_tickets=minimum_tickets,
                minimum_stake_yen=minimum_stake,
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
            int(row["profit_yen"]) - 0.25 * int(row["max_drawdown_yen"]),
            float(row["roi"]),
            -int(row["tickets"]),
        ),
    )
    return dict(selected["policy"]), rows


def policy_calibration_eligible(
    result: dict[str, Any],
    *,
    minimum_tickets: int,
    minimum_stake_yen: int,
) -> bool:
    race_days = int(result["race_days"])
    minimum_winning_days = min(
        race_days,
        max(1, math.ceil(race_days * 0.60)),
    )
    return bool(
        int(result["tickets"]) >= minimum_tickets
        and int(result["stake_yen"]) >= minimum_stake_yen
        and int(result["profit_yen"]) > 0
        and float(result["roi"]) >= 1.05
        and int(result["winning_days"]) >= minimum_winning_days
        and int(result["max_drawdown_yen"]) <= int(result["stake_yen"]) * 0.75
    )


def summarize_policy_candidates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in rows if not row["policy"].get("no_bet")]
    funded = [row for row in candidates if int(row["stake_yen"]) > 0]

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
            "max_drawdown_yen": int(row["max_drawdown_yen"]),
            "winning_days": int(row["winning_days"]),
            "race_days": int(row["race_days"]),
        }

    return {
        "candidate_count": len(candidates),
        "funded_candidate_count": len(funded),
        "profitable_candidate_count": sum(int(row["profit_yen"] > 0) for row in funded),
        "eligible_candidate_count": sum(bool(row["eligible"]) for row in candidates),
        "best_profit": compact(
            max(funded, key=lambda row: (row["profit_yen"], row["roi"]), default=None)
        ),
        "best_roi": compact(
            max(funded, key=lambda row: (row["roi"], row["profit_yen"]), default=None)
        ),
    }


def probability_metrics(
    races: list[dict[str, Any]],
    *,
    calibrator: dict[str, float] | None = None,
) -> dict[str, float | int | None]:
    losses = {"model": [], "market": [], "calibrated": []}
    top5_hits = {key: 0 for key in losses}
    for race in races:
        sources = {
            "model": race["model_probabilities"],
            "market": race["market_probabilities"],
        }
        if calibrator is not None:
            sources["calibrated"] = blend_probabilities(
                sources["model"],
                sources["market"],
                model_weight=float(calibrator["model_weight"]),
                temperature=float(calibrator["temperature"]),
            )
        actual = str(race["actual_combination"])
        for name, probabilities in sources.items():
            losses[name].append(-math.log(max(EPSILON, probabilities.get(actual, 0.0))))
            top5 = sorted(probabilities, key=probabilities.get, reverse=True)[:5]
            top5_hits[name] += int(actual in top5)
    result: dict[str, float | int | None] = {"evaluated_races": len(races)}
    for name in ("model", "market", "calibrated"):
        values = losses[name]
        result[f"{name}_trifecta_log_loss"] = sum(values) / len(values) if values else None
        result[f"{name}_trifecta_top5_hit_rate"] = (
            top5_hits[name] / len(values) if values else None
        )
    return result


def walk_forward_evaluate(
    races: list[dict[str, Any]],
    *,
    daily_budget_yen: int = 10_000,
    min_calibration_days: int = 2,
) -> dict[str, Any]:
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for race in races:
        by_day[str(race["race_date"])].append(race)
    dates = sorted(by_day)
    if len(dates) <= min_calibration_days:
        raise ValueError("not enough full days for walk-forward calibration and evaluation")

    folds = []
    evaluation_races: list[dict[str, Any]] = []
    daily_rows = []
    for index in range(min_calibration_days, len(dates)):
        calibration_dates = dates[:index]
        evaluation_date = dates[index]
        calibration_races = [race for date in calibration_dates for race in by_day[date]]
        holdout = by_day[evaluation_date]
        calibrator, calibrator_grid = select_calibrator(calibration_races)
        policy, policy_grid = select_policy(
            calibration_races,
            calibrator=calibrator,
            daily_budget_yen=daily_budget_yen,
        )
        bankroll = simulate_policy(
            holdout,
            calibrator=calibrator,
            policy=policy,
            daily_budget_yen=daily_budget_yen,
        )
        metrics = probability_metrics(holdout, calibrator=calibrator)
        folds.append(
            {
                "fold": len(folds) + 1,
                "calibration_dates": calibration_dates,
                "evaluation_date": evaluation_date,
                "calibration_races": len(calibration_races),
                "evaluation_races": len(holdout),
                "calibrator": calibrator,
                "selected_policy": policy,
                "calibrator_candidates": len(calibrator_grid),
                "policy_candidates": len(policy_grid),
                "calibrator_top5": sorted(
                    calibrator_grid,
                    key=lambda row: (
                        row["trifecta_log_loss"],
                        -row["trifecta_top5_hit_rate"],
                    ),
                )[:5],
                "policy_diagnostics": summarize_policy_candidates(policy_grid),
                "probability_metrics": metrics,
                "bankroll": {key: value for key, value in bankroll.items() if key != "daily"},
            }
        )
        daily_rows.extend(bankroll["daily"])
        evaluation_races.extend(holdout)

    stake_yen = sum(int(row["stake_yen"]) for row in daily_rows)
    return_yen = sum(int(row["return_yen"]) for row in daily_rows)
    cumulative_profit = peak_profit = max_drawdown_yen = 0
    for row in daily_rows:
        cumulative_profit += int(row["profit_yen"])
        peak_profit = max(peak_profit, cumulative_profit)
        max_drawdown_yen = max(max_drawdown_yen, peak_profit - cumulative_profit)
        row["cumulative_profit_yen"] = cumulative_profit
    profitable_folds = sum(int(fold["bankroll"]["profit_yen"] > 0) for fold in folds)
    aggregate_metrics = _aggregate_fold_probability_metrics(folds)
    promotion_gate = {
        "minimum_evaluation_races": 1000,
        "minimum_evaluation_days": 30,
        "minimum_profitable_fold_fraction": 0.60,
        "sample_size_pass": len(evaluation_races) >= 1000 and len(daily_rows) >= 30,
        "positive_profit_pass": return_yen > stake_yen and stake_yen > 0,
        "roi_pass": return_yen / stake_yen > 1.0 if stake_yen else False,
        "fold_stability_pass": profitable_folds >= math.ceil(len(folds) * 0.60),
        "calibration_pass": _calibration_gate_pass(aggregate_metrics),
        "no_lookahead_pass": True,
    }
    return {
        "model": MODEL_NAME,
        "comparison_role": "real_t5_odds_nested_daily_walk_forward_shadow",
        "validation_design": (
            "Each evaluation day is untouched; calibration and policy selection use only earlier full days"
        ),
        "daily_budget_yen": daily_budget_yen,
        "available_races": len(races),
        "available_days": len(dates),
        "evaluated_races": len(evaluation_races),
        "evaluation_races": len(evaluation_races),
        "evaluation_days": len(daily_rows),
        "probability_metrics": aggregate_metrics,
        "calibrated_trifecta_log_loss": aggregate_metrics.get(
            "calibrated_trifecta_log_loss"
        ),
        "trifecta_top5_hit_rate": aggregate_metrics.get(
            "calibrated_trifecta_top5_hit_rate"
        ),
        "tickets": sum(int(row["tickets"]) for row in daily_rows),
        "hit_tickets": sum(int(row["hit_tickets"]) for row in daily_rows),
        "stake_yen": stake_yen,
        "return_yen": return_yen,
        "profit_yen": return_yen - stake_yen,
        "roi": return_yen / stake_yen if stake_yen else 0.0,
        "max_drawdown_yen": max_drawdown_yen,
        "profitable_folds": profitable_folds,
        "folds": folds,
        "daily": daily_rows,
        "promotion_gate": promotion_gate,
        "promotion_eligible": all(
            value for key, value in promotion_gate.items() if key.endswith("_pass")
        ),
    }


def _aggregate_fold_probability_metrics(folds: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(int(fold["evaluation_races"]) for fold in folds)
    result: dict[str, Any] = {"evaluated_races": total}
    for source in ("model", "market", "calibrated"):
        for metric in ("trifecta_log_loss", "trifecta_top5_hit_rate"):
            key = f"{source}_{metric}"
            result[key] = (
                sum(
                    float(fold["probability_metrics"][key])
                    * int(fold["evaluation_races"])
                    for fold in folds
                )
                / total
                if total
                else None
            )
    return result


def _calibration_gate_pass(metrics: dict[str, Any]) -> bool:
    calibrated = metrics.get("calibrated_trifecta_log_loss")
    model = metrics.get("model_trifecta_log_loss")
    market = metrics.get("market_trifecta_log_loss")
    return bool(
        calibrated is not None
        and model is not None
        and market is not None
        and float(calibrated) <= min(float(model), float(market))
    )


def score_real_odds_races(
    conn,
    *,
    artifact: dict[str, Any],
    from_date: str,
    through_date: str | None = None,
    max_snapshot_age_seconds: float = 60.0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    _validate_artifact_before_period(artifact, from_date=from_date)
    model = artifact.get("model")
    hasher = artifact.get("hasher")
    if not isinstance(model, ListwiseLinearModel) or hasher is None:
        raise ValueError("model artifact must contain a listwise model and hasher")
    race_keys = load_complete_race_ids(conn)
    target_ids = {
        str(race_id)
        for race_id, race_date, _jcd, _rno in race_keys
        if str(race_date) >= from_date
        and (through_date is None or str(race_date) <= through_date)
    }
    payouts = _load_trifecta_payouts(conn)
    races = []
    skipped_no_odds = skipped_stale_odds = skipped_no_payout = 0
    for feature_rows in iter_artifact_feature_rows(
        conn,
        target_ids=target_ids,
        artifact=artifact,
    ):
        meta_rows = [item["meta"] for item in feature_rows]
        race_id = str(meta_rows[0]["race_id"])
        payout = payouts.get(race_id)
        if payout is None:
            skipped_no_payout += 1
            continue
        snapshot = latest_trifecta_odds_before_deadline(
            conn,
            race_id,
            min_combinations=120,
        )
        if snapshot is None or len(snapshot.get("odds") or {}) != 120:
            skipped_no_odds += 1
            continue
        snapshot_age = snapshot_age_seconds(snapshot)
        if (
            snapshot_age is None
            or snapshot_age < 0.0
            or snapshot_age > max_snapshot_age_seconds
        ):
            skipped_stale_odds += 1
            continue
        matrix = _ensure_sparse_index32(
            hasher.transform([to_hashable(item["features"]) for item in feature_rows])
        )
        scores = np.asarray(model.scaler.transform(matrix).dot(model.weights)).reshape(6)
        lane_probabilities = stable_softmax(scores)
        model_probabilities = {
            row["combination"]: float(row["probability"])
            for row in trifecta_predictions(
                {lane: float(lane_probabilities[lane - 1]) for lane in range(1, 7)}
            )
        }
        odds = {key: float(value) for key, value in snapshot["odds"].items()}
        market_probabilities = normalized_market_probabilities(odds)
        if set(model_probabilities) != set(odds) or set(market_probabilities) != set(odds):
            skipped_no_odds += 1
            continue
        races.append(
            {
                "race_id": race_id,
                "race_date": str(meta_rows[0]["race_date"]),
                "jcd": str(meta_rows[0]["jcd"]),
                "rno": int(meta_rows[0]["rno"]),
                "actual_combination": str(payout["combination"]),
                "actual_payout_yen": int(payout["payout_yen"]),
                "model_probabilities": model_probabilities,
                "market_probabilities": market_probabilities,
                "odds": odds,
                "snapshot_id": snapshot.get("snapshot_id"),
                "captured_at": snapshot.get("captured_at"),
                "odds_deadline_at": snapshot.get("odds_deadline_at"),
            }
        )
    return races, {
        "target_complete_races": len(target_ids),
        "eligible_real_odds_races": len(races),
        "skipped_no_real_odds": skipped_no_odds,
        "skipped_stale_real_odds": skipped_stale_odds,
        "skipped_no_payout": skipped_no_payout,
    }


def snapshot_age_seconds(snapshot: dict[str, Any]) -> float | None:
    try:
        captured = datetime.fromisoformat(str(snapshot["captured_at"]))
        deadline = datetime.fromisoformat(str(snapshot["odds_deadline_at"]))
    except (KeyError, TypeError, ValueError):
        return None
    if captured.tzinfo is None:
        captured = captured.replace(tzinfo=deadline.tzinfo or timezone.utc)
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=captured.tzinfo or timezone.utc)
    return (deadline - captured.astimezone(deadline.tzinfo)).total_seconds()


def _validate_artifact_before_period(artifact: dict[str, Any], *, from_date: str) -> None:
    trained_through = artifact.get("trained_through")
    if not isinstance(trained_through, (list, tuple)) or len(trained_through) < 2:
        raise ValueError("model artifact lacks trained_through leakage metadata")
    trained_date = str(trained_through[1])
    if trained_date >= from_date:
        raise ValueError(
            f"model training overlaps evaluation period: trained_through={trained_date} "
            f"from_date={from_date}"
        )


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scored_cache_contract(
    *,
    model_path: Path,
    artifact: dict[str, Any],
    from_date: str,
    through_date: str | None,
    max_snapshot_age_seconds: float,
) -> dict[str, Any]:
    return {
        "version": 3,
        "model_sha256": file_sha256(model_path),
        "trained_through": tuple(artifact.get("trained_through") or ()),
        "feature_variant": artifact.get("feature_variant"),
        "drop_feature_groups": artifact_drop_feature_groups(artifact),
        "from_date": from_date,
        "through_date": through_date,
        "max_snapshot_age_seconds": max_snapshot_age_seconds,
    }


def load_scored_cache(
    path: Path,
    *,
    contract: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]] | None:
    if not path.exists():
        return None
    try:
        payload = joblib.load(path)
    except (OSError, ValueError, EOFError):
        return None
    if not isinstance(payload, dict) or payload.get("contract") != contract:
        return None
    races = payload.get("races")
    dataset = payload.get("dataset")
    if not isinstance(races, list) or not isinstance(dataset, dict):
        return None
    return races, {str(key): int(value) for key, value in dataset.items()}


def write_scored_cache(
    path: Path,
    *,
    contract: dict[str, Any],
    races: list[dict[str, Any]],
    dataset: dict[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        joblib.dump(
            {"contract": contract, "races": races, "dataset": dataset},
            temporary,
            compress=3,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Leakage-safe market calibration and bankroll shadow evaluation."
    )
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--model", default="data/models/listwise_newton_cg_v1.joblib")
    parser.add_argument(
        "--output",
        default="data/models/listwise_market_calibrated_shadow.json",
    )
    parser.add_argument("--from-date", required=True)
    parser.add_argument("--through-date")
    parser.add_argument("--daily-budget-yen", type=int, default=10_000)
    parser.add_argument("--min-calibration-days", type=int, default=2)
    parser.add_argument("--scored-cache")
    parser.add_argument("--max-snapshot-age-seconds", type=float, default=60.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    init_db(args.db)
    model_path = Path(args.model)
    output_path = Path(args.output)
    cache_path = (
        Path(args.scored_cache)
        if args.scored_cache
        else output_path.with_suffix(".races.joblib")
    )
    artifact = joblib.load(model_path)
    contract = scored_cache_contract(
        model_path=model_path,
        artifact=artifact,
        from_date=args.from_date,
        through_date=args.through_date,
        max_snapshot_age_seconds=args.max_snapshot_age_seconds,
    )
    cached = load_scored_cache(cache_path, contract=contract)
    if cached is None:
        with connection(args.db) as conn:
            races, dataset = score_real_odds_races(
                conn,
                artifact=artifact,
                from_date=args.from_date,
                through_date=args.through_date,
                max_snapshot_age_seconds=args.max_snapshot_age_seconds,
            )
        write_scored_cache(
            cache_path,
            contract=contract,
            races=races,
            dataset=dataset,
        )
        cache_source = "built"
    else:
        races, dataset = cached
        cache_source = "disk"
    result = walk_forward_evaluate(
        races,
        daily_budget_yen=args.daily_budget_yen,
        min_calibration_days=args.min_calibration_days,
    )
    result.update(
        {
            "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "source_model": str(args.model),
            "source_model_trained_through": artifact.get("trained_through"),
            "from_date": args.from_date,
            "through_date": args.through_date,
            "dataset": dataset,
            "scored_cache": str(cache_path),
            "scored_cache_source": cache_source,
        }
    )
    write_json_atomic(output_path, result)
    compact = {key: value for key, value in result.items() if key not in {"folds", "daily"}}
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
