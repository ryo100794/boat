from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np

from .closing_odds_quantile import (
    _fit_daily_cross_conformal_model,
    closing_odds_quantile_metrics,
    forecast_closing_odds_quantiles,
)
from .decision_market_residual_v38 import (
    _iso_date,
    validate_decision_scored_cache_contract,
)
from .decision_stacked_market_v44 import fit_decision_time_stacked_market
from .stacked_market_residual_v42 import stacked_probabilities


MODEL_NAME = "decision_time_stacked_terminal_market_v46"
PRICE_MODEL_TYPE = "ridge_log_location_odds_path_context_v3"
MINIMUM_PRICE_TRAINING_DAYS = 10
PRICE_REGULARIZATION = 0.001
PURCHASE_MAX_PROBABILITY_RANK = 20
BOOTSTRAP_SAMPLES = 5_000


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _price_teacher_race(race: Mapping[str, Any]) -> dict[str, Any]:
    official = race.get("official_closing_odds")
    if not isinstance(official, Mapping) or len(official) != 120:
        raise ValueError("V46 requires 120 official closing odds per price teacher")
    item = dict(race)
    item["closing_odds"] = {
        str(key): float(value) for key, value in official.items()
    }
    return item


def _fit_price_model(calibration: list[dict[str, Any]]) -> dict[str, Any]:
    dates = sorted({str(race["race_date"]) for race in calibration})
    if len(dates) < MINIMUM_PRICE_TRAINING_DAYS:
        raise ValueError("V46 requires at least ten price training days")
    by_day = {
        day: [race for race in calibration if str(race["race_date"]) == day]
        for day in dates
    }
    return _fit_daily_cross_conformal_model(
        by_day,
        dates,
        regularization=PRICE_REGULARIZATION,
        use_trend_features=True,
        trend_context_features=True,
    )


def _day_block_lcb(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    by_day: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_day[str(row["race_date"])].append(float(row["return_multiple"]))
    days = sorted(by_day)
    stakes = np.asarray([len(by_day[day]) for day in days], dtype=np.float64)
    returned = np.asarray(
        [sum(by_day[day]) for day in days], dtype=np.float64
    )
    rng = np.random.default_rng(460300)
    draws = rng.integers(0, len(days), size=(BOOTSTRAP_SAMPLES, len(days)))
    roi = returned[draws].sum(axis=1) / stakes[draws].sum(axis=1)
    return float(np.quantile(roi, 0.05, method="inverted_cdf"))


def _candidate_diagnostic(
    evaluation: list[dict[str, Any]],
    probability_artifact: Mapping[str, Any],
    price_model: Mapping[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for race in evaluation:
        probabilities = stacked_probabilities(race, probability_artifact)
        ranked = sorted(
            probabilities,
            key=lambda key: (-float(probabilities[key]), str(key)),
        )[:PURCHASE_MAX_PROBABILITY_RANK]
        forecast = forecast_closing_odds_quantiles(race, dict(price_model))
        median = forecast["q50"]
        lower = forecast["q10"]
        selected = max(
            ranked,
            key=lambda key: (
                float(probabilities[key]) * float(median[key]),
                float(probabilities[key]),
                str(key),
            ),
        )
        probability = float(probabilities[selected])
        actual_odds = float(race["official_closing_odds"][selected])
        hit = selected == str(race["actual_combination"])
        rows.append({
            "race_date": str(race["race_date"]),
            "probability": probability,
            "median_forecast_ev": probability * float(median[selected]),
            "lower_forecast_ev": probability * float(lower[selected]),
            "oracle_closing_ev": probability * actual_odds,
            "lower_price_covered": actual_odds >= float(lower[selected]),
            "hit": hit,
            "return_multiple": (
                float(race["actual_payout_yen"]) / 100.0 if hit else 0.0
            ),
        })
    tickets = len(rows)
    returned = sum(float(row["return_multiple"]) for row in rows)
    largest = max((float(row["return_multiple"]) for row in rows), default=0.0)
    return {
        "status": "outer_holdout_diagnostic_only_not_for_selection",
        "selection_rule": "max_probability_x_conditional_median_closing_odds_within_probability_top20",
        "candidate_price_target": "conditional_median",
        "purchase_gate": "disabled_pending_strict_prior_realized_roi_lcb",
        "evaluated_days": len({str(row["race_date"]) for row in rows}),
        "tickets": tickets,
        "hits": sum(bool(row["hit"]) for row in rows),
        "predicted_hits": sum(float(row["probability"]) for row in rows),
        "mean_median_forecast_ev": (
            sum(float(row["median_forecast_ev"]) for row in rows) / tickets
            if tickets else None
        ),
        "mean_lower_forecast_ev": (
            sum(float(row["lower_forecast_ev"]) for row in rows) / tickets
            if tickets else None
        ),
        "mean_oracle_closing_ev": (
            sum(float(row["oracle_closing_ev"]) for row in rows) / tickets
            if tickets else None
        ),
        "selected_lower_price_coverage": (
            sum(bool(row["lower_price_covered"]) for row in rows) / tickets
            if tickets else None
        ),
        "realized_roi": returned / tickets if tickets else None,
        "roi_without_largest_hit": (
            (returned - largest) / (tickets - 1) if tickets > 1 else None
        ),
        "day_block_roi_lcb95": _day_block_lcb(rows),
        "outer_used_for_threshold_or_model_selection": False,
    }


def fit_decision_time_stacked_terminal_market(
    races: list[dict[str, Any]],
    *,
    calibration_through: str,
    minimum_training_days: int,
    minimum_training_races: int,
    num_threads: int = 4,
) -> dict[str, Any]:
    cutoff = _iso_date(calibration_through, "calibration_through")
    probability = fit_decision_time_stacked_market(
        races,
        calibration_through=cutoff,
        minimum_training_days=minimum_training_days,
        minimum_training_races=minimum_training_races,
        num_threads=num_threads,
    )
    if probability.get("status") != "ready":
        return {
            **probability,
            "model": MODEL_NAME,
            "probability_model": probability.get("model"),
            "price_training_status": "not_started_probability_not_ready",
        }
    calibration = [
        _price_teacher_race(race)
        for race in races
        if _iso_date(race["race_date"], "race_date") <= cutoff
    ]
    evaluation = [
        _price_teacher_race(race)
        for race in races
        if _iso_date(race["race_date"], "race_date") > cutoff
    ]
    price_model = _fit_price_model(calibration)
    probability_artifact = probability["artifact"]
    combined_artifact = {
        "model": MODEL_NAME,
        "role": "frozen_probability_and_terminal_price_candidate_ranker",
        "probability_artifact": probability_artifact,
        "probability_artifact_sha256": probability_artifact.get(
            "artifact_sha256"
        ),
        "closing_odds_model": price_model,
        "closing_odds_model_type": price_model.get("model_type"),
        "candidate_price_target": "conditional_median",
        "purchase_gate": "strict_prior_realized_roi_lcb_required",
        "purchase_max_probability_rank": PURCHASE_MAX_PROBABILITY_RANK,
        "trained_through": cutoff,
    }
    combined_artifact["artifact_sha256"] = _canonical_sha256(combined_artifact)
    price_metrics = closing_odds_quantile_metrics(evaluation, price_model)
    return {
        **probability,
        "model": MODEL_NAME,
        "probability_model": probability.get("model"),
        "price_training_status": "ready",
        "price_teacher": "official_closing_odds_strictly_on_or_before_calibration_through",
        "price_feature_probability_source": "source_cache_frozen_model_probabilities",
        "price_model_outer_period_used_for_selection": False,
        "closing_odds_model": price_model,
        "closing_odds_holdout_metrics": price_metrics,
        "probability_artifact": probability_artifact,
        "artifact": combined_artifact,
        "terminal_value_candidate_diagnostic": _candidate_diagnostic(
            evaluation,
            probability_artifact,
            price_model,
        ),
        "promotion_eligible": False,
        "real_betting_enabled": False,
    }


def train_from_scored_cache(
    cache_path: Path,
    *,
    calibration_through: str,
    minimum_training_days: int,
    minimum_training_races: int,
    num_threads: int = 4,
) -> dict[str, Any]:
    source = joblib.load(cache_path)
    if not isinstance(source, Mapping):
        raise ValueError("decision-time V46 cache must contain a mapping")
    races = source.get("races")
    contract = source.get("contract")
    if not isinstance(races, list) or not isinstance(contract, Mapping):
        raise ValueError("decision-time V46 cache is missing races or contract")
    validate_decision_scored_cache_contract(contract)
    fitted = fit_decision_time_stacked_terminal_market(
        races,
        calibration_through=calibration_through,
        minimum_training_days=minimum_training_days,
        minimum_training_races=minimum_training_races,
        num_threads=num_threads,
    )
    training_status = str(fitted.pop("status"))
    return {
        "status": "completed",
        "model": MODEL_NAME,
        "evaluation_version": 1,
        "training_status": training_status,
        "decision": (
            "freeze_candidate_ranker_for_strict_prior_realized_roi_calibration"
            if training_status == "ready"
            else "insufficient_data"
        ),
        "promotion_eligible": False,
        "real_betting_enabled": False,
        "source_scored_cache": str(cache_path),
        "source_scored_cache_sha256": _file_sha256(cache_path),
        "source_cache_contract": dict(contract),
        **fitted,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze V44 probabilities with contextual terminal-price ranking"
    )
    parser.add_argument("--scored-cache", type=Path, required=True)
    parser.add_argument("--calibration-through", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-training-days", type=int, default=30)
    parser.add_argument("--minimum-training-races", type=int, default=3000)
    parser.add_argument("--num-threads", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = train_from_scored_cache(
        args.scored_cache,
        calibration_through=args.calibration_through,
        minimum_training_days=args.minimum_training_days,
        minimum_training_races=args.minimum_training_races,
        num_threads=args.num_threads,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps({
        "output": str(args.output),
        "training_status": result["training_status"],
        "selected_stack": result.get("selected_stack"),
        "price_training_status": result.get("price_training_status"),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
