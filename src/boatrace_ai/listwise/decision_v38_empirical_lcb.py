from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from datetime import date, datetime
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

import joblib

from ..bankroll_bootstrap import bootstrap_daily_roi
from .contextual_empirical_ev_calibration import (
    fit_contextual_empirical_ev_calibration,
)
from .closing_odds_quantile import forecast_closing_odds_quantiles
from .empirical_lcb_policy import (
    policy_edge_records,
    simulate_empirical_lcb_policy,
)
from .nonlinear_market_residual_v38 import nonlinear_residual_probabilities
from .decision_market_residual_v38 import decision_time_race
from .stacked_market_residual_v42 import stacked_probabilities


MODEL_NAME = "decision_stack_contextual_strict_prior_lcb_v45"
SETTLEMENT_ENGINE_CONTRACT = (
    "official_result_gross_roi_v1_previous_calendar_dates_only"
)
PURCHASE_RESIDUAL_SHRINKAGE = 1.0
PURCHASE_MAX_PROBABILITY_RANK = 5
MINIMUM_LEDGER_DAYS = 30
MINIMUM_LEDGER_CANDIDATES = 300
MINIMUM_LEDGER_CANDIDATE_DAYS = 20
MINIMUM_RANK_DAYS = 30
MINIMUM_RANK_CANDIDATES = 300
MINIMUM_CELL_DAYS = 20
MINIMUM_CELL_CANDIDATES = 100


def _iso_date(value: object, name: str) -> str:
    text = str(value).strip()[:10]
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise ValueError(f"{name} must start with an ISO date") from exc


def _stable_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _aware_timestamp(value: object, name: str) -> datetime:
    text = str(value or "").strip()
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return parsed


def _warmup_denial_reasons(reasons: tuple[str, ...]) -> list[str]:
    mapping = {
        "insufficient_training_days": "WARMUP_CALENDAR_DAYS",
        "insufficient_tickets": "WARMUP_CANDIDATES",
        "insufficient_candidate_days": "WARMUP_CANDIDATE_DAYS",
        "insufficient_calendar_span_days": "WARMUP_CALENDAR_DAYS",
    }
    return [mapping.get(reason, str(reason).upper()) for reason in reasons]


def _calendar_span_days(records: list[dict[str, Any]]) -> int:
    dates = sorted({str(row["race_date"]) for row in records})
    if not dates:
        return 0
    return (date.fromisoformat(dates[-1]) - date.fromisoformat(dates[0])).days + 1


def _identity_probability_blender(
    model: Mapping[str, float],
    _market: Mapping[str, float],
    *,
    model_weight: float,
    temperature: float,
) -> dict[str, float]:
    if model_weight != 1.0 or temperature != 1.0:
        raise ValueError("V39 requires the frozen V38 probability distribution")
    values = {str(key): float(value) for key, value in model.items()}
    total = sum(values.values())
    if total <= 0.0:
        raise ValueError("V39 probability distribution has no mass")
    return {key: value / total for key, value in values.items()}


def score_frozen_v38_races(
    races: list[dict[str, Any]],
    frozen: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if frozen.get("training_status") != "ready":
        raise ValueError("V39 requires a ready frozen decision-time V38 artifact")
    if frozen.get("official_closing_fields_used") is not False:
        raise ValueError("V39 refuses an artifact without a no-closing-fields audit")
    artifact = frozen.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError("V39 frozen V38 artifact is missing")
    model = str(frozen.get("model") or "")
    if model == "decision_time_stacked_terminal_market_v46":
        probability_artifact = artifact.get("probability_artifact")
        price_model = artifact.get("closing_odds_model")
        if not isinstance(probability_artifact, Mapping):
            raise ValueError("V46 frozen probability artifact is missing")
        if not isinstance(price_model, Mapping):
            raise ValueError("V46 frozen closing-odds model is missing")
        scored = []
        for race in races:
            # Forecast with the source-cache probability feature before
            # replacing it with the frozen V44 probability distribution.
            terminal = forecast_closing_odds_quantiles(
                race, dict(price_model)
            )["q50"]
            probabilities = stacked_probabilities(
                race, probability_artifact
            )
            scored.append({
                **race,
                "model_probabilities": probabilities,
                "estimated_final_odds": terminal,
                "closing_odds_forecast_target": "conditional_median",
            })
        return scored
    if model == "decision_time_stacked_market_residual_v44":
        scorer = lambda race: stacked_probabilities(race, artifact)
    elif model == "decision_time_nonlinear_market_residual_v38":
        scorer = lambda race: nonlinear_residual_probabilities(
            race,
            artifact,
            shrinkage=PURCHASE_RESIDUAL_SHRINKAGE,
        )
    else:
        raise ValueError(f"V39 refuses unsupported frozen model: {model}")
    return [
        {**race, "model_probabilities": scorer(race)}
        for race in races
    ]


def _aggregate_daily(daily: list[dict[str, Any]]) -> dict[str, Any]:
    stake = sum(int(row.get("stake_yen") or 0) for row in daily)
    returned = sum(int(row.get("return_yen") or 0) for row in daily)
    tickets = sum(int(row.get("tickets") or 0) for row in daily)
    hit_tickets = sum(int(row.get("hit_tickets") or 0) for row in daily)
    largest_hit_return = max(
        (int(row.get("largest_hit_return_yen") or 0) for row in daily),
        default=0,
    )
    roi_without_largest_hit = (returned - largest_hit_return) / stake if stake else None
    confidence = (
        bootstrap_daily_roi(daily)
        if stake > 0
        else {
            "roi": None,
            "roi_ci95_lower": None,
            "roi_ci95_upper": None,
            "probability_roi_above_one": None,
        }
    )
    cumulative = peak = max_drawdown = 0
    for row in daily:
        cumulative += int(row.get("profit_yen") or 0)
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)
    return {
        "status": "ready" if stake > 0 else "no_authorized_purchases",
        "evaluation_days": len(daily),
        "tickets": tickets,
        "hit_tickets": hit_tickets,
        "stake_yen": stake,
        "return_yen": returned,
        "profit_yen": returned - stake,
        "roi": returned / stake if stake else None,
        "roi_display": returned / stake if stake else "N/A",
        "largest_hit_return_yen": largest_hit_return,
        "roi_without_largest_hit": roi_without_largest_hit,
        "largest_hit_excluded_roi_above_one": (
            roi_without_largest_hit > 1.0
            if roi_without_largest_hit is not None else None
        ),
        "roi_ci95_lower": confidence.get("roi_ci95_lower"),
        "roi_ci95_upper": confidence.get("roi_ci95_upper"),
        "probability_roi_above_one": confidence.get(
            "probability_roi_above_one"
        ),
        "max_drawdown_yen": max_drawdown,
        "daily": daily,
    }


def _value_decile_calibration(
    decisions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    eligible = [
        row for row in decisions
        if row.get("calibrated_roi") is not None
        and isfinite(float(row["calibrated_roi"]))
        and row.get("raw_estimated_ev") is not None
        and isfinite(float(row["raw_estimated_ev"]))
    ]
    eligible.sort(key=lambda row: float(row["raw_estimated_ev"]))
    if not eligible:
        return []
    rows: list[dict[str, Any]] = []
    size = len(eligible)
    for decile in range(1, 11):
        group = [
            row for index, row in enumerate(eligible)
            if min(9, index * 10 // size) == decile - 1
        ]
        if not group:
            continue
        daily: dict[str, dict[str, Any]] = {}
        for row in group:
            day = str(row["race_date"])
            aggregate = daily.setdefault(day, {
                "race_date": day,
                "stake_yen": 0,
                "return_yen": 0,
            })
            aggregate["stake_yen"] += 100
            aggregate["return_yen"] += int(
                round(100.0 * float(row["realized_gross_roi"]))
            )
        confidence = bootstrap_daily_roi(list(daily.values()))
        stake = 100 * len(group)
        returned = sum(
            int(round(100.0 * float(row["realized_gross_roi"])))
            for row in group
        )
        rows.append({
            "decile": decile,
            "candidate_portfolios": len(group),
            "evaluation_days": len(daily),
            "minimum_purchase_value": min(float(row["raw_estimated_ev"]) for row in group),
            "maximum_purchase_value": max(float(row["raw_estimated_ev"]) for row in group),
            "predicted_roi": sum(float(row["calibrated_roi"]) for row in group) / len(group),
            "realized_roi": returned / stake,
            "daily_block_roi_lower_95": confidence.get("roi_ci95_lower"),
            "profit_yen": returned - stake,
        })
    return rows


def walk_forward_decision_v38_lcb(
    races: list[dict[str, Any]],
    frozen: Mapping[str, Any],
    *,
    registered_after: str,
    daily_budget_yen: int = 10_000,
    minimum_ledger_days: int = MINIMUM_LEDGER_DAYS,
    minimum_ledger_candidates: int = MINIMUM_LEDGER_CANDIDATES,
    minimum_ledger_candidate_days: int = MINIMUM_LEDGER_CANDIDATE_DAYS,
    minimum_local_candidates: int = 50,
    minimum_local_candidate_days: int = 20,
    minimum_local_ess: float = 10.0,
    bootstrap_samples: int = 5_000,
    purchase_safety_margin: float = 0.0,
    purchase_max_probability_rank: int = PURCHASE_MAX_PROBABILITY_RANK,
) -> dict[str, Any]:
    if not isfinite(float(purchase_safety_margin)) or purchase_safety_margin < 0:
        raise ValueError("purchase_safety_margin must be finite and nonnegative")
    if not 1 <= int(purchase_max_probability_rank) <= 120:
        raise ValueError("purchase_max_probability_rank must be between 1 and 120")
    purchase_max_rank = int(purchase_max_probability_rank)
    buy_threshold = 1.0 + float(purchase_safety_margin)
    training_through = _iso_date(frozen.get("training_through"), "training_through")
    registration = _iso_date(registered_after, "registered_after")
    if registration < training_through:
        raise ValueError("V39 registration cannot precede frozen model training")
    selection_through = _iso_date(
        frozen.get("evaluation_through") or training_through,
        "evaluation_through",
    )
    if registration < selection_through:
        raise ValueError("V39 registration cannot precede challenger selection data")
    scored = score_frozen_v38_races(races, frozen)
    by_day: dict[str, list[dict[str, Any]]] = {}
    for race in scored:
        race_date = _iso_date(race.get("race_date"), "race_date")
        # Neither model-fitting data nor the holdout used to choose/register
        # this frozen challenger may seed its purchase calibration ledger.
        if race_date <= registration:
            continue
        by_day.setdefault(race_date, []).append(race)

    ledger: list[dict[str, Any]] = []
    prospective_daily: list[dict[str, Any]] = []
    fold_audit: list[dict[str, Any]] = []
    all_candidate_decisions: list[dict[str, Any]] = []
    latest_calibrator = None
    calibrator = {"model_weight": 1.0, "temperature": 1.0}
    source_hash = str(
        frozen.get("source_scored_cache_sha256")
        or frozen.get("artifact", {}).get("booster_sha256")
        or ""
    )
    settlement_engine_hash = _stable_hash(SETTLEMENT_ENGINE_CONTRACT)
    parent_artifact_hash = str(
        frozen.get("artifact_sha256") or _stable_hash(frozen)
    )
    joint_scenario_model_hash = frozen.get("joint_scenario_model_hash")
    portfolio_policy_hash = _stable_hash({
        "policy_model": MODEL_NAME,
        "maximum_probability_rank": purchase_max_rank,
        "residual_shrinkage": PURCHASE_RESIDUAL_SHRINKAGE,
        "daily_budget_yen": daily_budget_yen,
        "purchase_safety_margin": purchase_safety_margin,
    })
    evaluation_protocol_id = _stable_hash({
        "registration": registration,
        "selection_through": selection_through,
        "strict_prior": True,
        "warmup": [
            minimum_ledger_days,
            minimum_ledger_candidates,
            minimum_ledger_candidate_days,
        ],
        "calibration_target": "gross_roi_including_principal",
    })
    resampling_condition_id = _stable_hash({
        "method": "ordinary_day_cluster_bootstrap",
        "samples": bootstrap_samples,
        "confidence": 0.95,
        "local_minimums": [
            minimum_local_candidates,
            minimum_local_candidate_days,
            minimum_local_ess,
        ],
    })
    latest_calibrator_hash: str | None = None
    latest_ledger_hash: str | None = None
    for evaluation_date in sorted(by_day):
        if any(str(row["race_date"]) >= evaluation_date for row in ledger):
            raise AssertionError("V39 ledger contains a non-prior settlement")
        latest_calibrator = fit_contextual_empirical_ev_calibration(
            ledger,
            prediction_date=evaluation_date,
            bootstrap_samples=bootstrap_samples,
            min_days=min(minimum_ledger_days, minimum_ledger_candidate_days),
            min_tickets=minimum_ledger_candidates,
            min_candidate_days=minimum_ledger_candidate_days,
            min_local_candidates=minimum_local_candidates,
            min_local_candidate_days=minimum_local_candidate_days,
            min_local_ess=minimum_local_ess,
            candidate_min_raw_ev=0.0,
            min_rank_days=max(minimum_ledger_days, MINIMUM_RANK_DAYS),
            min_rank_tickets=max(
                minimum_ledger_candidates, MINIMUM_RANK_CANDIDATES
            ),
            min_cell_days=max(
                minimum_local_candidate_days, MINIMUM_CELL_DAYS
            ),
            min_cell_tickets=max(
                minimum_local_candidates, MINIMUM_CELL_CANDIDATES
            ),
            rank_prior_tickets=500.0,
            cell_prior_tickets=200.0,
        )
        prior_calendar_span_days = _calendar_span_days(ledger)
        prior_candidate_dates = {
            str(row["race_date"]) for row in ledger
        }
        strict_ready_reasons = list(latest_calibrator.ready_reasons)
        if prior_calendar_span_days < minimum_ledger_days:
            strict_ready_reasons.append("insufficient_calendar_span_days")
        strict_ready_reasons = list(dict.fromkeys(strict_ready_reasons))
        strict_calibration_ready = bool(
            latest_calibrator.ready and not strict_ready_reasons
        )
        day_races = by_day[evaluation_date]
        simulation = simulate_empirical_lcb_policy(
            day_races,
            calibrator,
            _identity_probability_blender,
            latest_calibrator,
            daily_budget_yen,
            max_rank=purchase_max_rank,
            buy_threshold=buy_threshold,
            purchase_gate_enabled=strict_calibration_ready,
            purchase_gate_denial_reason=(
                _warmup_denial_reasons(tuple(strict_ready_reasons))[0]
                if strict_ready_reasons else "CALIBRATION_NOT_READY"
            ),
        )
        current = policy_edge_records(
            day_races,
            calibrator,
            _identity_probability_blender,
            max_rank=purchase_max_rank,
        )
        if any(str(row["race_date"]) != evaluation_date for row in current):
            raise AssertionError("V39 admitted a mismatched settlement batch")
        decision_times = [
            _aware_timestamp(row.get("decision_time"), "decision_time")
            for row in current
        ]
        prior_settlement_times = [
            _aware_timestamp(row.get("settlement_time"), "settlement_time")
            for row in ledger
        ]
        earliest_decision = min(decision_times) if decision_times else None
        latest_decision = max(decision_times) if decision_times else None
        max_prior_settlement = (
            max(prior_settlement_times) if prior_settlement_times else None
        )
        strict_prior_violation_count = sum(
            max_prior_settlement is not None and max_prior_settlement >= decision
            for decision in decision_times
        )
        if strict_prior_violation_count:
            raise AssertionError("V45 calibration ledger violates decision time")
        calibrator_hash = _stable_hash(latest_calibrator.as_dict())
        ledger_hash = _stable_hash(ledger)
        latest_calibrator_hash = calibrator_hash
        latest_ledger_hash = ledger_hash
        decision_contract_hash = _stable_hash({
            "model_hash": source_hash,
            "calibrator_hash": calibrator_hash,
            "ledger_hash": ledger_hash,
            "purchase_threshold": buy_threshold,
            "purchase_threshold_unit": "gross_roi_including_principal",
            "maximum_probability_rank": purchase_max_rank,
            "residual_shrinkage": PURCHASE_RESIDUAL_SHRINKAGE,
            "settlement_engine_hash": settlement_engine_hash,
        })
        candidate_decisions = [
            dict(row)
            for day in simulation.get("daily") or ()
            for row in day.get("candidate_decision_audit") or ()
            if isinstance(row, Mapping)
        ]
        current_by_key = {
            (str(row["race_id"]), str(row["combination"])): row
            for row in current
        }
        if not strict_calibration_ready:
            warmup_reasons = _warmup_denial_reasons(
                tuple(strict_ready_reasons)
            ) or ["CALIBRATION_NOT_READY"]
            candidate_decisions = [
                {
                    "race_id": row["race_id"],
                    "combination": row["combination"],
                    "probability_rank": row["probability_rank"],
                    "forecast_odds": row["forecast_odds"],
                    "raw_estimated_ev": row["raw_estimated_ev"],
                    "calibrated_roi": None,
                    "calibrated_roi_lcb95": None,
                    "buy_threshold": buy_threshold,
                    "purchase_gate_approved": False,
                    "denial_reason": warmup_reasons[0],
                    "denial_reasons": warmup_reasons,
                }
                for row in current
            ]
        for decision in candidate_decisions:
            source = current_by_key.get(
                (str(decision.get("race_id")), str(decision.get("combination")))
            )
            if source is None:
                raise AssertionError("V45 decision is absent from pregate ledger")
            decision_time = _aware_timestamp(
                source.get("decision_time"), "decision_time"
            )
            decision.update({
                "race_date": source["race_date"],
                "realized_gross_roi": source["gross_return_per_yen"],
                "hit": source["hit"],
                "decision_time_source": source["decision_time_source"],
                "settlement_time": source["settlement_time"],
                "settlement_time_source": source["settlement_time_source"],
                "decision_time": decision_time.isoformat(),
                "max_prior_settlement_time": (
                    max_prior_settlement.isoformat()
                    if max_prior_settlement is not None else None
                ),
                "strict_prior_check": bool(
                    max_prior_settlement is None
                    or max_prior_settlement < decision_time
                ),
                "strict_prior_violation_count": int(
                    max_prior_settlement is not None
                    and max_prior_settlement >= decision_time
                ),
                "future_candidate_in_calibration_count": 0,
                "calibrator_hash": calibrator_hash,
                "calibration_ledger_hash": ledger_hash,
                "decision_contract_hash": decision_contract_hash,
                "prior_calendar_span_days": prior_calendar_span_days,
                "prior_observed_race_days": len(prior_candidate_dates),
                "prior_settled_candidate_days": len(prior_candidate_dates),
                "prior_calibration_eligible_days": len(prior_candidate_dates),
                "prior_candidates": len(ledger),
                "prior_candidate_days": len(prior_candidate_dates),
                "required_calendar_days": minimum_ledger_days,
                "required_candidates": minimum_ledger_candidates,
                "required_candidate_days": minimum_ledger_candidate_days,
                "calibration_target": "gross_roi_including_principal",
                "candidate_population": "all_pregate_probability_ranked",
            })
        all_candidate_decisions.extend(candidate_decisions)
        denial_reasons = Counter(
            str(row.get("denial_reason") or "unspecified")
            for row in candidate_decisions
            if row.get("purchase_gate_approved") is not True
        )
        approved_decisions = sum(
            row.get("purchase_gate_approved") is True
            for row in candidate_decisions
        )
        numeric_points = [
            float(row[key])
            for row in candidate_decisions
            for key in ("calibrated_roi",)
            if row.get(key) is not None and isfinite(float(row[key]))
        ]
        numeric_raw = [
            float(row["raw_estimated_ev"])
            for row in candidate_decisions
            if row.get("raw_estimated_ev") is not None
            and isfinite(float(row["raw_estimated_ev"]))
        ]
        numeric_lcbs = [
            float(row[key])
            for row in candidate_decisions
            for key in ("calibrated_roi_lcb95",)
            if row.get(key) is not None and isfinite(float(row[key]))
        ]
        if evaluation_date > registration:
            prospective_daily.extend(simulation["daily"])
        race_calibrator_counts = {
            race_id: len({str(row["calibrator_hash"]) for row in candidate_decisions
                         if str(row["race_id"]) == race_id})
            for race_id in {str(row["race_id"]) for row in candidate_decisions}
        }
        fold_audit.append({
            "evaluation_date": evaluation_date,
            "prospective_evidence": evaluation_date > registration,
            "calibration_cutoff_date": (
                latest_calibrator.trained_through_date
            ),
            "max_training_settlement_date": (
                latest_calibrator.trained_through_date
            ),
            "decision_time_earliest": (
                earliest_decision.isoformat() if earliest_decision else None
            ),
            "decision_time_latest": (
                latest_decision.isoformat() if latest_decision else None
            ),
            "max_prior_settlement_time": (
                max_prior_settlement.isoformat()
                if max_prior_settlement is not None else None
            ),
            "strict_prior_check": strict_prior_violation_count == 0,
            "prior_candidates": latest_calibrator.tickets,
            "prior_days": latest_calibrator.training_days,
            "prior_candidate_days": latest_calibrator.candidate_days,
            "prior_calendar_span_days": prior_calendar_span_days,
            "prior_observed_race_days": len(prior_candidate_dates),
            "prior_settled_candidate_days": len(prior_candidate_dates),
            "prior_calibration_eligible_days": len(prior_candidate_dates),
            "required_calendar_days": minimum_ledger_days,
            "required_candidates": minimum_ledger_candidates,
            "required_candidate_days": minimum_ledger_candidate_days,
            "calibration_ready": strict_calibration_ready,
            "ready_reasons": strict_ready_reasons,
            "authorized_tickets": simulation.get("tickets"),
            "stake_yen": simulation.get("stake_yen"),
            "pregate_candidates": len(current),
            "race_count": len(day_races),
            "candidate_count": len(current),
            "candidate_decisions": (
                len(candidate_decisions)
                if strict_calibration_ready
                else len(current)
            ),
            "purchase_gate_approved_candidates": approved_decisions,
            "purchase_gate_denied_candidates": sum(denial_reasons.values()),
            "denial_reason_counts": dict(sorted(denial_reasons.items())),
            "maximum_calibrated_roi": (
                max(numeric_points) if numeric_points else None
            ),
            "maximum_raw_estimated_ev": (
                max(numeric_raw) if numeric_raw else None
            ),
            "maximum_calibrated_roi_lcb95": (
                max(numeric_lcbs) if numeric_lcbs else None
            ),
            "buy_threshold": buy_threshold,
            "purchase_safety_margin": purchase_safety_margin,
            "purchase_threshold_unit": "gross_roi_including_principal",
            "approval_rule": (
                "warmup_and_local_support_and_calibrated_gross_roi_"
                "lcb95_above_one_plus_margin"
            ),
            "strict_prior_violation_count": strict_prior_violation_count,
            "future_candidate_in_calibration_count": 0,
            "same_race_calibrator_hash_count_max": (
                max(race_calibrator_counts.values())
                if race_calibrator_counts else 0
            ),
            "same_race_mid_decision_update_count": 0,
            "same_race_result_leakage_count": 0,
            "frozen_model_hash": source_hash,
            "calibrator_hash": calibrator_hash,
            "calibration_ledger_hash": ledger_hash,
            "settlement_engine_hash": settlement_engine_hash,
            "decision_contract_hash": decision_contract_hash,
        })
        ledger.extend(current)

    bankroll = _aggregate_daily(prospective_daily)
    result = {
        "model": MODEL_NAME,
        "status": "completed",
        "promotion_eligible": False,
        "real_betting_enabled": False,
        "registered_after": registration,
        "selection_evaluation_through": selection_through,
        "frozen_model_training_through": training_through,
        "frozen_model_hash": source_hash,
        "frozen_probability_model": frozen.get("model"),
        "settlement_engine_contract": SETTLEMENT_ENGINE_CONTRACT,
        "settlement_engine_hash": settlement_engine_hash,
        "purchase_residual_shrinkage": PURCHASE_RESIDUAL_SHRINKAGE,
        "candidate_population": (
            f"all_probability_top{purchase_max_rank}_before_purchase_gate"
        ),
        "candidate_population_includes_denied": True,
        "purchase_max_probability_rank": purchase_max_rank,
        "same_race_update_rule": (
            "one prior calibrator for the complete race batch; settlement added "
            "only after every decision in that date batch"
        ),
        "warmup": {
            "logical_operator": "AND",
            "minimum_training_calendar_days": minimum_ledger_days,
            "minimum_pregate_candidates": minimum_ledger_candidates,
            "minimum_candidate_days": minimum_ledger_candidate_days,
        },
        "calibration_target": "gross ROI including principal",
        "purchase_safety_margin": purchase_safety_margin,
        "buy_threshold": buy_threshold,
        "purchase_threshold": (
            "calibrated_gross_ROI_LCB95 > 1 + purchase_safety_margin"
        ),
        "formal_purchase_rule": {
            "warmup_operator": "AND",
            "warmup_calendar_days": minimum_ledger_days,
            "warmup_candidates": minimum_ledger_candidates,
            "warmup_candidate_days": minimum_ledger_candidate_days,
            "calibration_target": "gross_roi_including_principal",
            "lcb_confidence": 0.95,
            "threshold": buy_threshold,
            "raw_v_buy_is_diagnostic_only": True,
        },
        "range_policy": (
            "deny outside local isotonic block support or when rank-by-odds "
            "cell support is not ready"
        ),
        "contextual_dimensions": ["probability_rank", "forecast_odds"],
        "contextual_hierarchy": "global -> probability-rank -> rank-by-odds",
        "bootstrap_cluster_unit": "race_date",
        "ticket_level_independence_assumed": False,
        "ledger_candidates": len(ledger),
        "race_count": sum(len(rows) for rows in by_day.values()),
        "candidate_count": len(ledger),
        "calendar_span_days": _calendar_span_days(ledger),
        "observed_race_days": len({
            str(row["race_date"]) for row in ledger
        }),
        "candidate_days": len({str(row["race_date"]) for row in ledger}),
        "settled_candidate_days": len({
            str(row["race_date"]) for row in ledger
        }),
        "calibration_eligible_days": len({str(row["race_date"]) for row in ledger}),
        "ledger_hash": _stable_hash(ledger),
        "candidate_decision_audit": all_candidate_decisions,
        "purchase_value_calibration": _value_decile_calibration(all_candidate_decisions),
        "strict_prior_violation_count": sum(
            int(row["strict_prior_violation_count"])
            for row in all_candidate_decisions
        ),
        "future_candidate_in_calibration_count": 0,
        "same_race_calibrator_hash_count_max": max(
            (
                int(row["same_race_calibrator_hash_count_max"])
                for row in fold_audit
            ),
            default=0,
        ),
        "same_race_mid_decision_update_count": 0,
        "same_race_result_leakage_count": 0,
        "fold_audit": fold_audit,
        "latest_calibrator": (
            latest_calibrator.as_dict() if latest_calibrator is not None else None
        ),
        "bankroll": bankroll,
        "artifact_lineage": {
            "parent_artifact_hash": parent_artifact_hash,
            "prediction_model_hash": source_hash or None,
            "joint_scenario_model_hash": joint_scenario_model_hash,
            "calibrator_hash": latest_calibrator_hash,
            "calibration_ledger_hash": latest_ledger_hash,
            "portfolio_policy_hash": portfolio_policy_hash,
            "payout_engine_hash": settlement_engine_hash,
            "evaluation_protocol_id": evaluation_protocol_id,
            "resampling_condition_id": resampling_condition_id,
            "source_revision": frozen.get("source_revision"),
            "outer_draw_definition": (
                frozen.get("outer_draw_definition")
            ),
            "inner_scenario_definition": (
                frozen.get("inner_scenario_definition")
            ),
            "lineage_complete": all((
                source_hash,
                joint_scenario_model_hash,
                frozen.get("source_revision"),
                frozen.get("outer_draw_definition"),
                frozen.get("inner_scenario_definition"),
            )),
        },
    }
    drawdown_limit = frozen.get("maximum_drawdown_limit_yen")
    prediction_noninferiority = bool(
        frozen.get("prediction_noninferiority_pass")
        or frozen.get("challenger_selection_gate_pass")
        or frozen.get("prediction_deployment_eligible")
    )
    result["formal_promotion_gate"] = {
        "purchase_contract_audit_passed": bool(
            result["strict_prior_violation_count"] == 0
            and result["future_candidate_in_calibration_count"] == 0
            and result["same_race_calibrator_hash_count_max"] <= 1
            and result["same_race_mid_decision_update_count"] == 0
            and result["same_race_result_leakage_count"] == 0
        ),
        "warmup_completed": any(
            bool(row["calibration_ready"]) for row in fold_audit
        ),
        "post_warmup_approval_observed": any(
            row.get("purchase_gate_approved") is True
            for row in all_candidate_decisions
        ),
        "minimum_evaluation_days_passed": (
            int(bankroll.get("evaluation_days") or 0) >= 30
        ),
        "day_block_roi_q05_above_one": (
            bankroll.get("roi_ci95_lower") is not None
            and float(bankroll["roi_ci95_lower"]) > 1.0
        ),
        "positive_profit": int(bankroll.get("profit_yen") or 0) > 0,
        "prediction_noninferiority_passed": prediction_noninferiority,
        "maximum_drawdown_audit_passed": (
            drawdown_limit is not None
            and int(bankroll.get("max_drawdown_yen") or 0)
            <= int(drawdown_limit)
        ),
        "largest_hit_excluded_roi_above_one": bankroll.get(
            "largest_hit_excluded_roi_above_one"
        ),
    }
    result["formal_promotion_rule"] = {
        "day_block_roi_quantile": 0.05,
        "day_block_roi_threshold": 1.0,
        "probability_roi_above_one_is_diagnostic_only": True,
        "maximum_drawdown_limit_yen": drawdown_limit,
        "requires_positive_profit": True,
        "requires_prediction_noninferiority": True,
        "requires_largest_hit_excluded_roi_above_one": True,
    }
    result["promotion_eligible"] = all(
        value is True for value in result["formal_promotion_gate"].values()
    )
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def evaluate_from_files(
    frozen_path: Path,
    cache_path: Path,
    *,
    registered_after: str,
    daily_budget_yen: int = 10_000,
    purchase_max_probability_rank: int = PURCHASE_MAX_PROBABILITY_RANK,
) -> dict[str, Any]:
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    cache = joblib.load(cache_path)
    if not isinstance(frozen, Mapping):
        raise ValueError("V39 frozen artifact JSON must contain a mapping")
    if not isinstance(cache, Mapping) or not isinstance(cache.get("races"), list):
        raise ValueError("V39 scored cache is missing races")
    races = [decision_time_race(race) for race in cache["races"]]
    result = walk_forward_decision_v38_lcb(
        races,
        frozen,
        registered_after=registered_after,
        daily_budget_yen=daily_budget_yen,
        purchase_max_probability_rank=purchase_max_probability_rank,
    )
    return {
        **result,
        "source_frozen_artifact": str(frozen_path),
        "source_frozen_artifact_sha256": _file_sha256(frozen_path),
        "source_scored_cache": str(cache_path),
        "source_scored_cache_sha256": _file_sha256(cache_path),
        "source_cache_contract": dict(cache.get("contract") or {}),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen decision V38 with strict-prior empirical LCB"
    )
    parser.add_argument("--frozen-artifact", type=Path, required=True)
    parser.add_argument("--scored-cache", type=Path, required=True)
    parser.add_argument("--registered-after", required=True)
    parser.add_argument("--daily-budget-yen", type=int, default=10_000)
    parser.add_argument(
        "--purchase-max-probability-rank",
        type=int,
        default=PURCHASE_MAX_PROBABILITY_RANK,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate_from_files(
        args.frozen_artifact,
        args.scored_cache,
        registered_after=args.registered_after,
        daily_budget_yen=args.daily_budget_yen,
        purchase_max_probability_rank=args.purchase_max_probability_rank,
    )
    _write_json_atomic(args.output, result)
    print(json.dumps({
        "output": str(args.output),
        "evaluation_days": result["bankroll"]["evaluation_days"],
        "tickets": result["bankroll"]["tickets"],
        "roi": result["bankroll"]["roi"],
        "promotion_eligible": result["promotion_eligible"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
