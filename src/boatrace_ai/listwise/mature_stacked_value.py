from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Mapping

from ..bankroll_bootstrap import (
    BOOTSTRAP_QUANTILE_METHOD,
    bootstrap_daily_roi,
)
from .closing_odds import decision_odds
from .contextual_empirical_ev_calibration import (
    fit_contextual_empirical_ev_calibration,
)
from .empirical_lcb_policy import (
    empirical_bankroll_promotion_eligible,
    policy_edge_records,
    race_settlement_map,
    simulate_empirical_lcb_policy,
)
from .nested_nonlinear_value_v40 import value_decile_audit
from .stacked_market_residual_v42 import (
    fit_temporal_stacked_market_residual,
    stacked_metrics,
    stacked_probabilities,
)


MODEL_NAME = "mature_stacked_contextual_value_rank20"
DAILY_REFIT_MODEL_NAME = "mature_stacked_contextual_value_daily_refit_rank20"
MODEL_TRAINING_MINIMUM_DAYS = 60
VALUE_STACK_SELECTION_DAYS = 60
VALUE_CALIBRATION_DAYS = 60
PURCHASE_MAX_RANK = 20
PURCHASE_MAX_TICKETS_PER_RACE = 1
CONTEXT_AUDIT_BOOTSTRAP_SAMPLES = 5_000
VALUE_ALIGNED_STACK_POLICY_ID = (
    "top20_max_raw_ev_familywise_q01_top5_noninferiority_disjoint_v2"
)
VALUE_ALIGNED_STACK_BOOTSTRAP_SAMPLES = 5_000
VALUE_ALIGNED_STACK_SELECTION_LOWER_QUANTILE = 0.01
VALUE_ALIGNED_STACK_MIN_DAYS = 50
VALUE_ALIGNED_STACK_MIN_TICKETS = 300
CONTEXT_RANK_GROUPS = (("top5", 1, 5), ("6-20", 6, 20))
CONTEXT_ODDS_BANDS = (
    ("<20", 0.0, 20.0),
    ("20-50", 20.0, 50.0),
    ("50-101", 50.0, 101.0),
    (">=101", 101.0, float("inf")),
)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ledger_sha256(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _parse_timestamp(value: Any) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _strict_prior_artifact_audit(
    calibration_ledger: list[dict[str, Any]],
    evaluation_ledger: list[dict[str, Any]],
) -> dict[str, Any]:
    maximum_prior_settlement = max(
        (_parse_timestamp(row["settlement_time"]) for row in calibration_ledger),
        default=None,
    )
    evaluation_decisions = [
        _parse_timestamp(row["decision_time"]) for row in evaluation_ledger
    ]
    calibration_races = {str(row["race_id"]) for row in calibration_ledger}
    evaluation_races = {str(row["race_id"]) for row in evaluation_ledger}
    violations = sum(
        maximum_prior_settlement is not None
        and maximum_prior_settlement >= decision
        for decision in evaluation_decisions
    )
    return {
        "calibration_cutoff_time": (
            maximum_prior_settlement.isoformat()
            if maximum_prior_settlement is not None else None
        ),
        "max_training_settlement_time": (
            maximum_prior_settlement.isoformat()
            if maximum_prior_settlement is not None else None
        ),
        "minimum_evaluation_decision_time": (
            min(evaluation_decisions).isoformat() if evaluation_decisions else None
        ),
        "strict_prior_check": violations == 0,
        "strict_prior_violation_count": int(violations),
        "same_race_overlap_count": len(calibration_races & evaluation_races),
        "same_race_calibrator_hash_count_max": 1 if evaluation_races else 0,
        "all_pregate_candidates_registered": all(
            row.get("candidate_population") == "all_pregate_probability_ranked"
            for row in calibration_ledger
        ),
    }


def _context_value_rows(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank_group, minimum_rank, maximum_rank in CONTEXT_RANK_GROUPS:
        for odds_band, minimum_odds, maximum_odds in CONTEXT_ODDS_BANDS:
            bucket = [
                record for record in records
                if minimum_rank <= int(record["probability_rank"]) <= maximum_rank
                and minimum_odds <= float(record["forecast_odds"]) < maximum_odds
            ]
            daily: dict[str, dict[str, Any]] = {}
            gross_returns: list[float] = []
            for record in bucket:
                gross = 100.0 * float(record["gross_return_per_yen"])
                gross_returns.append(gross)
                day = str(record["race_date"])
                aggregate = daily.setdefault(day, {
                    "race_date": day,
                    "stake_yen": 0,
                    "return_yen": 0.0,
                })
                aggregate["stake_yen"] += 100
                aggregate["return_yen"] += gross
            confidence = (
                bootstrap_daily_roi(
                    list(daily.values()),
                    samples=CONTEXT_AUDIT_BOOTSTRAP_SAMPLES,
                )
                if bucket
                else {
                    "roi": None,
                    "roi_ci95_lower": None,
                    "probability_roi_above_one": None,
                }
            )
            stake = 100 * len(bucket)
            returned = sum(gross_returns)
            largest_hit = max(gross_returns, default=0.0)
            rows.append({
                "rank_group": rank_group,
                "odds_band": odds_band,
                "minimum_probability_rank": minimum_rank,
                "maximum_probability_rank": maximum_rank,
                "minimum_forecast_odds": minimum_odds,
                "maximum_forecast_odds": (
                    maximum_odds if maximum_odds != float("inf") else None
                ),
                "candidates": len(bucket),
                "candidate_days": len(daily),
                "mean_predicted_raw_ev": (
                    sum(float(record["raw_estimated_ev"]) for record in bucket)
                    / len(bucket)
                    if bucket else None
                ),
                "realized_roi": confidence.get("roi"),
                "realized_roi_lcb95": confidence.get("roi_ci95_lower"),
                "probability_roi_above_one": confidence.get(
                    "probability_roi_above_one"
                ),
                "largest_hit_return_yen": largest_hit if bucket else None,
                "roi_excluding_largest_hit": (
                    (returned - largest_hit) / stake if stake else None
                ),
            })
    return rows


def context_value_audit(
    calibration_records: list[dict[str, Any]],
    evaluation_records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "status": "completed",
        "context_definition": "predeclared_probability_rank_by_forecast_odds",
        "evaluation_used_for_context_definition": False,
        "bootstrap_cluster_unit": "race_date",
        "bootstrap_samples": CONTEXT_AUDIT_BOOTSTRAP_SAMPLES,
        "calibration": _context_value_rows(calibration_records),
        "evaluation": _context_value_rows(evaluation_records),
    }


def _identity_probability_blender(
    model: Mapping[str, float],
    _market: Mapping[str, float],
    *,
    model_weight: float,
    temperature: float,
) -> dict[str, float]:
    if model_weight != 1.0 or temperature != 1.0:
        raise ValueError("mature stacked value requires its frozen distribution")
    return {str(key): float(value) for key, value in model.items()}


def _score(
    races: list[dict[str, Any]], artifact: Mapping[str, Any]
) -> list[dict[str, Any]]:
    return [
        {**race, "model_probabilities": stacked_probabilities(race, artifact)}
        for race in races
    ]


def _value_stack_shortlist(
    probability: Mapping[str, Any],
) -> list[dict[str, Any]]:
    source = probability.get("stack_candidates")
    if not isinstance(source, list):
        return []
    candidates = [
        dict(candidate) for candidate in source
        if isinstance(candidate, Mapping)
        and isinstance(candidate.get("weights"), Mapping)
        and isinstance(candidate.get("metrics"), Mapping)
    ]
    if not candidates:
        return []

    def best(predicate: Any) -> dict[str, Any] | None:
        matches = [candidate for candidate in candidates if predicate(candidate)]
        if not matches:
            return None
        return min(
            matches,
            key=lambda candidate: (
                float(candidate["metrics"]["trifecta_log_loss"]),
                float(candidate["weights"]["linear"])
                + float(candidate["weights"]["nonlinear"]),
                str(candidate["name"]),
            ),
        )

    selected_name = str(probability.get("selected_stack") or "")
    selected: list[dict[str, Any] | None] = [
        best(lambda candidate: str(candidate["name"]) == "market"),
        best(lambda candidate: str(candidate["name"]) == selected_name),
        best(
            lambda candidate: float(candidate["weights"]["linear"]) > 0.0
            and float(candidate["weights"]["nonlinear"]) == 0.0
        ),
        best(
            lambda candidate: float(candidate["weights"]["nonlinear"]) > 0.0
            and float(candidate["weights"]["linear"]) == 0.0
        ),
        best(
            lambda candidate: float(candidate["weights"]["linear"]) > 0.0
            and float(candidate["weights"]["nonlinear"]) > 0.0
        ),
    ]
    unique: dict[str, dict[str, Any]] = {}
    for candidate in selected:
        if candidate is not None:
            unique[str(candidate["name"])] = candidate
    return list(unique.values())


def _value_aligned_artifact(
    base_artifact: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    base_selected_stack: str,
) -> dict[str, Any]:
    artifact = {
        str(key): value
        for key, value in base_artifact.items()
        if str(key) != "artifact_sha256"
    }
    weights = candidate.get("weights")
    if not isinstance(weights, Mapping):
        raise ValueError("value-aligned stack candidate weights are missing")
    artifact.update({
        "selected_stack": str(candidate["name"]),
        "weights": {
            key: float(weights[key])
            for key in ("market", "linear", "nonlinear")
        },
        "pre_value_alignment_stack": base_selected_stack,
        "value_alignment_policy_id": VALUE_ALIGNED_STACK_POLICY_ID,
    })
    encoded = json.dumps(
        artifact, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    artifact["artifact_sha256"] = hashlib.sha256(encoded).hexdigest()
    return artifact


def _value_alignment_metrics(
    races: list[dict[str, Any]],
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    scored = _score(races, artifact)
    daily: dict[str, dict[str, Any]] = {}
    top5_hits = market_top5_hits = hit_tickets = 0
    total_raw_ev = total_return = largest_return = 0.0
    for race in scored:
        probabilities = {
            str(key): float(value)
            for key, value in race["model_probabilities"].items()
        }
        odds = decision_odds(race)
        multipliers = race.get("historical_return_multipliers") or {}
        ranked = sorted(
            probabilities,
            key=lambda combination: (
                -probabilities[combination],
                combination,
            ),
        )[:PURCHASE_MAX_RANK]
        if not ranked:
            continue

        def value_key(combination: str) -> tuple[float, float, str]:
            raw_ev = (
                probabilities[combination]
                * float(odds[combination])
                * float(multipliers.get(combination, 1.0))
            )
            return (-raw_ev, -probabilities[combination], combination)

        selected = min(ranked, key=value_key)
        raw_ev = -value_key(selected)[0]
        settlements = race_settlement_map(race)
        payout_yen = settlements.get(selected)
        hit = payout_yen is not None
        returned = float(payout_yen) if hit else 0.0
        day = str(race["race_date"])
        aggregate = daily.setdefault(day, {
            "race_date": day,
            "stake_yen": 0,
            "return_yen": 0.0,
        })
        aggregate["stake_yen"] += 100
        aggregate["return_yen"] += returned
        total_raw_ev += raw_ev
        total_return += returned
        largest_return = max(largest_return, returned)
        hit_tickets += int(hit)
        top5_hits += int(bool(set(settlements) & set(ranked[:5])))
        market = race["market_probabilities"]
        market_top5 = sorted(
            market,
            key=lambda combination: (-float(market[combination]), combination),
        )[:5]
        market_top5_hits += int(bool(set(settlements) & set(market_top5)))

    tickets = sum(int(row["stake_yen"]) // 100 for row in daily.values())
    confidence = (
        bootstrap_daily_roi(
            list(daily.values()),
            samples=VALUE_ALIGNED_STACK_BOOTSTRAP_SAMPLES,
            lower_quantile=VALUE_ALIGNED_STACK_SELECTION_LOWER_QUANTILE,
        )
        if tickets
        else {
            "roi": None,
            "roi_ci95_lower": None,
            "roi_lower_quantile": VALUE_ALIGNED_STACK_SELECTION_LOWER_QUANTILE,
            "quantile_method": BOOTSTRAP_QUANTILE_METHOD,
            "probability_roi_above_one": None,
        }
    )
    stake_yen = tickets * 100
    return {
        "tickets": tickets,
        "hit_tickets": hit_tickets,
        "candidate_days": len(daily),
        "mean_selected_raw_ev": (
            total_raw_ev / tickets if tickets else None
        ),
        "roi": confidence.get("roi"),
        "roi_lcb95": confidence.get("roi_ci95_lower"),
        "roi_lower_quantile": confidence.get("roi_lower_quantile"),
        "roi_quantile_method": confidence.get("quantile_method"),
        "probability_roi_above_one": confidence.get(
            "probability_roi_above_one"
        ),
        "roi_excluding_largest_hit": (
            (total_return - largest_return) / stake_yen
            if stake_yen else None
        ),
        "trifecta_top5_hit_rate": (
            top5_hits / tickets if tickets else None
        ),
        "market_trifecta_top5_hit_rate": (
            market_top5_hits / tickets if tickets else None
        ),
    }


def select_value_aligned_stack(
    probability: Mapping[str, Any],
    stack_selection_races: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    base_artifact = probability.get("artifact")
    if not isinstance(base_artifact, Mapping):
        raise ValueError("mature probability artifact is missing")
    base_selected = str(probability.get("selected_stack") or "")
    shortlist = _value_stack_shortlist(probability)
    if not shortlist:
        return dict(base_artifact), {
            "policy_id": VALUE_ALIGNED_STACK_POLICY_ID,
            "status": "not_available",
            "base_selected_stack": base_selected,
            "selected_stack": base_selected,
            "candidate_family_size": 0,
            "shortlisted_candidates": 0,
            "stack_selection_shared_with_empirical_gate_training": False,
            "search_validation_draw_sets_disjoint": True,
            "outer_period_used": False,
        }

    rows: list[dict[str, Any]] = []
    artifacts: dict[str, dict[str, Any]] = {}
    for candidate in shortlist:
        name = str(candidate["name"])
        artifact = _value_aligned_artifact(
            base_artifact,
            candidate,
            base_selected_stack=base_selected,
        )
        artifacts[name] = artifact
        metrics = _value_alignment_metrics(stack_selection_races, artifact)
        support_ready = (
            int(metrics["candidate_days"]) >= VALUE_ALIGNED_STACK_MIN_DAYS
            and int(metrics["tickets"]) >= VALUE_ALIGNED_STACK_MIN_TICKETS
        )
        top5_noninferior = (
            metrics["trifecta_top5_hit_rate"] is not None
            and metrics["market_trifecta_top5_hit_rate"] is not None
            and float(metrics["trifecta_top5_hit_rate"])
            >= float(metrics["market_trifecta_top5_hit_rate"])
        )
        rows.append({
            "name": name,
            "weights": dict(candidate["weights"]),
            **metrics,
            "support_ready": support_ready,
            "top5_noninferior_to_market": top5_noninferior,
            "eligible": support_ready and top5_noninferior,
        })

    eligible = [
        row for row in rows
        if row["eligible"] and row["roi_lcb95"] is not None
    ]
    if eligible:
        selected_row = max(
            eligible,
            key=lambda row: (
                float(row["roi_lcb95"]),
                float(row["roi"] if row["roi"] is not None else -1.0),
                float(
                    row["roi_excluding_largest_hit"]
                    if row["roi_excluding_largest_hit"] is not None
                    else -1.0
                ),
                float(row["weights"]["market"]),
                str(row["name"]),
            ),
        )
        selected_name = str(selected_row["name"])
        status = "selected"
    else:
        selected_name = base_selected
        status = "fallback_base_no_eligible_value_candidate"
        if selected_name not in artifacts:
            return dict(base_artifact), {
                "policy_id": VALUE_ALIGNED_STACK_POLICY_ID,
                "status": status,
                "base_selected_stack": base_selected,
                "selected_stack": base_selected,
                "candidate_family_size": len(
                    probability.get("stack_candidates") or []
                ),
                "shortlisted_candidates": len(rows),
                "candidates": rows,
                "stack_selection_shared_with_empirical_gate_training": False,
                "search_validation_draw_sets_disjoint": True,
                "outer_period_used": False,
            }

    for row in rows:
        row["selected"] = str(row["name"]) == selected_name
    return artifacts[selected_name], {
        "policy_id": VALUE_ALIGNED_STACK_POLICY_ID,
        "status": status,
        "selection_objective": (
            "maximum day-block Q05 ROI for one max-raw-EV ticket per race "
            "within probability top20"
        ),
        "base_selected_stack": base_selected,
        "selected_stack": selected_name,
        "candidate_family_size": len(probability.get("stack_candidates") or []),
        "shortlisted_candidates": len(rows),
        "selection_lower_quantile": (
            VALUE_ALIGNED_STACK_SELECTION_LOWER_QUANTILE
        ),
        "selection_quantile_method": BOOTSTRAP_QUANTILE_METHOD,
        "familywise_candidate_cap": 5,
        "minimum_candidate_days": VALUE_ALIGNED_STACK_MIN_DAYS,
        "minimum_tickets": VALUE_ALIGNED_STACK_MIN_TICKETS,
        "top5_noninferiority_required": True,
        "stack_selection_shared_with_empirical_gate_training": False,
        "search_validation_draw_sets_disjoint": True,
        "outer_period_used": False,
        "candidates": rows,
    }


def _fit_mature_contextual_calibrator(
    ledger: list[dict[str, Any]],
    *,
    prediction_date: str,
):
    return fit_contextual_empirical_ev_calibration(
        ledger,
        prediction_date=prediction_date,
        bootstrap_samples=5_000,
        min_days=VALUE_CALIBRATION_DAYS,
        min_tickets=1_000,
        min_candidate_days=40,
        candidate_min_raw_ev=0.0,
        min_rank_days=45,
        min_rank_tickets=1_000,
        min_cell_days=30,
        min_cell_tickets=200,
        rank_prior_tickets=500.0,
        cell_prior_tickets=200.0,
    )


def _simulate_daily_strict_prior_refit(
    evaluation_scored: list[dict[str, Any]],
    initial_ledger: list[dict[str, Any]],
    probability_calibrator: Mapping[str, float],
    *,
    daily_budget_yen: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    """Refit once per day, then admit that day's teachers only after settlement."""
    races_by_day: dict[str, list[dict[str, Any]]] = {}
    for race in evaluation_scored:
        races_by_day.setdefault(str(race["race_date"]), []).append(race)

    ledger = list(initial_ledger)
    daily: list[dict[str, Any]] = []
    refit_audit: list[dict[str, Any]] = []
    totals = {
        "eligible_tickets": 0,
        "tickets": 0,
        "hit_tickets": 0,
        "stake_yen": 0,
        "return_yen": 0,
    }
    cumulative_profit = peak_profit = max_drawdown = 0
    strict_prior_violations = 0
    final_payload: dict[str, Any] = {}
    all_ready = True

    for race_date in sorted(races_by_day):
        day_races = races_by_day[race_date]
        artifact = _fit_mature_contextual_calibrator(
            ledger,
            prediction_date=race_date,
        )
        final_payload = dict(artifact.as_dict())
        all_ready = all_ready and bool(artifact.ready)
        artifact_hash = _canonical_sha256(final_payload)
        prior_ledger_hash = _ledger_sha256(ledger)
        max_prior_record = max(
            ledger,
            key=lambda row: _parse_timestamp(row["settlement_time"]),
        )
        max_prior_settlement_time = str(max_prior_record["settlement_time"])
        minimum_decision_time = min(
            str(
                race.get("odds_deadline_at")
                or race.get("captured_at")
                or f"{race_date}T00:00:00+09:00"
            )
            for race in day_races
        )
        strict_prior = _parse_timestamp(
            max_prior_settlement_time
        ) < _parse_timestamp(minimum_decision_time)
        strict_prior_violations += int(not strict_prior)

        simulation = simulate_empirical_lcb_policy(
            day_races,
            probability_calibrator,
            _identity_probability_blender,
            artifact,
            daily_budget_yen,
            max_rank=PURCHASE_MAX_RANK,
            max_tickets_per_race=PURCHASE_MAX_TICKETS_PER_RACE,
            compute_confidence=False,
        )
        if len(simulation["daily"]) != 1:
            raise ValueError("daily strict-prior refit must produce one day")
        day = dict(simulation["daily"][0])
        day.update({
            "calibrator_hash": artifact_hash,
            "calibration_ledger_hash": prior_ledger_hash,
            "max_training_settlement_time": max_prior_settlement_time,
            "minimum_decision_time": minimum_decision_time,
            "strict_prior_check": strict_prior,
            "same_race_calibrator_hash_count": 1,
        })
        for decision in day.get("candidate_decision_audit") or []:
            decision.update({
                "calibrator_hash": artifact_hash,
                "calibration_ledger_hash": prior_ledger_hash,
                "max_training_settlement_time": max_prior_settlement_time,
                "strict_prior_check": strict_prior,
            })
        cumulative_profit += int(day["profit_yen"])
        peak_profit = max(peak_profit, cumulative_profit)
        max_drawdown = max(max_drawdown, peak_profit - cumulative_profit)
        day["cumulative_profit_yen"] = cumulative_profit
        daily.append(day)
        totals["eligible_tickets"] += int(simulation["eligible_tickets"])
        for key in ("tickets", "hit_tickets", "stake_yen", "return_yen"):
            totals[key] += int(simulation[key])

        settled_day_records = policy_edge_records(
            day_races,
            probability_calibrator,
            _identity_probability_blender,
            max_rank=PURCHASE_MAX_RANK,
        )
        ledger.extend(settled_day_records)
        refit_audit.append({
            "race_date": race_date,
            "calibrator_hash": artifact_hash,
            "calibration_ledger_hash": prior_ledger_hash,
            "prior_candidates": len(ledger) - len(settled_day_records),
            "teachers_admitted_after_day": len(settled_day_records),
            "max_training_settlement_time": max_prior_settlement_time,
            "minimum_decision_time": minimum_decision_time,
            "strict_prior_check": strict_prior,
            "same_day_teachers_excluded": True,
            "same_race_calibrator_hash_count": 1,
        })

    stake_yen = totals["stake_yen"]
    return_yen = totals["return_yen"]
    final_ledger_hash = _ledger_sha256(ledger)
    bankroll = {
        "status": "ready" if all_ready else "calibration_not_ready",
        "allocation_policy": {
            "allocation_mode": "kelly_floor",
            "min_daily_exposure_fraction": 0.0,
            "max_daily_exposure_fraction": 0.30,
        },
        "calibration": final_payload,
        "calibration_update_mode": "daily_strict_prior_refit",
        "evaluation_days": len(daily),
        "evaluated_races": len(evaluation_scored),
        "eligible_days": sum(
            int(bool(day["eligible_candidate_audit"])) for day in daily
        ),
        "no_bet_days": sum(int(day["tickets"] == 0) for day in daily),
        **totals,
        "profit_yen": return_yen - stake_yen,
        "roi": return_yen / stake_yen if stake_yen else None,
        "roi_ci95_lower": None,
        "probability_roi_above_one": None,
        "max_drawdown_yen": max_drawdown,
        "daily": daily,
    }
    audit = {
        "update_rule": "refit_before_each_day_then_admit_all_pregate_candidates_after_day_settlement",
        "refit_days": len(refit_audit),
        "strict_prior_violation_count": strict_prior_violations,
        "strict_prior_check": strict_prior_violations == 0,
        "same_day_teachers_excluded": True,
        "same_race_calibrator_hash_count_max": 1,
        "initial_ledger_candidates": len(initial_ledger),
        "final_ledger_candidates": len(ledger),
        "evaluation_teachers_admitted": len(ledger) - len(initial_ledger),
        "final_calibration_ledger_hash": final_ledger_hash,
        "daily_refits": refit_audit,
    }
    return bankroll, audit, final_payload, final_ledger_hash


def evaluate_mature_stacked_value(
    calibration: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
    *,
    daily_budget_yen: int,
    num_threads: int = 4,
    calibration_update_mode: str = "fixed",
) -> dict[str, Any]:
    if calibration_update_mode not in {"fixed", "daily_strict_prior_refit"}:
        raise ValueError(
            "calibration_update_mode must be fixed or daily_strict_prior_refit"
        )
    dates = sorted({str(race["race_date"]) for race in calibration})
    required_days = (
        MODEL_TRAINING_MINIMUM_DAYS
        + VALUE_STACK_SELECTION_DAYS
        + VALUE_CALIBRATION_DAYS
    )
    if len(dates) < required_days:
        return {
            "model": MODEL_NAME,
            "status": "insufficient_nested_days",
            "calibration_days": len(dates),
            "required_days": required_days,
            "model_training_minimum_days": MODEL_TRAINING_MINIMUM_DAYS,
            "value_stack_selection_required_days": VALUE_STACK_SELECTION_DAYS,
            "value_calibration_required_days": VALUE_CALIBRATION_DAYS,
            "promotion_eligible": False,
            "real_betting_enabled": False,
        }
    if not evaluation:
        raise ValueError("mature stacked value requires untouched outer races")

    value_period_days = VALUE_STACK_SELECTION_DAYS + VALUE_CALIBRATION_DAYS
    value_period_dates = dates[-value_period_days:]
    stack_selection_dates = set(
        value_period_dates[:VALUE_STACK_SELECTION_DAYS]
    )
    value_dates = set(value_period_dates[VALUE_STACK_SELECTION_DAYS:])
    model_dates = set(dates[:-value_period_days])
    model_training = [
        race for race in calibration if str(race["race_date"]) in model_dates
    ]
    stack_selection = [
        race
        for race in calibration
        if str(race["race_date"]) in stack_selection_dates
    ]
    value_calibration = [
        race for race in calibration if str(race["race_date"]) in value_dates
    ]
    probability = fit_temporal_stacked_market_residual(
        model_training,
        [],
        num_threads=num_threads,
    )
    selection_artifact, value_aligned_selection = select_value_aligned_stack(
        probability,
        stack_selection,
    )
    probability_refit_training = model_training + stack_selection
    probability_refit = fit_temporal_stacked_market_residual(
        probability_refit_training,
        [],
        num_threads=num_threads,
    )
    selected_weights = selection_artifact.get("weights")
    if not isinstance(selected_weights, Mapping):
        raise ValueError("selected mature stack weights are missing")
    selected_stack = str(selection_artifact.get("selected_stack") or "")
    if not selected_stack:
        raise ValueError("selected mature stack name is missing")
    artifact = _value_aligned_artifact(
        probability_refit["artifact"],
        {
            "name": selected_stack,
            "weights": selected_weights,
        },
        base_selected_stack=str(probability.get("selected_stack") or ""),
    )
    value_aligned_selection = dict(value_aligned_selection)
    value_aligned_selection.update({
        "probability_component_refit_after_selection": True,
        "selected_stack_fixed_before_refit": True,
        "refit_excludes_empirical_gate_calibration": True,
        "refit_training_from": min(
            str(race["race_date"]) for race in probability_refit_training
        ),
        "refit_training_through": max(
            str(race["race_date"]) for race in probability_refit_training
        ),
        "refit_training_days": len(
            {
                str(race["race_date"])
                for race in probability_refit_training
            }
        ),
        "refit_training_races": len(probability_refit_training),
        "selection_artifact_sha256": selection_artifact.get(
            "artifact_sha256"
        ),
        "refit_artifact_sha256": artifact.get("artifact_sha256"),
    })
    value_scored = _score(value_calibration, artifact)
    evaluation_scored = _score(evaluation, artifact)
    calibrator = {"model_weight": 1.0, "temperature": 1.0}
    ledger = policy_edge_records(
        value_scored,
        calibrator,
        _identity_probability_blender,
        max_rank=PURCHASE_MAX_RANK,
    )
    evaluation_ledger = policy_edge_records(
        evaluation_scored,
        calibrator,
        _identity_probability_blender,
        max_rank=PURCHASE_MAX_RANK,
    )
    first_evaluation_date = min(
        str(race["race_date"]) for race in evaluation
    )
    empirical = _fit_mature_contextual_calibrator(
        ledger,
        prediction_date=first_evaluation_date,
    )
    empirical_payload = empirical.as_dict()
    strict_prior_audit = _strict_prior_artifact_audit(
        ledger, evaluation_ledger
    )
    calibrator_hash = _canonical_sha256(empirical_payload)
    calibration_ledger_hash = _ledger_sha256(ledger)
    calibration_update_audit = None
    final_calibrator_payload = empirical_payload
    final_calibration_ledger_hash = calibration_ledger_hash
    if calibration_update_mode == "daily_strict_prior_refit":
        (
            bankroll,
            calibration_update_audit,
            final_calibrator_payload,
            final_calibration_ledger_hash,
        ) = _simulate_daily_strict_prior_refit(
            evaluation_scored,
            ledger,
            calibrator,
            daily_budget_yen=daily_budget_yen,
        )
        calibrator_hash = _canonical_sha256(final_calibrator_payload)
    else:
        bankroll = simulate_empirical_lcb_policy(
            evaluation_scored,
            calibrator,
            _identity_probability_blender,
            empirical,
            daily_budget_yen,
            max_rank=PURCHASE_MAX_RANK,
            max_tickets_per_race=PURCHASE_MAX_TICKETS_PER_RACE,
        )
    confidence = (
        bootstrap_daily_roi(bankroll["daily"])
        if bankroll.get("stake_yen")
        else {
            "roi": None,
            "roi_ci95_lower": None,
            "roi_lower_quantile": 0.05,
            "quantile_method": BOOTSTRAP_QUANTILE_METHOD,
            "probability_roi_above_one": None,
        }
    )
    bankroll.update({
        "roi": confidence.get("roi"),
        "roi_display": (
            confidence.get("roi")
            if confidence.get("roi") is not None
            else "N/A"
        ),
        "roi_ci95_lower": confidence.get("roi_ci95_lower"),
        "roi_lower_quantile": confidence.get("roi_lower_quantile"),
        "roi_quantile_method": confidence.get("quantile_method"),
        "probability_roi_above_one": confidence.get(
            "probability_roi_above_one"
        ),
        "evaluation_days": len(
            {str(race["race_date"]) for race in evaluation}
        ),
    })
    effective_strict_prior_violations = (
        int(calibration_update_audit["strict_prior_violation_count"])
        if calibration_update_audit is not None
        else int(strict_prior_audit["strict_prior_violation_count"])
    )
    effective_same_race_hash_count = (
        int(calibration_update_audit["same_race_calibrator_hash_count_max"])
        if calibration_update_audit is not None
        else int(strict_prior_audit["same_race_calibrator_hash_count_max"])
    )
    final_calibration_ledger_candidates = (
        int(calibration_update_audit["final_ledger_candidates"])
        if calibration_update_audit is not None
        else len(ledger)
    )
    result = {
        "model": (
            DAILY_REFIT_MODEL_NAME
            if calibration_update_mode == "daily_strict_prior_refit"
            else MODEL_NAME
        ),
        "status": "completed",
        "evidence_role": (
            "retrospective_research_only_candidate_universe_search"
        ),
        "validation_design": (
            "earliest 60 or more days for nested V42 component and probability "
            "stack selection; following 60 untouched days for predeclared "
            "value-aligned stack reweighting; following 60 untouched days for "
            "top20 contextual rank-by-odds value calibration; final outer "
            "days used once"
        ),
        "stack_selection_calibration_disjoint": True,
        "search_validation_draw_sets_disjoint": True,
        "outer_period_used_for_selection": False,
        "model_training_from": min(model_dates),
        "model_training_through": max(model_dates),
        "model_training_days": len(model_dates),
        "model_training_races": len(model_training),
        "value_stack_selection_from": min(stack_selection_dates),
        "value_stack_selection_through": max(stack_selection_dates),
        "value_stack_selection_days": len(stack_selection_dates),
        "value_stack_selection_races": len(stack_selection),
        "value_calibration_from": min(value_dates),
        "value_calibration_through": max(value_dates),
        "value_calibration_days": len(value_dates),
        "value_calibration_races": len(value_calibration),
        "evaluation_from": first_evaluation_date,
        "evaluation_through": max(
            str(race["race_date"]) for race in evaluation
        ),
        "evaluation_races": len(evaluation),
        "purchase_max_rank": PURCHASE_MAX_RANK,
        "purchase_max_tickets_per_race": PURCHASE_MAX_TICKETS_PER_RACE,
        "formal_roi_gate": {
            "cluster_unit": "complete_race_date",
            "lower_quantile": 0.05,
            "quantile_method": BOOTSTRAP_QUANTILE_METHOD,
            "condition": "day_block_roi_lower_quantile_strictly_above_one",
        },
        "candidate_population": (
            "all_stacked_probability_top20_before_purchase_gate"
        ),
        "calibration_update_mode": calibration_update_mode,
        "calibration_update_audit": calibration_update_audit,
        "probability_selection": {
            key: probability.get(key)
            for key in (
                "base_training_through",
                "stack_validation_from",
                "raw_selected_stack",
                "stack_selection_gate",
                "selected_stack",
                "selected_weights",
                "component_selection",
            )
        },
        "probability_artifact": artifact,
        "probability_refit": {
            "training_from": min(
                str(race["race_date"])
                for race in probability_refit_training
            ),
            "training_through": max(
                str(race["race_date"])
                for race in probability_refit_training
            ),
            "training_days": len(
                {
                    str(race["race_date"])
                    for race in probability_refit_training
                }
            ),
            "training_races": len(probability_refit_training),
            "selected_stack_fixed_before_refit": True,
            "empirical_gate_calibration_used": False,
            "component_selection": probability_refit.get(
                "component_selection"
            ),
        },
        "value_aligned_stack_selection": value_aligned_selection,
        "value_stack_selection_probability_metrics": stacked_metrics(
            stack_selection, artifact
        ),
        "value_calibration_probability_metrics": stacked_metrics(
            value_calibration, artifact
        ),
        "evaluation_probability_metrics": stacked_metrics(
            evaluation, artifact
        ),
        "empirical_ev_calibration": empirical_payload,
        "final_empirical_ev_calibration": final_calibrator_payload,
        "calibration_ledger_candidates": len(ledger),
        "final_calibration_ledger_candidates": (
            final_calibration_ledger_candidates
        ),
        "evaluation_ledger_candidates": len(evaluation_ledger),
        "strict_prior_artifact_audit": strict_prior_audit,
        "strict_prior_violation_count": effective_strict_prior_violations,
        "same_race_calibrator_hash_count_max": effective_same_race_hash_count,
        "calibrator_hash": calibrator_hash,
        "initial_calibrator_hash": _canonical_sha256(empirical_payload),
        "initial_calibration_ledger_hash": calibration_ledger_hash,
        "calibration_ledger_hash": final_calibration_ledger_hash,
        "artifact_lineage": {
            "parent_artifact_hash": artifact.get("artifact_sha256"),
            "calibrator_hash": calibrator_hash,
            "calibration_ledger_hash": final_calibration_ledger_hash,
        },
        "value_decile_audit": value_decile_audit(
            ledger, evaluation_ledger
        ),
        "context_value_audit": context_value_audit(
            ledger, evaluation_ledger
        ),
        "bankroll": bankroll,
        "statistical_gate_passed": False,
        "promotion_eligible": False,
        "real_betting_enabled": False,
    }
    result["statistical_gate_passed"] = empirical_bankroll_promotion_eligible(
        bankroll
    )
    return result
