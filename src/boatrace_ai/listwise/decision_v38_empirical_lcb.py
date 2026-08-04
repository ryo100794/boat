from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import joblib

from ..bankroll_bootstrap import bootstrap_daily_roi
from .empirical_ev_calibration import fit_empirical_ev_calibration
from .empirical_lcb_policy import (
    empirical_bankroll_promotion_eligible,
    policy_edge_records,
    simulate_empirical_lcb_policy,
)
from .nonlinear_market_residual_v38 import nonlinear_residual_probabilities
from .decision_market_residual_v38 import decision_time_race


MODEL_NAME = "decision_v38_strict_prior_empirical_lcb_v39"
SETTLEMENT_ENGINE_CONTRACT = (
    "official_result_gross_roi_v1_previous_calendar_dates_only"
)
PURCHASE_RESIDUAL_SHRINKAGE = 1.0
PURCHASE_MAX_PROBABILITY_RANK = 5
MINIMUM_LEDGER_DAYS = 30
MINIMUM_LEDGER_CANDIDATES = 300
MINIMUM_LEDGER_CANDIDATE_DAYS = 20


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
    return [
        {
            **race,
            "model_probabilities": nonlinear_residual_probabilities(
                race,
                artifact,
                shrinkage=PURCHASE_RESIDUAL_SHRINKAGE,
            ),
        }
        for race in races
    ]


def _aggregate_daily(daily: list[dict[str, Any]]) -> dict[str, Any]:
    stake = sum(int(row.get("stake_yen") or 0) for row in daily)
    returned = sum(int(row.get("return_yen") or 0) for row in daily)
    tickets = sum(int(row.get("tickets") or 0) for row in daily)
    hit_tickets = sum(int(row.get("hit_tickets") or 0) for row in daily)
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
        "roi_ci95_lower": confidence.get("roi_ci95_lower"),
        "roi_ci95_upper": confidence.get("roi_ci95_upper"),
        "probability_roi_above_one": confidence.get(
            "probability_roi_above_one"
        ),
        "max_drawdown_yen": max_drawdown,
        "daily": daily,
    }


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
) -> dict[str, Any]:
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
    latest_calibrator = None
    calibrator = {"model_weight": 1.0, "temperature": 1.0}
    source_hash = str(
        frozen.get("source_scored_cache_sha256")
        or frozen.get("artifact", {}).get("booster_sha256")
        or ""
    )
    settlement_engine_hash = _stable_hash(SETTLEMENT_ENGINE_CONTRACT)
    for evaluation_date in sorted(by_day):
        if any(str(row["race_date"]) >= evaluation_date for row in ledger):
            raise AssertionError("V39 ledger contains a non-prior settlement")
        latest_calibrator = fit_empirical_ev_calibration(
            ledger,
            bootstrap_samples=bootstrap_samples,
            min_days=minimum_ledger_days,
            min_tickets=minimum_ledger_candidates,
            min_candidate_days=minimum_ledger_candidate_days,
            min_local_candidates=minimum_local_candidates,
            min_local_candidate_days=minimum_local_candidate_days,
            min_local_ess=minimum_local_ess,
            candidate_min_raw_ev=1.0,
        )
        day_races = by_day[evaluation_date]
        simulation = simulate_empirical_lcb_policy(
            day_races,
            calibrator,
            _identity_probability_blender,
            latest_calibrator,
            daily_budget_yen,
            max_rank=PURCHASE_MAX_PROBABILITY_RANK,
        )
        if evaluation_date > registration:
            prospective_daily.extend(simulation["daily"])
        calibrator_hash = _stable_hash(latest_calibrator.as_dict())
        ledger_hash = _stable_hash(ledger)
        decision_contract_hash = _stable_hash({
            "model_hash": source_hash,
            "calibrator_hash": calibrator_hash,
            "ledger_hash": ledger_hash,
            "purchase_threshold": "empirical_ROI_LCB95 > 1.0",
            "maximum_probability_rank": PURCHASE_MAX_PROBABILITY_RANK,
            "residual_shrinkage": PURCHASE_RESIDUAL_SHRINKAGE,
            "settlement_engine_hash": settlement_engine_hash,
        })
        fold_audit.append({
            "evaluation_date": evaluation_date,
            "prospective_evidence": evaluation_date > registration,
            "calibration_cutoff_date": (
                latest_calibrator.trained_through_date
            ),
            "max_training_settlement_date": (
                latest_calibrator.trained_through_date
            ),
            "strict_prior_check": bool(
                latest_calibrator.trained_through_date is None
                or latest_calibrator.trained_through_date < evaluation_date
            ),
            "prior_candidates": latest_calibrator.tickets,
            "prior_days": latest_calibrator.training_days,
            "prior_candidate_days": latest_calibrator.candidate_days,
            "calibration_ready": latest_calibrator.ready,
            "ready_reasons": list(latest_calibrator.ready_reasons),
            "authorized_tickets": simulation.get("tickets"),
            "stake_yen": simulation.get("stake_yen"),
            "frozen_model_hash": source_hash,
            "calibrator_hash": calibrator_hash,
            "calibration_ledger_hash": ledger_hash,
            "settlement_engine_hash": settlement_engine_hash,
            "decision_contract_hash": decision_contract_hash,
        })
        current = policy_edge_records(
            day_races,
            calibrator,
            _identity_probability_blender,
            max_rank=PURCHASE_MAX_PROBABILITY_RANK,
        )
        if any(str(row["race_date"]) != evaluation_date for row in current):
            raise AssertionError("V39 admitted a mismatched settlement batch")
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
        "settlement_engine_contract": SETTLEMENT_ENGINE_CONTRACT,
        "settlement_engine_hash": settlement_engine_hash,
        "purchase_residual_shrinkage": PURCHASE_RESIDUAL_SHRINKAGE,
        "candidate_population": "all_probability_top5_before_purchase_gate",
        "purchase_max_probability_rank": PURCHASE_MAX_PROBABILITY_RANK,
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
        "purchase_threshold": "empirical_ROI_LCB95 > 1.0",
        "range_policy": "deny outside local isotonic block support",
        "bootstrap_cluster_unit": "race_date",
        "ticket_level_independence_assumed": False,
        "ledger_candidates": len(ledger),
        "ledger_hash": _stable_hash(ledger),
        "fold_audit": fold_audit,
        "latest_calibrator": (
            latest_calibrator.as_dict() if latest_calibrator is not None else None
        ),
        "bankroll": bankroll,
    }
    result["promotion_eligible"] = empirical_bankroll_promotion_eligible(bankroll)
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
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate_from_files(
        args.frozen_artifact,
        args.scored_cache,
        registered_after=args.registered_after,
        daily_budget_yen=args.daily_budget_yen,
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
