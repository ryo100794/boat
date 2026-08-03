from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib

from .joint_bankroll_evaluation import (
    PURCHASE_UNIT_YEN,
    _instant,
    _realized_receipt,
    _release_matured_receipts,
    build_block_bootstrap_evidence,
)
from .listwise.empirical_ev_calibration import fit_empirical_ev_calibration


MODEL_VERSION = "joint_edge_calibrated_replay_v2"
CALIBRATION_VERSION = "strict_prior_independent_validation_isotonic_lcb_v2"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("base evaluation artifact must be a mapping")
    daily = payload.get("daily")
    if not isinstance(daily, list) or not daily:
        raise ValueError("base evaluation artifact must contain daily rows")
    return payload


def _load_decision_times(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = joblib.load(path)
    races = payload.get("races") if isinstance(payload, Mapping) else None
    if not isinstance(races, list):
        raise ValueError("scored cache must contain race rows")
    return {
        str(row["race_id"]): str(row.get("captured_at") or "")
        for row in races
        if isinstance(row, Mapping) and row.get("race_id")
    }


def _candidate_record(
    race_date: str,
    race: Mapping[str, Any],
    *,
    buy_margin: float,
) -> dict[str, object] | None:
    stake = int(race.get("best_search_stake_yen") or 0)
    separate_validation = bool(
        race.get("validation_uses_separate_draw_set")
    )
    if separate_validation:
        validated_edge = race.get("portfolio_lower_quantile")
        if validated_edge is None:
            return None
        edge_excess = float(validated_edge) - buy_margin
        raw_value_source = "independent_validation_portfolio_lower_quantile"
        structural_feasible = bool(
            race.get("purchase_value_gate_passed")
            and race.get("bankroll_growth_lower_quantile") is not None
            and float(race["bankroll_growth_lower_quantile"]) > 0.0
        )
    else:
        edge_excess = race.get("best_search_edge_excess")
        if edge_excess is None:
            return None
        edge_excess = float(edge_excess)
        raw_value_source = "legacy_search_edge_fallback"
        structural_feasible = bool(
            edge_excess > 0.0
            and float(race.get("best_search_growth_excess") or 0.0) > 0.0
            and (
                race.get("best_search_constraint_violation") is None
                or float(race["best_search_constraint_violation"]) <= 0.0
            )
        )
    if stake <= 0:
        return None
    raw_gross_return = max(0.0, 1.0 + edge_excess + buy_margin)
    realized_return = int(
        race.get("best_search_hypothetical_return_yen") or 0
    )
    return {
        "race_date": race_date,
        "raw_estimated_ev": raw_gross_return,
        "gross_return_per_yen": realized_return / stake,
        "raw_value_source": raw_value_source,
        "structural_feasible": structural_feasible,
    }


def _fit_bets_to_cash(
    bets_yen: Mapping[str, Any],
    available_cash_yen: int,
) -> dict[str, int]:
    units = {
        str(ticket): int(stake) // PURCHASE_UNIT_YEN
        for ticket, stake in bets_yen.items()
        if int(stake) >= PURCHASE_UNIT_YEN
        and int(stake) % PURCHASE_UNIT_YEN == 0
    }
    total_units = sum(units.values())
    budget_units = max(0, available_cash_yen // PURCHASE_UNIT_YEN)
    if total_units <= budget_units:
        return {
            ticket: value * PURCHASE_UNIT_YEN
            for ticket, value in sorted(units.items())
        }
    if budget_units <= 0 or total_units <= 0:
        return {}
    exact = {
        ticket: value * budget_units / total_units
        for ticket, value in units.items()
    }
    allocated = {
        ticket: min(units[ticket], int(exact[ticket]))
        for ticket in units
    }
    remainder = budget_units - sum(allocated.values())
    order = sorted(
        units,
        key=lambda ticket: (
            -(exact[ticket] - int(exact[ticket])),
            -units[ticket],
            ticket,
        ),
    )
    for ticket in order:
        if remainder <= 0:
            break
        if allocated[ticket] < units[ticket]:
            allocated[ticket] += 1
            remainder -= 1
    return {
        ticket: value * PURCHASE_UNIT_YEN
        for ticket, value in sorted(allocated.items())
        if value > 0
    }


def _decision_time(
    race: Mapping[str, Any],
    decision_times: Mapping[str, str],
) -> datetime:
    value = race.get("evaluation_time_t") or decision_times.get(
        str(race.get("race_id") or "")
    )
    return _instant(value, "evaluation_time_t")


def run_joint_edge_calibrated_replay(
    base_artifact: Path,
    *,
    scored_cache: Path | None = None,
    initial_daily_bankroll_yen: int = 10_000,
    calibration_margin: float = 0.0,
    calibration_bootstrap_samples: int = 2_000,
    calibration_min_training_days: int = 3,
    calibration_min_portfolios: int = 200,
    calibration_min_candidate_days: int = 3,
    bootstrap_samples: int = 2_000,
    seed: int = 43_041,
) -> dict[str, Any]:
    if initial_daily_bankroll_yen < PURCHASE_UNIT_YEN:
        raise ValueError("initial bankroll must cover one purchase unit")
    if initial_daily_bankroll_yen % PURCHASE_UNIT_YEN:
        raise ValueError("initial bankroll must use 100-yen units")
    if calibration_margin < 0.0:
        raise ValueError("calibration margin must be non-negative")
    payload = _load_payload(base_artifact)
    configuration = payload.get("configuration")
    configuration = configuration if isinstance(configuration, dict) else {}
    buy_margin = float(configuration.get("buy_margin") or 0.0)
    cache_path = scored_cache
    if cache_path is None and payload.get("scored_cache"):
        candidate = Path(str(payload["scored_cache"]))
        cache_path = candidate if candidate.is_file() else None
    decision_times = _load_decision_times(cache_path)

    training_records: list[dict[str, object]] = []
    replay_days = []
    ready_days = 0
    ready_races = 0
    calibrated_candidates = 0
    rejected_reasons: dict[str, int] = {}
    calibration_folds = []

    for day_index, source_day in enumerate(payload["daily"]):
        race_date = str(source_day.get("race_date") or "")
        calibrator = fit_empirical_ev_calibration(
            training_records,
            bootstrap_samples=calibration_bootstrap_samples,
            seed=seed + day_index,
            min_days=calibration_min_training_days,
            min_tickets=calibration_min_portfolios,
            min_candidate_days=calibration_min_candidate_days,
            candidate_min_raw_ev=1.0 + buy_margin,
            shape_constraint="isotonic",
            quantile_method="inverted_cdf",
        )
        calibration_folds.append({
            "evaluation_date": race_date,
            **calibrator.as_dict(),
        })
        if calibrator.trained_through_date is not None and (
            calibrator.trained_through_date >= race_date
        ):
            raise ValueError("calibration teacher is not strictly prior")

        balance = initial_daily_bankroll_yen
        peak = balance
        maximum_drawdown_yen = 0
        stake_yen = 0
        return_yen = 0
        hits = 0
        pending: list[tuple[datetime, int]] = []
        replay_races = []
        current_training_records = []
        source_races = source_day.get("races") or []
        if calibrator.ready:
            ready_days += 1
            ready_races += len(source_races)

        for race in source_races:
            purchase_at = _decision_time(race, decision_times)
            pending, matured = _release_matured_receipts(
                pending, asof=purchase_at
            )
            balance += matured
            peak = max(peak, balance)
            record = _candidate_record(
                race_date, race, buy_margin=buy_margin
            )
            if record is not None:
                current_training_records.append(record)

            raw_gross = (
                float(record["raw_estimated_ev"])
                if record is not None else None
            )
            prediction = (
                calibrator.predict(raw_gross)
                if calibrator.ready and raw_gross is not None else {}
            )
            calibrated_lcb = prediction.get("empirical_ev_lcb95")
            structural_feasible = bool(
                record is not None and record["structural_feasible"]
            )
            authorized = bool(
                calibrator.ready
                and structural_feasible
                and calibrated_lcb is not None
                and float(calibrated_lcb) > 1.0 + calibration_margin
            )
            reason = None
            if not calibrator.ready:
                reason = "calibration_not_ready"
            elif not structural_feasible:
                reason = "base_joint_gate_not_feasible"
            elif calibrated_lcb is None:
                reason = "calibration_lcb_missing"
            elif float(calibrated_lcb) <= 1.0 + calibration_margin:
                reason = "calibration_lcb_not_above_margin"

            proposed_bets = race.get("best_search_bets_yen") or {}
            bets = (
                _fit_bets_to_cash(proposed_bets, balance)
                if authorized else {}
            )
            if authorized:
                calibrated_candidates += 1
            if authorized and not bets:
                reason = "daily_cash_exhausted"
                authorized = False
            if reason is not None:
                rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1

            stake = sum(bets.values())
            receipt = _realized_receipt(
                bets,
                actual_combination=str(race.get("actual_combination") or ""),
                actual_payout_yen=int(race.get("actual_payout_yen") or 0),
            )
            balance -= stake
            maximum_drawdown_yen = max(
                maximum_drawdown_yen, peak - balance
            )
            stake_yen += stake
            return_yen += receipt
            if receipt:
                hits += 1
                pending.append((
                    _instant(
                        race.get("settlement_available_at"),
                        "settlement_available_at",
                    ),
                    receipt,
                ))
            replay_races.append({
                "race_id": race.get("race_id"),
                "evaluation_time_t": purchase_at.isoformat(),
                "raw_portfolio_gross_return_estimate": raw_gross,
                "raw_value_source": (
                    record.get("raw_value_source") if record else None
                ),
                "calibrated_gross_return": prediction.get("empirical_ev"),
                "calibrated_gross_return_lcb95": calibrated_lcb,
                "calibration_support": prediction.get("support"),
                "calibration_support_days": prediction.get("support_days"),
                "calibration_trained_through_date": (
                    calibrator.trained_through_date
                ),
                "base_joint_gate_feasible": structural_feasible,
                "purchase_authorized": authorized,
                "rejection_reason": reason,
                "bets_yen": bets,
                "stake_yen": stake,
                "return_yen": receipt,
                "profit_yen": receipt - stake,
            })
        for _available_at, receipt in pending:
            balance += receipt
            peak = max(peak, balance)
        training_records.extend(current_training_records)
        replay_days.append({
            "race_date": race_date,
            "calibration_ready": calibrator.ready,
            "calibration_trained_through_date": calibrator.trained_through_date,
            "opening_bankroll_yen": initial_daily_bankroll_yen,
            "closing_bankroll_yen": balance,
            "stake_yen": stake_yen,
            "return_yen": return_yen,
            "profit_yen": return_yen - stake_yen,
            "roi": return_yen / stake_yen if stake_yen else None,
            "hits": hits,
            "max_drawdown_yen": maximum_drawdown_yen,
            "evaluated_races": len(source_races),
            "races": replay_races,
        })

    confidence = build_block_bootstrap_evidence(
        replay_days, samples=bootstrap_samples, seed=seed
    )
    total_stake = sum(day["stake_yen"] for day in replay_days)
    total_return = sum(day["return_yen"] for day in replay_days)
    purchased = [
        race
        for day in replay_days
        for race in day["races"]
        if race["stake_yen"] > 0
    ]
    returns = [race["return_yen"] for race in purchased]
    largest_return = max(returns, default=0)
    lcb_values = [
        float(race["calibrated_gross_return_lcb95"])
        for race in purchased
        if race.get("calibrated_gross_return_lcb95") is not None
    ]
    bankroll = {
        "initial_daily_bankroll_yen": initial_daily_bankroll_yen,
        "stake_yen": total_stake,
        "return_yen": total_return,
        "profit_yen": total_return - total_stake,
        "roi": total_return / total_stake if total_stake else None,
        "tickets": sum(len(race["bets_yen"]) for race in purchased),
        "selected_races": len(purchased),
        "hit_races": sum(race["return_yen"] > 0 for race in purchased),
        "evaluation_days": len(replay_days),
        "calibration_ready_days": ready_days,
        "calibration_ready_races": ready_races,
        "roi_without_largest_hit": (
            (total_return - largest_return) / total_stake
            if total_stake else None
        ),
        "daily_cluster_bootstrap_roi_lower_95": confidence["roi_lower"],
        "roi_ci95_lower": confidence["roi_lower"],
        "roi_ci95_upper": confidence["roi_upper"],
        "probability_roi_above_one": confidence[
            "probability_roi_above_one"
        ],
        "max_drawdown_yen": max(
            int(day["max_drawdown_yen"]) for day in replay_days
        ),
    }
    calibration_input_sources = {
        source: sum(
            record.get("raw_value_source") == source
            for record in training_records
        )
        for source in sorted({
            str(record.get("raw_value_source") or "unknown")
            for record in training_records
        })
    }
    formal_gate = {
        "independent_validation_value_only": bool(
            purchased
            and all(
                race.get("raw_value_source")
                == "independent_validation_portfolio_lower_quantile"
                for race in purchased
            )
        ),
        "minimum_30_calibration_ready_days": ready_days >= 30,
        "minimum_1000_ready_races": ready_races >= 1_000,
        "minimum_200_tickets": bankroll["tickets"] >= 200,
        "minimum_20_hits": bankroll["hit_races"] >= 20,
        "positive_profit": bankroll["profit_yen"] > 0,
        "roi_lower_bound_above_one": bool(
            confidence["roi_lower"] is not None
            and confidence["roi_lower"] > 1.0
        ),
        "roi_without_largest_hit_above_one": bool(
            bankroll["roi_without_largest_hit"] is not None
            and bankroll["roi_without_largest_hit"] > 1.0
        ),
        "calibrated_lcb_above_margin": bool(
            purchased
            and len(lcb_values) == len(purchased)
            and min(lcb_values) > 1.0 + calibration_margin
        ),
    }
    base_protocol = payload.get("evaluation_protocol")
    base_protocol = base_protocol if isinstance(base_protocol, dict) else {}
    protocol = {
        "version": "joint_edge_calibrated_replay_protocol_v2",
        "model": MODEL_VERSION,
        "base_artifact_sha256": _sha256_file(base_artifact),
        "base_evaluation_protocol_id": payload.get("evaluation_protocol_id"),
        "evaluation_time_t": base_protocol.get("evaluation_time_t"),
        "population": base_protocol.get("population"),
        "calibration": {
            "version": CALIBRATION_VERSION,
            "margin": calibration_margin,
            "bootstrap_samples": calibration_bootstrap_samples,
            "min_training_days": calibration_min_training_days,
            "min_portfolios": calibration_min_portfolios,
            "min_candidate_days": calibration_min_candidate_days,
            "shape_constraint": "isotonic",
            "quantile_method": "inverted_cdf",
            "teacher": "fixed_candidate_realized_gross_return_per_yen",
            "information_boundary": "strictly_prior_complete_days_only",
            "primary_input": (
                "independent_validation_portfolio_lower_quantile"
            ),
            "legacy_input": "search_edge_explicit_fallback_only",
        },
        "bankroll": {
            "initial_daily_bankroll_yen": initial_daily_bankroll_yen,
            "purchase_unit_yen": PURCHASE_UNIT_YEN,
            "profit_reuse": "after_recorded_settlement_available_at",
            "cash_shortfall": "proportional_integer_unit_downscale",
        },
        "resampling_condition_id": confidence["condition_id"],
        "seed": seed,
    }
    return {
        "model": MODEL_VERSION,
        "status": (
            "promotion_candidate"
            if all(formal_gate.values())
            else "provisional_learned_edge_calibration"
        ),
        "promotion_eligible": all(formal_gate.values()),
        "deployment_eligible": False,
        "base_artifact": str(base_artifact),
        "base_model": payload.get("model"),
        "joint_value_audit": payload.get("joint_value_audit"),
        "settlement_audit": payload.get("settlement_audit"),
        "joint_audit_inherited_from_base_artifact": True,
        "probability_metrics": payload.get("probability_metrics"),
        "evaluation_protocol_id": _canonical_sha256(protocol),
        "evaluation_protocol": protocol,
        "evaluation_from": replay_days[0]["race_date"] if replay_days else None,
        "evaluation_through": replay_days[-1]["race_date"] if replay_days else None,
        "evaluation_days": len(replay_days),
        "evaluated_races": sum(day["evaluated_races"] for day in replay_days),
        "calibration_ready_days": ready_days,
        "calibration_ready_races": ready_races,
        "calibrated_candidates": calibrated_candidates,
        "calibration_training_records": len(training_records),
        "calibration_input_sources": calibration_input_sources,
        "calibration_folds": calibration_folds,
        "rejection_reasons": dict(sorted(rejected_reasons.items())),
        "primary_bankroll": bankroll,
        "bankroll_confidence": confidence,
        "formal_purchase_value": {
            "definition": "strict_prior_empirical_gross_return_isotonic_day_LCB95",
            "safety_margin": calibration_margin,
            "minimum": (
                min(lcb_values) - 1.0 if lcb_values else None
            ),
            "selected_portfolios": len(purchased),
            "all_above_safety_margin": formal_gate[
                "calibrated_lcb_above_margin"
            ],
        },
        "promotion_gate": formal_gate,
        "promotion_gate_passed": sum(formal_gate.values()),
        "promotion_gate_total": len(formal_gate),
        "promotion_gate_failed": [
            key for key, passed in formal_gate.items() if not passed
        ],
        "daily": replay_days,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-artifact", type=Path, required=True)
    parser.add_argument("--scored-cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-daily-bankroll-yen", type=int, default=10_000)
    parser.add_argument("--calibration-margin", type=float, default=0.0)
    parser.add_argument("--calibration-bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--calibration-min-training-days", type=int, default=3)
    parser.add_argument("--calibration-min-portfolios", type=int, default=200)
    parser.add_argument("--calibration-min-candidate-days", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=43_041)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    options = vars(args)
    output = options.pop("output")
    result = run_joint_edge_calibrated_replay(**options)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps({
        "model": result["model"],
        "evaluation_days": result["evaluation_days"],
        "evaluated_races": result["evaluated_races"],
        **result["primary_bankroll"],
        "promotion_eligible": result["promotion_eligible"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
