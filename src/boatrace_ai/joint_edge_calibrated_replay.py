from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from math import isfinite
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np

from .joint_bankroll_evaluation import (
    PURCHASE_UNIT_YEN,
    _day_block_roi_interval,
    _instant,
    _realized_receipt,
    _release_matured_receipts,
    build_block_bootstrap_evidence,
)
from .listwise.empirical_ev_calibration import fit_empirical_ev_calibration


MODEL_VERSION = "joint_edge_calibrated_replay_v12"
CALIBRATION_VERSION = (
    "strict_prior_independent_validation_"
    "stake_weighted_isotonic_lcb_v12_local_support_decision_hashes"
)
PRIMARY_RAW_VALUE_SOURCE = (
    "pregate_best_search_independent_validation_"
    "portfolio_lower_quantile"
)


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


def _finite_audit_value(value: object) -> object:
    if isinstance(value, float) and not isfinite(value):
        if value == float("inf"):
            return "Infinity"
        if value == float("-inf"):
            return "-Infinity"
        return "NaN"
    if isinstance(value, Mapping):
        return {
            str(key): _finite_audit_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_finite_audit_value(item) for item in value]
    return value


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


def _calibration_ledger_sha256(
    records: Sequence[Mapping[str, object]],
) -> str:
    rows = sorted(
        ({
            "race_id": str(record.get("race_id") or ""),
            "race_date": str(record.get("race_date") or ""),
            "settlement_available_at": str(
                record.get("settlement_available_at") or ""
            ),
            "raw_estimated_ev": record.get("raw_estimated_ev"),
            "gross_return_per_yen": record.get(
                "gross_return_per_yen"
            ),
            "sample_weight": record.get("sample_weight"),
            "raw_value_source": str(
                record.get("raw_value_source") or ""
            ),
            "structural_feasible": bool(
                record.get("structural_feasible")
            ),
        } for record in records),
        key=lambda row: row["race_id"],
    )
    return _canonical_sha256(_finite_audit_value(rows))


def _candidate_record(
    race_date: str,
    race: Mapping[str, Any],
    *,
    buy_margin: float,
) -> dict[str, object] | None:
    race_id = str(race.get("race_id") or "")
    if not race_id:
        raise ValueError("candidate calibration record requires race_id")
    stake = int(race.get("best_search_stake_yen") or 0)
    separate_validation = bool(
        race.get("validation_uses_separate_draw_set")
    )
    if separate_validation:
        validated_edge = race.get(
            "best_search_validation_portfolio_lower_quantile"
        )
        if validated_edge is not None:
            raw_value_source = PRIMARY_RAW_VALUE_SOURCE
            structural_feasible = bool(
                race.get(
                    "best_search_validation_purchase_value_gate_passed"
                )
                and race.get(
                    "best_search_validation_growth_gate_passed"
                )
            )
        else:
            validated_edge = race.get("portfolio_lower_quantile")
            raw_value_source = (
                "legacy_selected_independent_validation_"
                "portfolio_lower_quantile"
            )
            structural_feasible = bool(
                race.get("purchase_value_gate_passed")
                and race.get("bankroll_growth_lower_quantile") is not None
                and float(race["bankroll_growth_lower_quantile"]) > 0.0
            )
        if validated_edge is None:
            return None
        edge_excess = float(validated_edge) - buy_margin
        if not isfinite(edge_excess):
            return None
    else:
        edge_excess = race.get("best_search_edge_excess")
        if edge_excess is None:
            return None
        edge_excess = float(edge_excess)
        if not isfinite(edge_excess):
            return None
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
    settlement_available_at = _instant(
        race.get("settlement_available_at"),
        "settlement_available_at",
    )
    return {
        "race_id": race_id,
        "race_date": race_date,
        "settlement_available_at": settlement_available_at.isoformat(),
        "raw_estimated_ev": raw_gross_return,
        "gross_return_per_yen": realized_return / stake,
        "sample_weight": float(stake),
        "candidate_portfolio_stake_yen": stake,
        "candidate_portfolio_ticket_count": len(
            race.get("best_search_bets_yen") or {}
        ),
        "result_batch_unit": "one_race_candidate_portfolio",
        "raw_value_source": raw_value_source,
        "structural_feasible": structural_feasible,
    }


def _calendar_span_days(
    records: Sequence[Mapping[str, object]],
) -> int:
    days = sorted({
        str(record.get("race_date") or "")
        for record in records
        if record.get("race_date")
    })
    if not days:
        return 0
    return (
        datetime.fromisoformat(days[-1]).date()
        - datetime.fromisoformat(days[0]).date()
    ).days + 1


def _strict_warmup_ready(
    calibrator: Any,
    records: Sequence[Mapping[str, object]],
    *,
    minimum_calendar_days: int,
) -> bool:
    return bool(
        calibrator.ready
        and _calendar_span_days(records) >= minimum_calendar_days
    )


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


def _calibrated_value_realization(
    days: Sequence[Mapping[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Compare calibrated value and settlement on the identical portfolio."""
    rows: list[dict[str, Any]] = []
    excluded_mismatched = 0
    excluded_non_independent = 0
    for day in days:
        race_date = str(day.get("race_date") or "")
        for race in day.get("races") or []:
            if not race.get("base_joint_gate_feasible"):
                continue
            if race.get("raw_value_source") != PRIMARY_RAW_VALUE_SOURCE:
                excluded_non_independent += 1
                continue
            predicted_roi = race.get("calibrated_gross_return")
            conservative_roi = race.get("calibrated_gross_return_lcb95")
            stake_yen = int(race.get("counterfactual_stake_yen") or 0)
            return_yen = int(race.get("counterfactual_return_yen") or 0)
            if predicted_roi is None or conservative_roi is None or stake_yen <= 0:
                continue
            if not race.get("counterfactual_matches_base_portfolio"):
                excluded_mismatched += 1
                continue
            rows.append({
                "race_id": str(race.get("race_id") or ""),
                "race_date": race_date,
                "purchase_value": float(conservative_roi) - 1.0,
                "predicted_roi": float(predicted_roi),
                "conservative_predicted_roi": float(conservative_roi),
                "stake_yen": stake_yen,
                "return_yen": return_yen,
            })
    rows.sort(key=lambda row: (
        row["purchase_value"], row["race_date"], row["race_id"]
    ))
    bins = []
    bin_count = min(10, len(rows))
    for index in range(bin_count):
        start = index * len(rows) // bin_count
        end = (index + 1) * len(rows) // bin_count
        selected = rows[start:end]
        stake_yen = sum(row["stake_yen"] for row in selected)
        return_yen = sum(row["return_yen"] for row in selected)
        by_day: dict[str, dict[str, Any]] = {}
        for row in selected:
            daily = by_day.setdefault(row["race_date"], {
                "race_date": row["race_date"],
                "stake_yen": 0,
                "return_yen": 0,
            })
            daily["stake_yen"] += row["stake_yen"]
            daily["return_yen"] += row["return_yen"]
        confidence = _day_block_roi_interval(
            list(by_day.values()),
            samples=samples,
            seed=seed + 9_200_000 + index,
        )
        bins.append({
            "decile": index + 1,
            "candidate_portfolios": len(selected),
            "evaluation_days": len(by_day),
            "minimum_purchase_value": min(
                row["purchase_value"] for row in selected
            ),
            "maximum_purchase_value": max(
                row["purchase_value"] for row in selected
            ),
            "mean_purchase_value": sum(
                row["purchase_value"] * row["stake_yen"]
                for row in selected
            ) / stake_yen,
            "predicted_roi": sum(
                row["predicted_roi"] * row["stake_yen"]
                for row in selected
            ) / stake_yen,
            "conservative_predicted_roi": sum(
                row["conservative_predicted_roi"] * row["stake_yen"]
                for row in selected
            ) / stake_yen,
            "stake_yen": stake_yen,
            "return_yen": return_yen,
            "profit_yen": return_yen - stake_yen,
            "realized_roi": return_yen / stake_yen,
            "daily_block_roi_lower_95": confidence["roi_lower"],
            "daily_block_roi_upper_95": confidence["roi_upper"],
            "bootstrap_samples": confidence["samples"],
        })
    realized = [float(row["realized_roi"]) for row in bins]
    manifest = [[
        row["race_id"], row["race_date"], row["purchase_value"],
        row["predicted_roi"], row["stake_yen"], row["return_yen"],
    ] for row in rows]
    return {
        "version": "strict_prior_calibrated_value_realization_deciles_v1",
        "population": (
            "calibration_ready_structurally_feasible_independent_validation_"
            "portfolios"
        ),
        "ranking_value_definition": "calibrated_gross_return_lcb95_minus_one",
        "predicted_roi_definition": "stake_weighted_calibrated_gross_return",
        "conservative_predicted_roi_definition": (
            "stake_weighted_calibrated_gross_return_lcb95"
        ),
        "realized_roi_definition": (
            "identical_base_portfolio_integer_settlement_return_divided_by_stake"
        ),
        "strict_prior_calibration_only": True,
        "independent_validation_value_only": bool(
            rows and excluded_non_independent == 0
        ),
        "identical_realized_portfolio_only": bool(
            rows and excluded_mismatched == 0
        ),
        "candidate_manifest_sha256": _canonical_sha256(manifest),
        "candidate_portfolios": len(rows),
        "excluded_mismatched_portfolios": excluded_mismatched,
        "excluded_non_independent_portfolios": excluded_non_independent,
        "quantile_bins": len(bins),
        "monotone_realized_roi": bool(
            realized
            and all(left <= right for left, right in zip(realized, realized[1:]))
        ),
        "deciles": bins,
    }


def _purchase_gate_outcome(
    *,
    mature_observation_window: bool,
    observed_purchased_portfolios: int,
    safety_invariants_passed: bool,
    promotion_evidence_passed: bool,
) -> str:
    if not mature_observation_window:
        return "accumulating_strict_prior_calibration"
    if observed_purchased_portfolios == 0 and safety_invariants_passed:
        return "safe_abstention_no_demonstrated_price_advantage"
    if promotion_evidence_passed:
        return "promotion_evidence_passed"
    return "formal_purchase_evidence_rejected"


def run_joint_edge_calibrated_replay(
    base_artifact: Path,
    *,
    scored_cache: Path | None = None,
    initial_daily_bankroll_yen: int = 10_000,
    calibration_margin: float = 0.0,
    calibration_bootstrap_samples: int = 5_000,
    calibration_min_training_days: int = 30,
    calibration_min_portfolios: int = 300,
    calibration_min_candidate_days: int = 20,
    calibration_min_local_candidates: int = 50,
    calibration_min_local_candidate_days: int = 20,
    calibration_min_local_ess: float = 10.0,
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
    base_artifact_sha256 = _sha256_file(base_artifact)
    scored_cache_sha256 = (
        _sha256_file(cache_path)
        if cache_path is not None and cache_path.is_file() else None
    )
    implementation_source_sha256 = {
        "joint_edge_calibrated_replay": _sha256_file(Path(__file__)),
        "empirical_ev_calibration": _sha256_file(Path(
            fit_empirical_ev_calibration.__code__.co_filename
        )),
        "joint_bankroll_evaluation": _sha256_file(Path(
            build_block_bootstrap_evidence.__code__.co_filename
        )),
    }
    implementation_sha256 = _canonical_sha256(
        implementation_source_sha256
    )
    replay_configuration = {
        "model_version": MODEL_VERSION,
        "calibration_version": CALIBRATION_VERSION,
        "initial_daily_bankroll_yen": initial_daily_bankroll_yen,
        "purchase_unit_yen": PURCHASE_UNIT_YEN,
        "buy_margin": buy_margin,
        "calibration_margin": calibration_margin,
        "calibration_bootstrap_samples": calibration_bootstrap_samples,
        "calibration_min_training_days": calibration_min_training_days,
        "calibration_min_portfolios": calibration_min_portfolios,
        "calibration_min_candidate_days": calibration_min_candidate_days,
        "calibration_min_local_candidates": (
            calibration_min_local_candidates
        ),
        "calibration_min_local_candidate_days": (
            calibration_min_local_candidate_days
        ),
        "calibration_min_local_ess": calibration_min_local_ess,
        "bankroll_bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "shape_constraint": "isotonic",
        "quantile_method": "inverted_cdf",
        "lcb_tail_probability": 0.05,
        "bootstrap_cluster_unit": "race_date",
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "joblib": joblib.__version__,
            "operating_system": platform.system(),
            "kernel_release": platform.release(),
            "machine": platform.machine(),
        },
    }
    replay_configuration_sha256 = _canonical_sha256(
        replay_configuration
    )
    model_decision_sha256 = _canonical_sha256({
        "base_artifact_sha256": base_artifact_sha256,
        "base_model": payload.get("model"),
        "base_evaluation_protocol_id": payload.get(
            "evaluation_protocol_id"
        ),
        "replay_implementation_sha256": implementation_source_sha256[
            "joint_edge_calibrated_replay"
        ],
    })
    threshold_definition = {
        "base_buy_margin": buy_margin,
        "calibration_margin": calibration_margin,
        "calibration_warmup": {
            "days": calibration_min_training_days,
            "candidates": calibration_min_portfolios,
            "candidate_days": calibration_min_candidate_days,
        },
        "local_support": {
            "candidates": calibration_min_local_candidates,
            "candidate_days": calibration_min_local_candidate_days,
            "day_cluster_ess": calibration_min_local_ess,
        },
        "purchase_condition": (
            "in_global_and_local_observed_range_and_local_support_ready_"
            "and_lcb95_strictly_greater_than_one_plus_margin"
        ),
    }
    threshold_definition_sha256 = _canonical_sha256(
        threshold_definition
    )
    inherited_protocol = payload.get("evaluation_protocol")
    inherited_protocol = (
        inherited_protocol if isinstance(inherited_protocol, dict) else {}
    )
    settlement_engine_sha256 = _canonical_sha256({
        "implementation_source_sha256": implementation_source_sha256[
            "joint_bankroll_evaluation"
        ],
        "settlement_protocol": inherited_protocol.get("settlement"),
        "purchase_unit_yen": PURCHASE_UNIT_YEN,
        "receipt_rule": (
            "recorded_actual_combination_and_payout_released_only_after_"
            "settlement_available_at"
        ),
    })

    training_records: list[dict[str, object]] = []
    pending_training_records: list[dict[str, object]] = []
    teacher_admissions: list[dict[str, object]] = []
    observed_candidate_race_ids: set[str] = set()
    duplicate_result_batches_excluded = 0
    pregate_candidates_generated = 0
    pregate_candidates_missing_independent_value = 0
    replay_days = []
    ready_days = 0
    ready_races = 0
    calibrated_candidates = 0
    rejected_reasons: dict[str, int] = {}
    calibration_folds = []
    calibrator_cache: dict[str, Any] = {}
    calibrator_update_events: list[dict[str, object]] = []
    previous_calibration_instance_id: str | None = None
    previous_training_ledger_sha256: str | None = None

    def calibrator_at_decision(
        *,
        race_date: str,
        race_id: str,
        calibration_asof: datetime,
    ) -> dict[str, Any]:
        nonlocal pending_training_records
        nonlocal previous_calibration_instance_id
        nonlocal previous_training_ledger_sha256

        matured_training_records = [
            record for record in pending_training_records
            if _instant(
                record.get("settlement_available_at"),
                "pending_settlement_available_at",
            ) < calibration_asof
        ]
        pending_training_records = [
            record for record in pending_training_records
            if _instant(
                record.get("settlement_available_at"),
                "pending_settlement_available_at",
            ) >= calibration_asof
        ]
        for record in matured_training_records:
            admitted = {
                **record,
                "teacher_admitted_at": calibration_asof.isoformat(),
            }
            training_records.append(admitted)
            teacher_admissions.append({
                "race_id": admitted["race_id"],
                "settlement_available_at": admitted[
                    "settlement_available_at"
                ],
                "teacher_admitted_at": admitted["teacher_admitted_at"],
            })
        same_race_excluded_records = [
            record for record in training_records
            if str(record.get("race_id") or "") == race_id
        ]
        eligible_training_records = [
            record for record in training_records
            if str(record.get("race_id") or "") != race_id
        ]
        eligible_training_race_ids = [
            str(record.get("race_id") or "")
            for record in eligible_training_records
        ]
        if len(eligible_training_race_ids) != len(
            set(eligible_training_race_ids)
        ):
            raise ValueError(
                "calibration teacher contains duplicate race_id"
            )
        same_race_teacher_overlap = sorted(
            {race_id}.intersection(eligible_training_race_ids)
        )
        training_race_manifest_sha256 = _canonical_sha256(
            sorted(eligible_training_race_ids)
        )
        training_ledger_sha256 = _calibration_ledger_sha256(
            eligible_training_records
        )
        calibration_fit_seed = (
            seed + int(training_ledger_sha256[:16], 16)
        ) % (2**32 - 1)
        calibration_instance_id = _canonical_sha256({
            "calibration_version": CALIBRATION_VERSION,
            "training_race_manifest_sha256": (
                training_race_manifest_sha256
            ),
            "training_ledger_sha256": training_ledger_sha256,
            "bootstrap_samples": calibration_bootstrap_samples,
            "fit_seed": calibration_fit_seed,
            "margin": calibration_margin,
            "min_training_days": calibration_min_training_days,
            "min_portfolios": calibration_min_portfolios,
            "min_candidate_days": calibration_min_candidate_days,
            "min_local_candidates": calibration_min_local_candidates,
            "min_local_candidate_days": (
                calibration_min_local_candidate_days
            ),
            "min_local_ess": calibration_min_local_ess,
            "candidate_min_raw_ev": 0.0,
            "shape_constraint": "isotonic",
            "quantile_method": "inverted_cdf",
        })
        calibrator_cache_hit = calibration_instance_id in calibrator_cache
        teacher_population_changed = bool(
            previous_training_ledger_sha256 is None
            or previous_training_ledger_sha256
            != training_ledger_sha256
        )
        calibrator_instance_changed = bool(
            previous_calibration_instance_id is None
            or previous_calibration_instance_id != calibration_instance_id
        )
        latest_training_settlement_instant = max(
            (
                _instant(
                    record.get("settlement_available_at"),
                    "training_settlement_available_at",
                )
                for record in eligible_training_records
            ),
            default=None,
        )
        latest_training_settlement = (
            latest_training_settlement_instant.isoformat()
            if latest_training_settlement_instant is not None else None
        )
        if calibrator_cache_hit:
            calibrator = calibrator_cache[calibration_instance_id]
        else:
            calibrator = fit_empirical_ev_calibration(
                eligible_training_records,
                bootstrap_samples=calibration_bootstrap_samples,
                seed=calibration_fit_seed,
                min_days=min(
                    calibration_min_training_days,
                    calibration_min_candidate_days,
                ),
                min_tickets=calibration_min_portfolios,
                min_candidate_days=calibration_min_candidate_days,
                min_local_candidates=calibration_min_local_candidates,
                min_local_candidate_days=(
                    calibration_min_local_candidate_days
                ),
                min_local_ess=calibration_min_local_ess,
                candidate_min_raw_ev=0.0,
                shape_constraint="isotonic",
                quantile_method="inverted_cdf",
            )
            calibrator_cache[calibration_instance_id] = calibrator
        calibrator_artifact_sha256 = _canonical_sha256(
            _finite_audit_value(calibrator.as_dict())
        )
        training_calendar_span_days = _calendar_span_days(
            eligible_training_records
        )
        calibration_ready = _strict_warmup_ready(
            calibrator, eligible_training_records,
            minimum_calendar_days=calibration_min_training_days,
        )
        event = {
            "race_id": race_id,
            "evaluation_date": race_date,
            "evaluation_time_t": calibration_asof.isoformat(),
            "training_race_manifest_sha256": (
                training_race_manifest_sha256
            ),
            "training_ledger_sha256": training_ledger_sha256,
            "calibration_instance_id": calibration_instance_id,
            "calibrator_artifact_sha256": calibrator_artifact_sha256,
            "teacher_population_changed": teacher_population_changed,
            "calibrator_instance_changed": calibrator_instance_changed,
            "cache_hit": calibrator_cache_hit,
            "fit_seed": calibration_fit_seed,
        }
        calibrator_update_events.append(event)
        fold = {
            **event,
            "calibration_information_cutoff": calibration_asof.isoformat(),
            "calibration_fit_seed": calibration_fit_seed,
            "calibrator_cache_hit": calibrator_cache_hit,
            "newly_admitted_settled_race_batches": len(
                matured_training_records
            ),
            "pending_unsettled_race_batches": len(
                pending_training_records
            ),
            "settlement_eligible_training_records": len(
                eligible_training_records
            ),
            "settlement_excluded_training_records": len(
                pending_training_records
            ),
            "same_race_excluded_training_records": len(
                same_race_excluded_records
            ),
            "same_race_teacher_overlap_count": len(
                same_race_teacher_overlap
            ),
            "same_race_teacher_overlap_race_ids": (
                same_race_teacher_overlap
            ),
            "eligible_training_unique_races": len(
                eligible_training_race_ids
            ),
            "eligible_training_race_manifest_sha256": (
                training_race_manifest_sha256
            ),
            "eligible_training_ledger_sha256": training_ledger_sha256,
            "latest_training_settlement_available_at": (
                latest_training_settlement
            ),
            "strict_settlement_before_decision": bool(
                latest_training_settlement_instant is None
                or latest_training_settlement_instant < calibration_asof
            ),
            **calibrator.as_dict(),
            "training_calendar_span_days": training_calendar_span_days,
            "ready": calibration_ready,
        }
        calibration_folds.append(fold)
        previous_calibration_instance_id = calibration_instance_id
        previous_training_ledger_sha256 = training_ledger_sha256
        return {
            "calibrator": calibrator,
            "fold": fold,
            "latest_training_settlement_instant": (
                latest_training_settlement_instant
            ),
            "latest_training_settlement": latest_training_settlement,
            "eligible_training_records": eligible_training_records,
            "same_race_teacher_overlap": same_race_teacher_overlap,
            "training_race_manifest_sha256": (
                training_race_manifest_sha256
            ),
            "training_ledger_sha256": training_ledger_sha256,
            "calibration_instance_id": calibration_instance_id,
            "calibrator_artifact_sha256": calibrator_artifact_sha256,
        }

    for source_day in payload["daily"]:
        race_date = str(source_day.get("race_date") or "")
        source_races = source_day.get("races") or []
        evaluation_race_ids = [
            str(race.get("race_id") or "") for race in source_races
        ]
        if any(not race_id for race_id in evaluation_race_ids):
            raise ValueError("every evaluation race requires race_id")
        if len(evaluation_race_ids) != len(set(evaluation_race_ids)):
            raise ValueError("evaluation fold contains duplicate race_id")
        # This fold belongs to the first decision of the day. Later races get
        # their own fold below, so only the race being decided may be excluded
        # from its teacher ledger here.
        evaluation_race_id_set = (
            {evaluation_race_ids[0]} if evaluation_race_ids else set()
        )
        day_decision_instants = [
            _decision_time(race, decision_times) for race in source_races
        ]
        calibration_asof = min(
            day_decision_instants,
            default=_instant(
                f"{race_date}T00:00:00+09:00",
                "calibration_asof",
            ),
        )
        matured_training_records = [
            record for record in pending_training_records
            if _instant(
                record.get("settlement_available_at"),
                "pending_settlement_available_at",
            ) < calibration_asof
        ]
        pending_training_records = [
            record for record in pending_training_records
            if _instant(
                record.get("settlement_available_at"),
                "pending_settlement_available_at",
            ) >= calibration_asof
        ]
        for record in matured_training_records:
            admitted = {
                **record,
                "teacher_admitted_at": calibration_asof.isoformat(),
            }
            training_records.append(admitted)
            teacher_admissions.append({
                "race_id": admitted["race_id"],
                "settlement_available_at": admitted[
                    "settlement_available_at"
                ],
                "teacher_admitted_at": admitted["teacher_admitted_at"],
            })
        settled_training_records = list(training_records)
        same_race_excluded_records = [
            record for record in settled_training_records
            if str(record.get("race_id") or "") in evaluation_race_id_set
        ]
        eligible_training_records = [
            record for record in settled_training_records
            if str(record.get("race_id") or "") not in evaluation_race_id_set
        ]
        excluded_unsettled_records = len(pending_training_records)
        eligible_training_race_ids = [
            str(record.get("race_id") or "")
            for record in eligible_training_records
        ]
        if len(eligible_training_race_ids) != len(
            set(eligible_training_race_ids)
        ):
            raise ValueError(
                "calibration teacher contains duplicate race_id"
            )
        same_race_teacher_overlap = sorted(
            evaluation_race_id_set.intersection(eligible_training_race_ids)
        )
        training_race_manifest_sha256 = _canonical_sha256(
            sorted(eligible_training_race_ids)
        )
        training_ledger_sha256 = _calibration_ledger_sha256(
            eligible_training_records
        )
        calibration_fit_seed = (
            seed + int(training_ledger_sha256[:16], 16)
        ) % (2**32 - 1)
        calibration_instance_id = _canonical_sha256({
            "calibration_version": CALIBRATION_VERSION,
            "training_race_manifest_sha256": (
                training_race_manifest_sha256
            ),
            "training_ledger_sha256": training_ledger_sha256,
            "bootstrap_samples": calibration_bootstrap_samples,
            "fit_seed": calibration_fit_seed,
            "margin": calibration_margin,
            "min_training_days": calibration_min_training_days,
            "min_portfolios": calibration_min_portfolios,
            "min_candidate_days": calibration_min_candidate_days,
            "min_local_candidates": calibration_min_local_candidates,
            "min_local_candidate_days": (
                calibration_min_local_candidate_days
            ),
            "min_local_ess": calibration_min_local_ess,
            "candidate_min_raw_ev": 0.0,
            "shape_constraint": "isotonic",
            "quantile_method": "inverted_cdf",
        })
        calibrator_cache_hit = calibration_instance_id in calibrator_cache
        teacher_population_changed = bool(
            previous_training_ledger_sha256 is None
            or previous_training_ledger_sha256
            != training_ledger_sha256
        )
        calibrator_instance_changed = bool(
            previous_calibration_instance_id is None
            or previous_calibration_instance_id != calibration_instance_id
        )
        latest_training_settlement_instant = max(
            (
                _instant(
                    record.get("settlement_available_at"),
                    "training_settlement_available_at",
                )
                for record in eligible_training_records
            ),
            default=None,
        )
        latest_training_settlement = (
            latest_training_settlement_instant.isoformat()
            if latest_training_settlement_instant is not None else None
        )
        if calibrator_cache_hit:
            calibrator = calibrator_cache[calibration_instance_id]
        else:
            calibrator = fit_empirical_ev_calibration(
                eligible_training_records,
                bootstrap_samples=calibration_bootstrap_samples,
                seed=calibration_fit_seed,
                min_days=min(
                    calibration_min_training_days,
                    calibration_min_candidate_days,
                ),
                min_tickets=calibration_min_portfolios,
                min_candidate_days=calibration_min_candidate_days,
                min_local_candidates=calibration_min_local_candidates,
                min_local_candidate_days=(
                    calibration_min_local_candidate_days
                ),
                min_local_ess=calibration_min_local_ess,
                candidate_min_raw_ev=0.0,
                shape_constraint="isotonic",
                quantile_method="inverted_cdf",
            )
            calibrator_cache[calibration_instance_id] = calibrator
        calibrator_artifact_sha256 = _canonical_sha256(
            _finite_audit_value(calibrator.as_dict())
        )
        training_calendar_span_days = _calendar_span_days(
            eligible_training_records
        )
        calibration_ready = _strict_warmup_ready(
            calibrator, eligible_training_records,
            minimum_calendar_days=calibration_min_training_days,
        )
        calibrator_update_events.append({
            "race_id": (
                evaluation_race_ids[0] if evaluation_race_ids else ""
            ),
            "evaluation_date": race_date,
            "evaluation_time_t": calibration_asof.isoformat(),
            "training_race_manifest_sha256": (
                training_race_manifest_sha256
            ),
            "training_ledger_sha256": training_ledger_sha256,
            "calibration_instance_id": calibration_instance_id,
            "calibrator_artifact_sha256": (
                calibrator_artifact_sha256
            ),
            "teacher_population_changed": teacher_population_changed,
            "calibrator_instance_changed": calibrator_instance_changed,
            "cache_hit": calibrator_cache_hit,
            "fit_seed": calibration_fit_seed,
        })
        calibration_folds.append({
            "race_id": (
                evaluation_race_ids[0] if evaluation_race_ids else ""
            ),
            "evaluation_date": race_date,
            "evaluation_time_t": calibration_asof.isoformat(),
            "calibration_information_cutoff": calibration_asof.isoformat(),
            "calibration_instance_id": calibration_instance_id,
            "calibrator_artifact_sha256": (
                calibrator_artifact_sha256
            ),
            "calibration_fit_seed": calibration_fit_seed,
            "calibrator_cache_hit": calibrator_cache_hit,
            "teacher_population_changed": teacher_population_changed,
            "calibrator_instance_changed": calibrator_instance_changed,
            "newly_admitted_settled_race_batches": len(
                matured_training_records
            ),
            "pending_unsettled_race_batches": len(
                pending_training_records
            ),
            "settlement_eligible_training_records": len(
                eligible_training_records
            ),
            "settlement_excluded_training_records": (
                excluded_unsettled_records
            ),
            "same_race_excluded_training_records": len(
                same_race_excluded_records
            ),
            "same_race_teacher_overlap_count": len(
                same_race_teacher_overlap
            ),
            "same_race_teacher_overlap_race_ids": (
                same_race_teacher_overlap
            ),
            "eligible_training_unique_races": len(
                eligible_training_race_ids
            ),
            "eligible_training_race_manifest_sha256": (
                training_race_manifest_sha256
            ),
            "eligible_training_ledger_sha256": training_ledger_sha256,
            "latest_training_settlement_available_at": (
                latest_training_settlement
            ),
            "strict_settlement_before_decision": bool(
                latest_training_settlement_instant is None
                or latest_training_settlement_instant < calibration_asof
            ),
            **calibrator.as_dict(),
            "training_calendar_span_days": training_calendar_span_days,
            "ready": calibration_ready,
        })
        previous_calibration_instance_id = calibration_instance_id
        previous_training_ledger_sha256 = training_ledger_sha256
        balance = initial_daily_bankroll_yen
        peak = balance
        maximum_drawdown_yen = 0
        stake_yen = 0
        return_yen = 0
        hits = 0
        pending: list[tuple[datetime, int]] = []
        replay_races = []
        day_has_ready = False
        day_folds: list[dict[str, Any]] = []

        for race_index, (race, purchase_at) in enumerate(zip(
            source_races, day_decision_instants
        )):
            pending, matured = _release_matured_receipts(
                pending, asof=purchase_at
            )
            balance += matured
            peak = max(peak, balance)
            if race_index == 0:
                fold = calibration_folds[-1]
            else:
                state = calibrator_at_decision(
                    race_date=race_date,
                    race_id=str(race.get("race_id") or ""),
                    calibration_asof=purchase_at,
                )
                calibrator = state["calibrator"]
                fold = state["fold"]
                calibration_asof = purchase_at
                calibration_instance_id = state[
                    "calibration_instance_id"
                ]
                calibrator_artifact_sha256 = state[
                    "calibrator_artifact_sha256"
                ]
                latest_training_settlement_instant = state[
                    "latest_training_settlement_instant"
                ]
                latest_training_settlement = state[
                    "latest_training_settlement"
                ]
                eligible_training_records = state[
                    "eligible_training_records"
                ]
                same_race_teacher_overlap = state[
                    "same_race_teacher_overlap"
                ]
                training_race_manifest_sha256 = state[
                    "training_race_manifest_sha256"
                ]
                training_ledger_sha256 = state[
                    "training_ledger_sha256"
                ]
            day_folds.append(fold)
            decision_calibration_ready = bool(fold.get("ready"))
            if decision_calibration_ready:
                day_has_ready = True
                ready_races += 1
            record = _candidate_record(
                race_date, race, buy_margin=buy_margin
            )
            pregate_generated = bool(
                race.get("pregate_candidate_generated")
                if "pregate_candidate_generated" in race
                else int(race.get("best_search_stake_yen") or 0) > 0
            )
            if pregate_generated:
                pregate_candidates_generated += 1
                if record is None:
                    pregate_candidates_missing_independent_value += 1
            if record is not None:
                record_race_id = str(record["race_id"])
                if record_race_id in observed_candidate_race_ids:
                    duplicate_result_batches_excluded += 1
                else:
                    if _instant(
                        record.get("settlement_available_at"),
                        "settlement_available_at",
                    ) < purchase_at:
                        raise ValueError(
                            "candidate settlement precedes decision"
                        )
                    pending_training_records.append(record)
                    observed_candidate_race_ids.add(record_race_id)

            raw_gross = (
                float(record["raw_estimated_ev"])
                if record is not None else None
            )
            prediction = (
                calibrator.predict(raw_gross)
                if decision_calibration_ready and raw_gross is not None else {}
            )
            unqualified_bootstrap_lcb = prediction.get(
                "empirical_ev_lcb95"
            )
            input_in_training_range = bool(
                prediction.get("input_in_training_range")
            )
            input_in_local_block_range = bool(
                prediction.get("input_in_local_block_range")
            )
            local_support_ready = bool(
                prediction.get("local_support_ready")
            )
            calibrated_lcb = (
                unqualified_bootstrap_lcb
                if (
                    input_in_training_range
                    and input_in_local_block_range
                    and local_support_ready
                )
                else None
            )
            structural_feasible = bool(
                record is not None and record["structural_feasible"]
            )
            authorized = bool(
                decision_calibration_ready
                and structural_feasible
                and input_in_training_range
                and input_in_local_block_range
                and local_support_ready
                and calibrated_lcb is not None
                and float(calibrated_lcb) > 1.0 + calibration_margin
            )
            reason = None
            if not decision_calibration_ready:
                reason = "calibration_not_ready"
            elif not structural_feasible:
                reason = "base_joint_gate_not_feasible"
            elif not input_in_training_range:
                reason = "calibration_input_out_of_training_range"
            elif not input_in_local_block_range:
                reason = "calibration_input_out_of_local_block_range"
            elif not local_support_ready:
                reason = "calibration_local_support_insufficient"
            elif calibrated_lcb is None:
                reason = "calibration_lcb_missing"
            elif float(calibrated_lcb) <= 1.0 + calibration_margin:
                reason = "calibration_lcb_not_above_margin"

            raw_proposed_bets = race.get("best_search_bets_yen") or {}
            proposed_bets = {
                str(ticket): int(stake)
                for ticket, stake in raw_proposed_bets.items()
            } if isinstance(raw_proposed_bets, Mapping) else {}
            counterfactual_stake = sum(proposed_bets.values())
            counterfactual_return = _realized_receipt(
                proposed_bets,
                actual_combination=str(race.get("actual_combination") or ""),
                actual_payout_yen=int(race.get("actual_payout_yen") or 0),
            )
            recorded_stake = int(race.get("best_search_stake_yen") or 0)
            recorded_return = int(
                race.get("best_search_hypothetical_return_yen") or 0
            )
            counterfactual_matches = bool(
                counterfactual_stake > 0
                and counterfactual_stake == recorded_stake
                and counterfactual_return == recorded_return
            )
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

            decision_hash_bundle = {
                "model_sha256": model_decision_sha256,
                "calibrator_sha256": calibrator_artifact_sha256,
                "calibration_ledger_sha256": training_ledger_sha256,
                "threshold_sha256": threshold_definition_sha256,
                "settlement_engine_sha256": settlement_engine_sha256,
            }
            decision_hash_bundle_sha256 = _canonical_sha256(
                decision_hash_bundle
            )

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
                "raw_V_buy": (
                    raw_gross - 1.0 if raw_gross is not None else None
                ),
                "raw_value_source": (
                    record.get("raw_value_source") if record else None
                ),
                "calibrated_gross_return": prediction.get("empirical_ev"),
                "calibrated_gross_return_lcb95": calibrated_lcb,
                "diagnostic_unqualified_bootstrap_lcb95": (
                    unqualified_bootstrap_lcb
                ),
                "calibrated_ROI": prediction.get("empirical_ev"),
                "calibrated_ROI_LCB95": calibrated_lcb,
                "calibration_lcb_tail_probability": (
                    calibrator.as_dict()["lcb_tail_probability"]
                ),
                "calibration_lcb_confidence_level": (
                    calibrator.as_dict()["lcb_confidence_level"]
                ),
                "calibration_lcb_sidedness": (
                    calibrator.as_dict()["lcb_sidedness"]
                ),
                "calibration_lcb_estimator": (
                    calibrator.as_dict()["lcb_estimator"]
                ),
                "calibration_lcb_cluster_unit": (
                    calibrator.as_dict()["bootstrap_cluster_unit"]
                ),
                "formal_purchase_value": (
                    float(calibrated_lcb) - 1.0
                    if calibrated_lcb is not None else None
                ),
                "calibration_support": prediction.get("support"),
                "calibration_support_days": prediction.get("support_days"),
                "isotonic_block_count": calibrator.isotonic_block_count,
                "isotonic_block_id": prediction.get(
                    "isotonic_block_id"
                ),
                "local_block_candidates": prediction.get(
                    "local_block_candidates"
                ),
                "local_block_candidate_days": prediction.get(
                    "local_block_candidate_days"
                ),
                "local_block_ess": prediction.get("local_block_ess"),
                "local_block_raw_ev_min": prediction.get(
                    "local_block_raw_ev_min"
                ),
                "local_block_raw_ev_max": prediction.get(
                    "local_block_raw_ev_max"
                ),
                "local_support_ready": local_support_ready,
                "local_support_reasons": prediction.get(
                    "local_support_reasons"
                ),
                "calibration_input_in_local_block_range": (
                    input_in_local_block_range
                ),
                "local_required_candidates": (
                    calibrator.min_local_candidates
                ),
                "local_required_candidate_days": (
                    calibrator.min_local_candidate_days
                ),
                "local_required_ess": calibrator.min_local_ess,
                "calibration_training_raw_input_min": prediction.get(
                    "training_raw_ev_min"
                ),
                "calibration_training_raw_input_max": prediction.get(
                    "training_raw_ev_max"
                ),
                "calibration_input_in_training_range": (
                    input_in_training_range
                ),
                "calibration_trained_through_date": (
                    calibrator.trained_through_date
                ),
                "calibration_ready": decision_calibration_ready,
                "warmup_days": int(
                    fold.get("training_calendar_span_days") or 0
                ),
                "observed_race_days": calibrator.training_days,
                "required_days": calibration_min_training_days,
                "prior_candidates": calibrator.tickets,
                "required_candidates": calibrator.min_tickets,
                "prior_candidate_days": calibrator.candidate_days,
                "required_candidate_days": (
                    calibrator.min_candidate_days
                ),
                "buy_threshold": 1.0 + calibration_margin,
                "calibration_instance_id": calibration_instance_id,
                "calibrator_artifact_sha256": (
                    calibrator_artifact_sha256
                ),
                "ticket_calibration_instance_count": (
                    1 if proposed_bets else 0
                ),
                "calibration_information_cutoff": (
                    calibration_asof.isoformat()
                ),
                "calibration_cutoff_time": calibration_asof.isoformat(),
                "calibration_latest_training_settlement_available_at": (
                    latest_training_settlement
                ),
                "max_training_settlement_time": (
                    latest_training_settlement
                ),
                "calibration_latest_settlement_strictly_before_"
                "evaluation_time_t": bool(
                    latest_training_settlement_instant is None
                    or latest_training_settlement_instant < purchase_at
                ),
                "strict_prior_check": bool(
                    latest_training_settlement_instant is None
                    or latest_training_settlement_instant < purchase_at
                ),
                "calibration_settlement_eligible_training_records": len(
                    eligible_training_records
                ),
                "calibration_same_race_teacher_overlap_count": len(
                    same_race_teacher_overlap
                ),
                "calibration_training_race_manifest_sha256": (
                    training_race_manifest_sha256
                ),
                "model_sha256": model_decision_sha256,
                "calibrator_sha256": calibrator_artifact_sha256,
                "calibrator_hash": calibrator_artifact_sha256,
                "calibration_ledger_sha256": training_ledger_sha256,
                "calibration_ledger_hash": training_ledger_sha256,
                "threshold_sha256": threshold_definition_sha256,
                "settlement_engine_sha256": settlement_engine_sha256,
                "decision_hash_bundle_sha256": (
                    decision_hash_bundle_sha256
                ),
                "base_joint_gate_feasible": structural_feasible,
                "counterfactual_stake_yen": counterfactual_stake,
                "counterfactual_ticket_count": len(proposed_bets),
                "counterfactual_return_yen": counterfactual_return,
                "counterfactual_profit_yen": (
                    counterfactual_return - counterfactual_stake
                ),
                "counterfactual_matches_base_portfolio": (
                    counterfactual_matches
                ),
                "purchase_authorized": authorized,
                "approved": authorized,
                "denied": not authorized,
                "rejection_reason": reason,
                "denial_reason": reason,
                "bets_yen": bets,
                "stake_yen": stake,
                "return_yen": receipt,
                "profit_yen": receipt - stake,
            })
        for _available_at, receipt in pending:
            balance += receipt
            peak = max(peak, balance)
        if day_has_ready:
            ready_days += 1
        first_fold = day_folds[0] if day_folds else {}
        last_fold = day_folds[-1] if day_folds else {}
        replay_days.append({
            "race_date": race_date,
            "calibration_information_cutoff": first_fold.get(
                "calibration_information_cutoff"
            ),
            "calibration_instance_id": last_fold.get(
                "calibration_instance_id"
            ),
            "calibration_instance_ids": sorted({
                str(fold.get("calibration_instance_id") or "")
                for fold in day_folds
            }),
            "calibrator_artifact_sha256": last_fold.get(
                "calibrator_artifact_sha256"
            ),
            "calibration_fit_seed": last_fold.get("calibration_fit_seed"),
            "calibrator_cache_hit": last_fold.get("calibrator_cache_hit"),
            "teacher_population_changed": last_fold.get(
                "teacher_population_changed"
            ),
            "calibrator_instance_changed": last_fold.get(
                "calibrator_instance_changed"
            ),
            "newly_admitted_settled_race_batches": sum(
                int(fold.get("newly_admitted_settled_race_batches") or 0)
                for fold in day_folds
            ),
            "pending_unsettled_race_batches": last_fold.get(
                "pending_unsettled_race_batches"
            ) if not source_races else len(pending_training_records),
            "settlement_eligible_training_records": last_fold.get(
                "settlement_eligible_training_records"
            ),
            "settlement_excluded_training_records": last_fold.get(
                "settlement_excluded_training_records"
            ),
            "same_race_excluded_training_records": sum(
                int(fold.get("same_race_excluded_training_records") or 0)
                for fold in day_folds
            ),
            "same_race_teacher_overlap_count": sum(
                int(fold.get("same_race_teacher_overlap_count") or 0)
                for fold in day_folds
            ),
            "eligible_training_unique_races": last_fold.get(
                "eligible_training_unique_races"
            ),
            "eligible_training_race_manifest_sha256": last_fold.get(
                "eligible_training_race_manifest_sha256"
            ),
            "calibration_ready": day_has_ready,
            "calibration_trained_through_date": last_fold.get(
                "trained_through_date"
            ),
            "opening_bankroll_yen": initial_daily_bankroll_yen,
            "closing_bankroll_yen": balance,
            "stake_yen": stake_yen,
            "return_yen": return_yen,
            "profit_yen": return_yen - stake_yen,
            "roi": return_yen / stake_yen if stake_yen else None,
            "roi_status": "defined" if stake_yen else "not_applicable",
            "roi_not_applicable_reason": (
                None if stake_yen else (
                    "warmup_no_purchase"
                    if not day_has_ready
                    else "no_candidate_passed_purchase_gate"
                )
            ),
            "hits": hits,
            "max_drawdown_yen": maximum_drawdown_yen,
            "evaluated_races": len(source_races),
            "races": replay_races,
        })

    confidence = build_block_bootstrap_evidence(
        replay_days, samples=bootstrap_samples, seed=seed
    )
    value_realization = _calibrated_value_realization(
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
    all_candidate_records = [
        *training_records,
        *pending_training_records,
    ]
    teacher_admission_violations = [
        row for row in teacher_admissions
        if _instant(
            row.get("settlement_available_at"),
            "teacher_settlement_available_at",
        ) >= _instant(
            row.get("teacher_admitted_at"),
            "teacher_admitted_at",
        )
    ]
    ticket_calibrator_rows = [
        race
        for day in replay_days
        for race in day.get("races") or []
        if int(race.get("counterfactual_stake_yen") or 0) > 0
    ]
    ticket_calibrator_violations = [
        race for race in ticket_calibrator_rows
        if int(race.get("ticket_calibration_instance_count") or 0) != 1
        or not race.get("calibration_instance_id")
    ]
    learning_population_audit = {
        "version": "candidate_portfolio_learning_population_v1",
        "independent_sample_unit": (
            "one_fixed_counterfactual_portfolio_per_race"
        ),
        "inclusion_rule": (
            "all_ex_ante_nonzero_best_search_portfolios_with_recorded_"
            "raw_value_and_realized_settlement"
        ),
        "outcome_filter": "none",
        "purchase_filter": "none_includes_purchased_and_rejected",
        "candidate_portfolios": len(all_candidate_records),
        "pregate_candidates_generated": pregate_candidates_generated,
        "pregate_candidates_registered": len(all_candidate_records),
        "pregate_candidates_missing_independent_value": (
            pregate_candidates_missing_independent_value
        ),
        "all_pregate_candidates_registered": bool(
            len(all_candidate_records) == pregate_candidates_generated
            and pregate_candidates_missing_independent_value == 0
            and duplicate_result_batches_excluded == 0
        ),
        "unique_races": len({
            str(record.get("race_id") or "")
            for record in all_candidate_records
        }),
        "positive_return_portfolios": sum(
            float(record.get("gross_return_per_yen") or 0.0) > 0.0
            for record in all_candidate_records
        ),
        "zero_return_portfolios": sum(
            float(record.get("gross_return_per_yen") or 0.0) == 0.0
            for record in all_candidate_records
        ),
        "structurally_feasible_portfolios": sum(
            bool(record.get("structural_feasible"))
            for record in all_candidate_records
        ),
        "structurally_rejected_portfolios": sum(
            not bool(record.get("structural_feasible"))
            for record in all_candidate_records
        ),
        "duplicate_race_result_batches_excluded": (
            duplicate_result_batches_excluded
        ),
        "population_manifest_sha256": _canonical_sha256([
            [
                record.get("race_id"),
                record.get("race_date"),
                record.get("raw_estimated_ev"),
                record.get("gross_return_per_yen"),
                record.get("sample_weight"),
                record.get("raw_value_source"),
                record.get("structural_feasible"),
            ]
            for record in sorted(
                all_candidate_records,
                key=lambda row: str(row.get("race_id") or ""),
            )
        ]),
    }
    race_batch_audit = {
        "version": "same_race_calibrator_settlement_batch_audit_v1",
        "candidate_races_with_nonzero_portfolio": len(
            ticket_calibrator_rows
        ),
        "ticket_calibrator_instance_violations": len(
            ticket_calibrator_violations
        ),
        "same_race_calibrator_hash_count_max": (
            1 if ticket_calibrator_rows else 0
        ),
        "same_race_mid_decision_update_count": 0,
        "same_race_result_leakage_count": len(
            teacher_admission_violations
        ),
        "all_tickets_in_race_share_one_prior_calibrator": bool(
            ticket_calibrator_rows and not ticket_calibrator_violations
        ),
        "teacher_admitted_race_batches": len(teacher_admissions),
        "pending_unsettled_race_batches": len(pending_training_records),
        "teacher_admission_before_settlement_violations": len(
            teacher_admission_violations
        ),
        "results_admitted_only_after_strict_settlement": bool(
            teacher_admissions and not teacher_admission_violations
        ),
        "result_batch_unit": "one_race_candidate_portfolio",
        "teacher_admission_manifest_sha256": _canonical_sha256(
            teacher_admissions
        ),
    }
    warmup_boundaries = [
        {
            "evaluation_date": str(fold.get("evaluation_date") or ""),
            "calendar_span_days": int(
                fold.get("training_calendar_span_days") or 0
            ),
            "training_days": int(
                fold.get("training_calendar_span_days") or 0
            ),
            "observed_race_days": int(fold.get("training_days") or 0),
            "candidate_portfolios": int(fold.get("tickets") or 0),
            "candidate_days": int(fold.get("candidate_days") or 0),
            "minimum_training_days_passed": int(
                fold.get("training_calendar_span_days") or 0
            ) >= calibration_min_training_days,
            "minimum_candidate_portfolios_passed": int(
                fold.get("tickets") or 0
            ) >= calibration_min_portfolios,
            "minimum_candidate_days_passed": int(
                fold.get("candidate_days") or 0
            ) >= calibration_min_candidate_days,
            "ready": bool(fold.get("ready")),
        }
        for fold in calibration_folds
    ]
    for boundary in warmup_boundaries:
        boundary["conjunction_passed"] = bool(
            boundary["minimum_training_days_passed"]
            and boundary["minimum_candidate_portfolios_passed"]
            and boundary["minimum_candidate_days_passed"]
        )
    warmup_logic_violations = [
        row for row in warmup_boundaries
        if row["ready"] != row["conjunction_passed"]
    ]
    first_ready_boundary = next(
        (row for row in warmup_boundaries if row["ready"]),
        None,
    )
    warmup_audit = {
        "version": "all_pregate_candidate_warmup_audit_v1",
        "logical_operator": "AND",
        "minimum_training_calendar_days": calibration_min_training_days,
        "minimum_pregate_candidate_portfolios": (
            calibration_min_portfolios
        ),
        "minimum_candidate_days": calibration_min_candidate_days,
        "candidate_day_definition": (
            "day_with_at_least_one_pregate_portfolio_raw_gross_return_"
            "at_or_above_one_plus_base_buy_margin"
        ),
        "population_source": (
            "all_pregate_candidates_not_only_purchased_candidates"
        ),
        "folds": len(warmup_boundaries),
        "logic_violations": len(warmup_logic_violations),
        "ready_exactly_when_all_thresholds_pass": bool(
            warmup_boundaries and not warmup_logic_violations
        ),
        "first_ready_boundary": first_ready_boundary,
        "boundary_manifest_sha256": _canonical_sha256(
            warmup_boundaries
        ),
    }
    bankroll = {
        "initial_daily_bankroll_yen": initial_daily_bankroll_yen,
        "stake_yen": total_stake,
        "return_yen": total_return,
        "profit_yen": total_return - total_stake,
        "roi": total_return / total_stake if total_stake else None,
        "roi_status": "defined" if total_stake else "not_applicable",
        "roi_not_applicable_reason": (
            None if total_stake else (
                "warmup_no_purchase"
                if ready_races == 0
                else "no_candidate_passed_purchase_gate"
            )
        ),
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
            for record in all_candidate_records
        )
        for source in sorted({
            str(record.get("raw_value_source") or "unknown")
            for record in all_candidate_records
        })
    }
    base_protocol = payload.get("evaluation_protocol")
    base_protocol = base_protocol if isinstance(base_protocol, dict) else {}
    joint_distribution = base_protocol.get("training_and_joint_distribution")
    joint_distribution = (
        joint_distribution if isinstance(joint_distribution, dict) else {}
    )
    ready_boundaries = [
        {
            "evaluation_date": str(fold.get("evaluation_date") or ""),
            "trained_through_date": str(
                fold.get("trained_through_date") or ""
            ),
            "calibration_information_cutoff": str(
                fold.get("calibration_information_cutoff") or ""
            ),
            "latest_training_settlement_available_at": (
                fold.get("latest_training_settlement_available_at")
            ),
            "strict_settlement_before_decision": bool(
                fold.get("strict_settlement_before_decision")
            ),
            "same_race_teacher_overlap_count": int(
                fold.get("same_race_teacher_overlap_count") or 0
            ),
            "eligible_training_unique_races": int(
                fold.get("eligible_training_unique_races") or 0
            ),
            "eligible_training_race_manifest_sha256": str(
                fold.get("eligible_training_race_manifest_sha256") or ""
            ),
        }
        for fold in calibration_folds
        if fold.get("ready")
    ]
    strict_prior_violations = [
        row for row in ready_boundaries
        if not row["calibration_information_cutoff"]
        or (
            row["latest_training_settlement_available_at"] is not None
            and _instant(
                row["latest_training_settlement_available_at"],
                "latest_training_settlement_available_at",
            ) >= _instant(
                row["calibration_information_cutoff"],
                "calibration_information_cutoff",
            )
        )
    ]
    strict_settlement_violations = [
        row for row in ready_boundaries
        if not row["strict_settlement_before_decision"]
        or (
            row["latest_training_settlement_available_at"] is not None
            and _instant(
                row["latest_training_settlement_available_at"],
                "latest_training_settlement_available_at",
            ) >= _instant(
                row["calibration_information_cutoff"],
                "calibration_information_cutoff",
            )
        )
    ]
    all_candidate_boundaries = [
        {
            "race_id": str(race.get("race_id") or ""),
            "evaluation_time_t": str(
                race.get("evaluation_time_t") or ""
            ),
            "latest_training_settlement_available_at": race.get(
                "calibration_latest_training_settlement_available_at"
            ),
            "strict_settlement_before_evaluation_time_t": bool(
                race.get(
                    "calibration_latest_settlement_strictly_before_"
                    "evaluation_time_t"
                )
            ),
            "same_race_teacher_overlap_count": int(
                race.get(
                    "calibration_same_race_teacher_overlap_count"
                ) or 0
            ),
            "training_race_manifest_sha256": str(
                race.get("calibration_training_race_manifest_sha256") or ""
            ),
            "calibration_ledger_sha256": str(
                race.get("calibration_ledger_sha256") or ""
            ),
            "calibration_ready": bool(race.get("calibration_ready")),
        }
        for day in replay_days
        for race in day.get("races") or []
    ]
    ready_candidate_boundaries = [
        row for row in all_candidate_boundaries
        if row["calibration_ready"]
    ]
    all_candidate_settlement_violations = [
        row for row in all_candidate_boundaries
        if not row["strict_settlement_before_evaluation_time_t"]
        or (
            row["latest_training_settlement_available_at"] is not None
            and _instant(
                row["latest_training_settlement_available_at"],
                "latest_training_settlement_available_at",
            ) >= _instant(
                row["evaluation_time_t"], "evaluation_time_t"
            )
        )
    ]
    candidate_settlement_violations = [
        row for row in ready_candidate_boundaries
        if not row["strict_settlement_before_evaluation_time_t"]
        or (
            row["latest_training_settlement_available_at"] is not None
            and _instant(
                row["latest_training_settlement_available_at"],
                "latest_training_settlement_available_at",
            ) >= _instant(
                row["evaluation_time_t"], "evaluation_time_t"
            )
        )
    ]
    same_race_teacher_violations = [
        row for row in ready_boundaries
        if row["same_race_teacher_overlap_count"] != 0
    ]
    independence_audit = {
        "version": "strict_prior_calibrated_value_independence_audit_v3",
        "calibration_folds": len(calibration_folds),
        "calibration_ready_folds": len(ready_boundaries),
        "all_candidate_calibration_boundaries": len(
            all_candidate_boundaries
        ),
        "all_candidate_settlement_boundary_violations": len(
            all_candidate_settlement_violations
        ),
        "strict_prior_violation_count": len(
            all_candidate_settlement_violations
        ),
        "future_candidate_in_calibration_count": 0,
        "strict_prior_for_every_candidate": bool(
            all_candidate_boundaries
            and not all_candidate_settlement_violations
        ),
        "strict_prior_fold_violations": len(strict_prior_violations),
        "strict_prior_training_for_every_ready_fold": bool(
            ready_boundaries and not strict_prior_violations
        ),
        "strict_settlement_fold_violations": len(
            strict_settlement_violations
        ),
        "settlement_before_decision_for_every_ready_fold": bool(
            ready_boundaries and not strict_settlement_violations
        ),
        "ready_candidate_calibration_boundaries": len(
            ready_candidate_boundaries
        ),
        "candidate_settlement_boundary_violations": len(
            candidate_settlement_violations
        ),
        "settlement_before_decision_for_every_ready_candidate": bool(
            ready_candidate_boundaries
            and not candidate_settlement_violations
        ),
        "candidate_boundary_manifest_sha256": _canonical_sha256(
            all_candidate_boundaries
        ),
        "settlement_boundary_definition": (
            "candidate_settlement_available_at_strictly_before_"
            "earliest_evaluation_time_t_of_fold"
        ),
        "same_race_teacher_fold_violations": len(
            same_race_teacher_violations
        ),
        "same_race_excluded_for_every_ready_fold": bool(
            ready_boundaries and not same_race_teacher_violations
        ),
        "same_race_rule": (
            "evaluation_race_id_must_not_appear_in_calibration_teacher_"
            "and_one_candidate_portfolio_per_race"
        ),
        "fold_boundary_manifest_sha256": _canonical_sha256(ready_boundaries),
        "search_validation_draw_sets_disjoint": (
            joint_distribution.get("search_validation_draw_sets_disjoint")
        ),
        "value_population_manifest_sha256": value_realization[
            "candidate_manifest_sha256"
        ],
        "value_population_candidate_portfolios": value_realization[
            "candidate_portfolios"
        ],
        "value_population_independent_validation_only": value_realization[
            "independent_validation_value_only"
        ],
        "value_population_identical_realized_portfolios_only": (
            value_realization["identical_realized_portfolio_only"]
        ),
    }
    calibrator_transitions = calibrator_update_events[1:]
    calibrator_update_logic_violations = [
        row for row in calibrator_transitions
        if bool(row["teacher_population_changed"])
        != bool(row["calibrator_instance_changed"])
    ]
    unchanged_population_reuse_violations = [
        row for row in calibrator_transitions
        if not row["teacher_population_changed"]
        and (
            row["calibrator_instance_changed"]
            or not row["cache_hit"]
        )
    ]
    calibrator_decision_rows = [
        {
            "race_id": str(race.get("race_id") or ""),
            "evaluation_time_t": str(
                race.get("evaluation_time_t") or ""
            ),
                "training_race_manifest_sha256": str(
                    race.get("calibration_training_race_manifest_sha256") or ""
                ),
                "calibration_ledger_sha256": str(
                    race.get("calibration_ledger_sha256") or ""
                ),
                "calibration_instance_id": str(
                race.get("calibration_instance_id") or ""
            ),
            "calibrator_artifact_sha256": str(
                race.get("calibrator_artifact_sha256") or ""
            ),
            "model_sha256": str(race.get("model_sha256") or ""),
            "threshold_sha256": str(race.get("threshold_sha256") or ""),
            "settlement_engine_sha256": str(
                race.get("settlement_engine_sha256") or ""
            ),
            "decision_hash_bundle_sha256": str(
                race.get("decision_hash_bundle_sha256") or ""
            ),
        }
        for day in replay_days
        for race in day.get("races") or []
    ]
    missing_decision_calibrator_bindings = [
        row for row in calibrator_decision_rows
        if not all(row.values())
    ]
    instance_artifact_sets: dict[str, set[str]] = {}
    instance_ledger_sets: dict[str, set[str]] = {}
    for row in calibrator_decision_rows:
        instance_id = row["calibration_instance_id"]
        instance_artifact_sets.setdefault(instance_id, set()).add(
            row["calibrator_artifact_sha256"]
        )
        instance_ledger_sets.setdefault(instance_id, set()).add(
            row["calibration_ledger_sha256"]
        )
    instance_artifact_collisions = {
        key: sorted(values)
        for key, values in instance_artifact_sets.items()
        if key and len(values) != 1
    }
    instance_ledger_collisions = {
        key: sorted(values)
        for key, values in instance_ledger_sets.items()
        if key and len(values) != 1
    }
    decision_hash_bundle_violations = [
        row for row in calibrator_decision_rows
        if row["decision_hash_bundle_sha256"] != _canonical_sha256({
            "model_sha256": row["model_sha256"],
            "calibrator_sha256": row["calibrator_artifact_sha256"],
            "calibration_ledger_sha256": row[
                "calibration_ledger_sha256"
            ],
            "threshold_sha256": row["threshold_sha256"],
            "settlement_engine_sha256": row[
                "settlement_engine_sha256"
            ],
        })
    ]
    event_binding_rows: dict[tuple[str, str], list[dict[str, object]]] = {}
    for event in calibrator_update_events:
        key = (
            str(event.get("race_id") or ""),
            str(event.get("evaluation_time_t") or ""),
        )
        event_binding_rows.setdefault(key, []).append(event)
    duplicate_event_bindings = {
        key: rows for key, rows in event_binding_rows.items()
        if len(rows) != 1
    }
    decision_event_binding_violations = []
    fixed_component_hash_violations = []
    for row in calibrator_decision_rows:
        key = (row["race_id"], row["evaluation_time_t"])
        events = event_binding_rows.get(key) or []
        if (
            len(events) != 1
            or row["calibration_instance_id"]
            != str(events[0].get("calibration_instance_id") or "")
            or row["calibrator_artifact_sha256"]
            != str(events[0].get("calibrator_artifact_sha256") or "")
            or row["calibration_ledger_sha256"]
            != str(events[0].get("training_ledger_sha256") or "")
            or row["training_race_manifest_sha256"]
            != str(events[0].get("training_race_manifest_sha256") or "")
        ):
            decision_event_binding_violations.append(row)
        if (
            row["model_sha256"] != model_decision_sha256
            or row["threshold_sha256"] != threshold_definition_sha256
            or row["settlement_engine_sha256"]
            != settlement_engine_sha256
        ):
            fixed_component_hash_violations.append(row)
    calibrator_update_audit = {
        "version": "strict_prior_calibrator_update_audit_v2",
        "folds": len(calibrator_update_events),
        "initializations": int(bool(calibrator_update_events)),
        "updates_after_initialization": sum(
            bool(row["calibrator_instance_changed"])
            for row in calibrator_transitions
        ),
        "unchanged_population_reuses": sum(
            not bool(row["teacher_population_changed"])
            for row in calibrator_transitions
        ),
        "unique_calibrator_instances": len({
            str(row["calibration_instance_id"])
            for row in calibrator_update_events
        }),
        "calibrator_fits": sum(
            not bool(row["cache_hit"])
            for row in calibrator_update_events
        ),
        "update_logic_violations": len(
            calibrator_update_logic_violations
        ),
        "unchanged_population_reuse_violations": len(
            unchanged_population_reuse_violations
        ),
        "updates_only_when_eligible_teacher_population_changes": bool(
            calibrator_update_events
            and not calibrator_update_logic_violations
        ),
        "unchanged_population_reuses_identical_calibrator": bool(
            calibrator_update_events
            and not unchanged_population_reuse_violations
        ),
        "instance_definition": (
            "calibration_version_plus_eligible_teacher_race_manifest_"
            "plus_fixed_hyperparameters_and_manifest_derived_seed"
        ),
        "event_manifest_sha256": _canonical_sha256(
            calibrator_update_events
        ),
        "decision_bindings": len(calibrator_decision_rows),
        "missing_decision_calibrator_bindings": len(
            missing_decision_calibrator_bindings
        ),
        "instance_artifact_collisions": len(
            instance_artifact_collisions
        ),
        "instance_ledger_collisions": len(instance_ledger_collisions),
        "decision_hash_bundle_violations": len(
            decision_hash_bundle_violations
        ),
        "duplicate_decision_event_bindings": len(
            duplicate_event_bindings
        ),
        "decision_event_binding_violations": len(
            decision_event_binding_violations
        ),
        "fixed_component_hash_violations": len(
            fixed_component_hash_violations
        ),
        "decision_hashes_present": bool(
            calibrator_decision_rows
            and not missing_decision_calibrator_bindings
        ),
        "model_sha256": model_decision_sha256,
        "threshold_sha256": threshold_definition_sha256,
        "settlement_engine_sha256": settlement_engine_sha256,
        "every_decision_bound_to_full_prior_ledger_artifact": bool(
            calibrator_decision_rows
            and not missing_decision_calibrator_bindings
            and not instance_artifact_collisions
            and not instance_ledger_collisions
            and not decision_hash_bundle_violations
            and not duplicate_event_bindings
            and not decision_event_binding_violations
            and not fixed_component_hash_violations
        ),
        "reconstruction_mode": (
            "full_refit_from_complete_eligible_prior_ledger_when_manifest_"
            "changes_exact_artifact_reuse_when_manifest_is_unchanged"
        ),
        "decision_binding_manifest_sha256": _canonical_sha256(
            calibrator_decision_rows
        ),
    }
    range_evaluable_candidates = [
        race
        for day in replay_days
        for race in day.get("races") or []
        if race.get("calibration_ready")
        and race.get("raw_portfolio_gross_return_estimate") is not None
    ]
    out_of_range_candidates = [
        race for race in range_evaluable_candidates
        if not race.get("calibration_input_in_training_range")
    ]
    out_of_range_purchases = [
        race for race in out_of_range_candidates
        if int(race.get("stake_yen") or 0) > 0
        or race.get("purchase_authorized")
        or race.get("bets_yen")
    ]
    calibration_input_range_audit = {
        "version": "calibration_raw_input_range_guard_v1",
        "ready_candidates_with_raw_input": len(
            range_evaluable_candidates
        ),
        "in_range_candidates": (
            len(range_evaluable_candidates) - len(out_of_range_candidates)
        ),
        "out_of_range_candidates": len(out_of_range_candidates),
        "out_of_range_purchase_violations": len(out_of_range_purchases),
        "out_of_range_action": "reject_purchase_no_extrapolation",
        "all_out_of_range_inputs_rejected": not out_of_range_purchases,
        "candidate_manifest_sha256": _canonical_sha256([
            [
                race.get("race_id"),
                race.get("raw_portfolio_gross_return_estimate"),
                race.get("calibration_training_raw_input_min"),
                race.get("calibration_training_raw_input_max"),
                race.get("calibration_input_in_training_range"),
                race.get("rejection_reason"),
            ]
            for race in range_evaluable_candidates
        ]),
    }
    local_evaluable_candidates = [
        race for race in range_evaluable_candidates
        if race.get("calibration_input_in_training_range")
    ]
    local_range_rejections = [
        race for race in local_evaluable_candidates
        if not race.get("calibration_input_in_local_block_range")
    ]
    local_support_rejections = [
        race for race in local_evaluable_candidates
        if race.get("calibration_input_in_local_block_range")
        and not race.get("local_support_ready")
    ]
    local_support_purchase_violations = [
        race for race in local_evaluable_candidates
        if (
            not race.get("calibration_input_in_local_block_range")
            or not race.get("local_support_ready")
        )
        and (
            int(race.get("stake_yen") or 0) > 0
            or race.get("purchase_authorized")
            or race.get("bets_yen")
        )
    ]
    calibration_local_support_audit = {
        "version": "isotonic_local_support_gate_v1",
        "minimum_local_candidates": calibration_min_local_candidates,
        "minimum_local_candidate_days": (
            calibration_min_local_candidate_days
        ),
        "minimum_local_day_cluster_ess": calibration_min_local_ess,
        "ess_definition": (
            "square_of_total_block_exposure_divided_by_sum_of_squared_"
            "race_date_block_exposures"
        ),
        "ready_in_global_range_candidates": len(
            local_evaluable_candidates
        ),
        "outside_observed_local_block_range": len(
            local_range_rejections
        ),
        "insufficient_local_support_candidates": len(
            local_support_rejections
        ),
        "local_support_purchase_violations": len(
            local_support_purchase_violations
        ),
        "all_local_range_and_support_failures_rejected": (
            not local_support_purchase_violations
        ),
        "candidate_manifest_sha256": _canonical_sha256([
            [
                race.get("race_id"),
                race.get("isotonic_block_id"),
                race.get("local_block_candidates"),
                race.get("local_block_candidate_days"),
                race.get("local_block_ess"),
                race.get("local_block_raw_ev_min"),
                race.get("local_block_raw_ev_max"),
                race.get("local_support_ready"),
                race.get("calibration_input_in_local_block_range"),
                race.get("rejection_reason"),
            ]
            for race in local_evaluable_candidates
        ]),
    }
    lcb_evaluable_candidates = [
        race for race in range_evaluable_candidates
        if race.get("calibration_input_in_training_range")
        and race.get("calibration_input_in_local_block_range")
        and race.get("local_support_ready")
    ]
    lcb_invalid_candidates = [
        race for race in lcb_evaluable_candidates
        if (
            race.get("calibrated_gross_return_lcb95") is None
            or race.get("calibrated_gross_return") is None
            or not isfinite(float(race["calibrated_gross_return_lcb95"]))
            or not isfinite(float(race["calibrated_gross_return"]))
            or float(race["calibrated_gross_return_lcb95"])
            > float(race["calibrated_gross_return"]) + 1e-12
        )
    ]
    lcb_purchase_violations = [
        race
        for day in replay_days
        for race in day.get("races") or []
        if (
            int(race.get("stake_yen") or 0) > 0
            or race.get("purchase_authorized")
            or race.get("bets_yen")
        )
        and (
            race.get("calibrated_gross_return_lcb95") is None
            or not isfinite(
                float(race["calibrated_gross_return_lcb95"])
            )
            or float(race["calibrated_gross_return_lcb95"])
            <= 1.0 + calibration_margin
        )
    ]
    lcb_definition_fold_violations = [
        fold for fold in calibration_folds
        if (
            float(fold.get("lcb_tail_probability") or -1.0) != 0.05
            or float(fold.get("lcb_confidence_level") or -1.0) != 0.95
            or fold.get("lcb_sidedness") != "one_sided_lower"
            or fold.get("lcb_estimator") != (
                "nonparametric_race_date_cluster_percentile_bootstrap"
            )
            or fold.get("bootstrap_cluster_unit") != "race_date"
            or not fold.get("within_day_candidates_resampled_together")
            or fold.get("ticket_level_independence_assumed") is not False
            or fold.get("quantile_method") != "inverted_cdf"
            or not fold.get("lcb_capped_at_point_estimate")
        )
    ]
    calibration_lcb_audit = {
        "version": "strict_prior_one_sided_lcb95_audit_v1",
        "tail_probability": 0.05,
        "confidence_level": 0.95,
        "sidedness": "one_sided_lower",
        "estimator": (
            "nonparametric_race_date_cluster_percentile_bootstrap"
        ),
        "cluster_unit": "race_date",
        "within_day_candidates_resampled_together": True,
        "ticket_level_independence_assumed": False,
        "dependence_structure": (
            "all_candidate_returns_and_exposures_on_one_race_date_are_"
            "resampled_as_one_joint_vector"
        ),
        "quantile_method": "inverted_cdf",
        "bootstrap_samples_per_fit": calibration_bootstrap_samples,
        "ready_in_range_locally_supported_candidates": len(
            lcb_evaluable_candidates
        ),
        "invalid_or_above_point_candidate_bounds": len(
            lcb_invalid_candidates
        ),
        "definition_fold_violations": len(
            lcb_definition_fold_violations
        ),
        "purchase_threshold": 1.0 + calibration_margin,
        "threshold_comparison": "strictly_greater_than",
        "missing_nonfinite_or_below_threshold_purchase_violations": len(
            lcb_purchase_violations
        ),
        "all_evaluable_bounds_finite_and_not_above_point": (
            not lcb_invalid_candidates
        ),
        "one_sided_95_definition_consistent_for_every_fold": bool(
            calibration_folds and not lcb_definition_fold_violations
        ),
        "strict_lcb_purchase_threshold_enforced": (
            not lcb_purchase_violations
        ),
        "fail_closed_rule": (
            "reject_when_lcb_missing_nonfinite_or_not_strictly_above_"
            "one_plus_calibration_margin"
        ),
        "candidate_manifest_sha256": _canonical_sha256([
            [
                race.get("race_id"),
                race.get("calibrated_gross_return"),
                race.get("calibrated_gross_return_lcb95"),
                race.get("purchase_authorized"),
                race.get("stake_yen"),
            ]
            for race in lcb_evaluable_candidates
        ]),
    }
    reproducibility_instance_rows = sorted({
        (
            str(row.get("training_ledger_sha256") or ""),
            str(row.get("training_race_manifest_sha256") or ""),
            int(row.get("fit_seed") or 0),
            str(row.get("calibration_instance_id") or ""),
            str(row.get("calibrator_artifact_sha256") or ""),
        )
        for row in calibrator_update_events
    })
    instance_seed_sets: dict[str, set[int]] = {}
    for row in calibrator_update_events:
        instance_seed_sets.setdefault(
            str(row.get("calibration_instance_id") or ""), set()
        ).add(int(row.get("fit_seed") or 0))
    instance_seed_collisions = {
        key: sorted(values)
        for key, values in instance_seed_sets.items()
        if key and len(values) != 1
    }
    incomplete_reproducibility_instances = [
        row for row in reproducibility_instance_rows
        if not row[0] or not row[1] or not row[3] or not row[4]
    ]
    reproducibility_decision_outputs = [
        {
            "race_id": race.get("race_id"),
            "evaluation_time_t": race.get("evaluation_time_t"),
            "calibration_instance_id": race.get(
                "calibration_instance_id"
            ),
            "calibrator_artifact_sha256": race.get(
                "calibrator_artifact_sha256"
            ),
            "decision_hash_bundle_sha256": race.get(
                "decision_hash_bundle_sha256"
            ),
            "raw_portfolio_gross_return_estimate": race.get(
                "raw_portfolio_gross_return_estimate"
            ),
            "calibrated_gross_return_lcb95": race.get(
                "calibrated_gross_return_lcb95"
            ),
            "purchase_authorized": race.get("purchase_authorized"),
            "rejection_reason": race.get("rejection_reason"),
            "bets_yen": race.get("bets_yen"),
            "stake_yen": race.get("stake_yen"),
            "return_yen": race.get("return_yen"),
        }
        for day in replay_days
        for race in day.get("races") or []
    ]
    rerun_input_fingerprint = _canonical_sha256({
        "base_artifact_sha256": base_artifact_sha256,
        "scored_cache_sha256": scored_cache_sha256,
        "replay_configuration_sha256": replay_configuration_sha256,
        "implementation_sha256": implementation_sha256,
    })
    deterministic_output_fingerprint = _canonical_sha256({
        "calibrator_instances": reproducibility_instance_rows,
        "decision_outputs": reproducibility_decision_outputs,
        "primary_bankroll": bankroll,
        "bankroll_confidence": confidence,
        "value_realization": value_realization,
    })
    reproducibility_manifest_complete = bool(
        reproducibility_instance_rows
        and reproducibility_decision_outputs
        and not incomplete_reproducibility_instances
        and not instance_seed_collisions
        and not instance_artifact_collisions
        and not instance_ledger_collisions
        and not missing_decision_calibrator_bindings
    )
    replay_reproducibility_audit = {
        "version": "deterministic_replay_manifest_v1",
        "base_artifact_sha256": base_artifact_sha256,
        "scored_cache_sha256": scored_cache_sha256,
        "configuration": replay_configuration,
        "configuration_sha256": replay_configuration_sha256,
        "implementation_source_sha256": implementation_source_sha256,
        "implementation_sha256": implementation_sha256,
        "seed_rule": (
            "fit_seed_equals_base_seed_plus_first_64_bits_of_complete_"
            "calibration_ledger_sha256_modulo_2_pow_32_minus_1"
        ),
        "quantile_method": "inverted_cdf",
        "deterministic_calibrator_instances": len(
            reproducibility_instance_rows
        ),
        "instance_seed_collisions": len(instance_seed_collisions),
        "incomplete_calibrator_instances": len(
            incomplete_reproducibility_instances
        ),
        "rerun_input_fingerprint_sha256": rerun_input_fingerprint,
        "deterministic_output_fingerprint_sha256": (
            deterministic_output_fingerprint
        ),
        "reproducibility_contract": (
            "independent_rerun_with_identical_input_fingerprint_must_"
            "produce_identical_output_fingerprint"
        ),
        "manifest_complete": reproducibility_manifest_complete,
        "calibrator_instance_manifest_sha256": _canonical_sha256(
            reproducibility_instance_rows
        ),
        "decision_output_manifest_sha256": _canonical_sha256(
            reproducibility_decision_outputs
        ),
    }
    formal_gate = {
        "prediction_noninferiority": (
            payload.get("probability_metrics") is not None
            and float(
                (payload.get("probability_metrics") or {}).get(
                    "generated_log_loss_delta_vs_decision_model", 1.0
                )
            ) <= 0.0
        ),
        "maximum_drawdown_audit": (
            int(bankroll.get("max_drawdown_yen") or 0)
            <= initial_daily_bankroll_yen // 2
        ),
        "strict_prior_for_every_candidate": independence_audit[
            "strict_prior_for_every_candidate"
        ],
        "independent_validation_value_only": bool(
            purchased
            and all(
                race.get("raw_value_source") == PRIMARY_RAW_VALUE_SOURCE
                for race in purchased
            )
        ),
        "strict_prior_calibration_folds": independence_audit[
            "strict_prior_training_for_every_ready_fold"
        ],
        "strict_settlement_before_decision": independence_audit[
            "settlement_before_decision_for_every_ready_candidate"
        ],
        "same_race_calibration_independence": independence_audit[
            "same_race_excluded_for_every_ready_fold"
        ],
        "same_race_single_prior_calibrator": race_batch_audit[
            "all_tickets_in_race_share_one_prior_calibrator"
        ],
        "results_admitted_only_after_settlement": race_batch_audit[
            "results_admitted_only_after_strict_settlement"
        ],
        "all_pregate_candidates_registered": learning_population_audit[
            "all_pregate_candidates_registered"
        ],
        "warmup_conjunction_consistent": warmup_audit[
            "ready_exactly_when_all_thresholds_pass"
        ],
        "calibrator_updates_only_on_teacher_change": (
            calibrator_update_audit[
                "updates_only_when_eligible_teacher_population_changes"
            ]
            and calibrator_update_audit[
                "unchanged_population_reuses_identical_calibrator"
            ]
        ),
        "each_decision_has_equivalent_prior_ledger_calibrator": (
            calibrator_update_audit[
                "every_decision_bound_to_full_prior_ledger_artifact"
            ]
        ),
        "all_decision_component_hashes_persisted": bool(
            calibrator_update_audit["decision_hashes_present"]
            and not calibrator_update_audit[
                "decision_hash_bundle_violations"
            ]
            and not calibrator_update_audit[
                "decision_event_binding_violations"
            ]
            and not calibrator_update_audit[
                "fixed_component_hash_violations"
            ]
        ),
        "out_of_range_inputs_rejected": calibration_input_range_audit[
            "all_out_of_range_inputs_rejected"
        ],
        "local_isotonic_support_enforced": calibration_local_support_audit[
            "all_local_range_and_support_failures_rejected"
        ],
        "one_sided_lcb95_valid_and_enforced": bool(
            calibration_lcb_audit[
                "all_evaluable_bounds_finite_and_not_above_point"
            ]
            and calibration_lcb_audit[
                "one_sided_95_definition_consistent_for_every_fold"
            ]
            and calibration_lcb_audit[
                "strict_lcb_purchase_threshold_enforced"
            ]
        ),
        "day_cluster_dependence_reflected": bool(
            calibration_lcb_audit["cluster_unit"] == "race_date"
            and calibration_lcb_audit[
                "within_day_candidates_resampled_together"
            ]
            and not calibration_lcb_audit[
                "ticket_level_independence_assumed"
            ]
        ),
        "deterministic_replay_manifest_complete": (
            replay_reproducibility_audit["manifest_complete"]
        ),
        "independent_search_validation_draw_sets": bool(
            independence_audit["search_validation_draw_sets_disjoint"]
        ),
        "identical_value_realization_population": bool(
            value_realization["candidate_portfolios"]
            and value_realization["identical_realized_portfolio_only"]
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
    pre_ready_purchases = sum(
        int(race.get("stake_yen") or 0) > 0
        for day in replay_days
        for race in day.get("races") or []
        if not race.get("calibration_ready")
    )
    pre_ready_stake_yen = sum(
        int(race.get("stake_yen") or 0)
        for day in replay_days
        for race in day.get("races") or []
        if not race.get("calibration_ready")
    )
    pre_ready_nonempty_bet_vectors = sum(
        bool(race.get("bets_yen"))
        for day in replay_days
        for race in day.get("races") or []
        if not race.get("calibration_ready")
    )
    pre_ready_purchase_authorizations = sum(
        bool(race.get("purchase_authorized"))
        for day in replay_days
        for race in day.get("races") or []
        if not race.get("calibration_ready")
    )
    strict_zero_stake_before_warmup = not any((
        pre_ready_purchases,
        pre_ready_stake_yen,
        pre_ready_nonempty_bet_vectors,
        pre_ready_purchase_authorizations,
    ))
    formal_gate["strict_zero_stake_before_warmup"] = (
        strict_zero_stake_before_warmup
    )
    below_threshold_purchases = sum(
        int(race.get("stake_yen") or 0) > 0
        and (
            race.get("calibrated_gross_return_lcb95") is None
            or float(race["calibrated_gross_return_lcb95"])
            <= 1.0 + calibration_margin
        )
        for day in replay_days
        for race in day.get("races") or []
    )
    non_independent_value_purchases = sum(
        int(race.get("stake_yen") or 0) > 0
        and race.get("raw_value_source") != PRIMARY_RAW_VALUE_SOURCE
        for day in replay_days
        for race in day.get("races") or []
    )
    safety_invariants_passed = not any((
        pre_ready_purchases,
        below_threshold_purchases,
        non_independent_value_purchases,
        strict_prior_violations,
        strict_settlement_violations,
        candidate_settlement_violations,
        all_candidate_settlement_violations,
        same_race_teacher_violations,
        teacher_admission_violations,
        ticket_calibrator_violations,
        warmup_logic_violations,
        calibrator_update_logic_violations,
        unchanged_population_reuse_violations,
        missing_decision_calibrator_bindings,
        instance_artifact_collisions,
        instance_ledger_collisions,
        decision_hash_bundle_violations,
        duplicate_event_bindings,
        decision_event_binding_violations,
        fixed_component_hash_violations,
        out_of_range_purchases,
        local_support_purchase_violations,
        lcb_definition_fold_violations,
        lcb_purchase_violations,
        not reproducibility_manifest_complete,
        not strict_zero_stake_before_warmup,
        not learning_population_audit[
            "all_pregate_candidates_registered"
        ],
    ))
    mature_observation_window = ready_days >= 30 and ready_races >= 1_000
    purchase_gate_outcome = _purchase_gate_outcome(
        mature_observation_window=mature_observation_window,
        observed_purchased_portfolios=len(purchased),
        safety_invariants_passed=safety_invariants_passed,
        promotion_evidence_passed=all(formal_gate.values()),
    )
    purchase_gate_audit = {
        "version": "strict_prior_purchase_gate_operational_audit_v1",
        "outcome": purchase_gate_outcome,
        "safety_invariants_passed": safety_invariants_passed,
        "mature_observation_window": mature_observation_window,
        "safe_abstention": purchase_gate_outcome == (
            "safe_abstention_no_demonstrated_price_advantage"
        ),
        "pre_calibration_ready_purchases": pre_ready_purchases,
        "pre_calibration_ready_stake_yen": pre_ready_stake_yen,
        "pre_calibration_ready_nonempty_bet_vectors": (
            pre_ready_nonempty_bet_vectors
        ),
        "pre_calibration_ready_purchase_authorizations": (
            pre_ready_purchase_authorizations
        ),
        "below_calibrated_lcb_threshold_purchases": (
            below_threshold_purchases
        ),
        "lcb_definition_fold_violations": len(
            lcb_definition_fold_violations
        ),
        "lcb_invalid_or_above_point_candidates": len(
            lcb_invalid_candidates
        ),
        "lcb_missing_nonfinite_or_below_threshold_purchase_violations": (
            len(lcb_purchase_violations)
        ),
        "one_sided_lcb95_valid_and_enforced": formal_gate[
            "one_sided_lcb95_valid_and_enforced"
        ],
        "local_isotonic_support_enforced": formal_gate[
            "local_isotonic_support_enforced"
        ],
        "local_support_purchase_violations": len(
            local_support_purchase_violations
        ),
        "day_cluster_dependence_reflected": formal_gate[
            "day_cluster_dependence_reflected"
        ],
        "deterministic_replay_manifest_complete": (
            reproducibility_manifest_complete
        ),
        "non_independent_value_purchases": non_independent_value_purchases,
        "observed_purchased_portfolios": len(purchased),
        "interpretation": (
            "zero_purchases_is_safe_abstention_not_gate_failure"
        ),
    }
    warmup_audit.update({
        "pre_ready_purchases": pre_ready_purchases,
        "pre_ready_stake_yen": pre_ready_stake_yen,
        "pre_ready_nonempty_bet_vectors": (
            pre_ready_nonempty_bet_vectors
        ),
        "pre_ready_purchase_authorizations": (
            pre_ready_purchase_authorizations
        ),
        "no_purchases_before_ready": strict_zero_stake_before_warmup,
    })
    protocol = {
        "version": "joint_edge_calibrated_replay_protocol_v12",
        "model": MODEL_VERSION,
        "base_artifact_sha256": base_artifact_sha256,
        "base_evaluation_protocol_id": payload.get("evaluation_protocol_id"),
        "evaluation_time_t": base_protocol.get("evaluation_time_t"),
        "odds_snapshot_age": base_protocol.get("odds_snapshot_age"),
        "population": base_protocol.get("population"),
        "training_and_joint_distribution": joint_distribution,
        "purchase_rule": base_protocol.get("purchase_rule"),
        "settlement": base_protocol.get("settlement"),
        "calibration": {
            "version": CALIBRATION_VERSION,
            "margin": calibration_margin,
            "bootstrap_samples": calibration_bootstrap_samples,
            "min_training_days": calibration_min_training_days,
            "min_portfolios": calibration_min_portfolios,
            "min_candidate_days": calibration_min_candidate_days,
            "min_local_candidates": calibration_min_local_candidates,
            "min_local_candidate_days": (
                calibration_min_local_candidate_days
            ),
            "min_local_day_cluster_ess": calibration_min_local_ess,
            "local_ess_definition": (
                "square_of_total_block_exposure_divided_by_sum_of_"
                "squared_race_date_block_exposures"
            ),
            "shape_constraint": "isotonic",
            "quantile_method": "inverted_cdf",
            "lcb_tail_probability": 0.05,
            "lcb_confidence_level": 0.95,
            "lcb_sidedness": "one_sided_lower",
            "lcb_estimator": (
                "nonparametric_race_date_cluster_percentile_bootstrap"
            ),
            "bootstrap_cluster_unit": "race_date",
            "within_day_candidates_resampled_together": True,
            "ticket_level_independence_assumed": False,
            "bootstrap_resample_cluster_count": (
                "all_strict_prior_training_calendar_days"
            ),
            "lcb_point_estimate_cap": (
                "lower_bound_is_never_above_isotonic_point_estimate"
            ),
            "teacher": (
                "stake_weighted_fixed_candidate_"
                "realized_gross_return_per_yen"
            ),
            "target_unit": (
                "gross_return_per_staked_yen_including_returned_principal"
            ),
            "raw_input_unit": "gross_return_multiple_including_principal",
            "purchase_condition": (
                "calibrated_gross_return_lcb95_greater_than_"
                "one_plus_calibration_margin"
            ),
            "information_boundary": (
                "all_candidate_race_batches_with_settlement_available_at_"
                "strictly_before_each_candidate_decision_time"
            ),
            "same_race_rule": (
                "exclude_all_teacher_records_with_evaluation_race_id_"
                "and_reject_duplicate_race_ids_within_a_fold"
            ),
            "independent_sample_unit": (
                "one_stake_weighted_candidate_portfolio_per_race"
            ),
            "calibrator_scope": (
                "one_frozen_prior_calibrator_instance_per_race_fold_"
                "shared_by_every_ticket_in_the_portfolio"
            ),
            "result_admission": (
                "one_portfolio_result_batch_per_race_admitted_only_when_"
                "settlement_available_at_is_strictly_before_next_"
                "calibration_information_cutoff"
            ),
            "update_rule": (
                "at_each_race_decision_fit_or_switch_calibrator_only_when_"
                "eligible_complete_calibration_ledger_changes_otherwise_reuse_"
                "identical_instance"
            ),
            "bootstrap_seed_rule": (
                "deterministic_from_base_seed_and_complete_ledger_sha256"
            ),
            "out_of_range_input_rule": (
                "reject_purchase_when_raw_gross_return_is_outside_"
                "strict_prior_training_raw_min_max"
            ),
            "local_support_rule": (
                "reject_when_outside_observed_isotonic_block_range_or_"
                "local_candidates_days_or_day_cluster_ess_are_below_"
                "predeclared_thresholds"
            ),
            "learning_population": learning_population_audit,
            "sample_weight": "candidate_portfolio_stake_yen",
            "primary_input": (
                "pregate_best_search_independent_validation_"
                "portfolio_lower_quantile"
            ),
            "legacy_input": (
                "selected_independent_validation_then_search_edge_"
                "explicit_fallback_only"
            ),
        },
        "bankroll": {
            "initial_daily_bankroll_yen": initial_daily_bankroll_yen,
            "purchase_unit_yen": PURCHASE_UNIT_YEN,
            "profit_reuse": "after_recorded_settlement_available_at",
            "cash_shortfall": "proportional_integer_unit_downscale",
        },
        "reproducibility": {
            "rerun_input_fingerprint_sha256": rerun_input_fingerprint,
            "configuration_sha256": replay_configuration_sha256,
            "implementation_sha256": implementation_sha256,
            "base_artifact_sha256": base_artifact_sha256,
            "scored_cache_sha256": scored_cache_sha256,
            "contract": (
                "identical_input_fingerprint_requires_identical_"
                "deterministic_output_fingerprint"
            ),
        },
        "decision_component_hashes": {
            "model_sha256": model_decision_sha256,
            "threshold_sha256": threshold_definition_sha256,
            "threshold_definition": threshold_definition,
            "settlement_engine_sha256": settlement_engine_sha256,
            "per_decision_dynamic_hashes": [
                "calibrator_sha256",
                "calibration_ledger_sha256",
            ],
        },
        "resampling_condition_id": confidence["condition_id"],
        "seed": seed,
    }
    decision_rows = [
        race
        for day in replay_days
        for race in day.get("races") or []
    ]
    approval_examples = [
        race for race in decision_rows
        if race.get("purchase_authorized")
    ][:3]
    post_warmup_denial_examples = [
        race for race in decision_rows
        if race.get("calibration_ready")
        and not race.get("purchase_authorized")
    ][:3]
    denial_examples = post_warmup_denial_examples or [
        race for race in decision_rows
        if not race.get("purchase_authorized")
    ][:3]
    observed_dates = sorted({
        str(day.get("race_date") or "")
        for day in replay_days if day.get("race_date")
    })
    calendar_span_days = (
        (
            datetime.fromisoformat(observed_dates[-1]).date()
            - datetime.fromisoformat(observed_dates[0]).date()
        ).days + 1
        if observed_dates else 0
    )
    candidate_dates = {
        str(record.get("race_date") or "")
        for record in all_candidate_records
        if record.get("race_date")
    }
    settled_candidate_dates = {
        str(record.get("race_date") or "")
        for record in training_records
        if record.get("race_date")
    }
    outer_draw_definition = {
        "search_outer_draws": joint_distribution.get(
            "search_outer_draws"
        ),
        "validation_outer_draws": joint_distribution.get(
            "validation_outer_draws"
        ),
        "draw_sets_disjoint": joint_distribution.get(
            "search_validation_draw_sets_disjoint"
        ),
    }
    inner_scenario_definition = {
        "inner_scenarios": joint_distribution.get("inner_scenarios"),
        "scenario_weighting": joint_distribution.get(
            "scenario_weighting"
        ),
    }
    artifact_lineage = {
        "parent_artifact_hash": base_artifact_sha256,
        "prediction_model_hash": model_decision_sha256,
        "joint_scenario_model_hash": _canonical_sha256({
            "joint_distribution": joint_distribution,
            "joint_value_audit": payload.get("joint_value_audit"),
        }),
        "calibrator_hash": (
            decision_rows[-1].get("calibrator_sha256")
            if decision_rows else None
        ),
        "calibration_ledger_hash": (
            decision_rows[-1].get("calibration_ledger_sha256")
            if decision_rows else None
        ),
        "portfolio_policy_hash": _canonical_sha256({
            "purchase_rule": base_protocol.get("purchase_rule"),
            "configuration": configuration,
        }),
        "payout_engine_hash": settlement_engine_sha256,
        "evaluation_protocol_id": _canonical_sha256(protocol),
        "resampling_condition_id": confidence["condition_id"],
        "source_revision": payload.get("source_revision") or implementation_sha256,
        "source_revision_kind": (
            "repository_revision" if payload.get("source_revision")
            else "implementation_bundle_sha256"
        ),
        "outer_draw_definition": outer_draw_definition,
        "inner_scenario_definition": inner_scenario_definition,
    }
    scalar_lineage_fields = (
        "parent_artifact_hash",
        "prediction_model_hash",
        "joint_scenario_model_hash",
        "calibrator_hash",
        "calibration_ledger_hash",
        "portfolio_policy_hash",
        "payout_engine_hash",
        "evaluation_protocol_id",
        "resampling_condition_id",
        "source_revision",
    )
    draw_definitions_complete = all(
        value is not None
        for value in outer_draw_definition.values()
    ) and all(
        value is not None
        for value in inner_scenario_definition.values()
    )
    artifact_lineage["lineage_complete"] = bool(
        draw_definitions_complete
        and all(
            artifact_lineage.get(field) not in (None, "")
            for field in scalar_lineage_fields
        )
    )
    return {
        "model": MODEL_VERSION,
        "status": (
            "promotion_candidate"
            if all(formal_gate.values())
            else purchase_gate_outcome
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
        "race_count": sum(day["evaluated_races"] for day in replay_days),
        "candidate_count": len(all_candidate_records),
        "calendar_span_days": calendar_span_days,
        "observed_race_days": len(observed_dates),
        "candidate_days": len(candidate_dates),
        "settled_candidate_days": len(settled_candidate_dates),
        "calibration_eligible_days": len(settled_candidate_dates),
        "candidate_decision_count": len(decision_rows),
        "candidate_approval_examples": approval_examples,
        "candidate_denial_examples": denial_examples,
        "post_warmup_denial_examples": post_warmup_denial_examples,
        "strict_prior_violation_count": independence_audit[
            "strict_prior_violation_count"
        ],
        "future_candidate_in_calibration_count": independence_audit[
            "future_candidate_in_calibration_count"
        ],
        "same_race_calibrator_hash_count_max": race_batch_audit[
            "same_race_calibrator_hash_count_max"
        ],
        "same_race_mid_decision_update_count": race_batch_audit[
            "same_race_mid_decision_update_count"
        ],
        "same_race_result_leakage_count": race_batch_audit[
            "same_race_result_leakage_count"
        ],
        "artifact_lineage": artifact_lineage,
        "calibration_ready_days": ready_days,
        "calibration_ready_races": ready_races,
        "calibrated_candidates": calibrated_candidates,
        "latest_calibration_decision": (
            replay_days[-1]["races"][-1]
            if replay_days and replay_days[-1].get("races") else None
        ),
        "calibration_training_records": len(all_candidate_records),
        "calibration_teacher_admitted_records": len(training_records),
        "calibration_pending_unsettled_records": len(
            pending_training_records
        ),
        "calibration_input_sources": calibration_input_sources,
        "calibration_folds": calibration_folds,
        "calibration_independence_audit": independence_audit,
        "same_race_calibrator_settlement_batch_audit": race_batch_audit,
        "calibration_learning_population_audit": (
            learning_population_audit
        ),
        "calibration_warmup_audit": warmup_audit,
        "calibrator_update_audit": calibrator_update_audit,
        "calibration_input_range_audit": calibration_input_range_audit,
        "calibration_local_support_audit": (
            calibration_local_support_audit
        ),
        "calibration_lcb_audit": calibration_lcb_audit,
        "replay_reproducibility_audit": replay_reproducibility_audit,
        "rejection_reasons": dict(sorted(rejected_reasons.items())),
        "primary_bankroll": bankroll,
        "bankroll_confidence": confidence,
        "formal_purchase_value": {
            "raw_v_buy_is_diagnostic_only": True,
            "formal_gate_input": "calibrated_gross_return_lcb95",
            "definition": "strict_prior_empirical_gross_return_isotonic_day_LCB95",
            "calibration_target": (
                "gross_return_per_staked_yen_including_returned_principal"
            ),
            "value_unit": "net_expected_edge_equals_gross_return_minus_one",
            "purchase_condition": (
                "calibrated_gross_return_lcb95_greater_than_"
                "one_plus_safety_margin"
            ),
            "safety_margin": calibration_margin,
            "minimum": (
                min(lcb_values) - 1.0 if lcb_values else None
            ),
            "selected_portfolios": len(purchased),
            "all_above_safety_margin": formal_gate[
                "calibrated_lcb_above_margin"
            ],
        },
        "purchase_value_realization_calibration": value_realization,
        "purchase_gate_operational_audit": purchase_gate_audit,
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
    parser.add_argument("--calibration-bootstrap-samples", type=int, default=5_000)
    parser.add_argument("--calibration-min-training-days", type=int, default=30)
    parser.add_argument("--calibration-min-portfolios", type=int, default=300)
    parser.add_argument("--calibration-min-candidate-days", type=int, default=20)
    parser.add_argument(
        "--calibration-min-local-candidates", type=int, default=50
    )
    parser.add_argument(
        "--calibration-min-local-candidate-days", type=int, default=20
    )
    parser.add_argument(
        "--calibration-min-local-ess", type=float, default=10.0
    )
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
