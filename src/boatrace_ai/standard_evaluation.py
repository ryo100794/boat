from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from .db import connection, init_db


PROTOCOL_ID = "standard_365d_v2"
INCUMBENT_MODEL_ID = "no_odds_v8"
PREDICTION_METRICS = (
    "entry_log_loss",
    "entry_brier",
    "winner_top1_accuracy",
    "trifecta_top1_hit_rate",
    "trifecta_top5_hit_rate",
    "ranking_log_loss",
)
POLICY = {
    "daily_budget_yen": 10_000,
    "bet_type": "3連単",
    "include_odds": False,
    "require_real_odds": False,
    "ev_threshold": 1.20,
    "payout_prior_weight": 30.0,
    "fractional_kelly": 0.25,
    "max_daily_exposure_fraction": 0.60,
    "min_daily_exposure_fraction": 0.40,
    "race_cap_fraction": 0.10,
    "ticket_cap_fraction": 0.03,
    "max_daily_tickets": 30,
    "allocation_mode": "normalized_kelly",
    "stake_granularity_yen": 100,
    "min_stake_yen": 100,
}
PROMOTION_CRITERIA = {
    "minimum_roi": 1.0,
    "minimum_profit_yen_exclusive": 0,
    "entry_log_loss_not_worse_than_incumbent": True,
    "winner_top1_not_worse_than_incumbent": True,
    "trifecta_top5_not_worse_than_incumbent": True,
    "all_model_validations_required": True,
}


@dataclass(frozen=True)
class ModelSource:
    model_id: str
    prediction_file: str
    bankroll_file: str
    prediction_section: str | None = None
    role: str = "candidate"


@dataclass(frozen=True)
class ModelRegistration:
    model_id: str
    category: str
    comparison_status: str
    reason: str


MODEL_SOURCES = (
    ModelSource(
        "no_odds_v8",
        "no_odds_v8_prediction.json",
        "no_odds_v8_bankroll.json",
        role="incumbent",
    ),
    ModelSource("pastlog_v7", "pastlog_v7_prediction.json", "pastlog_v7_bankroll.json"),
    ModelSource(
        "pastlog_v9_research",
        "pastlog_v9_research_prediction.json",
        "pastlog_v9_research_bankroll.json",
    ),
    ModelSource("calibrated_linear", "calibrated_linear.json", "calibrated_linear.json"),
    ModelSource("calibrated_mlp", "calibrated_mlp.json", "calibrated_mlp.json"),
    ModelSource(
        "listwise_feature_teacher",
        "listwise_feature_teacher.json",
        "listwise_feature_teacher.json",
        "holdout",
    ),
    ModelSource(
        "listwise_newton",
        "listwise_newton.json",
        "listwise_newton.json",
        "holdout_after_newton",
    ),
)

MODEL_REGISTRY = (
    *(
        ModelRegistration(
            source.model_id,
            "prediction_model",
            "included",
            "same historical no-odds universe and fixed bankroll policy",
        )
        for source in MODEL_SOURCES
    ),
    ModelRegistration(
        "realtime_odds_shadow",
        "prediction_model",
        "separate_protocol",
        "requires deadline-time real odds; a complete 365-day odds universe is unavailable",
    ),
    ModelRegistration(
        "listwise_temporal_stability",
        "selection_diagnostic",
        "diagnostic_only",
        "latest interval was reused diagnostically and is not an untouched promotion holdout",
    ),
    ModelRegistration(
        "adaptive_no_bet_policy",
        "bankroll_policy_search",
        "policy_variant",
        "changes the bankroll policy and therefore cannot be mixed with the fixed-policy model comparison",
    ),
)

_PROTOCOL_FINGERPRINT_KEYS = (
    "protocol_id",
    "as_of_date_jst",
    "holdout_start",
    "holdout_end",
    "calendar_days",
    "holdout_date_count",
    "training_races",
    "prediction_races",
    "bankroll_evaluable_races",
    "prediction_universe",
    "bankroll_universe",
    "race_set_sha256",
    "full_day_boundary",
    "model_selection",
    "policy",
)


def race_set_sha256(race_ids: Iterable[str]) -> str:
    return hashlib.sha256(
        "\n".join(sorted({str(value) for value in race_ids})).encode("utf-8")
    ).hexdigest()


def protocol_sha256(protocol: dict[str, Any]) -> str:
    payload = {key: protocol.get(key) for key in _PROTOCOL_FINGERPRINT_KEYS}
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_protocol(
    conn,
    *,
    days: int = 365,
    as_of_date: date | None = None,
) -> dict[str, Any]:
    evaluation_cutoff = as_of_date or datetime.now(ZoneInfo("Asia/Tokyo")).date()
    rows = conn.execute(
        """
        SELECT r.race_id, r.race_date, r.jcd, r.rno
        FROM races r
        WHERE (SELECT COUNT(*) FROM entries e WHERE e.race_id = r.race_id) = 6
          AND (SELECT COUNT(*) FROM race_results rr
               WHERE rr.race_id = r.race_id AND rr.rank IS NOT NULL) = 6
          AND r.race_date < ?
        ORDER BY r.race_date, r.jcd, r.rno
        """,
        (evaluation_cutoff.isoformat(),),
    ).fetchall()
    if not rows:
        raise ValueError("no complete races for standardized evaluation")
    holdout_end = date.fromisoformat(str(rows[-1]["race_date"]))
    holdout_start = holdout_end - timedelta(days=max(1, int(days)) - 1)
    train_rows = [row for row in rows if str(row["race_date"]) < holdout_start.isoformat()]
    holdout_rows = [
        row
        for row in rows
        if holdout_start.isoformat() <= str(row["race_date"]) <= holdout_end.isoformat()
    ]
    holdout_ids = [str(row["race_id"]) for row in holdout_rows]
    payout_count = int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM races r
            WHERE r.race_date BETWEEN ? AND ?
              AND (SELECT COUNT(*) FROM entries e WHERE e.race_id = r.race_id) = 6
              AND (SELECT COUNT(*) FROM race_results rr
                   WHERE rr.race_id = r.race_id AND rr.rank IS NOT NULL) = 6
              AND EXISTS (
                  SELECT 1 FROM payouts p
                  WHERE p.race_id = r.race_id
                    AND p.bet_type = '3連単'
                    AND p.payout_yen IS NOT NULL
              )
            """,
            (holdout_start.isoformat(), holdout_end.isoformat()),
        ).fetchone()[0]
    )
    holdout_dates = sorted({str(row["race_date"]) for row in holdout_rows})
    protocol = {
        "protocol_id": PROTOCOL_ID,
        "generated_at": _now(),
        "as_of_date_jst": evaluation_cutoff.isoformat(),
        "holdout_start": holdout_start.isoformat(),
        "holdout_end": holdout_end.isoformat(),
        "calendar_days": int(days),
        "holdout_date_count": len(holdout_dates),
        "training_races": len(train_rows),
        "prediction_races": len(holdout_rows),
        "bankroll_evaluable_races": payout_count,
        "prediction_universe": "exactly 6 entries and 6 non-null finish ranks",
        "bankroll_universe": "prediction universe with a non-null official trifecta payout",
        "race_set_sha256": race_set_sha256(holdout_ids),
        "full_day_boundary": holdout_end < evaluation_cutoff,
        "model_selection": "training period only; final 365-day holdout untouched",
        "policy": dict(POLICY),
    }
    protocol["protocol_sha256"] = protocol_sha256(protocol)
    return protocol


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = _read_json(path)
    _validate_protocol_shape(protocol)
    return protocol


def verify_protocol_against_database(conn, protocol: dict[str, Any]) -> None:
    expected = build_protocol(
        conn,
        days=int(protocol["calendar_days"]),
        as_of_date=date.fromisoformat(str(protocol["as_of_date_jst"])),
    )
    if expected["protocol_sha256"] != protocol["protocol_sha256"]:
        raise ValueError(
            "frozen evaluation protocol no longer matches the database: "
            f"expected {protocol['protocol_sha256']}, got {expected['protocol_sha256']}"
        )


def consolidate_model(
    *,
    protocol: dict[str, Any],
    source: ModelSource,
    prediction: dict[str, Any],
    bankroll: dict[str, Any],
) -> dict[str, Any]:
    prediction_metrics = prediction
    if source.prediction_section:
        prediction_metrics = prediction.get(source.prediction_section) or {}
    nested_bankroll = bankroll.get("bankroll") or {}
    daily = list(bankroll.get("daily") or [])
    aggregate = _aggregate_daily(daily)
    policy = bankroll.get("policy") or {}
    prediction_races = _first_int(
        prediction_metrics.get("evaluated_races"),
        prediction.get("holdout_races"),
        prediction.get("evaluated_races"),
    )
    bankroll_races = _first_int(
        nested_bankroll.get("evaluated_races"),
        bankroll.get("bankroll_evaluated_races"),
        bankroll.get("evaluated_races"),
        aggregate.get("evaluated_races"),
    )
    normalized_policy = {
        key: policy.get(key, False if key in {"include_odds", "require_real_odds"} else None)
        for key in POLICY
    }
    policy_mismatches = [
        key
        for key, expected in POLICY.items()
        if not _same_value(normalized_policy.get(key), expected)
    ]
    daily_dates = sorted({str(row.get("race_date")) for row in daily if row.get("race_date")})
    prediction_hash = _race_hash(prediction_metrics, prediction)
    bankroll_hash = _race_hash(nested_bankroll, bankroll)
    expected_hash = str(protocol["race_set_sha256"])
    validation = {
        "prediction_race_count": prediction_races,
        "expected_prediction_races": int(protocol["prediction_races"]),
        "bankroll_race_count": bankroll_races,
        "expected_bankroll_races": int(protocol["bankroll_evaluable_races"]),
        "prediction_race_set_sha256": prediction_hash,
        "bankroll_race_set_sha256": bankroll_hash,
        "expected_race_set_sha256": expected_hash,
        "daily_date_count": len(daily_dates),
        "expected_daily_date_count": int(protocol["holdout_date_count"]),
        "daily_start": daily_dates[0] if daily_dates else None,
        "daily_end": daily_dates[-1] if daily_dates else None,
        "policy_mismatches": policy_mismatches,
    }
    validation["passed"] = all(
        (
            prediction_races == int(protocol["prediction_races"]),
            bankroll_races == int(protocol["bankroll_evaluable_races"]),
            prediction_hash == expected_hash,
            bankroll_hash == expected_hash,
            len(daily_dates) == int(protocol["holdout_date_count"]),
            validation["daily_start"] == protocol["holdout_start"],
            validation["daily_end"] == protocol["holdout_end"],
            not policy_mismatches,
        )
    )
    bank = _bankroll_metrics(bankroll, nested_bankroll, aggregate)
    return {
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "evaluation_scope": PROTOCOL_ID,
        "generated_at": _now(),
        "model_id": source.model_id,
        "model": prediction.get("model") or bankroll.get("model") or source.model_id,
        "role": source.role,
        "feature_set": prediction.get("feature_set") or bankroll.get("feature_set"),
        "include_odds": False,
        "protocol": protocol,
        "policy": dict(POLICY),
        "validation": validation,
        "evaluated_races": prediction_races,
        "bankroll_evaluated_races": bankroll_races,
        "evaluation_race_set_sha256": expected_hash,
        "daily": daily,
        "bankroll": bank,
        **{
            key: prediction_metrics.get(key)
            for key in PREDICTION_METRICS
            if prediction_metrics.get(key) is not None
        },
        **bank,
    }


def validate_model_source(
    *,
    protocol: dict[str, Any],
    source: ModelSource,
    raw_dir: Path,
) -> dict[str, Any]:
    result = consolidate_model(
        protocol=protocol,
        source=source,
        prediction=_read_json(raw_dir / source.prediction_file),
        bankroll=_read_json(raw_dir / source.bankroll_file),
    )
    return {
        "model_id": source.model_id,
        "validation": result["validation"],
    }


def evaluate_promotions(
    results: list[dict[str, Any]],
    *,
    comparison_ready: bool,
) -> dict[str, Any]:
    by_id = {str(row["model_id"]): row for row in results}
    incumbent = by_id.get(INCUMBENT_MODEL_ID)
    if incumbent is None:
        raise ValueError(f"incumbent result missing: {INCUMBENT_MODEL_ID}")
    candidates = []
    eligible = []
    for source in MODEL_SOURCES:
        row = by_id[source.model_id]
        if source.role == "incumbent":
            row["promotion"] = {
                "role": "incumbent",
                "eligible": False,
                "checks": {"validation_passed": bool(row["validation"]["passed"])},
            }
            continue
        checks = {
            "comparison_ready": comparison_ready,
            "validation_passed": bool(row["validation"]["passed"]),
            "roi_at_least_one": _number(row.get("roi"), -1.0) >= 1.0,
            "profit_positive": _number(row.get("profit_yen"), 0.0) > 0.0,
            "entry_log_loss_not_worse": _number(row.get("entry_log_loss"), float("inf"))
            <= _number(incumbent.get("entry_log_loss"), float("-inf")),
            "winner_top1_not_worse": _number(row.get("winner_top1_accuracy"), -1.0)
            >= _number(incumbent.get("winner_top1_accuracy"), float("inf")),
            "trifecta_top5_not_worse": _number(row.get("trifecta_top5_hit_rate"), -1.0)
            >= _number(incumbent.get("trifecta_top5_hit_rate"), float("inf")),
        }
        candidate_eligible = all(checks.values())
        promotion = {
            "role": "candidate",
            "eligible": candidate_eligible,
            "checks": checks,
            "failed_checks": [key for key, passed in checks.items() if not passed],
        }
        row["promotion"] = promotion
        candidate = {
            "model_id": source.model_id,
            "eligible": candidate_eligible,
            "checks": checks,
            "failed_checks": promotion["failed_checks"],
            "roi": row.get("roi"),
            "profit_yen": row.get("profit_yen"),
        }
        candidates.append(candidate)
        if candidate_eligible:
            eligible.append(row)
    selected = max(
        eligible,
        key=lambda row: (
            _number(row.get("roi"), -1.0),
            -_number(row.get("entry_log_loss"), float("inf")),
            _number(row.get("winner_top1_accuracy"), -1.0),
            _number(row.get("trifecta_top5_hit_rate"), -1.0),
        ),
        default=None,
    )
    return {
        "criteria": dict(PROMOTION_CRITERIA),
        "incumbent_model_id": INCUMBENT_MODEL_ID,
        "status": "promote" if selected else "retain_incumbent",
        "selected_model_id": str(selected["model_id"]) if selected else INCUMBENT_MODEL_ID,
        "eligible_candidate_ids": [str(row["model_id"]) for row in eligible],
        "candidates": candidates,
        "reason": (
            "one or more candidates passed every unified prediction and bankroll gate"
            if selected
            else "no candidate passed every unified prediction and bankroll gate"
        ),
    }


def consolidate(
    conn,
    *,
    raw_dir: Path,
    output_dir: Path,
    protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    protocol = protocol or build_protocol(conn)
    _validate_protocol_shape(protocol)
    verify_protocol_against_database(conn, protocol)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    failures = []
    for source in MODEL_SOURCES:
        result = consolidate_model(
            protocol=protocol,
            source=source,
            prediction=_read_json(raw_dir / source.prediction_file),
            bankroll=_read_json(raw_dir / source.bankroll_file),
        )
        results.append(result)
        if not result["validation"]["passed"]:
            failures.append(source.model_id)
    included_registry = {
        row.model_id for row in MODEL_REGISTRY if row.comparison_status == "included"
    }
    source_ids = {row.model_id for row in MODEL_SOURCES}
    registry_coverage_passed = included_registry == source_ids
    comparison_ready = (
        not failures
        and len(results) == len(MODEL_SOURCES)
        and registry_coverage_passed
    )
    promotion_decision = evaluate_promotions(results, comparison_ready=comparison_ready)
    models = []
    for result in results:
        path = output_dir / f"{result['model_id']}.json"
        _write_json_atomic(path, result)
        models.append(
            {
                "model_id": result["model_id"],
                "file": str(path),
                "role": result["role"],
                "validation": result["validation"],
                "promotion": result["promotion"],
                "entry_log_loss": result.get("entry_log_loss"),
                "winner_top1_accuracy": result.get("winner_top1_accuracy"),
                "trifecta_top5_hit_rate": result.get("trifecta_top5_hit_rate"),
                "roi": result.get("roi"),
                "profit_yen": result.get("profit_yen"),
            }
        )
    registry = [asdict(row) for row in MODEL_REGISTRY]
    manifest = {
        **protocol,
        "model_registry": registry,
        "comparison_model_ids": [source.model_id for source in MODEL_SOURCES],
        "excluded_models": [
            row for row in registry if row["comparison_status"] != "included"
        ],
        "registry_coverage_passed": registry_coverage_passed,
        "models": models,
        "model_count": len(models),
        "valid_model_count": len(models) - len(failures),
        "failed_models": failures,
        "comparison_ready": comparison_ready,
        "promotion_decision": promotion_decision,
    }
    _write_json_atomic(output_dir / "manifest.json", manifest)
    if failures:
        raise ValueError(f"standardized evaluation validation failed: {failures}")
    if not registry_coverage_passed:
        raise ValueError("standardized evaluation model registry does not match model sources")
    return manifest


def _validate_protocol_shape(protocol: dict[str, Any]) -> None:
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"unexpected protocol_id: {protocol.get('protocol_id')}")
    if protocol.get("policy") != POLICY:
        raise ValueError("frozen protocol policy does not match the standardized policy")
    actual = str(protocol.get("protocol_sha256") or "")
    expected = protocol_sha256(protocol)
    if not actual or actual != expected:
        raise ValueError(f"protocol fingerprint mismatch: expected {expected}, got {actual or '-'}")


def _race_hash(*values: dict[str, Any]) -> str | None:
    for value in values:
        candidate = value.get("evaluation_race_set_sha256") or value.get("race_set_sha256")
        if candidate:
            return str(candidate)
    return None


def _aggregate_daily(daily: list[dict[str, Any]]) -> dict[str, int]:
    keys = (
        "evaluated_races",
        "candidate_tickets",
        "tickets",
        "races_bet",
        "hit_tickets",
        "hit_races",
        "stake_yen",
        "return_yen",
        "profit_yen",
    )
    result = {key: sum(int(row.get(key) or 0) for row in daily) for key in keys}
    result["winning_days"] = sum(int(row.get("profit_yen") or 0) > 0 for row in daily)
    result["losing_days"] = sum(int(row.get("profit_yen") or 0) < 0 for row in daily)
    return result


def _bankroll_metrics(
    root: dict[str, Any],
    nested: dict[str, Any],
    aggregate: dict[str, int],
) -> dict[str, Any]:
    def value(key: str, default: Any = None) -> Any:
        if nested.get(key) is not None:
            return nested[key]
        if root.get(key) is not None:
            return root[key]
        return aggregate.get(key, default)

    stake = int(value("stake_yen", 0) or 0)
    returned = int(value("return_yen", 0) or 0)
    tickets = int(value("tickets", value("selected_tickets", 0)) or 0)
    selected_races = int(value("selected_races", value("races_bet", 0)) or 0)
    hit_tickets = int(value("hit_tickets", 0) or 0)
    return {
        "race_days": len(root.get("daily") or []),
        "candidate_tickets": int(value("candidate_tickets", 0) or 0),
        "selected_races": selected_races,
        "tickets": tickets,
        "hit_tickets": hit_tickets,
        "ticket_hit_rate": float(value("ticket_hit_rate", hit_tickets / tickets if tickets else 0.0) or 0.0),
        "race_hit_rate": float(value("race_hit_rate", 0.0) or 0.0),
        "stake_yen": stake,
        "return_yen": returned,
        "profit_yen": int(value("profit_yen", returned - stake) or 0),
        "roi": float(value("roi", returned / stake if stake else 0.0) or 0.0),
        "winning_days": int(value("winning_days", 0) or 0),
        "losing_days": int(value("losing_days", 0) or 0),
        "budget_utilization": float(value("budget_utilization", 0.0) or 0.0),
        "max_drawdown_yen": int(value("max_drawdown_yen", 0) or 0),
    }


def _first_int(*values: Any) -> int:
    for value in values:
        if value is not None:
            return int(value)
    return 0


def _same_value(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) <= 1e-12
        except (TypeError, ValueError):
            return False
    return actual == expected


def _number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze, validate and consolidate all models under one 365-day protocol."
    )
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--raw-dir", default="data/models/standardized_365d_v2/raw", type=Path)
    parser.add_argument("--output-dir", default="data/models/standardized_365d_v2", type=Path)
    parser.add_argument("--protocol-file", type=Path)
    parser.add_argument("--as-of-date", type=date.fromisoformat)
    parser.add_argument("--days", type=int, default=365)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument(
        "--validate-source",
        choices=[source.model_id for source in MODEL_SOURCES],
    )
    parser.add_argument(
        "--artifacts-only",
        action="store_true",
        help="validate artifacts against an already database-verified protocol",
    )
    args = parser.parse_args(argv)
    if args.artifacts_only:
        if not args.validate_source:
            parser.error("--artifacts-only requires --validate-source")
        if not args.protocol_file or not args.protocol_file.exists():
            parser.error("--artifacts-only requires an existing --protocol-file")
        protocol = load_protocol(args.protocol_file)
        if (
            args.as_of_date
            and str(protocol["as_of_date_jst"]) != args.as_of_date.isoformat()
        ):
            raise ValueError("existing protocol as-of date differs from --as-of-date")
        source = next(
            item for item in MODEL_SOURCES if item.model_id == args.validate_source
        )
        result = validate_model_source(
            protocol=protocol,
            source=source,
            raw_dir=args.raw_dir,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0 if result["validation"]["passed"] else 1
    init_db(args.db)
    with connection(args.db) as conn:
        if args.protocol_file and args.protocol_file.exists():
            protocol = load_protocol(args.protocol_file)
            if args.as_of_date and str(protocol["as_of_date_jst"]) != args.as_of_date.isoformat():
                raise ValueError("existing protocol as-of date differs from --as-of-date")
        else:
            protocol = build_protocol(conn, days=args.days, as_of_date=args.as_of_date)
            if args.protocol_file:
                _write_json_atomic(args.protocol_file, protocol)
        verify_protocol_against_database(conn, protocol)
        if args.prepare_only:
            result = protocol
        elif args.validate_source:
            source = next(
                item for item in MODEL_SOURCES if item.model_id == args.validate_source
            )
            result = validate_model_source(
                protocol=protocol,
                source=source,
                raw_dir=args.raw_dir,
            )
        else:
            result = consolidate(
                conn,
                raw_dir=args.raw_dir,
                output_dir=args.output_dir,
                protocol=protocol,
            )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if args.validate_source and not result["validation"]["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
