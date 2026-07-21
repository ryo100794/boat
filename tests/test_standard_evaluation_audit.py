from __future__ import annotations

from datetime import date, timedelta
import importlib.util
import json
from pathlib import Path
import sqlite3

import pytest

from boatrace_ai.standard_evaluation import (
    MODEL_SOURCES,
    POLICY,
    evaluate_promotions,
    protocol_sha256,
    race_set_sha256,
)


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_standardized_evaluation.py"
SPEC = importlib.util.spec_from_file_location("audit_standardized_evaluation", SCRIPT)
assert SPEC and SPEC.loader
AUDIT_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT_MODULE)


HOLDOUT_START = date(2025, 1, 1)
HOLDOUT_END = date(2025, 12, 31)
RACE_IDS = [f"race-{offset:03d}" for offset in range(365)]


def _protocol() -> dict:
    payload = {
        "protocol_id": "standard_365d_v2",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "as_of_date_jst": "2026-01-01",
        "holdout_start": HOLDOUT_START.isoformat(),
        "holdout_end": HOLDOUT_END.isoformat(),
        "calendar_days": 365,
        "holdout_date_count": 365,
        "training_races": 0,
        "prediction_races": 365,
        "bankroll_evaluable_races": 365,
        "prediction_universe": "exactly 6 entries and 6 non-null finish ranks",
        "bankroll_universe": (
            "prediction universe with a non-null official trifecta payout"
        ),
        "race_set_sha256": race_set_sha256(RACE_IDS),
        "full_day_boundary": True,
        "model_selection": "training period only; final 365-day holdout untouched",
        "policy": dict(POLICY),
    }
    payload["protocol_sha256"] = protocol_sha256(payload)
    return payload


def _daily() -> list[dict]:
    return [
        {
            "race_date": (HOLDOUT_START + timedelta(days=offset)).isoformat(),
            "evaluated_races": 1,
            "tickets": 1,
            "stake_yen": 100,
            "return_yen": 200,
            "profit_yen": 100,
            "roi": 2.0,
            "cumulative_profit_yen": (offset + 1) * 100,
        }
        for offset in range(365)
    ]


def _model(model_id: str, *, incumbent: bool) -> dict:
    aggregate = {
        "stake_yen": 36_500,
        "return_yen": 73_000,
        "profit_yen": 36_500,
        "tickets": 365,
        "roi": 2.0,
    }
    return {
        "protocol_id": "standard_365d_v2",
        "protocol_sha256": _protocol()["protocol_sha256"],
        "model_id": model_id,
        "policy": dict(POLICY),
        "validation": {
            "passed": True,
            "policy_mismatches": [],
            "prediction_race_set_sha256": race_set_sha256(RACE_IDS),
            "bankroll_race_set_sha256": race_set_sha256(RACE_IDS),
            "expected_race_set_sha256": race_set_sha256(RACE_IDS),
            "prediction_race_count": 365,
            "bankroll_race_count": 365,
        },
        "evaluated_races": 365,
        "bankroll_evaluated_races": 365,
        "entry_log_loss": 0.5 if incumbent else 0.4,
        "winner_top1_accuracy": 0.4 if incumbent else 0.5,
        "trifecta_top5_hit_rate": 0.2 if incumbent else 0.3,
        "max_drawdown_yen": 0,
        "daily": _daily(),
        "bankroll": dict(aggregate),
        **aggregate,
    }


def _write_bundle(root: Path) -> None:
    protocol = _protocol()
    models = [
        _model(source.model_id, incumbent=source.role == "incumbent")
        for source in MODEL_SOURCES
    ]
    promotion_decision = evaluate_promotions(models, comparison_ready=True)
    manifest = {
        **protocol,
        "comparison_model_ids": [source.model_id for source in MODEL_SOURCES],
        "registry_coverage_passed": True,
        "model_count": len(models),
        "valid_model_count": len(models),
        "failed_models": [],
        "comparison_ready": True,
        "promotion_decision": promotion_decision,
    }
    root.mkdir()
    (root / "protocol.json").write_text(
        json.dumps(protocol),
        encoding="utf-8",
    )
    (root / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    for model in models:
        (root / f"{model['model_id']}.json").write_text(
            json.dumps(model),
            encoding="utf-8",
        )


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_database(path: Path, *, omit_last_day: bool = False) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE races (
            race_id TEXT,
            race_date TEXT,
            jcd TEXT,
            rno INTEGER
        );
        CREATE TABLE entries (race_id TEXT, lane INTEGER);
        CREATE TABLE race_results (race_id TEXT, lane INTEGER, rank INTEGER);
        CREATE TABLE payouts (
            race_id TEXT,
            bet_type TEXT,
            payout_yen INTEGER
        );
        """
    )
    limit = 364 if omit_last_day else 365
    for offset, race_id in enumerate(RACE_IDS[:limit]):
        race_date = (HOLDOUT_START + timedelta(days=offset)).isoformat()
        conn.execute(
            "INSERT INTO races VALUES (?, ?, '01', 1)",
            (race_id, race_date),
        )
        conn.executemany(
            "INSERT INTO entries VALUES (?, ?)",
            ((race_id, lane) for lane in range(1, 7)),
        )
        conn.executemany(
            "INSERT INTO race_results VALUES (?, ?, ?)",
            ((race_id, lane, lane) for lane in range(1, 7)),
        )
        conn.execute(
            "INSERT INTO payouts VALUES (?, '3連単', 1000)",
            (race_id,),
        )
    conn.commit()
    conn.close()


def test_audit_accepts_consistent_bundle(tmp_path) -> None:
    root = tmp_path / "artifacts"
    _write_bundle(root)

    result = AUDIT_MODULE.audit(root)

    assert result["passed"] is True
    assert result["errors"] == []


def test_audit_uses_protocol_instead_of_manifest_as_expectation(tmp_path) -> None:
    root = tmp_path / "artifacts"
    _write_bundle(root)
    manifest_path = root / "manifest.json"
    manifest = _read(manifest_path)
    manifest["training_races"] = 999
    manifest["protocol_sha256"] = protocol_sha256(manifest)
    _write(manifest_path, manifest)

    result = AUDIT_MODULE.audit(root)

    assert result["passed"] is False
    assert "manifest training_races mismatch" in result["errors"]
    assert "protocol fingerprint mismatch" in result["errors"]


@pytest.mark.parametrize(
    ("field", "delta", "message"),
    (
        ("stake_yen", 100, "stake_yen aggregate mismatch"),
        ("return_yen", 100, "return_yen aggregate mismatch"),
        ("profit_yen", 100, "profit_yen aggregate mismatch"),
        ("tickets", 1, "tickets aggregate mismatch"),
        (
            "bankroll_evaluated_races",
            1,
            "bankroll evaluated race total mismatch",
        ),
        ("roi", 0.1, "roi aggregate mismatch"),
    ),
)
def test_audit_rejects_model_aggregate_not_backed_by_daily_ledger(
    tmp_path,
    field: str,
    delta: float,
    message: str,
) -> None:
    root = tmp_path / "artifacts"
    _write_bundle(root)
    model_path = root / f"{MODEL_SOURCES[0].model_id}.json"
    model = _read(model_path)
    model[field] += delta
    _write(model_path, model)

    result = AUDIT_MODULE.audit(root)

    assert result["passed"] is False
    assert any(message in error for error in result["errors"])


def test_audit_rejects_broken_cumulative_profit(tmp_path) -> None:
    root = tmp_path / "artifacts"
    _write_bundle(root)
    model_path = root / f"{MODEL_SOURCES[0].model_id}.json"
    model = _read(model_path)
    model["daily"][10]["cumulative_profit_yen"] += 1
    _write(model_path, model)

    result = AUDIT_MODULE.audit(root)

    assert result["passed"] is False
    assert any("cumulative profit mismatch" in error for error in result["errors"])


@pytest.mark.parametrize("tamper", ("criteria", "candidates", "checks", "reason"))
def test_audit_recalculates_the_complete_promotion_decision(
    tmp_path,
    tamper: str,
) -> None:
    root = tmp_path / "artifacts"
    _write_bundle(root)
    manifest_path = root / "manifest.json"
    manifest = _read(manifest_path)
    decision = manifest["promotion_decision"]
    if tamper == "criteria":
        decision["criteria"]["minimum_roi"] = 99.0
    elif tamper == "candidates":
        decision["candidates"] = decision["candidates"][1:]
    elif tamper == "checks":
        decision["candidates"][0]["checks"]["roi_at_least_one"] = False
    else:
        decision["reason"] = "tampered"
    _write(manifest_path, manifest)

    result = AUDIT_MODULE.audit(root)

    assert result["passed"] is False
    assert "promotion decision mismatch" in result["errors"]


def test_audit_db_option_rebuilds_matching_protocol(tmp_path) -> None:
    root = tmp_path / "artifacts"
    db_path = tmp_path / "evaluation.sqlite"
    _write_bundle(root)
    _write_database(db_path)

    result = AUDIT_MODULE.audit(root, db=db_path)

    assert result["passed"] is True


def test_audit_db_option_rejects_database_boundary_mismatch(tmp_path) -> None:
    root = tmp_path / "artifacts"
    db_path = tmp_path / "evaluation.sqlite"
    _write_bundle(root)
    _write_database(db_path, omit_last_day=True)

    result = AUDIT_MODULE.audit(root, db=db_path)

    assert result["passed"] is False
    assert any(
        error.startswith("database protocol verification failed:")
        for error in result["errors"]
    )
