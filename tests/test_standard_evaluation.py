from __future__ import annotations

from copy import deepcopy
from datetime import date
import json
import sqlite3

from boatrace_ai.standard_evaluation import (
    POLICY,
    ModelSource,
    build_protocol,
    consolidate_model,
    protocol_sha256,
    race_set_sha256,
    validate_model_source,
)


def protocol() -> dict:
    payload = {
        "protocol_id": "standard_365d_v2",
        "holdout_start": "2026-01-01",
        "holdout_end": "2026-01-02",
        "holdout_date_count": 2,
        "prediction_races": 2,
        "bankroll_evaluable_races": 2,
        "race_set_sha256": race_set_sha256(("race-1", "race-2")),
        "policy": dict(POLICY),
    }
    payload["protocol_sha256"] = protocol_sha256(payload)
    return payload


def prediction() -> dict:
    return {
        "model": "candidate",
        "feature_set": "features",
        "evaluated_races": 2,
        "evaluation_race_set_sha256": race_set_sha256(("race-1", "race-2")),
        "entry_log_loss": 0.31,
        "entry_brier": 0.09,
        "winner_top1_accuracy": 0.57,
        "trifecta_top5_hit_rate": 0.32,
    }


def bankroll() -> dict:
    return {
        "policy": dict(POLICY),
        "evaluation_race_set_sha256": race_set_sha256(("race-1", "race-2")),
        "stake_yen": 400,
        "return_yen": 500,
        "profit_yen": 100,
        "roi": 1.25,
        "max_drawdown_yen": 200,
        "daily": [
            {
                "race_date": "2026-01-01",
                "evaluated_races": 1,
                "candidate_tickets": 3,
                "tickets": 2,
                "races_bet": 1,
                "hit_tickets": 0,
                "hit_races": 0,
                "stake_yen": 200,
                "return_yen": 0,
                "profit_yen": -200,
            },
            {
                "race_date": "2026-01-02",
                "evaluated_races": 1,
                "candidate_tickets": 4,
                "tickets": 2,
                "races_bet": 1,
                "hit_tickets": 1,
                "hit_races": 1,
                "stake_yen": 200,
                "return_yen": 500,
                "profit_yen": 300,
            },
        ],
    }


def test_consolidated_model_requires_one_protocol() -> None:
    result = consolidate_model(
        protocol=protocol(),
        source=ModelSource("candidate", "prediction.json", "bankroll.json"),
        prediction=prediction(),
        bankroll=bankroll(),
    )

    assert result["validation"]["passed"] is True
    assert result["evaluation_scope"] == "standard_365d_v2"
    assert result["evaluated_races"] == 2
    assert result["bankroll_evaluated_races"] == 2
    assert result["roi"] == 1.25
    assert result["tickets"] == 4


def test_policy_mismatch_blocks_comparison() -> None:
    changed = deepcopy(bankroll())
    changed["policy"]["ev_threshold"] = 1.10

    result = consolidate_model(
        protocol=protocol(),
        source=ModelSource("candidate", "prediction.json", "bankroll.json"),
        prediction=prediction(),
        bankroll=changed,
    )

    assert result["validation"]["passed"] is False
    assert result["validation"]["policy_mismatches"] == ["ev_threshold"]


def test_missing_day_blocks_comparison() -> None:
    changed = deepcopy(bankroll())
    changed["daily"] = changed["daily"][:1]

    result = consolidate_model(
        protocol=protocol(),
        source=ModelSource("candidate", "prediction.json", "bankroll.json"),
        prediction=prediction(),
        bankroll=changed,
    )

    assert result["validation"]["passed"] is False
    assert result["validation"]["daily_date_count"] == 1


def test_protocol_excludes_partial_current_jst_day() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE races (race_id TEXT, race_date TEXT, jcd TEXT, rno INTEGER);
        CREATE TABLE entries (race_id TEXT, lane INTEGER);
        CREATE TABLE race_results (race_id TEXT, lane INTEGER, rank INTEGER);
        CREATE TABLE payouts (race_id TEXT, bet_type TEXT, payout_yen INTEGER);
        """
    )
    for race_id, race_date in (
        ("previous", "2026-07-19"),
        ("current", "2026-07-20"),
    ):
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
        conn.execute("INSERT INTO payouts VALUES (?, '3連単', 1000)", (race_id,))

    result = build_protocol(conn, days=1, as_of_date=date(2026, 7, 20))

    assert result["holdout_start"] == "2026-07-19"
    assert result["holdout_end"] == "2026-07-19"
    assert result["prediction_races"] == 1
    assert result["full_day_boundary"] is True


def test_validate_model_source_reads_and_checks_the_pair(tmp_path) -> None:
    source = ModelSource("candidate", "prediction.json", "bankroll.json")
    (tmp_path / source.prediction_file).write_text(
        json.dumps(prediction()),
        encoding="utf-8",
    )
    (tmp_path / source.bankroll_file).write_text(
        json.dumps(bankroll()),
        encoding="utf-8",
    )

    result = validate_model_source(
        protocol=protocol(),
        source=source,
        raw_dir=tmp_path,
    )

    assert result["model_id"] == "candidate"
    assert result["validation"]["passed"] is True
