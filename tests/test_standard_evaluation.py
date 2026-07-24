from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
import json
import sqlite3

import pytest

from boatrace_ai.feature_schema import FEATURE_SCHEMA_VERSION
from boatrace_ai.standard_evaluation import (
    MODEL_SOURCES,
    POLICY,
    PROMOTION_CRITERIA,
    ModelSource,
    build_protocol,
    consolidate_model,
    evaluate_promotions,
    load_protocol,
    main,
    protocol_sha256,
    race_set_sha256,
    validate_model_source,
)


HOLDOUT_START = date(2025, 1, 3)
HOLDOUT_END = date(2026, 1, 2)


def protocol() -> dict:
    payload = {
        "protocol_id": "standard_365d_v2",
        "generated_at": "2026-01-03T00:00:00+00:00",
        "as_of_date_jst": "2026-01-03",
        "holdout_start": HOLDOUT_START.isoformat(),
        "holdout_end": HOLDOUT_END.isoformat(),
        "calendar_days": 365,
        "holdout_date_count": 365,
        "training_races": 10,
        "prediction_races": 2,
        "bankroll_evaluable_races": 2,
        "prediction_universe": "test prediction universe",
        "bankroll_universe": "test bankroll universe",
        "race_set_sha256": race_set_sha256(("race-1", "race-2")),
        "full_day_boundary": True,
        "model_selection": "test selection policy",
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
    daily = []
    cumulative_profit = 0
    for offset in range(365):
        row = {
            "race_date": (HOLDOUT_START + timedelta(days=offset)).isoformat(),
            "evaluated_races": 0,
            "candidate_tickets": 0,
            "tickets": 0,
            "races_bet": 0,
            "hit_tickets": 0,
            "hit_races": 0,
            "stake_yen": 0,
            "return_yen": 0,
            "profit_yen": 0,
            "roi": None,
        }
        if offset == 0:
            row.update(
                {
                    "evaluated_races": 1,
                    "candidate_tickets": 3,
                    "tickets": 2,
                    "races_bet": 1,
                    "stake_yen": 200,
                    "profit_yen": -200,
                    "roi": 0.0,
                }
            )
        elif offset == 364:
            row.update(
                {
                    "evaluated_races": 1,
                    "candidate_tickets": 4,
                    "tickets": 2,
                    "races_bet": 1,
                    "hit_tickets": 1,
                    "hit_races": 1,
                    "stake_yen": 200,
                    "return_yen": 500,
                    "profit_yen": 300,
                    "roi": 2.5,
                }
            )
        cumulative_profit += int(row["profit_yen"])
        row["cumulative_profit_yen"] = cumulative_profit
        daily.append(row)
    return {
        "policy": dict(POLICY),
        "evaluation_race_set_sha256": race_set_sha256(("race-1", "race-2")),
        "stake_yen": 400,
        "return_yen": 500,
        "profit_yen": 100,
        "roi": 1.25,
        "max_drawdown_yen": 200,
        "daily": daily,
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


def test_schema_dependent_source_rejects_stale_artifacts() -> None:
    source = ModelSource(
        "candidate",
        "prediction.json",
        "bankroll.json",
        requires_current_feature_schema=True,
    )
    stale = consolidate_model(
        protocol=protocol(),
        source=source,
        prediction=prediction(),
        bankroll=bankroll(),
    )
    assert stale["validation"]["passed"] is False
    assert stale["validation"]["feature_schema_mismatches"] == [
        "prediction",
        "bankroll",
    ]

    current_prediction = {
        **prediction(),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
    }
    current_bankroll = {
        **bankroll(),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
    }
    current = consolidate_model(
        protocol=protocol(),
        source=source,
        prediction=current_prediction,
        bankroll=current_bankroll,
    )
    assert current["validation"]["passed"] is True
    assert current["feature_schema_version"] == FEATURE_SCHEMA_VERSION


def test_all_feature_based_standard_sources_require_current_schema() -> None:
    requirements = {
        source.model_id: source.requires_current_feature_schema
        for source in MODEL_SOURCES
    }

    assert requirements["no_odds_v8"] is False
    assert all(
        required
        for model_id, required in requirements.items()
        if model_id != "no_odds_v8"
    )


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


def _protocol_connection(*race_dates: str) -> sqlite3.Connection:
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
    for index, race_date in enumerate(race_dates):
        race_id = f"race-{index}"
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
    return conn


def test_protocol_excludes_partial_current_jst_day() -> None:
    conn = _protocol_connection("2026-07-19", "2026-07-20")

    result = build_protocol(conn, days=1, as_of_date=date(2026, 7, 20))

    assert result["holdout_start"] == "2026-07-19"
    assert result["holdout_end"] == "2026-07-19"
    assert result["prediction_races"] == 1
    assert result["full_day_boundary"] is True


def test_protocol_refuses_to_slide_back_from_missing_previous_day() -> None:
    conn = _protocol_connection("2026-07-18")

    with pytest.raises(
        ValueError,
        match="expected 2026-07-19, got 2026-07-18",
    ):
        build_protocol(conn, days=365, as_of_date=date(2026, 7, 20))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("calendar_days", 364, "calendar_days must be 365"),
        ("holdout_date_count", 364, "holdout_date_count must be 365"),
        (
            "holdout_end",
            "2026-01-01",
            "holdout_end must be one day before",
        ),
        ("full_day_boundary", False, "full_day_boundary=true"),
    ),
)
def test_load_protocol_rejects_nonstandard_shape(
    tmp_path,
    field: str,
    value,
    message: str,
) -> None:
    changed = protocol()
    changed[field] = value
    changed["protocol_sha256"] = protocol_sha256(changed)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_protocol(path)


def test_promotion_thresholds_are_read_from_promotion_criteria(monkeypatch) -> None:
    results = []
    for source in MODEL_SOURCES:
        incumbent = source.role == "incumbent"
        results.append(
            {
                "model_id": source.model_id,
                "validation": {"passed": True},
                "roi": 1.25,
                "profit_yen": 100,
                "entry_log_loss": 0.5 if incumbent else 0.4,
                "winner_top1_accuracy": 0.4 if incumbent else 0.5,
                "trifecta_top5_hit_rate": 0.2 if incumbent else 0.3,
            }
        )
    monkeypatch.setitem(PROMOTION_CRITERIA, "minimum_roi", 1.5)

    decision = evaluate_promotions(results, comparison_ready=True)

    assert decision["criteria"]["minimum_roi"] == 1.5
    assert decision["status"] == "retain_incumbent"
    assert all(
        candidate["checks"]["roi_at_least_one"] is False
        for candidate in decision["candidates"]
    )


def test_combined_listwise_sources_use_distinct_holdout_contracts() -> None:
    sources = {source.model_id: source for source in MODEL_SOURCES}

    teacher = sources["listwise_combined_feature_teacher"]
    assert teacher.prediction_file == "listwise_combined_feature_teacher.json"
    assert teacher.bankroll_file == "listwise_combined_feature_teacher.json"
    assert teacher.prediction_section == "holdout"

    newton = sources["listwise_combined_newton"]
    assert newton.prediction_file == "listwise_combined_newton.json"
    assert newton.bankroll_file == "listwise_combined_newton.json"
    assert newton.prediction_section == "holdout_after_newton"


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


def test_artifacts_only_validation_does_not_open_database(tmp_path) -> None:
    source = MODEL_SOURCES[0]
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol()), encoding="utf-8")
    (tmp_path / source.prediction_file).write_text(
        json.dumps(prediction()),
        encoding="utf-8",
    )
    (tmp_path / source.bankroll_file).write_text(
        json.dumps(bankroll()),
        encoding="utf-8",
    )

    result = main(
        [
            "--db",
            str(tmp_path / "must-not-be-created.sqlite"),
            "--raw-dir",
            str(tmp_path),
            "--protocol-file",
            str(protocol_path),
            "--validate-source",
            source.model_id,
            "--artifacts-only",
        ]
    )

    assert result == 0
    assert not (tmp_path / "must-not-be-created.sqlite").exists()
