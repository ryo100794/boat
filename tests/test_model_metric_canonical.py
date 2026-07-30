from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from boatrace_ai.evaluation_queue import summarize_result
from boatrace_ai.evaluation_result_summary import canonicalize_primary_bankroll
from boatrace_ai.web.dashboard import (
    MODEL_REPORT_HTML,
    _bankroll_summary,
    _database_evaluation_status,
    _remote_evaluation_job_summaries,
)


def chronological_payload() -> dict[str, object]:
    return {
        "roi": 0.89,
        "stake_yen": 26_400,
        "return_yen": 23_550,
        "profit_yen": -2_850,
        "max_drawdown_yen": 5_100,
        "roi_without_largest_hit": 0.81,
        "daily_cluster_bootstrap_roi_lower_95": 0.68,
        "largest_hit_excluded_roi": 0.80,
        "roi_ci95_lower": 0.67,
        "probability_roi_above_one": 0.22,
        "promotion_gate": {
            "primary_bankroll": "chronological_bankroll",
            "positive_profit_pass": True,
        },
        "chronological_bankroll": {
            "stake_yen": 8_200,
            "return_yen": 12_100,
            "profit_yen": 3_900,
            "roi": 1.475609756,
            "max_drawdown_yen": 270,
            "tickets": 64,
            "hit_tickets": 8,
            "roi_without_largest_hit": 1.202439024,
            "daily_cluster_bootstrap_roi_lower_95": 1.114705882,
            "effective_hit_count": 7.0999,
            "bootstrap_probability_roi_above_one": 0.97,
        },
    }


def test_result_summary_uses_chronological_headlines_and_labels_legacy_batch() -> None:
    summary = summarize_result(chronological_payload())

    assert summary["primary_bankroll"] == "chronological"
    assert summary["roi"] == 1.475609756
    assert summary["stake_yen"] == 8_200
    assert summary["return_yen"] == 12_100
    assert summary["profit_yen"] == 3_900
    assert summary["max_drawdown_yen"] == 270
    assert summary["roi_without_largest_hit"] == 1.202439024
    assert summary["daily_cluster_bootstrap_roi_lower_95"] == 1.114705882
    assert summary["largest_hit_excluded_roi"] == 1.202439024
    assert summary["roi_ci95_lower"] == 1.114705882
    assert summary["probability_roi_above_one"] == 0.97
    assert summary["legacy_batch_roi"] == 0.89
    assert summary["legacy_batch_stake_yen"] == 26_400
    assert summary["legacy_batch_return_yen"] == 23_550
    assert summary["legacy_batch_profit_yen"] == -2_850
    assert summary["legacy_batch_max_drawdown_yen"] == 5_100
    assert summary["legacy_batch_largest_hit_excluded_roi"] == 0.80
    assert summary["legacy_batch_roi_ci95_lower"] == 0.67
    assert summary["legacy_batch_probability_roi_above_one"] == 0.22
    assert summary["legacy_batch_bankroll"]["roi"] == 0.89


def test_primary_bankroll_normalization_is_idempotent() -> None:
    first = canonicalize_primary_bankroll(chronological_payload())
    second = canonicalize_primary_bankroll(first)

    assert second == first
    assert second["legacy_batch_bankroll"]["roi"] == 0.89
    assert second["roi"] == 1.475609756


def test_database_model_performance_recovers_old_flattened_summary(tmp_path: Path) -> None:
    db_path = tmp_path / "queue.sqlite"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE model_evaluation_jobs (
          job_id INTEGER PRIMARY KEY, task_type TEXT, category TEXT,
          model_key TEXT, status TEXT, parameters TEXT,
          attempt INTEGER, max_attempts INTEGER,
          started_at TEXT, completed_at TEXT, decision TEXT,
          result_summary TEXT, result_path TEXT, error TEXT
        );
        CREATE TABLE model_improvement_candidates (
          job_id INTEGER PRIMARY KEY, metrics TEXT, parameters TEXT,
          created_at TEXT
        );
        """
    )
    stored = {
        "primary_bankroll": "chronological",
        "roi": 0.89,
        "stake_yen": 26_400,
        "return_yen": 23_550,
        "profit_yen": -2_850,
        "max_drawdown_yen": 5_100,
        "chronological_roi": 1.475609756,
        "chronological_stake_yen": 8_200,
        "chronological_return_yen": 12_100,
        "chronological_profit_yen": 3_900,
        "chronological_max_drawdown_yen": 270,
    }
    conn.execute(
        "INSERT INTO model_evaluation_jobs VALUES "
        "(8191, 'market_residual_walk_forward', 'evaluation', 'v18', "
        "'completed', '{}', 1, 1, NULL, NULL, 'accumulate_formal_evidence', "
        "?, 'v18.json', NULL)",
        (json.dumps(stored),),
    )
    conn.commit()
    conn.close()

    job = _database_evaluation_status(db_path)["jobs"][0]

    assert job["primary_bankroll"] == "chronological"
    assert job["roi"] == 1.475609756
    assert job["stake_yen"] == 8_200
    assert job["return_yen"] == 12_100
    assert job["profit_yen"] == 3_900
    assert job["max_drawdown_yen"] == 270
    assert job["legacy_batch_roi"] == 0.89
    assert job["legacy_batch_stake_yen"] == 26_400
    assert job["legacy_batch_return_yen"] == 23_550
    assert job["legacy_batch_profit_yen"] == -2_850
    assert job["legacy_batch_max_drawdown_yen"] == 5_100


def test_remote_model_performance_uses_chronological_primary() -> None:
    payload = chronological_payload()
    payload["primary_bankroll"] = "chronological"
    payload.update({f"chronological_{key}": value for key, value in payload.pop("chronological_bankroll").items()})
    rows = _remote_evaluation_job_summaries({
        "jobs": [{
            "name": "v18",
            "status": "完了",
            "running": False,
            "result": {"metrics": payload},
        }]
    })

    row = rows[0]
    assert row["primary_bankroll"] == "chronological"
    assert row["roi"] == 1.475609756
    assert row["stake_yen"] == 8_200
    assert row["return_yen"] == 12_100
    assert row["profit_yen"] == 3_900
    assert row["max_drawdown_yen"] == 270
    assert row["legacy_batch_roi"] == 0.89


def test_bankroll_report_row_uses_chronological_primary(tmp_path: Path) -> None:
    row = _bankroll_summary(tmp_path / "v18.json", "v18", chronological_payload())

    assert row["primary_bankroll"] == "chronological"
    assert row["roi"] == 1.475609756
    assert row["stake_yen"] == 8_200
    assert row["return_yen"] == 12_100
    assert row["profit_yen"] == 3_900
    assert row["max_drawdown_yen"] == 270
    assert row["legacy_batch_bankroll"]["roi"] == 0.89


def test_model_performance_labels_primary_and_legacy_bankrolls() -> None:
    assert "時系列 primary" in MODEL_REPORT_HTML
    assert "旧batch診断 ROI" in MODEL_REPORT_HTML
    assert "bankrollSourceDiagnostic" in MODEL_REPORT_HTML
