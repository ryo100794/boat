from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from boatrace_ai.evaluation_queue import summarize_result
from boatrace_ai.web.dashboard import (
    MODEL_REPORT_HTML,
    _database_evaluation_status,
    _odds_path_model_tracks,
    _remote_evaluation_job_summaries,
)


def market_comparison() -> dict[str, object]:
    return {
        "log_loss_difference_calibrated_minus_market": {
            "observations": 918,
            "mean_difference": -0.0583112107,
            "ci95_lower": -0.0792087605,
            "ci95_upper": -0.0373144813,
            "probability_less_than_zero": 1.0,
        },
        "top5_hit_difference_calibrated_minus_market": {
            "observations": 918,
            "mean_difference": 0.0087145969,
            "ci95_lower": -0.0076252723,
            "ci95_upper": 0.0261437908,
        },
        "day_cluster_log_loss_difference_calibrated_minus_market": {
            "observations": 918,
            "clusters": 6,
            "ci95_lower": -0.0770497442,
            "ci95_upper": -0.0397243953,
        },
        "day_cluster_top5_hit_difference_calibrated_minus_market": {
            "observations": 918,
            "clusters": 6,
            "ci95_lower": -0.0010593220,
            "ci95_upper": 0.0205850488,
        },
        "race_level_confidence_pass": False,
        "day_cluster_confidence_pass": False,
        "confidence_pass": False,
    }


def v20_payload() -> dict[str, object]:
    return {
        "model": "odds_path_observed_closing_return_schedule_quota_dual_head_v20",
        "winner_log_loss": 1.1649013962,
        "winner_top1_accuracy": 0.5620915033,
        "calibrated_trifecta_log_loss": 3.7089263912,
        "trifecta_top5_hit_rate": 0.3703703704,
        "probability_metrics": {
            "model_winner_log_loss": 1.1649013962,
            "model_winner_top1_accuracy": 0.5620915033,
            "model_trifecta_log_loss": 3.7089263912,
            "model_trifecta_top5_hit_rate": 0.3703703704,
            "market_winner_log_loss": 1.2006305169,
            "market_winner_top1_accuracy": 0.5599128540,
            "market_trifecta_log_loss": 3.7672376018,
            "market_trifecta_top5_hit_rate": 0.3616557734,
            "calibrated_winner_log_loss": 1.1649013962,
            "calibrated_winner_top1_accuracy": 0.5620915033,
            "calibrated_trifecta_log_loss": 3.7089263912,
            "calibrated_trifecta_top5_hit_rate": 0.3703703704,
        },
        "market_comparison": market_comparison(),
        "periods": {
            "outer_from": "2026-07-18",
            "outer_through": "2026-07-30",
        },
        "evaluation": {"races": 918},
        "formal_bankroll": {
            "policy": {
                "initial_bankroll_yen_per_day": 10_000,
                "allocation_api": "adaptive_discrete_log",
                "profit_reinvestment": True,
                "decision_odds": "complete_official_trifecta_snapshot_at_T-5",
            },
            "bankroll": {"evaluated_races": 918},
        },
        "promotion_gate": {
            "primary_bankroll": "chronological_bankroll",
            "sample_size_pass": False,
            "effective_hit_count_pass": False,
            "calibration_pass": True,
            "market_confidence_pass": False,
        },
        "chronological_bankroll": {
            "roi": 1.4756097561,
            "stake_yen": 8_200,
            "return_yen": 12_100,
            "profit_yen": 3_900,
            "roi_without_largest_hit": 1.2024390244,
            "daily_cluster_bootstrap_roi_lower_95": 1.1147058824,
        },
    }


def assert_v20_probability_fields(row: dict[str, object]) -> None:
    assert row["winner_log_loss"] == 1.1649013962
    assert row["winner_top1_accuracy"] == 0.5620915033
    assert row["calibrated_trifecta_log_loss"] == 3.7089263912
    assert row["trifecta_top5_hit_rate"] == 0.3703703704
    assert row["market_winner_log_loss"] == 1.2006305169
    assert row["market_trifecta_log_loss"] == 3.7672376018
    assert row["market_comparison_races"] == 918
    assert row["market_comparison_days"] == 6
    assert row["market_log_loss_delta"] == -0.0583112107
    assert row["market_log_loss_delta_ci95_lower"] == -0.0792087605
    assert row["market_log_loss_delta_ci95_upper"] == -0.0373144813
    assert row["market_improvement_probability"] == 1.0
    assert row["market_race_confidence_pass"] is False
    assert row["market_day_confidence_pass"] is False
    assert row["market_confidence_pass"] is False


def test_v20_result_summary_serializes_probability_head_and_market_confidence() -> None:
    summary = summarize_result(v20_payload())

    assert_v20_probability_fields(summary)
    assert summary["roi"] == 1.4756097561
    assert summary["market_comparison"] == market_comparison()
    assert summary["evaluation_from"] == "2026-07-18"
    assert summary["evaluation_through"] == "2026-07-30"
    assert summary["evaluated_races"] == 918
    assert summary["daily_budget_yen"] == 10_000
    assert summary["allocation_mode"] == "adaptive_discrete_log"
    assert summary["profit_reinvestment"] is True
    assert summary["odds_mode"].endswith("T-5")


def test_v20_database_api_exposes_canonical_probability_head(tmp_path: Path) -> None:
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
    summary = summarize_result(v20_payload())
    for key in (
        "sample_size_pass",
        "effective_hit_count_pass",
        "calibration_pass",
        "market_confidence_pass",
    ):
        summary.pop(key, None)
    conn.execute(
        "INSERT INTO model_evaluation_jobs VALUES "
        "(8458, 'market_residual_walk_forward', 'evaluation', 'v20', "
        "'completed', '{}', 1, 1, NULL, NULL, 'accumulate_formal_evidence', "
        "?, 'job-00008458.json', NULL)",
        (json.dumps(summary),),
    )
    conn.commit()
    conn.close()

    row = _database_evaluation_status(db_path)["jobs"][0]

    assert_v20_probability_fields(row)
    assert row["roi"] == 1.4756097561
    assert row["evaluation_from"] == "2026-07-18"
    assert row["evaluation_through"] == "2026-07-30"
    assert row["evaluated_races"] == 918
    assert row["daily_budget_yen"] == 10_000
    assert row["allocation_mode"] == "adaptive_discrete_log"
    assert row["profit_reinvestment"] is True
    assert row["odds_mode"].endswith("T-5")
    assert row["largest_hit_excluded_roi"] == 1.2024390244
    assert row["roi_ci95_lower"] == 1.1147058824
    assert row["promotion_gate_failed"] == [
        "sample_size_pass",
        "effective_hit_count_pass",
        "market_confidence_pass",
    ]
    assert row["sample_size_pass"] is False
    assert row["effective_hit_count_pass"] is False
    assert row["calibration_pass"] is True
    assert row["market_confidence_pass"] is False


def test_remote_and_existing_model_rows_share_probability_contract() -> None:
    summary = summarize_result(v20_payload())
    remote = _remote_evaluation_job_summaries({
        "jobs": [{
            "name": "v20",
            "status": "完了",
            "running": False,
            "result": {"metrics": summary},
        }]
    })[0]
    assert_v20_probability_fields(remote)

    existing = _remote_evaluation_job_summaries({
        "jobs": [{
            "name": "existing",
            "status": "完了",
            "running": False,
            "result": {"metrics": {
                "winner_log_loss": 1.20,
                "winner_top1_accuracy": 0.55,
                "calibrated_trifecta_log_loss": 3.80,
                "trifecta_top5_hit_rate": 0.34,
            }},
        }]
    })[0]
    assert existing["winner_log_loss"] == 1.20
    assert existing["winner_top1_accuracy"] == 0.55
    assert existing["calibrated_trifecta_log_loss"] == 3.80
    assert existing["trifecta_top5_hit_rate"] == 0.34
    assert existing["market_confidence_pass"] is None


def test_odds_path_track_preserves_probability_and_market_fields() -> None:
    job = {
        "db_job_id": 8458,
        "name": "odds_path_market_offset_selection_conformal_discrete_ev_v10",
        "status": "完了",
        **summarize_result(v20_payload()),
    }
    track = next(
        row
        for row in _odds_path_model_tracks([job])
        if row["id"]
        == "odds_path_market_offset_selection_conformal_discrete_ev_v10"
    )

    assert_v20_probability_fields(track)


def test_model_report_renders_probability_head_and_market_confidence() -> None:
    assert "probabilityHeadline" in MODEL_REPORT_HTML
    assert "較正3TLL" in MODEL_REPORT_HTML
    assert "市場差" in MODEL_REPORT_HTML
    assert "信頼未達" in MODEL_REPORT_HTML
