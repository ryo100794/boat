import json
import sqlite3

from boatrace_ai.web.dashboard import (
    MODEL_REPORT_HTML,
    _database_evaluation_artifacts,
    _database_evaluation_status,
    _remote_evaluation_job_summaries,
)


def test_remote_job_summary_reports_fold_progress_and_metrics() -> None:
    remote = {
        "jobs": [
            {
                "name": "kelly-sweep",
                "milestone": "M6",
                "kind": "bankroll_norm",
                "status": "実行中",
                "running": True,
                "process": {"elapsed": "00:10:00", "cmd": "runner --folds 5 --epochs 1"},
                "log_tail": [
                    '{"fold": 1, "evaluated_races": 100}',
                    '{"fold": 4, "evaluated_races": 400}',
                ],
                "result": None,
            },
            {
                "name": "baseline",
                "milestone": "M4",
                "kind": "backtest",
                "status": "完了",
                "running": False,
                "process": None,
                "result": {
                    "metrics": {
                        "roi": 0.91,
                        "profit_yen": -900,
                        "evaluated_races": 1000,
                    }
                },
            },
        ]
    }

    rows = _remote_evaluation_job_summaries(remote)

    assert rows[0]["completed_folds"] == 4
    assert rows[0]["expected_folds"] == 5
    assert rows[0]["elapsed"] == "00:10:00"
    assert rows[1]["roi"] == 0.91
    assert rows[1]["profit_yen"] == -900


def test_model_report_contains_live_evaluation_table() -> None:
    assert 'id="evaluationRows"' in MODEL_REPORT_HTML
    assert 'id="candidateRows"' in MODEL_REPORT_HTML
    assert "基準1着" in MODEL_REPORT_HTML
    assert "evaluation_jobs" in MODEL_REPORT_HTML


def test_database_evaluation_status_exposes_paired_payout_comparison(tmp_path) -> None:
    db_path = tmp_path / "queue.sqlite"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE model_evaluation_jobs (
          job_id INTEGER PRIMARY KEY, task_type TEXT, category TEXT,
          model_key TEXT, status TEXT, attempt INTEGER, max_attempts INTEGER,
          started_at TEXT, completed_at TEXT, decision TEXT,
          result_summary TEXT, result_path TEXT, error TEXT
        );
        CREATE TABLE model_improvement_candidates (
          job_id INTEGER PRIMARY KEY, metrics TEXT, parameters TEXT,
          created_at TEXT
        );
        """
    )
    metrics = {
        "roi": 0.94,
        "profit_yen": -600,
        "trifecta_log_loss": 3.79,
        "trifecta_top5_hit_rate": 0.35,
        "payout_feature_candidate_schema": "interactions_v2",
        "payout_feature_legacy_schema": "additive_v1",
        "payout_feature_candidate_roi": 1.03,
        "payout_feature_legacy_roi": 0.90,
        "payout_feature_roi_delta": 0.13,
        "payout_feature_roi_delta_ci95_lower": 0.02,
        "payout_feature_roi_delta_ci95_upper": 0.24,
        "payout_feature_probability_roi_delta_above_zero": 0.99,
    }
    conn.execute(
        "INSERT INTO model_evaluation_jobs VALUES (273, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "venue_conditional_order", "evaluation", "venue-v1", "completed",
            2, 2, "2026-07-23T00:00:00+00:00", "2026-07-23T01:00:00+00:00",
            "confirm_on_new_holdout", json.dumps(metrics), "result.json", None,
        ),
    )
    conn.execute(
        "INSERT INTO model_improvement_candidates VALUES (?, ?, ?, ?)",
        (273, json.dumps(metrics), "{}", "2026-07-23T01:00:00+00:00"),
    )
    conn.commit()
    conn.close()

    status = _database_evaluation_status(db_path)

    assert status["jobs"][0]["status"] == "完了"
    assert status["candidates"][0]["payout_feature_candidate_roi"] == 1.03
    assert status["candidates"][0]["payout_feature_roi_delta_ci95_lower"] == 0.02


def test_database_evaluation_artifact_exposes_daily_and_payout_walk_forward(
    tmp_path,
) -> None:
    model_dir = tmp_path / "models"
    result_path = model_dir / "evaluation_queue" / "job-00000932.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "model": "calibrated_mlp_recency_selected",
                "generated_at": "2026-07-24T00:00:00+00:00",
                "entry_log_loss": 0.32,
                "entry_brier": 0.09,
                "winner_top1_accuracy": 0.57,
                "trifecta_top5_hit_rate": 0.31,
                "evaluated_races": 100,
                "bankroll": {
                    "roi": 0.8,
                    "profit_yen": -200,
                    "stake_yen": 1000,
                    "return_yen": 800,
                },
                "daily": [
                    {
                        "race_date": "2026-07-23",
                        "stake_yen": 1000,
                        "return_yen": 800,
                    }
                ],
                "conditional_payout_walk_forward": {
                    "bankroll": {
                        "roi": 1.2,
                        "profit_yen": 200,
                        "stake_yen": 1000,
                        "return_yen": 1200,
                        "daily": [
                            {
                                "race_date": "2026-07-23",
                                "stake_yen": 1000,
                                "return_yen": 1200,
                            }
                        ],
                    },
                    "bankroll_confidence": {
                        "roi_ci95_lower": 1.01,
                        "roi_ci95_upper": 1.4,
                        "roi_delta_ci95_lower": 0.1,
                        "roi_delta_ci95_upper": 0.5,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    queue_status = {
        "candidates": [
            {
                "model_key": "calibrated_mlp_recency_selected",
                "result_path": str(result_path),
            }
        ]
    }

    backtests, bankroll, daily = _database_evaluation_artifacts(
        queue_status,
        model_dir,
    )

    assert [row["name"] for row in backtests] == [
        "calibrated_mlp_recency_selected"
    ]
    assert [row["name"] for row in bankroll] == [
        "calibrated_mlp_recency_selected",
        "calibrated_mlp_recency_selected_conditional_payout_walk_forward",
    ]
    assert daily["calibrated_mlp_recency_selected"][0]["roi_delta"] == -0.2
    assert daily[
        "calibrated_mlp_recency_selected_conditional_payout_walk_forward"
    ][0]["roi_delta"] == 0.2


def test_database_evaluation_artifact_rejects_paths_outside_model_dir(
    tmp_path,
) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")

    assert _database_evaluation_artifacts(
        {"candidates": [{"model_key": "outside", "result_path": str(outside)}]},
        tmp_path / "models",
    ) == ([], [], {})
