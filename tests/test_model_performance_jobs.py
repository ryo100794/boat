import json
import sqlite3
from datetime import datetime, timedelta, timezone

from boatrace_ai.web.dashboard import (
    MODEL_REPORT_HTML,
    _database_evaluation_artifacts,
    _database_evaluation_status,
    _nested_checkpoint_fold_progress,
    _remote_evaluation_job_summaries,
    genetic_evolution_report,
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


def test_nested_checkpoint_progress_counts_only_complete_valid_folds(tmp_path) -> None:
    checkpoint = (
        tmp_path / "data/models/evaluation_cache/nested_annual/job-00000077"
    )
    checkpoint.mkdir(parents=True)
    (checkpoint / "fold-01.npz").write_bytes(b"arrays")
    (checkpoint / "fold-01.json").write_text(json.dumps({
        "checkpoint_version": 1,
        "complete": True,
        "fold": 1,
        "npz_file": "fold-01.npz",
        "boundary_audit": {"passed": True},
    }), encoding="utf-8")
    (checkpoint / "fold-02.json").write_text(json.dumps({
        "checkpoint_version": 1,
        "complete": True,
        "fold": 2,
        "npz_file": "missing.npz",
        "boundary_audit": {"passed": True},
    }), encoding="utf-8")
    (checkpoint / "fold-03.json").write_text("not-json", encoding="utf-8")

    assert _nested_checkpoint_fold_progress(77, root=tmp_path) == 1


def test_model_report_contains_live_evaluation_table() -> None:
    assert 'id="evaluationRows"' in MODEL_REPORT_HTML
    assert 'id="candidateRows"' in MODEL_REPORT_HTML
    assert "基準1着" in MODEL_REPORT_HTML
    assert "evaluation_jobs" in MODEL_REPORT_HTML
    assert "<th>判定</th>" in MODEL_REPORT_HTML
    assert "<th>予測</th>" in MODEL_REPORT_HTML
    assert "<th>資金診断</th>" in MODEL_REPORT_HTML
    assert "top5_flat_roi" in MODEL_REPORT_HTML
    assert "<th>EV帯 証拠/診断</th>" in MODEL_REPORT_HTML
    assert "registered_ev_band_evaluation_days" in MODEL_REPORT_HTML
    assert "prospective_normalized_ev_evaluation_days" in MODEL_REPORT_HTML
    assert "prospective_normalized_ev_evaluation_days" in MODEL_REPORT_HTML
    assert "registeredSummary" in MODEL_REPORT_HTML
    assert "winner_log_loss" in MODEL_REPORT_HTML
    assert "候補損益" in MODEL_REPORT_HTML
    assert "候補最大DD" in MODEL_REPORT_HTML
    assert "P(ROI&gt;1)" in MODEL_REPORT_HTML
    assert 'id="geneticEvolution"' in MODEL_REPORT_HTML
    assert 'id="gaEvolutionChart"' in MODEL_REPORT_HTML
    assert "renderGeneticEvolution(jobs)" in MODEL_REPORT_HTML
    assert "drawGeneticEvolutionChart" in MODEL_REPORT_HTML
    assert "世代別fitness・アイランド内分散" in MODEL_REPORT_HTML
    assert "row.genetic_fitness==null?NaN" in MODEL_REPORT_HTML
    assert "/api/reports/genetic-evolution" in MODEL_REPORT_HTML
    assert "setInterval(refreshGeneticEvolution,10000)" in MODEL_REPORT_HTML
    assert "変異率" in MODEL_REPORT_HTML
    assert "投機fitnessは候補削減専用" in MODEL_REPORT_HTML
    assert "promotion_gate_passed" in MODEL_REPORT_HTML
    assert "gateTitle" in MODEL_REPORT_HTML
    assert "minimum_fold_roi" in MODEL_REPORT_HTML
    assert "largest_hit_excluded_roi" in MODEL_REPORT_HTML
    assert "probability_roi_above_one" in MODEL_REPORT_HTML
    assert "5F min" in MODEL_REPORT_HTML


def test_database_evaluation_status_exposes_paired_payout_comparison(tmp_path) -> None:
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
    metrics = {
        "roi": 0.94,
        "profit_yen": -600,
        "stake_yen": 10_000,
        "return_yen": 9_400,
        "max_drawdown_yen": 1_500,
        "tickets": 100,
        "hit_tickets": 8,
        "residual_purchase_policies": [
            {
                "name": "residual-top5",
                "tickets": 12,
                "hit_tickets": 2,
                "stake_yen": 1_200,
                "roi": 1.25,
            }
        ],
        "roi_without_largest_hit": 0.82,
        "trifecta_log_loss": 3.79,
        "winner_log_loss": 1.24,
        "winner_top1_accuracy": 0.53,
        "trifecta_top5_hit_rate": 0.35,
        "payout_feature_candidate_schema": "interactions_v2",
        "payout_feature_legacy_schema": "additive_v1",
        "payout_feature_candidate_roi": 1.03,
        "payout_feature_candidate_profit_yen": 300,
        "payout_feature_candidate_max_drawdown_yen": 1_200,
        "payout_feature_roi_ci95_lower": 1.01,
        "payout_feature_probability_roi_above_one": 0.96,
        "payout_feature_legacy_roi": 0.90,
        "payout_feature_roi_delta": 0.13,
        "payout_feature_roi_delta_ci95_lower": 0.02,
        "payout_feature_roi_delta_ci95_upper": 0.24,
        "payout_feature_probability_roi_delta_above_zero": 0.99,
        "promotion_gate_passed": 7,
        "promotion_gate_total": 10,
        "promotion_gate_failed": ["minimum_betting_days"],
        "holdout_temporal_minimum_roi": 0.94,
        "holdout_temporal_fold_rois": [1.10, 0.94, 1.03],
        "fold_count": 5,
        "fold_rois": [1.10, 1.04, 0.98, 1.02, 1.06],
        "minimum_fold_roi": 0.98,
        "largest_hit_excluded_roi": 1.01,
        "roi_ci95_lower": 0.97,
        "roi_ci95_upper": 1.09,
        "probability_roi_above_one": 0.91,
        "prospective_top5_narrow_ev_status": "evaluating",
        "prospective_top5_narrow_ev_evaluation_days": 2,
        "prospective_top5_narrow_ev_tickets": 112,
        "prospective_top5_narrow_ev_hit_tickets": 13,
        "prospective_top5_narrow_ev_roi": 1.298,
        "top5_narrow_retrospective_status": (
            "diagnostic_only_not_promotion_evidence"
        ),
        "top5_narrow_retrospective_evaluation_days": 8,
        "top5_narrow_retrospective_tickets": 609,
        "top5_narrow_retrospective_hit_tickets": 71,
        "top5_narrow_retrospective_roi": 1.312,
        "top5_narrow_retrospective_roi_without_largest_hit": 1.28,
    }
    conn.execute(
        "INSERT INTO model_evaluation_jobs VALUES (273, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "venue_conditional_order", "evaluation", "venue-v1", "completed",
            "{}", 2, 2, "2026-07-23T00:00:00+00:00", "2026-07-23T01:00:00+00:00",
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
    assert status["jobs"][0]["decision"] == "confirm_on_new_holdout"
    assert status["jobs"][0]["winner_log_loss"] == 1.24
    assert status["jobs"][0]["trifecta_log_loss"] == 3.79
    assert status["jobs"][0]["winner_top1_accuracy"] == 0.53
    assert status["jobs"][0]["stake_yen"] == 10_000
    assert status["jobs"][0]["return_yen"] == 9_400
    assert status["jobs"][0]["max_drawdown_yen"] == 1_500
    assert status["jobs"][0]["tickets"] == 100
    assert status["jobs"][0]["hit_tickets"] == 8
    assert status["jobs"][0]["residual_purchase_policies"] == [
        {
            "name": "residual-top5",
            "tickets": 12,
            "hit_tickets": 2,
            "stake_yen": 1_200,
            "roi": 1.25,
        }
    ]
    assert status["jobs"][0]["roi_without_largest_hit"] == 0.82
    assert status["jobs"][0]["promotion_gate_passed"] == 7
    assert status["jobs"][0]["promotion_gate_total"] == 10
    assert status["jobs"][0]["promotion_gate_failed"] == ["minimum_betting_days"]
    assert status["jobs"][0]["holdout_temporal_minimum_roi"] == 0.94
    assert status["jobs"][0]["fold_count"] == 5
    assert status["jobs"][0]["minimum_fold_roi"] == 0.98
    assert status["jobs"][0]["largest_hit_excluded_roi"] == 1.01
    assert status["jobs"][0]["roi_ci95_lower"] == 0.97
    assert status["jobs"][0]["probability_roi_above_one"] == 0.91
    assert status["jobs"][0]["prospective_top5_narrow_ev_evaluation_days"] == 2
    assert status["jobs"][0]["prospective_top5_narrow_ev_roi"] == 1.298
    assert status["jobs"][0]["top5_narrow_retrospective_evaluation_days"] == 8
    assert status["jobs"][0]["top5_narrow_retrospective_roi"] == 1.312
    assert (
        status["jobs"][0]["top5_narrow_retrospective_roi_without_largest_hit"]
        == 1.28
    )
    assert status["candidates"][0]["payout_feature_candidate_roi"] == 1.03
    assert status["candidates"][0]["payout_feature_candidate_profit_yen"] == 300
    assert status["candidates"][0]["payout_feature_candidate_max_drawdown_yen"] == 1_200
    assert status["candidates"][0]["payout_feature_roi_ci95_lower"] == 1.01
    assert status["candidates"][0]["payout_feature_probability_roi_above_one"] == 0.96
    assert status["candidates"][0]["payout_feature_roi_delta_ci95_lower"] == 0.02


def test_database_evaluation_status_includes_parent_of_recent_job(tmp_path) -> None:
    db_path = tmp_path / "queue.sqlite"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE model_evaluation_jobs (
          job_id INTEGER PRIMARY KEY, task_type TEXT, category TEXT,
          model_key TEXT, status TEXT, parameters TEXT,
          attempt INTEGER, max_attempts INTEGER,
          started_at TEXT, completed_at TEXT, decision TEXT,
          result_summary TEXT, result_path TEXT, error TEXT,
          parent_job_id INTEGER
        );
        CREATE TABLE model_improvement_candidates (
          job_id INTEGER PRIMARY KEY, metrics TEXT, parameters TEXT,
          created_at TEXT
        );
        """
    )
    rows = []
    for job_id in range(1, 103):
        parent_job_id = 1 if job_id == 102 else None
        summary = json.dumps({"roi": 0.81}) if job_id == 1 else "{}"
        rows.append(
            (
                job_id,
                "listwise_feature_search",
                "evaluation",
                f"job-{job_id}",
                "completed",
                "{}",
                1,
                2,
                None,
                "2026-07-28T00:00:00+00:00",
                None,
                summary,
                None,
                None,
                parent_job_id,
            )
        )
    conn.executemany(
        "INSERT INTO model_evaluation_jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()

    status = _database_evaluation_status(db_path)

    by_name = {row["name"]: row for row in status["jobs"]}
    assert "job-1" in by_name
    assert by_name["job-1"]["roi"] == 0.81
    assert "job-2" not in by_name


def test_database_evaluation_status_normalizes_duplicate_formal_jobs(tmp_path) -> None:
    db_path = tmp_path / "queue.sqlite"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE model_evaluation_jobs (
          job_id INTEGER PRIMARY KEY, task_type TEXT, category TEXT,
          model_key TEXT, status TEXT, parameters TEXT, priority INTEGER,
          attempt INTEGER, max_attempts INTEGER,
          started_at TEXT, completed_at TEXT, decision TEXT,
          result_summary TEXT, result_path TEXT, error TEXT,
          parent_job_id INTEGER
        );
        CREATE TABLE model_improvement_candidates (
          job_id INTEGER PRIMARY KEY, metrics TEXT, parameters TEXT,
          created_at TEXT
        );
        """
    )
    model_key = "triple_head_v21_daily:market_residual:20260718-29"
    metrics = {
        "evaluation_days": 6,
        "evaluated_races": 918,
        "roi": 1.4756097561,
        "stake_yen": 8200,
        "return_yen": 12100,
        "profit_yen": 3900,
        "winner_log_loss": 1.1649013962,
        "calibrated_trifecta_log_loss": 3.7089263912,
        "trifecta_top5_hit_rate": 0.3736383442,
        "comparison_role": "triple_head",
    }
    base_parameters = {
        "from_date": "2026-07-18",
        "through_date": "2026-07-29",
        "calibrator_strategy": "triple_head_v21",
    }
    rows = [
        (8624, base_parameters, 97, "accumulate_formal_evidence", None),
        (8666, base_parameters, 116, "accumulate_formal_evidence", 8458),
        (
            8667,
            {**base_parameters, "calibrator_strategy": "tail_diagnostic"},
            80,
            "accumulate_formal_evidence",
            None,
        ),
        (8668, base_parameters, 10, "research_only", None),
    ]
    for job_id, parameters, priority, decision, parent_job_id in rows:
        conn.execute(
            """
            INSERT INTO model_evaluation_jobs (
              job_id, task_type, category, model_key, status, parameters,
              priority, attempt, max_attempts, decision, result_summary,
              parent_job_id
            ) VALUES (?, 'market_residual_walk_forward', 'evaluation', ?,
                      'completed', ?, ?, 1, 2, ?, ?, ?)
            """,
            (
                job_id,
                model_key,
                json.dumps(parameters),
                priority,
                decision,
                json.dumps(metrics),
                parent_job_id,
            ),
        )
        conn.execute(
            "INSERT INTO model_improvement_candidates VALUES (?, ?, '{}', NULL)",
            (job_id, json.dumps(metrics)),
        )
    conn.commit()
    conn.close()

    status = _database_evaluation_status(db_path)

    assert [row["db_job_id"] for row in status["jobs"]] == [8668, 8667, 8666]
    assert status["jobs"][-1]["priority"] == 116
    assert status["jobs"][-1]["parent_job_id"] == 8458
    assert {row["job_id"] for row in status["candidates"]} == {8666, 8667, 8668}


def test_database_evaluation_status_quarantines_invalid_data_source(tmp_path) -> None:
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
    metrics = {
        "roi": 1.25,
        "profit_yen": 2500,
        "winner_top1_accuracy": 0.70,
        "data_source_validation_pass": False,
    }
    conn.execute(
        "INSERT INTO model_evaluation_jobs VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "archive_market_oracle", "evaluation", "invalid-oracle", "completed",
            "{}", 1, 1, "2026-07-28T00:00:00+00:00",
            "2026-07-28T01:00:00+00:00", "invalid_data_source",
            json.dumps(metrics), "invalid.json", None,
        ),
    )
    conn.execute(
        "INSERT INTO model_improvement_candidates VALUES (?, ?, ?, ?)",
        (1, json.dumps(metrics), "{}", "2026-07-28T01:00:00+00:00"),
    )
    conn.commit()
    conn.close()

    status = _database_evaluation_status(db_path)

    assert status["jobs"][0]["status"] == "無効"
    assert status["jobs"][0]["valid_for_comparison"] is False
    assert status["jobs"][0]["roi"] is None
    assert status["jobs"][0]["winner_top1_accuracy"] is None
    assert status["candidates"] == []


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
                "market_offset_multinomial_kelly_walk_forward": {
                    "evaluation_days": 6,
                    "evaluated_races": 918,
                    "tickets": 28,
                    "hit_tickets": 3,
                    "stake_yen": 3000,
                    "return_yen": 3130,
                    "profit_yen": 130,
                    "roi": 1.0433333333333332,
                    "daily": [{
                        "race_date": "2026-07-23",
                        "stake_yen": 3000,
                        "return_yen": 3130,
                    }],
                },
                "conservative_market_offset_kelly_walk_forward": {
                    "status": "waiting_for_first_unseen_day",
                    "registered_after": "2026-07-28",
                    "evaluation_days": 0,
                    "evaluated_races": 0,
                    "tickets": 0,
                    "stake_yen": 0,
                    "return_yen": 0,
                    "profit_yen": 0,
                    "roi": 0.0,
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
        "calibrated_mlp_recency_selected_market_offset_multinomial_kelly_walk_forward",
        "calibrated_mlp_recency_selected_conservative_market_offset_kelly_walk_forward",
    ]
    assert daily["calibrated_mlp_recency_selected"][0]["roi_delta"] == -0.2
    assert daily[
        "calibrated_mlp_recency_selected_conditional_payout_walk_forward"
    ][0]["roi_delta"] == 0.2
    assert daily[
        "calibrated_mlp_recency_selected_market_offset_multinomial_kelly_walk_forward"
    ][0]["roi_delta"] == 0.043333333333333335


def test_database_evaluation_artifact_rejects_paths_outside_model_dir(
    tmp_path,
) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")

    assert _database_evaluation_artifacts(
        {"candidates": [{"model_key": "outside", "result_path": str(outside)}]},
        tmp_path / "models",
    ) == ([], [], {})


def test_database_evaluation_artifact_prioritizes_bankroll_results(
    tmp_path,
) -> None:
    model_dir = tmp_path / "models"
    queue_dir = model_dir / "evaluation_queue"
    queue_dir.mkdir(parents=True)
    search_path = queue_dir / "job-00000001.json"
    search_path.write_text(
        json.dumps({"entry_log_loss": 0.34, "evaluated_races": 100}),
        encoding="utf-8",
    )
    bankroll_path = queue_dir / "job-00000002.json"
    bankroll_path.write_text(
        json.dumps(
            {
                "entry_log_loss": 0.32,
                "evaluated_races": 100,
                "bankroll": {"roi": 0.8, "stake_yen": 1000},
                "conditional_payout_walk_forward": {
                    "bankroll": {
                        "roi": 0.0,
                        "stake_yen": 0,
                        "policy": {
                            "no_bet": True,
                            "no_bet_reason": "selection_gate_no_bet",
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    queue_status = {
        "candidates": [
            {
                "model_key": "search-only",
                "result_path": str(search_path),
                "roi": None,
            },
            {
                "model_key": "bankroll-model",
                "result_path": str(bankroll_path),
                "roi": 0.8,
                "payout_feature_candidate_roi": 0.0,
            },
        ]
    }

    _, bankroll, _ = _database_evaluation_artifacts(
        queue_status,
        model_dir,
        maximum_artifacts=1,
    )

    assert [row["name"] for row in bankroll] == [
        "bankroll-model",
        "bankroll-model_conditional_payout_walk_forward",
    ]
    assert bankroll[1]["no_bet"] is True
    assert bankroll[1]["no_bet_reason"] == "selection_gate_no_bet"


def test_database_evaluation_status_uses_current_attempt_elapsed(tmp_path) -> None:
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
        CREATE TABLE model_evaluation_job_runs (
          job_id INTEGER, attempt INTEGER, status TEXT, started_at TEXT
        );
        """
    )
    original_started = datetime.now(timezone.utc) - timedelta(hours=8)
    current_started = datetime.now(timezone.utc) - timedelta(minutes=5)
    conn.execute(
        "INSERT INTO model_evaluation_jobs VALUES (3564, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "listwise_feature_search",
            "evaluation",
            "retrying-search",
            "running",
            "{}",
            4,
            4,
            original_started.isoformat(),
            None,
            None,
            "{}",
            None,
            None,
        ),
    )
    conn.execute(
        "INSERT INTO model_evaluation_job_runs VALUES (?, ?, ?, ?)",
        (3564, 4, "running", current_started.isoformat()),
    )
    conn.commit()
    conn.close()

    row = _database_evaluation_status(db_path)["jobs"][0]

    assert row["running"] is True
    assert row["elapsed"].startswith("00:")
