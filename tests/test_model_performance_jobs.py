from boatrace_ai.web.dashboard import MODEL_REPORT_HTML, _remote_evaluation_job_summaries


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
    assert "基準1着" in MODEL_REPORT_HTML
    assert "evaluation_jobs" in MODEL_REPORT_HTML
