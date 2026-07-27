from boatrace_ai.evaluation_queue import summarize_result


def test_closing_odds_forecast_metrics_are_persisted_in_db_summary() -> None:
    payload = {
        "roi": 0.8,
        "closing_odds_forecast": {
            "status": "evaluated",
            "closing_odds_log_mae": 0.12,
            "baseline_closing_odds_log_mae": 0.20,
            "closing_odds_rank_correlation": 0.91,
            "closing_odds_interval_coverage": 0.79,
            "closing_snapshot_age_seconds": 19.9,
            "closing_snapshot_age_seconds_p90": 42.0,
        },
    }

    summary = summarize_result(payload)

    assert summary["roi"] == 0.8
    assert summary["closing_odds_log_mae"] == 0.12
    assert summary["baseline_closing_odds_log_mae"] == 0.20
    assert summary["closing_odds_rank_correlation"] == 0.91
    assert summary["closing_odds_interval_coverage"] == 0.79
    assert summary["closing_snapshot_age_seconds"] == 19.9
    assert summary["closing_snapshot_age_seconds_p90"] == 42.0
