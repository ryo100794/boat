from boatrace_ai.evaluation_queue import summarize_result


def test_empirical_lcb_track_is_summarized_separately_from_legacy_bankroll():
    payload = {
        "roi": 0.83,
        "profit_yen": -670,
        "empirical_lcb_walk_forward": {
            "status": "evaluating",
            "evaluation_days": 61,
            "evaluated_races": 7_500,
            "calibration_ready_folds": 31,
            "minimum_ready_evaluation_days": 30,
            "minimum_tickets": 300,
            "sample_size_pass": True,
            "eligible_days": 24,
            "no_bet_days": 37,
            "profitable_days": 14,
            "tickets": 320,
            "hit_tickets": 25,
            "stake_yen": 32_000,
            "return_yen": 36_000,
            "profit_yen": 4_000,
            "roi": 1.125,
            "roi_without_largest_hit": 1.03,
            "largest_hit_return_share": 0.18,
            "max_drawdown_yen": 5_000,
            "tail_portfolio_diagnostics": {
                "normal": {
                    "daily_cluster_bootstrap_roi_lower_95": 1.01,
                },
                "tail": {
                    "daily_cluster_bootstrap_roi_lower_95": None,
                },
            },
        },
    }

    summary = summarize_result(payload)

    assert summary["roi"] == 0.83
    assert summary["profit_yen"] == -670
    assert summary["empirical_lcb_roi"] == 1.125
    assert summary["empirical_lcb_profit_yen"] == 4_000
    assert summary["empirical_lcb_sample_size_pass"] is True
    assert summary["empirical_lcb_roi_lower95"] == 1.01
    assert summary["empirical_lcb_tail_portfolio_diagnostics"] == (
        payload["empirical_lcb_walk_forward"]["tail_portfolio_diagnostics"]
    )


def test_empirical_lcb_summary_tolerates_no_purchase_track():
    summary = summarize_result(
        {
            "empirical_lcb_walk_forward": {
                "status": "calibration_not_ready",
                "evaluation_days": 6,
                "calibration_ready_folds": 0,
                "sample_size_pass": False,
                "tickets": 0,
                "stake_yen": 0,
                "return_yen": 0,
                "profit_yen": 0,
                "roi": None,
            }
        }
    )

    assert summary["empirical_lcb_status"] == "calibration_not_ready"
    assert summary["empirical_lcb_tickets"] == 0
    assert "empirical_lcb_roi" not in summary
