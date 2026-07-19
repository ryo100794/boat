from __future__ import annotations

from pathlib import Path

from boatrace_ai.web.dashboard import (
    _evaluation_scope,
    _fold_report_row,
    _quality_gates,
    _remote_backtest_report_summaries,
)


def test_standardized_result_is_labeled_separately() -> None:
    assert (
        _evaluation_scope(
            Path("standardized_365d_no_odds_v8_bankroll.json"),
            [],
        )
        == "standard_365d"
    )


def test_legacy_daily_range_is_explicit() -> None:
    assert _evaluation_scope(
        Path("listwise_newton_cg_v1.json"),
        [
            {"race_date": "2026-05-10"},
            {"race_date": "2026-07-18"},
        ],
    ) == "legacy:2026-05-10:2026-07-18:2"


def test_fold_row_keeps_evaluation_scope() -> None:
    row = _fold_report_row(
        "model",
        {"fold": 1, "train_races": 100, "test_races": 50},
        "standard_365d",
    )
    assert row["evaluation_scope"] == "standard_365d"


def test_standardized_remote_backtest_is_exposed_with_scope() -> None:
    rows = _remote_backtest_report_summaries(
        {
            "generated_at": "2026-07-19T18:00:00+00:00",
            "jobs": [
                {
                    "kind": "standardized_365d_backtest",
                    "name": "standardized_365d_no_odds_v8_backtest",
                    "result": {
                        "file": "standardized_365d_no_odds_v8_backtest.json",
                        "feature_set": "no_odds_v8",
                        "include_odds": False,
                        "metrics": {
                            "evaluated_races": 48294,
                            "entry_log_loss": 0.32514,
                            "winner_top1_accuracy": 0.56744,
                        },
                    },
                }
            ],
        }
    )

    assert len(rows) == 1
    assert rows[0]["evaluation_scope"] == "standard_365d"
    assert rows[0]["include_odds"] is False
    assert rows[0]["evaluated_races"] == 48294


def test_quality_gate_prefers_standardized_bankroll_over_legacy(tmp_path) -> None:
    remote = {
        "jobs": [
            {
                "kind": "bankroll_norm",
                "status": "完了",
                "result": {
                    "file": "legacy_high_roi.json",
                    "metrics": {"roi": 0.99, "profit_yen": -100},
                },
            },
            {
                "kind": "standardized_365d_bankroll",
                "status": "完了",
                "result": {
                    "file": "standardized_365d_pastlog_v7_bankroll.json",
                    "metrics": {"roi": 0.8808429, "profit_yen": -232130},
                },
            },
        ]
    }

    roi_gate = next(
        row for row in _quality_gates(tmp_path, remote) if row["target"] == "M6 ROI"
    )
    assert "ROI=0.881" in roi_gate["evidence"]
    assert "ROI=0.990" not in roi_gate["evidence"]
