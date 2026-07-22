from __future__ import annotations

import json

from boatrace_ai.web import dashboard


def _artifact(*, bankroll_pass: bool = False) -> dict:
    return {
        "generated_at": "2026-07-22T13:30:00+00:00",
        "model": "pastlog_conditional_order",
        "conditional_order": {
            "evaluated_races": 48_437,
            "trifecta_log_loss": 3.8099,
            "trifecta_top1_hit_rate": 0.0983,
            "trifecta_top5_hit_rate": 0.3447,
        },
        "listwise_baseline": {
            "evaluated_races": 48_437,
            "trifecta_log_loss": 3.9021,
            "trifecta_top5_hit_rate": 0.3293,
        },
        "bankroll": {
            "evaluated_races": 48_437,
            "roi": 1.02 if bankroll_pass else 0.91,
            "profit_yen": 2_000 if bankroll_pass else -9_000,
            "stake_yen": 100_000,
            "return_yen": 102_000 if bankroll_pass else 91_000,
            "selected_tickets": 120,
            "daily": [
                {
                    "race_date": "2025-07-20",
                    "evaluated_races": 130,
                    "tickets": 2,
                    "stake_yen": 600,
                    "return_yen": 0,
                    "profit_yen": -600,
                    "cumulative_profit_yen": -600,
                    "roi": 0.0,
                    "budget_used_fraction": 0.06,
                }
            ],
        },
        "bankroll_confidence": {
            "roi_ci95_lower": 0.87,
            "roi_ci95_upper": 1.08,
            "roi_delta_ci95_lower": -0.06,
            "roi_delta_ci95_upper": 0.09,
        },
        "structure_gate": {"pass": True},
        "bankroll_gate": {"pass": bankroll_pass},
        "promotion_gate": {
            "structure_pass": True,
            "bankroll_pass": bankroll_pass,
            "pass": bankroll_pass,
        },
        "promotion_eligible": bankroll_pass,
    }


def test_conditional_order_track_keeps_revenue_gate_open(tmp_path) -> None:
    path = tmp_path / "conditional_order_365d.json"
    path.write_text(json.dumps(_artifact()), encoding="utf-8")

    tracks = {
        row["id"]: row
        for row in dashboard._model_track_summaries(tmp_path, [], {"jobs": []})
    }
    candidate = tracks["conditional_order_365d"]

    assert candidate["status"] == "要改善/収益ゲート未達"
    assert candidate["entry_log_loss"] == 3.8099
    assert candidate["trifecta_top5_hit_rate"] == 0.3447
    assert candidate["roi"] == 0.91
    assert candidate["profit_yen"] == -9_000
    assert candidate["promotion_eligible"] is False
    assert candidate["backtest_available"] is True
    assert "条件付き遷移108係数" in candidate["training"]


def test_model_report_separates_conditional_metrics_and_bankroll(tmp_path) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    path = model_dir / "conditional_order_365d.json"
    path.write_text(json.dumps(_artifact()), encoding="utf-8")
    dashboard._MODEL_REPORT_CACHE.clear()

    report = dashboard.model_performance_report(
        tmp_path / "boatrace.sqlite",
        {"model_dir": [str(model_dir)]},
    )

    backtest = next(
        row for row in report["backtests"]
        if row["name"] == "pastlog_conditional_order"
    )
    bankroll = next(
        row for row in report["bankroll"]
        if row["name"] == "pastlog_conditional_order"
    )
    assert backtest["entry_log_loss"] == 3.8099
    assert backtest["trifecta_top5_hit_rate"] == 0.3447
    assert bankroll["entry_log_loss"] == 3.8099
    assert bankroll["roi"] == 0.91
    assert bankroll["profit_yen"] == -9_000
    assert bankroll["roi_ci95_lower"] == 0.87
    assert bankroll["roi_ci95_upper"] == 1.08
    assert bankroll["roi_delta_ci95_lower"] == -0.06
    assert len(report["bankroll_daily"]["pastlog_conditional_order"]) == 1


def test_local_result_does_not_treat_structure_only_as_promotion(tmp_path) -> None:
    path = tmp_path / "conditional_order_365d.json"
    path.write_text(json.dumps(_artifact()), encoding="utf-8")

    result = dashboard._local_evaluation_result(path)

    assert result is not None
    assert result["metrics"]["trifecta_log_loss"] == 3.8099
    assert result["metrics"]["roi"] == 0.91
    assert result["structure_gate"]["pass"] is True
    assert result["bankroll_gate"]["pass"] is False
    assert result["promotion_eligible"] is False
