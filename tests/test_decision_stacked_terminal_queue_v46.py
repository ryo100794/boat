from __future__ import annotations

from datetime import date

from boatrace_ai.evaluation_queue import (
    TASK_PROFILES,
    build_command,
    result_decision,
    summarize_result,
)
from boatrace_ai.listwise import decision_v38_empirical_lcb as lcb_subject


def _job(parameters: dict) -> dict:
    return {
        "job_id": 7,
        "task_type": "decision_stacked_terminal_market_v46",
        "parameters": parameters,
    }


def _race() -> dict:
    return {
        "race_id": "2026-08-06-01-01",
        "race_date": date(2026, 8, 6).isoformat(),
        "jcd": "01",
        "rno": 1,
        "odds": {"1-2-3": 3.0, "1-3-2": 4.0},
        "model_probabilities": {"1-2-3": 0.55, "1-3-2": 0.45},
        "market_probabilities": {"1-2-3": 0.6, "1-3-2": 0.4},
    }


def test_v46_command_is_cache_bounded(tmp_path) -> None:
    root = tmp_path / "boat"
    cache = (
        root
        / "data/models/evaluation_cache/market_scored/source.races.joblib"
    )
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"cache")
    command, output = build_command(
        _job({
            "scored_cache": (
                "data/models/evaluation_cache/market_scored/"
                "source.races.joblib"
            ),
            "calibration_through": "2026-08-18",
            "minimum_training_days": 30,
            "minimum_training_races": 3000,
            "num_threads": 4,
            "timeout_seconds": 14400,
        }),
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )
    assert command[1:3] == [
        "-m",
        "boatrace_ai.listwise.decision_stacked_terminal_market_v46",
    ]
    assert command[command.index("--scored-cache") + 1] == str(cache)
    assert output == root / "data/models/evaluation_queue/job-00000007.json"
    assert TASK_PROFILES["decision_stacked_terminal_market_v46"]["max_parallel"] == 1


def test_v46_lcb_scoring_uses_source_probability_for_price_then_v44_for_gate(
    monkeypatch,
) -> None:
    race = _race()
    observed = {}

    def fake_price(item, model):
        observed["price_probability"] = dict(item["model_probabilities"])
        observed["price_model"] = model
        return {
            "q10": {"1-2-3": 2.0, "1-3-2": 3.0},
            "q50": {"1-2-3": 2.5, "1-3-2": 3.5},
            "q90": {"1-2-3": 3.0, "1-3-2": 4.0},
        }

    monkeypatch.setattr(
        lcb_subject, "forecast_closing_odds_quantiles", fake_price
    )
    monkeypatch.setattr(
        lcb_subject,
        "stacked_probabilities",
        lambda item, artifact: {"1-2-3": 0.7, "1-3-2": 0.3},
    )
    frozen = {
        "model": "decision_time_stacked_terminal_market_v46",
        "training_status": "ready",
        "official_closing_fields_used": False,
        "artifact": {
            "probability_artifact": {"artifact_sha256": "a" * 64},
            "closing_odds_model": {"model_type": "test-price"},
        },
    }
    scored = lcb_subject.score_frozen_v38_races([race], frozen)
    assert observed["price_probability"] == race["model_probabilities"]
    assert observed["price_model"] == {"model_type": "test-price"}
    assert scored[0]["model_probabilities"] == {
        "1-2-3": 0.7,
        "1-3-2": 0.3,
    }
    assert scored[0]["estimated_final_odds"] == {
        "1-2-3": 2.5,
        "1-3-2": 3.5,
    }
    assert scored[0]["closing_odds_forecast_target"] == "conditional_median"


def test_v46_summary_exposes_terminal_price_and_keeps_purchase_disabled() -> None:
    payload = {
        "model": "decision_time_stacked_terminal_market_v46",
        "training_status": "ready",
        "training_days": 10,
        "training_races": 1447,
        "minimum_training_days": 10,
        "minimum_training_races": 1000,
        "official_closing_fields_used": False,
        "selected_stack": "linear50_nonlinear50",
        "selected_weights": {"market": 0.0, "linear": 0.5, "nonlinear": 0.5},
        "holdout_metrics": {
            "evaluated_days": 7,
            "evaluated_races": 979,
            "days_better_than_market": 7,
            "log_loss_delta_vs_market": -0.05,
            "trifecta_top5_hit_rate": 0.38,
            "market_trifecta_top5_hit_rate": 0.37,
        },
        "artifact": {
            "artifact_sha256": "b" * 64,
            "closing_odds_model_type": (
                "ridge_log_location_odds_path_context_v3"
            ),
        },
        "closing_odds_holdout_metrics": {
            "closing_odds_log_mae": 0.375,
            "baseline_closing_odds_log_mae": 0.514,
            "closing_odds_rank_correlation": 0.949,
        },
        "terminal_value_candidate_diagnostic": {
            "candidate_price_target": "conditional_median",
            "mean_median_forecast_ev": 1.019,
            "mean_oracle_closing_ev": 1.051,
            "realized_roi": 0.780,
            "day_block_roi_lcb95": 0.437,
            "purchase_gate": (
                "disabled_pending_strict_prior_realized_roi_lcb"
            ),
        },
    }
    summary = summarize_result(payload)
    assert summary["closing_odds_log_mae"] == 0.375
    assert summary["baseline_closing_odds_log_mae"] == 0.514
    assert summary["terminal_value_mean_oracle_ev"] == 1.051
    assert summary["terminal_value_realized_roi"] == 0.780
    assert summary["terminal_value_purchase_gate"] == (
        "disabled_pending_strict_prior_realized_roi_lcb"
    )
    assert summary["challenger_selection_gate_pass"] is False
    assert result_decision(
        "decision_stacked_terminal_market_v46", summary
    ) == "probability_challenger_gate_failed"
