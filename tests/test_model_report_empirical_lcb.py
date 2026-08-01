from __future__ import annotations

import json
from pathlib import Path

from boatrace_ai.web import dashboard


def _empirical_policy() -> dict:
    return {
        "status": "evaluating",
        "evaluation_days": 34,
        "calibration_ready_folds": 31,
        "minimum_ready_evaluation_days": 30,
        "tickets": 420,
        "stake_yen": 42_000,
        "return_yen": 47_250,
        "profit_yen": 5_250,
        "roi": 1.125,
        "roi_without_largest_hit": 1.031,
        "sample_size_pass": True,
        "tail_portfolio_diagnostics": {
            "normal": {
                "daily_cluster_bootstrap_roi_lower_95": 1.006,
            }
        },
        "daily": [],
    }


def test_model_report_exposes_empirical_lcb_separately_from_legacy(tmp_path) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    artifact = {
        "model": "market_calibrated_v23",
        "empirical_lcb_walk_forward": _empirical_policy(),
    }
    (model_dir / "market_calibrated_v23.json").write_text(
        json.dumps(artifact), encoding="utf-8"
    )
    dashboard._MODEL_REPORT_CACHE.clear()

    report = dashboard.model_performance_report(
        tmp_path / "boatrace.sqlite",
        {"model_dir": [str(model_dir)]},
    )

    assert report["bankroll"] == []
    assert len(report["empirical_lcb_walk_forward"]) == 1
    row = report["empirical_lcb_walk_forward"][0]
    assert row["name"] == "market_calibrated_v23"
    assert row["status"] == "evaluating"
    assert row["calibration_ready_folds"] == 31
    assert row["evaluation_days"] == 34
    assert row["tickets"] == 420
    assert row["stake_yen"] == 42_000
    assert row["return_yen"] == 47_250
    assert row["profit_yen"] == 5_250
    assert row["roi"] == 1.125
    assert row["roi_without_largest_hit"] == 1.031
    assert row["sample_size_pass"] is True
    assert row["tail_bootstrap_roi_lower95"] == 1.006

def test_model_report_exposes_v21_runtime_evidence(tmp_path) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    state_dir = tmp_path / "runtime" / "daily-shadow-models"
    state_dir.mkdir(parents=True)
    activation = {"status": "monitoring", "real_betting_enabled": False}
    evidence = {
        "bankroll": {"clean_days": 1, "races": 132, "roi": 1.1},
        "promotion_gate": {"pass": False, "failed_checks": ["minimum_clean_days"]},
    }
    stable_evidence = {
        "model_key": "stable_cell_daily",
        "bankroll": {"clean_days": 0, "races": 0, "roi": None},
        "promotion_gate": {"pass": False, "failed_checks": ["identity_fixed"]},
    }
    (state_dir / "activation-recovery.json").write_text(
        json.dumps(activation), encoding="utf-8"
    )
    (state_dir / "v21-prospective-evidence.json").write_text(
        json.dumps(evidence), encoding="utf-8"
    )
    (state_dir / "stable-cell-prospective-evidence.json").write_text(
        json.dumps(stable_evidence), encoding="utf-8"
    )
    dashboard._MODEL_REPORT_CACHE.clear()

    report = dashboard.model_performance_report(
        tmp_path / "boatrace.sqlite",
        {"model_dir": [str(model_dir)]},
    )

    assert report["v21_activation_recovery"] == activation
    assert report["v21_prospective_evidence"] == evidence
    assert report["stable_cell_prospective_evidence"] == stable_evidence
    public = dashboard.model_performance_public_report(report)
    assert public["v21_activation_recovery"] == activation
    assert public["v21_prospective_evidence"] == evidence
    assert public["stable_cell_prospective_evidence"] == stable_evidence


def test_model_report_exposes_direct_shadow_bankroll_components(tmp_path) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    artifact = {
        "model": "shared_market_calibrated_model_name",
        "generated_at": "2026-07-28T08:00:00+00:00",
        "winner_top1_accuracy": 0.56,
        "trifecta_top5_hit_rate": 0.36,
        "evaluated_races": 918,
        "calibrated_trifecta_log_loss": 3.735,
        "conservative_market_offset_kelly_walk_forward": {
            "status": "waiting_for_first_unseen_day",
            "evaluation_days": 0,
            "tickets": 0,
            "stake_yen": 0,
            "return_yen": 0,
            "profit_yen": 0,
            "roi": 0,
            "daily": [
                {
                    "race_date": "2026-07-29",
                    "stake_yen": 0,
                    "return_yen": 0,
                }
            ],
        },
        "conformal_lower_market_offset_kelly_diagnostic": {
            "status": "evaluated",
            "evaluation_days": 6,
            "tickets": 12,
            "stake_yen": 1_200,
            "return_yen": 1_050,
            "profit_yen": -150,
            "roi": 0.875,
            "promotion_eligible": False,
        },
        "conformal_lower_market_offset_kelly_walk_forward": {
            "status": "waiting_for_first_unseen_day",
            "evaluation_days": 0,
            "tickets": 0,
            "stake_yen": 0,
            "return_yen": 0,
            "profit_yen": 0,
            "roi": 0,
        },
    }
    artifact_path = model_dir / "stagewise_market_shadow.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    dashboard._MODEL_REPORT_CACHE.clear()

    report = dashboard.model_performance_report(
        tmp_path / "boatrace.sqlite",
        {"model_dir": [str(model_dir)]},
    )

    expected_name = (
        "stagewise_market_shadow_"
        "conservative_market_offset_kelly_walk_forward"
    )
    row = next(item for item in report["bankroll"] if item["name"] == expected_name)
    assert row["model"].endswith("conservative_market_offset_kelly_walk_forward")
    assert row["evaluated_races"] == 918
    assert row["winner_top1_accuracy"] == 0.56
    assert row["trifecta_top5_hit_rate"] == 0.36
    assert row["roi"] == 0
    assert row["entry_log_loss"] == 3.735
    assert expected_name in report["bankroll_daily"]
    conformal_names = {
        item["name"]
        for item in report["bankroll"]
        if "conformal_lower_market_offset" in item["name"]
    }
    assert conformal_names == {
        "stagewise_market_shadow_conformal_lower_market_offset_kelly_diagnostic",
        "stagewise_market_shadow_conformal_lower_market_offset_kelly_walk_forward",
    }


def test_empirical_lcb_database_metrics_restore_compact_result() -> None:
    metrics = {
        "empirical_lcb_status": "calibration_not_ready",
        "empirical_lcb_evaluation_days": 7,
        "empirical_lcb_calibration_ready_folds": 0,
        "empirical_lcb_minimum_ready_evaluation_days": 30,
        "empirical_lcb_tickets": 0,
        "empirical_lcb_stake_yen": 0,
        "empirical_lcb_return_yen": 0,
        "empirical_lcb_profit_yen": 0,
        "empirical_lcb_roi": None,
        "empirical_lcb_roi_without_largest_hit": None,
        "empirical_lcb_sample_size_pass": False,
        "empirical_lcb_roi_lower95": 0.81,
    }

    policy = dashboard._empirical_lcb_walk_forward_from_metrics(metrics)
    assert policy is not None
    row = dashboard._empirical_lcb_walk_forward_summary(
        Path("job.json"),
        "job-v23",
        policy,
    )

    assert row["status"] == "calibration_not_ready"
    assert row["tickets"] == 0
    assert row["stake_yen"] == 0
    assert row["sample_size_pass"] is False
    assert row["tail_bootstrap_roi_lower95"] == 0.81


def test_model_report_template_distinguishes_empirical_lcb_track() -> None:
    source = dashboard.MODEL_REPORT_HTML

    assert 'id="empiricalLcbRows"' in source
    assert "実証LCB 前進運用" in source
    assert "レガシー資金運用とは別評価" in source
    assert "data.empirical_lcb_walk_forward||[]" in source
    assert "function empiricalLcbRow(row)" in source
    for label in (
        "準備fold",
        "評価日",
        "点数",
        "投資",
        "払戻",
        "損益",
        "最大1的中除外ROI",
        "tail LCB95",
        "母数判定",
    ):
        assert label in source


def test_model_report_renders_stable_cell_prospective_evidence() -> None:
    template = (
        Path(__file__).parents[1]
        / "src"
        / "boatrace_ai"
        / "templates"
        / "model_report.html"
    ).read_text(encoding="utf-8")

    assert 'id="stableCellProspective"' in template
    assert "data.stable_cell_prospective_evidence" in template
