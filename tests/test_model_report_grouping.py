from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from boatrace_ai.web.dashboard import (
    MODEL_REPORT_HTML,
    _model_track_summaries,
    _production_trend_point_prospective_evidence,
    _repository_deployment_status,
)


def test_model_tables_are_the_selector_and_dropdown_is_removed() -> None:
    assert 'id="modelSelect"' not in MODEL_REPORT_HTML
    assert "<optgroup" not in MODEL_REPORT_HTML
    assert 'class="model-select-row"' in MODEL_REPORT_HTML
    assert 'role="button"' in MODEL_REPORT_HTML
    assert 'aria-selected="false"' in MODEL_REPORT_HTML
    assert 'event.key!=="Enter"&&event.key!==" "' in MODEL_REPORT_HTML
    assert "activateModel(row.dataset.modelKey)" in MODEL_REPORT_HTML


def test_model_selection_uses_backend_catalog_and_stable_key() -> None:
    source = MODEL_REPORT_HTML.split(
        "function configureModelSelection(data,comparisonRows){", 1
    )[1].split("function selectedCatalogEntry", 1)[0]

    assert "data.model_catalog||[]" in source
    assert "row.model_key" in source
    assert "comparisonRows||[]" in source
    assert "modelSelect" not in source


def test_production_trend_point_evidence_reports_missing_registration(
    tmp_path,
) -> None:
    evidence = _production_trend_point_prospective_evidence(
        [], app_root=tmp_path
    )

    assert evidence["status"] == "blocked_missing_or_invalid_source_evaluation"
    assert evidence["registered_after"] == "2026-08-03"
    assert evidence["model_identity"]["fixed"] is False
    assert evidence["error"]


def test_production_trend_point_evidence_proves_identity_before_first_job(
    tmp_path,
) -> None:
    model = tmp_path / "data/models/evaluation_queue/job-00012012.joblib"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"fixed model")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    (model.parent / "job-00012051.json").write_text(
        json.dumps({
            "source_model": str(model),
            "source_model_sha256": digest,
        }),
        encoding="utf-8",
    )

    evidence = _production_trend_point_prospective_evidence(
        [],
        app_root=tmp_path,
        now=datetime(2026, 8, 3, 15, tzinfo=timezone.utc),
    )

    assert evidence["status"] == "ready_waiting_for_first_unseen_day"
    assert evidence["next_evidence_date"] == "2026-08-04"
    assert evidence["first_scheduled_at_jst"] == "2026-08-05T03:00:00+09:00"
    assert evidence["model_identity"] == {
        "expected_model_sha256": digest,
        "observed_model_sha256": digest,
        "fixed": True,
    }


def test_production_trend_point_evidence_uses_latest_fixed_job() -> None:
    older = {
        "db_job_id": 13000,
        "model_key": "production_trend_point_job_12012",
        "status": "完了",
        "parameters": {"through_date": "2026-08-04"},
    }
    expected_sha256 = "a" * 64
    latest = {
        "db_job_id": 13001,
        "model_key": "production_trend_point_job_12012",
        "status": "完了",
        "parameters": {
            "through_date": "2026-08-05",
            "trend_point_registered_after": "2026-08-03",
            "expected_model_sha256": expected_sha256,
            "prospective_candidate": {
                "source_model_job_id": 12012,
                "source_evaluation_job_id": 12051,
                "selection_data_is_diagnostic_only": True,
                "real_betting_enabled": False,
            },
        },
        "source_model_sha256": expected_sha256,
        "trend_point_prospective_evaluation_days": 2,
        "trend_point_prospective_evaluated_races": 288,
        "trend_point_prospective_tickets": 51,
        "trend_point_prospective_hit_tickets": 8,
        "trend_point_prospective_roi": 1.21,
        "trend_point_prospective_promotion_gate": {"pass": False},
    }

    evidence = _production_trend_point_prospective_evidence([older, latest])

    assert evidence["latest_job_id"] == 13001
    assert evidence["evaluation_through"] == "2026-08-05"
    assert evidence["model_identity"] == {
        "expected_model_sha256": expected_sha256,
        "observed_model_sha256": expected_sha256,
        "fixed": True,
    }
    assert evidence["clean_days"] == 2
    assert evidence["races"] == 288
    assert evidence["tickets"] == 51
    assert evidence["roi"] == 1.21
    assert evidence["promotion_gate"] == {"pass": False}


def test_repository_deployment_status_exposes_only_verification_fields(
    tmp_path,
) -> None:
    state_path = tmp_path / "runtime" / "repository-deployment-state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        '{"status":"completed","head":"abc","completed_at":"now",'
        '"restarted":["boatrace-dashboard"],"secret":"must-not-leak"}',
        encoding="utf-8",
    )

    status = _repository_deployment_status(tmp_path / "boatrace.sqlite")

    assert status["status"] == "completed"
    assert status["head"] == "abc"
    assert status["restarted"] == ["boatrace-dashboard"]
    assert "secret" not in status


def test_daily_series_uses_canonical_backend_data_and_reason() -> None:
    source = MODEL_REPORT_HTML.split("async function renderDaily(data,key){", 1)[1].split(
        "function groupFoldSeries", 1
    )[0]

    assert "(data.model_daily||{})[key]" in source
    assert "daily.unavailable_reason" in source
    assert "r.cumulative_profit_yen" in source
    assert "r.cumulative_roi_delta" in source
    assert "r.roi_delta" in source
    assert "日次損益率" in MODEL_REPORT_HTML
    assert "累積損益率" in source
    assert "percent:true, baseline:0" in source
    assert "fmt(r.evaluated_races)" in source
    assert "data.bankroll_daily" not in source
    assert "daily.loaded===false" in source
    assert "/api/reports/model-performance/daily?model_key=" in source
    assert "encodeURIComponent(key)" in source
    assert "selectedModelKey!==key" in source


def test_client_model_key_keeps_v4_and_v6_as_single_model_families() -> None:
    source = MODEL_REPORT_HTML.split("function modelKey(value){", 1)[1].split(
        "function modelValues", 1
    )[0]

    assert "observed_closing_return_v4" in source
    assert "prequential_shrinkage_return_v6" in source


def test_unified_summary_and_promotion_display_are_explicit() -> None:
    render_source = MODEL_REPORT_HTML.split("function render(data){", 1)[1].split(
        "function renderStandardProtocol", 1
    )[0]
    protocol_source = MODEL_REPORT_HTML.split("function renderStandardProtocol(row){", 1)[
        1
    ].split("function modelKey", 1)[0]

    assert "const v2Ready=Boolean" in render_source
    assert "comparisonBank=v2Ready?v2Bank:v1Bank" in render_source
    assert "comparisonTests=v2Ready?v2Tests:v1Tests" in render_source
    assert "mergeComparisonRows(comparisonTests,comparisonBank)" in render_source
    assert "map(predictionRow)" in render_source
    assert "map(operationRow)" in render_source
    assert "map(compositeRow)" in render_source
    assert "summaryTests.map" not in render_source
    assert "summaryBank.map" not in render_source
    assert "function mergeComparisonRows" in MODEL_REPORT_HTML
    assert "function predictionRow" in MODEL_REPORT_HTML
    assert "function operationRow" in MODEL_REPORT_HTML
    assert "function compositeRow" in MODEL_REPORT_HTML
    assert "function trackLoss" in MODEL_REPORT_HTML
    assert "item.name||item.file||item.model" in MODEL_REPORT_HTML
    assert 'status==="retain_incumbent"' in protocol_source
    assert "判定状態不明" in protocol_source
    for text in ("policy odds", "Kelly", "露出", "上限", "単位・最低", "不合格:"):
        assert text in protocol_source
    for reason in ("ROI<1", "損益<=0", "艇Entry LL悪化", "1着悪化", "3T5悪化"):
        assert reason in protocol_source
    assert "EV帯 証拠/診断" in MODEL_REPORT_HTML
    assert "診断(昇格外)" in MODEL_REPORT_HTML
    assert "top5_narrow_retrospective_roi_without_largest_hit" in MODEL_REPORT_HTML


def test_model_metrics_are_split_into_vertical_full_width_tables() -> None:
    for row_id in (
        "predictionSummaryRows",
        "operationSummaryRows",
        "compositeSummaryRows",
        "predictionTrackRows",
        "operationTrackRows",
        "compositeTrackRows",
    ):
        assert f'id="{row_id}"' in MODEL_REPORT_HTML

    assert 'class="table-stack"' in MODEL_REPORT_HTML
    assert ".table-stack,.model-track-groups" in MODEL_REPORT_HTML
    assert "grid-template-columns:minmax(0,1fr)" in MODEL_REPORT_HTML
    assert ".table-scroll { overflow-x:visible; }" in MODEL_REPORT_HTML
    assert "min-width:1780px" not in MODEL_REPORT_HTML
    assert "min-width:1280px" not in MODEL_REPORT_HTML
    assert "教師・特徴量 / 学習" in MODEL_REPORT_HTML
    assert 'x.evaluation_group==="t5_walk_forward"' in MODEL_REPORT_HTML
    assert "predictionTrackRows([...prediction,...pendingOddsPath])" in MODEL_REPORT_HTML


def test_model_tracks_include_listwise_search_and_newton(tmp_path) -> None:
    remote = {
        "jobs": [
            {
                "kind": "feature_teacher_search",
                "status": "実行中",
                "result": None,
            },
            {
                "kind": "newton_listwise_bankroll",
                "status": "待機中",
                "result": None,
            },
        ]
    }
    rows = _model_track_summaries(tmp_path, [], remote)
    by_id = {row["id"]: row for row in rows}
    assert by_id["feature_teacher_search"]["status"] == "実行中"
    assert by_id["newton_listwise_bankroll"]["status"] == "待機中"
    assert "Plackett-Luce" in by_id["feature_teacher_search"]["teacher"]
    assert "Newton-CG" in by_id["newton_listwise_bankroll"]["training"]


def test_model_tracks_keep_missing_safe_and_legacy_ablation_separate(tmp_path) -> None:
    (tmp_path / "listwise_missing_safe_365d_5fold.json").write_text(
        '{"evaluated_races":48424,"entry_log_loss":0.33,'
        '"winner_top1_accuracy":0.57,"trifecta_top5_hit_rate":0.32,'
        '"roi":0.84,"profit_yen":-1000}',
        encoding="utf-8",
    )
    rows = _model_track_summaries(tmp_path, [], {"jobs": []})
    by_id = {row["id"]: row for row in rows}

    current = by_id["listwise_missing_safe_365d"]
    legacy = by_id["listwise_legacy_schema_365d"]
    assert current["status"] == "完了"
    assert current["eligible_races"] == 48424
    assert current["entry_log_loss"] == 0.33
    assert current["winner_top1_accuracy"] == 0.57
    assert current["trifecta_top5_hit_rate"] == 0.32
    assert current["roi"] == 0.84
    assert current["profit_yen"] == -1000
    assert "欠損値は順位0" in current["teacher"]
    assert legacy["status"] == "未登録"
    assert "旧スキーマ" in legacy["teacher"]


def test_odds_path_v4_v6_v7_v8_tracks_share_complete_web_metrics(tmp_path) -> None:
    jobs = [
        {
            "db_job_id": 101,
            "name": "odds_path_observed_closing_return_v4_daily:market_residual:20260718-28",
            "status": "完了",
            "evaluated_races": 918,
            "evaluation_days": 6,
            "winner_log_loss": 1.16,
            "trifecta_log_loss": 3.71,
            "winner_top1_accuracy": 0.561,
            "trifecta_top5_hit_rate": 0.373,
            "stake_yen": 15_700,
            "return_yen": 17_390,
            "roi": 1.1076,
            "profit_yen": 1_690,
            "max_drawdown_yen": 1_900,
            "tickets": 157,
            "hit_tickets": 15,
            "roi_without_largest_hit": 0.9707,
        },
        {
            "db_job_id": 102,
            "name": "odds_path_prequential_shrinkage_return_v6_daily:market_residual:20260718-28",
            "status": "実行中",
        },
        {
            "db_job_id": 103,
            "name": "odds_path_crossfit_conservative_ev_v7_daily:market_residual:20260718-28",
            "status": "待機中",
            "closing_q20_pinball_loss": 0.04,
            "closing_q20_lower_coverage": 0.81,
            "purchase_decision_diagnostics": {
                "threshold_pass_candidates": 0,
                "candidates_after_race_cap": 0,
                "purchases_after_allocation": 0,
                "safe_ev_max": 1.04,
                "safe_ev_p95": 0.98,
                "safe_ev_p99": 1.01,
            },
        },
        {
            "db_job_id": 104,
            "name": "odds_path_market_offset_crossfit_conservative_ev_v8_daily:market_residual:20260718-28",
            "status": "実行中",
            "trifecta_log_loss": 3.69,
            "closing_q20_pinball_loss": 0.039,
            "closing_q20_lower_coverage": 0.80,
            "purchase_decision_diagnostics": {
                "threshold_pass_candidates": 5,
                "candidates_after_race_cap": 2,
                "purchases_after_allocation": 0,
                "safe_ev_max": 1.08,
                "safe_ev_p95": 1.03,
                "safe_ev_p99": 1.06,
            },
        },
    ]

    rows = _model_track_summaries(
        tmp_path,
        [],
        {"jobs": []},
        evaluation_jobs=jobs,
    )
    by_id = {row["id"]: row for row in rows}
    v4 = by_id["odds_path_observed_closing_return_v4"]
    v6 = by_id["odds_path_prequential_shrinkage_return_v6"]
    v7 = by_id["odds_path_crossfit_conservative_ev_v7"]
    v8 = by_id["odds_path_market_offset_crossfit_conservative_ev_v8"]

    assert v4["trifecta_log_loss"] == 3.71
    assert v4["winner_top1_accuracy"] == 0.561
    assert v4["trifecta_top5_hit_rate"] == 0.373
    assert v4["roi"] == 1.1076
    assert v4["profit_yen"] == 1_690
    assert v4["roi_without_largest_hit"] == 0.9707
    assert set(
        (
            "closing_odds_log_mae",
            "closing_odds_rank_correlation",
            "closing_odds_interval_coverage",
            "closing_snapshot_age_seconds",
        )
    ).issubset(v4)
    assert v6["status"] == "実行中"
    assert v6["model_key"] == "odds_path_prequential_shrinkage_return_v6"
    assert "inner日だけ" in v6["teacher"]
    assert "11特徴" in v6["training"]
    assert "18候補" in v6["training"]
    assert v7["status"] == "待機中"
    assert v7["closing_q20_pinball_loss"] == 0.04
    assert v7["closing_q20_lower_coverage"] == 0.81
    assert "10年履歴base" in v7["teacher"]
    assert "固定safe EV" in v7["training"]
    assert v7["purchase_decision_diagnostics"]["threshold_pass_candidates"] == 0
    assert v8["status"] == "実行中"
    assert v8["trifecta_log_loss"] == 3.69
    assert v8["closing_q20_lower_coverage"] == 0.80
    assert "市場)固定offset" in v8["training"]
    assert "日付順nested L2" in v8["training"]
    assert v8["purchase_decision_diagnostics"]["candidates_after_race_cap"] == 2
    assert "function purchaseDiagnostic(x)" in MODEL_REPORT_HTML
    assert "閾値通過0" in MODEL_REPORT_HTML
    assert "配分後0" in MODEL_REPORT_HTML
    assert "safe_ev_p95" in MODEL_REPORT_HTML


def test_v9_report_track_exposes_discrete_allocation_diagnostics(tmp_path) -> None:
    diagnostics = {
        "threshold_pass_candidates": 11,
        "candidates_before_allocation": 6,
        "allocation_candidate_tickets": 4,
        "purchases_after_allocation": 3,
        "zero_reason_counts": {"no_positive_discrete_log_growth": 1},
    }
    rows = _model_track_summaries(
        tmp_path,
        [],
        {"jobs": []},
        evaluation_jobs=[{
            "db_job_id": 109,
            "name": "odds_path_market_offset_discrete_log_ev_v9_daily:market_residual:20260722-29",
            "status": "完了",
            "evaluated_races": 918,
            "evaluation_days": 6,
            "trifecta_log_loss": 3.68,
            "roi": 1.03,
            "purchase_decision_diagnostics": diagnostics,
        }],
    )
    row = next(
        item
        for item in rows
        if item["id"] == "odds_path_market_offset_discrete_log_ev_v9"
    )

    assert row["model_key"] == "odds_path_market_offset_discrete_log_ev_v9"
    assert row["trifecta_log_loss"] == 3.68
    assert row["roi"] == 1.03
    assert row["purchase_decision_diagnostics"] == diagnostics
    assert "離散期待対数効用" in row["training"]
    assert "配分前" in MODEL_REPORT_HTML
    assert "正効用" in MODEL_REPORT_HTML
    assert "zero_reason_counts" in MODEL_REPORT_HTML


def test_v10_report_track_exposes_selection_conformal_guard(tmp_path) -> None:
    diagnostics = {
        "raw_selected_candidates": 9,
        "guarded_threshold_candidates": 3,
        "purchases_after_allocation": 2,
        "zero_reason_counts": {"no_candidate_after_selection_conformal": 1},
    }
    conformal = {
        "selection_raw_closing_coverage": 2 / 9,
        "selection_guarded_closing_coverage": 8 / 9,
        "haircut_latest": 0.55,
        "training_days_latest": 6,
        "training_candidates_latest": 41,
    }
    rows = _model_track_summaries(
        tmp_path,
        [],
        {"jobs": []},
        evaluation_jobs=[{
            "db_job_id": 110,
            "name": "odds_path_market_offset_selection_conformal_discrete_ev_v10_daily:market_residual:20260718-29",
            "status": "完了",
            "evaluated_races": 918,
            "evaluation_days": 6,
            "trifecta_log_loss": 3.67,
            "roi": 1.04,
            "purchase_decision_diagnostics": diagnostics,
            "selection_conformal": conformal,
            **conformal,
        }],
    )
    row = next(
        item
        for item in rows
        if item["id"]
        == "odds_path_market_offset_selection_conformal_discrete_ev_v10"
    )

    assert row["purchase_decision_diagnostics"] == diagnostics
    assert row["selection_conformal"] == conformal
    assert row["selection_guarded_closing_coverage"] == 8 / 9
    assert row["haircut_latest"] == 0.55
    assert row["training_days_latest"] == 6
    assert row["training_candidates_latest"] == 41
    assert "選択条件付き有限標本conformal" in row["training"]
    assert "条件終値 raw" in MODEL_REPORT_HTML
    assert "haircut" in MODEL_REPORT_HTML
    assert "trainingCandidates" in MODEL_REPORT_HTML
    assert "条件補正未学習" in MODEL_REPORT_HTML
    assert "補正後候補0" in MODEL_REPORT_HTML
    assert "zero_reason_counts" in MODEL_REPORT_HTML
