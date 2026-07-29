from __future__ import annotations

from boatrace_ai.web.dashboard import MODEL_REPORT_HTML, _model_track_summaries


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
    assert "row.name||row.file||row.model" in MODEL_REPORT_HTML
    assert 'status==="retain_incumbent"' in protocol_source
    assert "判定状態不明" in protocol_source
    for text in ("policy odds", "Kelly", "露出", "上限", "単位・最低", "不合格:"):
        assert text in protocol_source
    for reason in ("ROI<1", "損益<=0", "艇Entry LL悪化", "1着悪化", "3T5悪化"):
        assert reason in protocol_source


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


def test_odds_path_v4_and_v6_tracks_share_complete_web_metrics(tmp_path) -> None:
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
