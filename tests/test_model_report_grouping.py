from __future__ import annotations

from boatrace_ai.web.dashboard import MODEL_REPORT_HTML, _model_track_summaries


def test_model_selector_groups_all_sources() -> None:
    assert '<optgroup label="' in MODEL_REPORT_HTML
    for label in (
        "モデル系統",
        "予測評価",
        "資金運用",
        "実行ジョブ",
        "特徴量診断",
        "候補スイープ",
    ):
        assert label in MODEL_REPORT_HTML
    assert "seenLabels" in MODEL_REPORT_HTML


def test_model_selector_normalizes_unified_keys_and_uses_file_identity() -> None:
    key_source = MODEL_REPORT_HTML.split("function modelKey(value){", 1)[1].split(
        "function modelValues", 1
    )[0]
    catalog_source = MODEL_REPORT_HTML.split("function modelCatalog(data){", 1)[1].split(
        "function configureModelSelect", 1
    )[0]

    assert "standardized_365d_v2" in key_source
    assert 'group==="bankroll"?(x.file||x.name||x.model)' in catalog_source


def test_daily_series_matches_normalized_key_and_uses_evaluated_races() -> None:
    source = MODEL_REPORT_HTML.split("function renderDaily(data,key){", 1)[1].split(
        "function groupFoldSeries", 1
    )[0]

    assert "filter(value=>modelKey(value)===key)" in source
    assert 'includes("standardized_365d_v2")' in source
    assert "fmt(r.evaluated_races)" in source
    assert "fmt(r.races)" not in source


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
    assert "comparisonRows.map(comparisonRow)" in render_source
    assert "summaryTests.map" not in render_source
    assert "summaryBank.map" not in render_source
    assert "function mergeComparisonRows" in MODEL_REPORT_HTML
    assert "function trackLoss" in MODEL_REPORT_HTML
    assert 'status==="retain_incumbent"' in protocol_source
    assert "判定状態不明" in protocol_source
    for text in ("policy odds", "Kelly", "露出", "上限", "単位・最低", "不合格:"):
        assert text in protocol_source
    for reason in ("ROI<1", "損益<=0", "艇Entry LL悪化", "1着悪化", "3T5悪化"):
        assert reason in protocol_source


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
