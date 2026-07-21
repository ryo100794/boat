import json

from boatrace_ai.web.dashboard import (
    HTML,
    MODEL_REPORT_HTML,
    _model_track_summaries,
)


def test_model_tracks_separate_main_and_realtime_odds_shadow(tmp_path) -> None:
    (tmp_path / "realtime_odds_shadow_state.json").write_text(
        json.dumps(
            {
                "eligible_races": 217,
                "required_races": 1000,
                "ready": False,
                "status": "waiting_for_data",
            }
        ),
        encoding="utf-8",
    )
    backtests = [
        {
            "file": "backtest_no_odds_v8.json",
            "evaluated_races": 93774,
            "winner_top1_accuracy": 0.5642,
            "trifecta_top5_hit_rate": 0.3079,
            "entry_log_loss": 0.44,
        }
    ]

    remote = {
        "jobs": [
            {"kind": "calibrated_linear", "status": "待機中"},
            {"kind": "calibrated_mlp", "status": "待機中"},
        ]
    }
    tracks = _model_track_summaries(tmp_path, backtests, remote)
    by_id = {row["id"]: row for row in tracks}
    main = by_id["historical_main"]
    shadow = by_id["realtime_odds_shadow"]
    linear = by_id["calibrated_linear"]
    mlp = by_id["calibrated_mlp"]

    assert main["role"] == "本番予測"
    assert main["include_odds"] is False
    assert shadow["role"] == "比較評価のみ"
    assert shadow["status"] == "学習待ち/蓄積中"
    assert shadow["eligible_races"] == 217
    assert shadow["backtest_available"] is False
    assert "1着=1" in main["teacher"]
    assert "C=0.20" in main["training"]
    assert linear["model_file"] == "calibrated_linear_shadow_2fold.json"
    assert "SGDClassifier" in linear["training"]
    assert mlp["model_file"] == "calibrated_mlp_shadow_2fold.json"
    assert "MLP 64-16" in mlp["training"]


def test_web_templates_identify_the_active_model_track() -> None:
    assert 'id="modelTrackRows"' in MODEL_REPORT_HTML
    assert "本番とshadowを分離" in MODEL_REPORT_HTML
    assert "主系予測" in HTML
    assert 'bt.model_label || "過去ログ主系"' in HTML


def test_model_report_separates_cross_model_and_selected_model_aggregates() -> None:
    header = MODEL_REPORT_HTML.split("</header>", 1)[0]

    assert 'id="modelSelect"' not in header
    assert MODEL_REPORT_HTML.index('id="allModelsGroup"') < MODEL_REPORT_HTML.index('id="summaryRows"')
    assert MODEL_REPORT_HTML.index('id="summaryRows"') < MODEL_REPORT_HTML.index('id="selectedModelGroup"')
    assert MODEL_REPORT_HTML.index('id="selectedModelGroup"') < MODEL_REPORT_HTML.index('id="modelSelect"')
    assert MODEL_REPORT_HTML.index('id="modelSelect"') < MODEL_REPORT_HTML.index('id="modelDetailRows"')
    assert "全モデル横断集計" in MODEL_REPORT_HTML
    assert "モデル個別集計" in MODEL_REPORT_HTML


def test_model_selector_catalog_includes_every_report_data_group() -> None:
    catalog_source = MODEL_REPORT_HTML.split("function modelCatalog(data){", 1)[1].split("function configureModelSelect", 1)[0]

    for group in (
        "model_tracks",
        "backtests",
        "bankroll",
        "fold_metrics",
        "evaluation_jobs",
        "feature_diagnostics",
        "sweeps",
        "bankroll_daily",
    ):
        assert group in catalog_source


def test_model_tracks_exposes_t5_safe_provisional_metrics(tmp_path) -> None:
    tracks = _model_track_summaries(
        tmp_path,
        [
            {
                "file": "realtime_odds_shadow_t5_safe_candidate_backtest.json",
                "evaluated_races": 95,
                "entry_log_loss": 0.345016,
                "winner_top1_accuracy": 0.589474,
                "trifecta_top5_hit_rate": 0.326316,
            }
        ],
        {"jobs": []},
    )

    shadow = next(row for row in tracks if row["id"] == "realtime_odds_shadow")
    assert shadow["status"] == "暫定評価済み"
    assert shadow["model_file"] == "realtime_odds_shadow_t5_safe_candidate.joblib"
    assert shadow["entry_log_loss"] == 0.345016
    assert shadow["winner_top1_accuracy"] == 0.589474
    assert shadow["trifecta_top5_hit_rate"] == 0.326316
