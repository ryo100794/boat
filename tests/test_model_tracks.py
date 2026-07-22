import json

from boatrace_ai.web.dashboard import (
    HTML,
    MODEL_REPORT_HTML,
    _model_track_summaries,
    _local_evaluation_result,
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
    (tmp_path / "realtime_odds_shadow_t5_safe_candidate_state.json").write_text(
        json.dumps(
            {
                "evaluation_version": 2,
                "eligible_races": 450,
                "required_races": 450,
                "last_evaluated_races": 450,
                "status": "evaluated",
            }
        ),
        encoding="utf-8",
    )
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
    assert "3fold暫定・評価95R" in shadow["training"]
    assert shadow["evaluated_races"] == 95


def test_model_tracks_exposes_market_calibrated_shadow(tmp_path) -> None:
    remote = {
        "jobs": [
            {
                "kind": "market_calibrated_shadow",
                "status": "完了",
                "result": {
                    "metrics": {
                        "evaluated_races": 279,
                        "calibrated_trifecta_log_loss": 4.1323,
                        "trifecta_top5_hit_rate": 0.2581,
                        "roi": 0.0,
                        "profit_yen": 0,
                        "promotion_eligible": False,
                    }
                },
            }
        ]
    }
    tracks = _model_track_summaries(tmp_path, [], remote)
    market = next(row for row in tracks if row["id"] == "market_calibrated_shadow")
    assert market["include_odds"] is True
    assert market["eligible_races"] == 279
    assert market["trifecta_log_loss"] == 4.1323
    assert market["trifecta_top5_hit_rate"] == 0.2581
    assert market["promotion_eligible"] is False


def test_model_tracks_exposes_cutoff_refit_and_market_shadow(tmp_path) -> None:
    remote = {
        "jobs": [
            {
                "kind": "listwise_cutoff_refit",
                "status": "完了",
                "result": {
                    "metrics": {
                        "evaluated_races": 632,
                        "entry_log_loss": 0.3398,
                        "winner_top1_accuracy": 0.5491,
                        "trifecta_top5_hit_rate": 0.3180,
                    }
                },
            },
            {
                "kind": "market_calibrated_cutoff_shadow",
                "status": "完了",
                "result": {
                    "metrics": {
                        "evaluated_races": 275,
                        "model_trifecta_log_loss": 3.9695,
                        "model_trifecta_top5_hit_rate": 0.3309,
                        "calibrated_trifecta_log_loss": 4.0013,
                        "trifecta_top5_hit_rate": 0.3091,
                        "roi": 0.0,
                        "profit_yen": 0,
                        "promotion_eligible": False,
                    }
                },
            },
        ]
    }
    tracks = {row["id"]: row for row in _model_track_summaries(tmp_path, [], remote)}
    refit = tracks["listwise_cutoff_refit"]
    market = tracks["market_calibrated_cutoff_shadow"]
    assert refit["winner_top1_accuracy"] == 0.5491
    assert refit["trifecta_top5_hit_rate"] == 0.3180
    assert market["model_trifecta_log_loss"] == 3.9695
    assert market["model_trifecta_top5_hit_rate"] == 0.3309
    assert market["promotion_eligible"] is False


def test_model_tracks_exposes_preselected_stagewise_blend(tmp_path) -> None:
    remote = {
        "jobs": [
            {
                "kind": "stagewise_blend_preselected",
                "status": "完了",
                "result": {
                    "metrics": {
                        "evaluated_races": 632,
                        "trifecta_log_loss": 3.9465,
                        "winner_top1_accuracy": 0.5459,
                        "trifecta_top5_hit_rate": 0.3307,
                    }
                },
            },
            {
                "kind": "market_calibrated_blend_shadow",
                "status": "完了",
                "result": {
                    "metrics": {
                        "evaluated_races": 275,
                        "model_trifecta_log_loss": 3.9081,
                        "model_trifecta_top5_hit_rate": 0.3527,
                        "calibrated_trifecta_log_loss": 3.8953,
                        "trifecta_top5_hit_rate": 0.3527,
                        "roi": 0.0,
                        "promotion_eligible": False,
                    }
                },
            },
        ]
    }

    tracks = {row["id"]: row for row in _model_track_summaries(tmp_path, [], remote)}

    blend = tracks["stagewise_blend_preselected"]
    market = tracks["market_calibrated_blend_shadow"]
    assert blend["trifecta_top5_hit_rate"] == 0.3307
    assert market["model_trifecta_log_loss"] == 3.9081
    assert market["trifecta_log_loss"] == 3.8953
    assert market["promotion_eligible"] is False


def test_model_tracks_reads_blend_artifacts_without_remote_poll(tmp_path) -> None:
    (tmp_path / "stagewise_blend_preselected_20260717.json").write_text(
        json.dumps(
            {
                "model": "blend",
                "final_evaluation": {
                    "selected_blend": {
                        "evaluated_races": 632,
                        "trifecta_log_loss": 3.9465,
                        "winner_top1_accuracy": 0.5459,
                        "trifecta_top5_hit_rate": 0.3307,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "stagewise_blend_market_shadow.json").write_text(
        json.dumps(
            {
                "evaluated_races": 275,
                "calibrated_trifecta_log_loss": 3.8953,
                "trifecta_top5_hit_rate": 0.3527,
                "promotion_eligible": False,
            }
        ),
        encoding="utf-8",
    )

    tracks = {row["id"]: row for row in _model_track_summaries(tmp_path, [], {"jobs": []})}

    assert tracks["stagewise_blend_preselected"]["status"] == "完了"
    assert tracks["stagewise_blend_preselected"]["trifecta_top5_hit_rate"] == 0.3307
    assert tracks["market_calibrated_blend_shadow"]["status"] == "完了"
    assert tracks["market_calibrated_blend_shadow"]["trifecta_log_loss"] == 3.8953


def test_market_track_uses_local_waiting_status_without_remote_job(tmp_path) -> None:
    (tmp_path / "stagewise_blend_market_shadow.json").write_text(
        json.dumps(
            {
                "status": "waiting_for_clean_evaluation_day",
                "available_races": 270,
                "available_days": 2,
                "evaluated_races": 0,
                "promotion_eligible": False,
            }
        ),
        encoding="utf-8",
    )

    tracks = {
        row["id"]: row
        for row in _model_track_summaries(tmp_path, [], {"jobs": []})
    }

    market = tracks["market_calibrated_blend_shadow"]
    assert market["status"] == "データ待ち"
    assert market["eligible_races"] == 0
    assert market["backtest_available"] is False


def test_local_evaluation_result_keeps_market_calibration_metrics(tmp_path) -> None:
    path = tmp_path / "market.json"
    path.write_text(
        json.dumps(
            {
                "model": "market",
                "evaluated_races": 279,
                "calibrated_trifecta_log_loss": 4.1323,
                "trifecta_top5_hit_rate": 0.2581,
                "evaluation_days": 2,
                "probability_metrics": {
                    "model_trifecta_log_loss": 3.9695,
                    "model_trifecta_top5_hit_rate": 0.3309,
                },
            }
        ),
        encoding="utf-8",
    )
    result = _local_evaluation_result(path)
    assert result is not None
    assert result["metrics"]["calibrated_trifecta_log_loss"] == 4.1323
    assert result["metrics"]["evaluated_races"] == 279
    assert result["metrics"]["evaluation_days"] == 2
    assert result["metrics"]["model_trifecta_log_loss"] == 3.9695
    assert result["metrics"]["model_trifecta_top5_hit_rate"] == 0.3309


def test_local_evaluation_result_uses_cutoff_after_refit_metrics(tmp_path) -> None:
    path = tmp_path / "cutoff.json"
    path.write_text(
        json.dumps(
            {
                "model": "cutoff",
                "evaluation_races": 632,
                "after_refit": {
                    "evaluated_races": 632,
                    "entry_log_loss": 0.3398,
                    "winner_top1_accuracy": 0.5491,
                    "trifecta_top5_hit_rate": 0.3180,
                },
            }
        ),
        encoding="utf-8",
    )
    result = _local_evaluation_result(path)
    assert result is not None
    assert result["metrics"]["evaluated_races"] == 632
    assert result["metrics"]["entry_log_loss"] == 0.3398
    assert result["metrics"]["winner_top1_accuracy"] == 0.5491
    assert result["metrics"]["trifecta_top5_hit_rate"] == 0.3180
