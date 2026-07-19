import json

from boatrace_ai.web_dashboard import (
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
    main, shadow, linear, mlp = _model_track_summaries(tmp_path, backtests, remote)

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
