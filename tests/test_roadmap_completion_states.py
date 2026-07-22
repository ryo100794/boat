from pathlib import Path

from boatrace_ai.web.dashboard import (
    _quality_gates,
    _roadmap_improvements,
    _roadmap_milestones,
    _v_file_inventory,
)


def completed_remote_evaluations() -> dict:
    return {
        "jobs": [
            {
                "kind": "calibrated_linear",
                "status": "完了",
                "result": {"metrics": {"entry_log_loss": 5.0566, "winner_top1_accuracy": 0.5602}},
            },
            {
                "kind": "calibrated_mlp",
                "status": "完了",
                "result": {
                    "metrics": {
                        "entry_log_loss": 0.3301,
                        "winner_top1_accuracy": 0.5633,
                        "trifecta_top5_hit_rate": 0.3054,
                    }
                },
            },
            {"kind": "feature_ablation", "status": "完了"},
            {"kind": "feature_teacher_search", "status": "完了"},
            {
                "kind": "newton_listwise_bankroll",
                "status": "完了",
                "result": {
                    "metrics": {
                        "winner_top1_accuracy": 0.564653,
                        "trifecta_top5_hit_rate": 0.327308,
                        "ranking_log_loss": 1.299785,
                        "roi": 0.799672,
                        "profit_yen": -36600,
                        "promotion_eligible": False,
                    }
                },
            },
            {"kind": "bankroll_sanity", "status": "完了"},
        ]
    }


def standardized_remote_evaluations() -> dict:
    jobs = [
        {
            "kind": "standardized_365d_backtest",
            "name": "standardized_365d_no_odds_v8_backtest",
            "status": "完了",
            "result": {
                "metrics": {
                    "winner_top1_accuracy": 0.567441,
                    "trifecta_top5_hit_rate": 0.315298,
                }
            },
        },
        {
            "kind": "standardized_365d_bankroll",
            "name": "standardized_365d_no_odds_v8_bankroll",
            "status": "完了",
            "result": {"metrics": {"roi": 0.761072, "profit_yen": -335120}},
        },
        {
            "kind": "standardized_365d_backtest",
            "name": "standardized_365d_pastlog_v7_backtest",
            "status": "完了",
            "result": {"metrics": {"winner_top1_accuracy": 0.558993}},
        },
        {
            "kind": "standardized_365d_bankroll",
            "name": "standardized_365d_pastlog_v7_bankroll",
            "status": "完了",
            "result": {"metrics": {"roi": 0.8808429, "profit_yen": -232130}},
        },
        {
            "kind": "standardized_365d_calibrated_linear",
            "name": "standardized_365d_calibrated_linear",
            "status": "完了",
            "result": {"metrics": {"entry_log_loss": 4.941831}},
        },
        {
            "kind": "standardized_365d_calibrated_mlp",
            "name": "standardized_365d_calibrated_mlp",
            "status": "完了",
            "result": {"metrics": {"entry_log_loss": 0.326569}},
        },
        {
            "kind": "standardized_365d_listwise_bankroll",
            "name": "standardized_365d_listwise_feature_teacher",
            "status": "完了",
            "result": {"metrics": {"roi": 0.678485, "profit_yen": -309490}},
        },
        {
            "kind": "standardized_365d_listwise_bankroll",
            "name": "standardized_365d_listwise_newton",
            "status": "完了",
            "result": {"metrics": {"roi": 0.717211, "profit_yen": -269470}},
        },
        {
            "kind": "bankroll_temporal_no_bet",
            "name": "bankroll_no_odds_v8_temporal_no_bet_5fold",
            "status": "実行中",
            "process": {"cmdline": "evaluate --folds 5"},
            "completed_folds": 4,
        },
    ]
    return {"jobs": jobs}


def test_completed_model_evaluations_show_final_outcomes() -> None:
    progress = {"realtime": {"eligible_races": 350, "readiness": 0.35}}
    improvements = {
        row["id"]: row
        for row in _roadmap_improvements(
            progress,
            [{"kind": "リアルタイムshadow"}],
            completed_remote_evaluations(),
        )
    }

    assert improvements["M4-1"]["progress"] == 100
    assert improvements["M4-1"]["status"] == "評価完了/昇格見送り"
    assert improvements["M4-2"]["progress"] == 60
    assert improvements["M4-2"]["status"] == "要再検証/欠損表現修正待ち"
    assert improvements["M4-3"]["progress"] == 100
    assert improvements["M4-3"]["status"] == "評価完了/昇格見送り"
    assert "ROI 0.7997" in improvements["M4-3"]["next"]
    assert improvements["M6-1"]["status"] == "蓄積待ち"
    assert improvements["M6-1"]["progress"] == 35
    assert improvements["M6-4"]["progress"] == 100


def test_milestones_separate_completed_evaluation_from_failed_gate() -> None:
    milestones = {
        row["id"]: row
        for row in _roadmap_milestones(
            {"realtime": {"eligible_races": 350, "readiness": 0.35}},
            [{"kind": "リアルタイムshadow"}],
            completed_remote_evaluations(),
        )
    }

    assert milestones["M4"]["status"] == "統一再評価中"
    assert milestones["M4"]["progress"] == 88
    assert milestones["M6"]["status"] == "未完了/収益ゲート未達"
    assert milestones["M6"]["progress"] < 100


def test_completed_remote_gate_has_no_stale_pid_instruction(tmp_path) -> None:
    remote = {
        "jobs": [
            {
                "kind": "bankroll_norm",
                "status": "完了",
                "result": {
                    "file": "best.json",
                    "modified_at": "2026-07-19T12:00:00+00:00",
                    "metrics": {
                        "roi": 0.89346,
                        "profit_yen": -112270,
                        "evaluated_races": 93774,
                    },
                },
            }
        ]
    }

    gates = _quality_gates(tmp_path, remote)
    assert all("PID" not in row["next"] for row in gates)


def test_standardized_comparison_and_temporal_progress_are_visible() -> None:
    remote = standardized_remote_evaluations()
    improvements = {
        row["id"]: row
        for row in _roadmap_improvements(
            {"realtime": {"eligible_races": 350, "readiness": 0.35}},
            [{"kind": "リアルタイムshadow"}],
            remote,
        )
    }

    assert improvements["M4-4"]["status"] == "評価中"
    assert improvements["M4-4"]["progress"] == 0
    assert "0/7件" in improvements["M4-4"]["next"]
    assert improvements["M6-9"]["status"] == "実行中"
    assert improvements["M6-9"]["progress"] == 80

    milestones = {
        row["id"]: row
        for row in _roadmap_milestones(
            {"realtime": {"eligible_races": 350, "readiness": 0.35}},
            [{"kind": "リアルタイムshadow"}],
            remote,
        )
    }
    assert "標準365日" in milestones["M4"]["next"]
    assert milestones["M6"]["progress"] == 75
    assert "同一holdout比較" in milestones["M6"]["next"]


def test_versioned_inventory_scans_nested_packages(tmp_path) -> None:
    nested = tmp_path / "runtime"
    nested.mkdir()
    (nested / "worker_v2.py").write_text("", encoding="utf-8")

    inventory = _v_file_inventory(tmp_path)

    assert inventory["count"] == 1
    assert inventory["sample"] == [str(Path("runtime") / "worker_v2.py")]


def test_local_artifacts_override_stale_remote_state(tmp_path) -> None:
    from boatrace_ai.web.dashboard import _hydrate_local_evaluation_jobs

    status_path = tmp_path / "data" / "remote_eval_status.json"
    artifact = tmp_path / "data" / "models" / "completed.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        '{"roi": 0.84, "profit_yen": -1600, "evaluated_races": 105}',
        encoding="utf-8",
    )
    hydrated = _hydrate_local_evaluation_jobs(
        {
            "status": "取得失敗",
            "error": "old worker unavailable",
            "jobs": [
                {
                    "kind": "bankroll_temporal_no_bet",
                    "output": "data/models/completed.json",
                }
            ],
        },
        status_path,
    )

    assert hydrated["status"] == "ローカル成果物で更新"
    assert "error" not in hydrated
    assert hydrated["jobs"][0]["status"] == "完了"
    assert hydrated["jobs"][0]["result"]["metrics"]["roi"] == 0.84


def test_local_waiting_artifact_is_not_marked_complete(tmp_path) -> None:
    from boatrace_ai.web.dashboard import _hydrate_local_evaluation_jobs

    status_path = tmp_path / "data" / "remote_eval_status.json"
    artifact = tmp_path / "data" / "models" / "waiting.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        "{\"status\":\"waiting_for_clean_evaluation_day\",\"available_races\":270,"
        "\"available_days\":2,\"required_additional_days\":1,\"evaluated_races\":0}",
        encoding="utf-8",
    )
    hydrated = _hydrate_local_evaluation_jobs(
        {
            "jobs": [
                {
                    "kind": "market_calibrated_blend_shadow",
                    "output": "data/models/waiting.json",
                }
            ]
        },
        status_path,
    )

    job = hydrated["jobs"][0]
    assert job["status"] == "データ待ち"
    assert job["result"]["status"] == "waiting_for_clean_evaluation_day"
    assert job["result"]["metrics"]["available_races"] == 270
    assert job["result"]["metrics"]["evaluated_races"] == 0


def test_collector_deployment_enables_predictions() -> None:
    script = Path("scripts/deployment/run-boatrace-collector-foreground.sh").read_text(
        encoding="utf-8"
    )

    assert "--predict" in script
    assert "LD_LIBRARY_PATH=/workspace/postgresql/runtime/lib" in script
