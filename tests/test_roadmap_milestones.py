from boatrace_ai.web.dashboard import (
    _remote_evaluation_job_summaries,
    _roadmap_improvements,
    _roadmap_milestones,
)

def milestones_by_id(progress, processes=None, remote=None):
    return {
        row["id"]: row
        for row in _roadmap_milestones(progress, processes or [], remote or {})
    }


def test_historical_milestone_completes_from_database_coverage() -> None:
    rows = milestones_by_id(
        {
            "historical": {
                "target_days": 3650,
                "program_days": 3653,
                "result_days": 3653,
                "program_remaining_days": 0,
                "result_remaining_days": 0,
            }
        }
    )

    assert rows["M3"]["status"] == "完了"
    assert rows["M3"]["progress"] == 100


def test_operational_milestones_follow_required_processes() -> None:
    processes = [
        {"kind": "Webサーバ"},
        {"kind": "リアルタイム収集"},
        {"kind": "予測ループ"},
    ]
    rows = milestones_by_id({}, processes)

    assert rows["M0"]["status"] == "完了/運用中"
    assert rows["M2"]["status"] == "完了/運用中"


def test_model_milestones_follow_remote_evaluation_state() -> None:
    remote = {
        "jobs": [
            {"kind": "feature_ablation", "status": "完了"},
            {"kind": "bankroll_sanity", "status": "実行中"},
        ]
    }
    rows = milestones_by_id({}, remote=remote)

    assert rows["M4"]["progress"] == 90
    assert rows["M6"]["progress"] == 78


def test_m6_tracks_operational_same_policy_fold_progress() -> None:
    remote = {
        "jobs": [
            {
                "kind": "bankroll_operational_same_policy",
                "status": "実行中",
                "running": True,
                "completed_folds": 2,
                "process": {"cmd": "python -m worker --folds 5"},
            }
        ]
    }

    improvement = next(
        row for row in _roadmap_improvements({}, [], remote) if row["id"] == "M6-8"
    )
    summary = _remote_evaluation_job_summaries(remote)[0]

    assert improvement["status"] == "実行中"
    assert improvement["progress"] == 45
    assert "2/5" in improvement["next"]
    assert summary["completed_folds"] == 2
    assert summary["expected_folds"] == 5


def test_m6_completed_operational_backtest_shows_metrics() -> None:
    remote = {
        "jobs": [
            {
                "kind": "bankroll_operational_same_policy",
                "status": "完了",
                "completed_folds": 5,
                "result": {
                    "folds": 5,
                    "metrics": {
                        "roi": 0.802518753,
                        "profit_yen": -573920,
                        "max_drawdown_yen": 574180,
                    },
                },
            }
        ]
    }

    improvement = next(
        row for row in _roadmap_improvements({}, [], remote) if row["id"] == "M6-8"
    )

    assert improvement["status"] == "完了"
    assert improvement["progress"] == 100
    assert "ROI 0.8025" in improvement["next"]
    assert "-573,920円" in improvement["next"]
    assert "評価中" not in improvement["next"]


def test_m8_completed_status_has_no_terminal_setup_instruction() -> None:
    teleboat = {
        "connection_status": "ログイン・ログアウト確認済み",
        "public": {"success": True},
    }
    improvement = next(
        row
        for row in _roadmap_improvements({}, [], {}, teleboat)
        if row["id"] == "M8-1"
    )
    milestone = next(
        row
        for row in _roadmap_milestones({}, [], {}, teleboat)
        if row["id"] == "M8"
    )

    assert improvement["progress"] == 100
    assert milestone["status"] == "完了"
    assert "端末" not in improvement["next"]
    assert "監査済み" in milestone["next"]
