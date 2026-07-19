from boatrace_ai.web_dashboard import _roadmap_milestones


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
