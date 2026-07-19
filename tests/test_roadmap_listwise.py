from __future__ import annotations

from boatrace_ai.web_dashboard import _roadmap_improvements, _roadmap_milestones


def remote_jobs(search_status: str, newton_status: str) -> dict:
    return {
        "jobs": [
            {"kind": "feature_teacher_search", "status": search_status},
            {"kind": "newton_listwise_bankroll", "status": newton_status},
        ]
    }


def test_roadmap_m43_tracks_feature_teacher_search() -> None:
    rows = _roadmap_improvements(
        progress={}, processes=[], remote_evaluations=remote_jobs("実行中", "待機中")
    )
    row = next(item for item in rows if item["id"] == "M4-3")
    assert row["status"] == "特徴量・教師探索中"
    assert row["progress"] == 70
    assert "ROI 0.7079" in row["next"]
    assert "Newton-CG" in row["item"]


def test_m4_milestone_does_not_treat_running_newton_as_complete() -> None:
    rows = _roadmap_milestones(
        progress={}, processes=[], remote_evaluations=remote_jobs("完了", "実行中")
    )
    row = next(item for item in rows if item["id"] == "M4")
    assert row["status"] == "listwise再設計を評価中"
    assert row["progress"] == 96
    assert "ROI 1.0" in row["next"]
