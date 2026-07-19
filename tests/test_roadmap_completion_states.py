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
    assert improvements["M4-2"]["progress"] == 100
    assert improvements["M4-2"]["status"] == "評価完了/安定根拠なし"
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

    assert milestones["M4"]["status"] == "評価完了/主系維持"
    assert milestones["M4"]["progress"] == 100
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


def test_versioned_inventory_scans_nested_packages(tmp_path) -> None:
    nested = tmp_path / "runtime"
    nested.mkdir()
    (nested / "worker_v2.py").write_text("", encoding="utf-8")

    inventory = _v_file_inventory(tmp_path)

    assert inventory["count"] == 1
    assert inventory["sample"] == [str(Path("runtime") / "worker_v2.py")]
