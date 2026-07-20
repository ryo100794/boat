from boatrace_ai.web.dashboard import _roadmap_improvements, _roadmap_milestones


MODEL_IDS = (
    "no_odds_v8",
    "pastlog_v7",
    "pastlog_v9_research",
    "calibrated_linear",
    "calibrated_mlp",
    "listwise_feature_teacher",
    "listwise_newton",
)


def v2_jobs(*, missing: str | None = None) -> dict:
    jobs = []
    for index, model_id in enumerate(MODEL_IDS):
        if model_id == missing:
            continue
        jobs.append(
            {
                "kind": "standardized_365d_v2_model",
                "name": f"standardized_365d_v2_{model_id}",
                "status": "完了",
                "result": {
                    "metrics": {
                        "roi": 0.80 + index / 100,
                        "profit_yen": -10_000 + index * 1_000,
                        "winner_top1_accuracy": 0.56,
                        "trifecta_top5_hit_rate": 0.31,
                    }
                },
            }
        )
    jobs.extend(
        [
            {
                "kind": "newton_listwise_bankroll",
                "status": "完了",
                "result": {"metrics": {"promotion_eligible": False, "roi": 0.8}},
            },
            {"kind": "bankroll_temporal_no_bet", "status": "完了"},
        ]
    )
    return {"jobs": jobs}


def test_all_six_v2_models_complete_unified_evaluation() -> None:
    improvements = {
        row["id"]: row for row in _roadmap_improvements({}, [], v2_jobs())
    }
    milestones = {
        row["id"]: row for row in _roadmap_milestones({}, [], v2_jobs())
    }

    assert improvements["M4-4"]["status"] == "完了"
    assert improvements["M4-4"]["progress"] == 100
    assert milestones["M4"]["status"] == "評価完了/主系維持"


def test_missing_v2_model_keeps_unified_evaluation_open() -> None:
    improvements = {
        row["id"]: row
        for row in _roadmap_improvements({}, [], v2_jobs(missing="calibrated_mlp"))
    }
    milestones = {
        row["id"]: row
        for row in _roadmap_milestones({}, [], v2_jobs(missing="calibrated_mlp"))
    }

    assert improvements["M4-4"]["status"] == "評価中"
    assert improvements["M4-4"]["progress"] == 85
    assert milestones["M4"]["status"] == "統一再評価中"
