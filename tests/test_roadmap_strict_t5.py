import json

from boatrace_ai.web.dashboard import (
    _quality_gates,
    _roadmap_improvements,
    _roadmap_milestones,
)


STRICT_PROGRESS = {
    "realtime": {
        "eligible_races": 756,
        "target_eligible_races": 1000,
        "readiness": 0.756,
    },
    "realtime_shadow_evaluation": {
        "available": True,
        "evaluated_races": 147,
        "state": {
            "eligible_races": 263,
            "required_races": 450,
            "readiness": 263 / 450,
            "baseline_reset_reason": "data_quality_contract_changed",
        },
    },
}


def test_dynamic_roadmap_prefers_strict_t5_population() -> None:
    processes = [
        {"kind": "リアルタイムshadow"},
        {"kind": "予測ループ"},
    ]

    milestones = {
        row["id"]: row
        for row in _roadmap_milestones(STRICT_PROGRESS, processes, {})
    }
    improvements = {
        row["id"]: row
        for row in _roadmap_improvements(STRICT_PROGRESS, processes, {})
    }

    assert milestones["M5"]["progress"] == 50
    assert "厳格T-5品質R 263/450" in milestones["M5"]["next"]
    assert improvements["M5-1"]["progress"] == 58
    assert "263/450" in improvements["M5-1"]["next"]
    assert improvements["M2-4"]["status"] == "完了/運用監視"
    assert improvements["M2-4"]["progress"] == 100


def test_strict_t5_zero_does_not_fall_back_to_legacy_population() -> None:
    progress = {
        **STRICT_PROGRESS,
        "realtime_shadow_evaluation": {
            "state": {
                "eligible_races": 0,
                "required_races": 450,
                "readiness": 0.0,
            }
        },
    }

    milestone = next(
        row
        for row in _roadmap_milestones(
            progress,
            [{"kind": "リアルタイムshadow"}],
            {},
        )
        if row["id"] == "M5"
    )

    assert milestone["progress"] == 10
    assert "厳格T-5品質R 0/450" in milestone["next"]


def test_quality_gates_expose_market_confidence_interval(tmp_path) -> None:
    (tmp_path / "listwise_market_residual_shadow.json").write_text(
        json.dumps(
            {
                "status": "waiting_for_clean_evaluation_day",
                "market_comparison": {
                    "evaluation_races": 0,
                    "confidence_pass": False,
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "listwise_market_residual_intraday_bootstrap.json").write_text(
        json.dumps(
            {
                "confidence_pass": False,
                "log_loss_difference_calibrated_minus_market": {
                    "observations": 113,
                    "mean_difference": -0.02843,
                    "ci95_lower": -0.05858,
                    "ci95_upper": 0.00212,
                    "probability_less_than_zero": 0.966,
                },
                "top5_hit_difference_calibrated_minus_market": {
                    "observations": 113,
                    "mean_difference": 0.00885,
                    "ci95_lower": 0.0,
                    "ci95_upper": 0.02655,
                },
            }
        ),
        encoding="utf-8",
    )

    gate = next(
        row
        for row in _quality_gates(tmp_path, {"jobs": []})
        if row["target"] == "M6 T-5市場信頼区間"
    )

    assert gate["status"] == "未達"
    assert "paired 113R" in gate["evidence"]
    assert "+0.00212" in gate["evidence"]
