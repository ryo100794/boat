from boatrace_ai.web.dashboard import _model_performance_report_contract


def test_model_performance_contract_exposes_role_specific_stages() -> None:
    report = _model_performance_report_contract(
        {"ready": True},
        [
            {
                "evaluation_group": "t5_formal",
                "formal_evaluation_from": "2026-07-01",
            }
        ],
    )

    assert report["version"] == "model-performance-v7"
    assert [stage["id"] for stage in report["stages"]] == [
        "outcome_prediction",
        "closing_odds",
        "purchase_policy",
        "promotion",
        "operational_evidence",
    ]
    assert report["stages"][1]["teacher"] == "締切直前の公式確定オッズ"
    assert "ROI片側95%下限" in report["stages"][2]["metrics"]
    assert "V_buy" in report["stages"][2]["metrics"]
    assert report["metric_definitions"]["formal_roi_gate"].startswith(
        "inverted_cdf"
    )
    assert report["metric_definitions"]["roi"].endswith("購入なしは未評価")
    assert report["groups"][1]["formal_from"] == "2026-07-01"
