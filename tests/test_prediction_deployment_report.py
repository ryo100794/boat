from boatrace_ai.evaluation_queue import summarize_result
from boatrace_ai.web.dashboard import MODEL_REPORT_HTML


def test_prediction_deployment_gate_is_summarized_separately() -> None:
    summary = summarize_result(
        {
            "prediction_deployment_eligible": True,
            "deployment_model_artifact_saved": True,
            "prediction_deployment_gate": {
                "prediction_pass": True,
                "conditional_order_converged": True,
                "protected_runtime_supported": True,
                "pass": True,
            },
            "promotion_gate": {
                "prediction_pass": True,
                "payout_policy_pass": False,
                "pass": False,
            },
        }
    )

    assert summary["prediction_deployment_eligible"] is True
    assert summary["deployment_model_artifact_saved"] is True
    assert summary["prediction_deployment_gate_passed"] == 3
    assert summary["prediction_deployment_gate_total"] == 3
    assert summary["prediction_deployment_gate_failed"] == []
    assert "payout_policy_pass" in summary["promotion_gate_failed"]


def test_model_report_labels_prediction_only_readiness() -> None:
    assert "prediction_deployment_eligible" in MODEL_REPORT_HTML
    assert "予測可" in MODEL_REPORT_HTML
    assert "予測未達" in MODEL_REPORT_HTML
