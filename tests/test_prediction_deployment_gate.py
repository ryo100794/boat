from boatrace_ai.recency_mlp_evaluation import build_prediction_deployment_gate


def test_prediction_deployment_is_independent_of_payout_policy() -> None:
    gate = build_prediction_deployment_gate(
        {"pass": True},
        {"final_fit": {"success": True}},
        {"candidate_weight": 1.0},
        protected_model_requested=True,
    )

    assert gate == {
        "prediction_pass": True,
        "conditional_order_converged": True,
        "protected_runtime_supported": True,
        "pass": True,
    }


def test_prediction_deployment_rejects_unsupported_partial_blend() -> None:
    gate = build_prediction_deployment_gate(
        {"pass": True},
        {"final_fit": {"success": True}},
        {"candidate_weight": 0.5},
        protected_model_requested=True,
    )

    assert not gate["protected_runtime_supported"]
    assert not gate["pass"]
