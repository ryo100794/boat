from boatrace_ai.recency_mlp_evaluation import protected_runtime_supported


def test_unblended_candidate_is_runtime_supported() -> None:
    assert protected_runtime_supported(None)
    assert protected_runtime_supported({"candidate_weight": 1.0})


def test_partial_or_baseline_blend_requires_runtime_support() -> None:
    assert not protected_runtime_supported({"candidate_weight": 0.5})
    assert not protected_runtime_supported({"candidate_weight": 0.0})
