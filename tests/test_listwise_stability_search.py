from __future__ import annotations

from boatrace_ai.listwise_stability_search import candidate_key, summarize_candidate


def fold(top1: float, loss: float, top5: float = 0.31) -> dict:
    return {
        "feature_variant": "full",
        "drop_feature_groups": [],
        "target": "top3_pl",
        "alpha": 0.0001,
        "winner_top1_accuracy": top1,
        "ranking_log_loss": loss,
        "trifecta_top5_hit_rate": top5,
    }


def test_stability_summary_penalizes_dispersion_and_checks_worst_fold() -> None:
    stable = summarize_candidate(
        [fold(0.565, 1.30), fold(0.562, 1.31), fold(0.568, 1.29)],
        baseline_top1=0.5642,
    )
    unstable = summarize_candidate(
        [fold(0.60, 1.20), fold(0.53, 1.40), fold(0.57, 1.30)],
        baseline_top1=0.5642,
    )
    assert stable["top1_stability_constraint_pass"] is True
    assert unstable["top1_stability_constraint_pass"] is False
    assert stable["stability_score"] < unstable["stability_score"]


def test_candidate_key_keeps_feature_teacher_and_regularization_distinct() -> None:
    row = fold(0.56, 1.3)
    assert candidate_key(row) == ("full", "top3_pl", 0.0001)
