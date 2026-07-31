from scripts.analyze_v23_policy_stability import _score, policy_grid


def test_policy_grid_is_small_and_contains_current_policy() -> None:
    policies = policy_grid()
    assert len(policies) == 81
    assert len({policy["name"] for policy in policies}) == len(policies)
    assert any(
        policy["max_model_rank"] == 5
        and policy["ev_threshold"] == 1.0
        and policy["max_estimated_ev"] == 1.05
        and policy["max_odds"] is None
        for policy in policies
    )


def test_score_rejects_sparse_or_unstable_discovery_results() -> None:
    assert _score({
        "tickets": 99,
        "winning_days": 6,
        "bootstrap_roi_ci95_lower": 2.0,
        "leave_one_day_out_min_roi": 2.0,
        "roi": 2.0,
    })[0] == float("-inf")
    assert _score({
        "tickets": 100,
        "winning_days": 2,
        "bootstrap_roi_ci95_lower": 2.0,
        "leave_one_day_out_min_roi": 2.0,
        "roi": 2.0,
    })[0] == float("-inf")
