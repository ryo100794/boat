from boatrace_ai.runtime.raw_guard_shadow_policy import registration


def test_raw_guard_is_shadow_only_and_fully_unseen_from_august_second() -> None:
    value = registration()
    assert value["candidate_policy"]["min_raw_ev"] == 0.95
    assert value["ticket_control"]["learned_daily_ticket_limit"] == 13
    assert value["ticket_control"]["schedule_quota_rounding"] == "ceil"
    assert value["ticket_control"]["schedule_quota_opportunity"] is None
    assert value["promotion_evidence_start_date"] == "2026-08-02"
    assert value["development_holdout_used_to_choose_policy"] is True
    assert value["real_betting_enabled"] is False
