from __future__ import annotations

from typing import Any


POLICY_NAME = "preregistered_v21_quota13_ceil_raw_ev095_20260802"
STRATEGY_NAME = "raw_guard_v21_t300"
MODEL_KEY = "raw_guard_daily"
REGISTERED_AFTER = "2026-07-31"
PROMOTION_EVIDENCE_START_DATE = "2026-08-02"
SOURCE_EVALUATION_JOB_ID = 9_906
AUDIT_EVALUATION_JOB_ID = 10_730
LEARNED_DAILY_TICKET_LIMIT = 13
MIN_RAW_EV = 0.95


def registration() -> dict[str, Any]:
    return {
        "policy_name": POLICY_NAME,
        "model_key": MODEL_KEY,
        "strategy_name": STRATEGY_NAME,
        "registered_after": REGISTERED_AFTER,
        "source_evaluation_job_id": SOURCE_EVALUATION_JOB_ID,
        "audit_evaluation_job_id": AUDIT_EVALUATION_JOB_ID,
        "candidate_policy": {
            "ev_threshold": 1.5,
            "min_raw_ev": MIN_RAW_EV,
            "max_odds": 40.0,
            "max_tickets_per_race": 1,
            "min_model_market_ratio": 1.2,
            "staking_mode": "kelly_100",
        },
        "ticket_control": {
            "method": "strict_prior_daily_ticket_lower_quantile",
            "learned_daily_ticket_limit": LEARNED_DAILY_TICKET_LIMIT,
            "schedule_quota_rounding": "ceil",
            "schedule_quota_opportunity": None,
        },
        "development_holdout_used_to_choose_policy": True,
        "promotion_evidence_start_date": PROMOTION_EVIDENCE_START_DATE,
        "real_betting_enabled": False,
    }


__all__ = [
    "AUDIT_EVALUATION_JOB_ID",
    "LEARNED_DAILY_TICKET_LIMIT",
    "MIN_RAW_EV",
    "MODEL_KEY",
    "POLICY_NAME",
    "PROMOTION_EVIDENCE_START_DATE",
    "REGISTERED_AFTER",
    "SOURCE_EVALUATION_JOB_ID",
    "STRATEGY_NAME",
    "registration",
]
