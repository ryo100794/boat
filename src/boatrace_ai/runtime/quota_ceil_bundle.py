from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import joblib

from .quota_ceil_shadow_policy import (
    LEARNED_DAILY_TICKET_LIMIT,
    REGISTERED_AFTER,
    SOURCE_EVALUATION_JOB_ID,
    registration,
)


EXPECTED_MODEL = "odds_path_observed_closing_return_schedule_quota_triple_head_v21"


def _validate_candidate_policy(value: object) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("quota-ceil candidate policy is missing")
    expected = registration()["candidate_policy"]
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"quota-ceil candidate policy differs at {key}")
    control = value.get("v18_ticket_control")
    if not isinstance(control, Mapping):
        raise ValueError("quota-ceil ticket control is missing")
    if (
        control.get("method") != "strict_prior_daily_ticket_lower_quantile"
        or int(control.get("learned_daily_ticket_limit") or 0)
        != LEARNED_DAILY_TICKET_LIMIT
        or control.get("schedule_quota_rounding") != "ceil"
        or control.get("schedule_quota_opportunity") is not None
        or control.get("result_or_payout_fields_used") is not False
    ):
        raise ValueError("quota-ceil ticket control differs from registration")


def build_registered_quota_ceil_bundle(
    source_result: Path, output: Path
) -> dict[str, Any]:
    source = json.loads(source_result.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or source.get("model") != EXPECTED_MODEL:
        raise ValueError("quota-ceil source result identity mismatch")
    if source.get("calibrator_strategy") != EXPECTED_MODEL:
        raise ValueError("quota-ceil calibrator identity mismatch")
    deployment = source.get("deployment_configuration")
    if not isinstance(deployment, dict):
        raise ValueError("quota-ceil deployment configuration is missing")
    if str(deployment.get("trained_through_date") or "") != REGISTERED_AFTER:
        raise ValueError("quota-ceil source training boundary mismatch")
    if deployment.get("real_betting_enabled") is not False:
        raise ValueError("quota-ceil source must disable real betting")
    if deployment.get("selected_policy") != {"name": "no_bet", "no_bet": True}:
        raise ValueError("quota-ceil formal source must retain no-bet")
    triple = deployment.get("triple_head_calibration")
    if not isinstance(triple, Mapping) or triple.get("outer_holdout_used") is not False:
        raise ValueError("quota-ceil source violates the outer information boundary")
    _validate_candidate_policy(deployment.get("candidate_policy"))

    value = copy.deepcopy(deployment)
    value.update(
        {
            "source_evaluation_job_id": SOURCE_EVALUATION_JOB_ID,
            "source_result_sha256": hashlib.sha256(source_result.read_bytes()).hexdigest(),
            "outer_result_or_payout_used": False,
            "real_betting_enabled": False,
            "deployment_mode": "evaluation_only",
            "prospective_policy_registration": registration(),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    joblib.dump({"deployment": value}, temporary, compress=3)
    os.replace(temporary, output)
    return {
        "output": str(output),
        "source_result": str(source_result),
        "source_result_sha256": value["source_result_sha256"],
        "trained_through_date": value["trained_through_date"],
        "registration": value["prospective_policy_registration"],
    }


__all__ = ["EXPECTED_MODEL", "build_registered_quota_ceil_bundle"]
