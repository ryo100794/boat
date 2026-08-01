from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import joblib

from .raw_guard_shadow_policy import (
    LEARNED_DAILY_TICKET_LIMIT,
    MIN_RAW_EV,
    REGISTERED_AFTER,
    SOURCE_EVALUATION_JOB_ID,
    registration,
)


EXPECTED_MODEL = "odds_path_observed_closing_return_schedule_quota_triple_head_v21"


def _validate_source_policy(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("raw-guard source candidate policy is missing")
    expected = registration()["candidate_policy"]
    for key, expected_value in expected.items():
        if key == "min_raw_ev":
            continue
        if value.get(key) != expected_value:
            raise ValueError(f"raw-guard source policy differs at {key}")
    control = value.get("v18_ticket_control")
    if not isinstance(control, Mapping):
        raise ValueError("raw-guard ticket control is missing")
    if (
        int(control.get("learned_daily_ticket_limit") or 0)
        != LEARNED_DAILY_TICKET_LIMIT
        or control.get("schedule_quota_rounding") != "ceil"
        or control.get("schedule_quota_opportunity") is not None
        or control.get("result_or_payout_fields_used") is not False
    ):
        raise ValueError("raw-guard ticket control differs from registration")
    return copy.deepcopy(dict(value))


def _validate_audit(value: object) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("raw-guard replay audit is invalid")
    policy = value.get("fixed_policy")
    bankroll = value.get("chronological_bankroll")
    boundary = value.get("information_boundary")
    if (
        value.get("comparison_role") != "fixed_policy_strict_prior_fold_replay"
        or not isinstance(policy, Mapping)
        or float(policy.get("min_raw_ev") or 0.0) != MIN_RAW_EV
        or not isinstance(bankroll, Mapping)
        or int(bankroll.get("race_days") or 0) != 8
        or int(bankroll.get("evaluated_races") or 0) != 1_242
        or not isinstance(boundary, Mapping)
        or boundary.get("outer_holdout_used_to_fit_or_select_policy") is not False
    ):
        raise ValueError("raw-guard replay audit does not match registration")


def build_registered_raw_guard_bundle(
    source_result: Path,
    audit_result: Path,
    output: Path,
) -> dict[str, Any]:
    source = json.loads(source_result.read_text(encoding="utf-8"))
    audit = json.loads(audit_result.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or source.get("model") != EXPECTED_MODEL:
        raise ValueError("raw-guard source result identity mismatch")
    deployment = source.get("deployment_configuration")
    if not isinstance(deployment, dict):
        raise ValueError("raw-guard deployment configuration is missing")
    if str(deployment.get("trained_through_date") or "") != REGISTERED_AFTER:
        raise ValueError("raw-guard source training boundary mismatch")
    if deployment.get("real_betting_enabled") is not False:
        raise ValueError("raw-guard source must disable real betting")
    if deployment.get("selected_policy") != {"name": "no_bet", "no_bet": True}:
        raise ValueError("raw-guard formal source must retain no-bet")
    triple = deployment.get("triple_head_calibration")
    if not isinstance(triple, Mapping) or triple.get("outer_holdout_used") is not False:
        raise ValueError("raw-guard source violates the outer information boundary")
    candidate = _validate_source_policy(deployment.get("candidate_policy"))
    _validate_audit(audit)
    candidate["min_raw_ev"] = MIN_RAW_EV

    value = copy.deepcopy(deployment)
    value.update(
        {
            "candidate_policy": candidate,
            "source_evaluation_job_id": SOURCE_EVALUATION_JOB_ID,
            "source_result_sha256": hashlib.sha256(source_result.read_bytes()).hexdigest(),
            "replay_audit_sha256": hashlib.sha256(audit_result.read_bytes()).hexdigest(),
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
        "source_result_sha256": value["source_result_sha256"],
        "replay_audit_sha256": value["replay_audit_sha256"],
        "trained_through_date": value["trained_through_date"],
        "registration": value["prospective_policy_registration"],
    }


__all__ = ["EXPECTED_MODEL", "build_registered_raw_guard_bundle"]
