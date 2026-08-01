from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import joblib

from .stable_cell_shadow_policy import (
    REGISTERED_AFTER,
    SOURCE_EVALUATION_JOB_ID,
    registration,
)


EXPECTED_MODEL = "odds_path_observed_closing_return_schedule_quota_triple_head_v21"


def build_registered_stable_cell_bundle(
    source_result: Path, output: Path
) -> dict[str, object]:
    source = json.loads(source_result.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or source.get("model") != EXPECTED_MODEL:
        raise ValueError("stable-cell source result identity mismatch")
    if source.get("calibrator_strategy") != EXPECTED_MODEL:
        raise ValueError("stable-cell calibrator identity mismatch")
    deployment = source.get("deployment_configuration")
    if not isinstance(deployment, dict):
        raise ValueError("stable-cell deployment configuration is missing")
    if str(deployment.get("trained_through_date") or "") != REGISTERED_AFTER:
        raise ValueError("stable-cell source training boundary mismatch")
    if deployment.get("real_betting_enabled") is not False:
        raise ValueError("stable-cell source must disable real betting")
    if deployment.get("selected_policy") != {"name": "no_bet", "no_bet": True}:
        raise ValueError("stable-cell formal source must retain no-bet")

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


__all__ = ["EXPECTED_MODEL", "build_registered_stable_cell_bundle"]
