import json
from pathlib import Path

import joblib
import pytest

from boatrace_ai.runtime.stable_cell_bundle import build_registered_stable_cell_bundle


MODEL = "odds_path_observed_closing_return_schedule_quota_triple_head_v21"


def _source(path: Path, *, trained_through: str = "2026-08-01") -> None:
    path.write_text(
        json.dumps(
            {
                "model": MODEL,
                "calibrator_strategy": MODEL,
                "deployment_configuration": {
                    "trained_through_date": trained_through,
                    "real_betting_enabled": False,
                    "selected_policy": {"name": "no_bet", "no_bet": True},
                },
            }
        )
    )


def test_builder_persists_fixed_registration_and_no_bet_boundary(tmp_path: Path) -> None:
    source = tmp_path / "job-00010730.json"
    output = tmp_path / "stable.joblib"
    _source(source)

    result = build_registered_stable_cell_bundle(source, output)
    deployment = joblib.load(output)["deployment"]

    assert result["trained_through_date"] == "2026-08-01"
    assert deployment["source_evaluation_job_id"] == 10_730
    assert deployment["real_betting_enabled"] is False
    assert deployment["prospective_policy_registration"][
        "promotion_evidence_start_date"
    ] == "2026-08-02"


def test_builder_rejects_same_day_or_future_training(tmp_path: Path) -> None:
    source = tmp_path / "job-00010730.json"
    _source(source, trained_through="2026-08-02")

    with pytest.raises(ValueError, match="training boundary"):
        build_registered_stable_cell_bundle(source, tmp_path / "stable.joblib")
