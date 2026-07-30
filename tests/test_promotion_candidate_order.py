import json
import os
from pathlib import Path

from boatrace_ai.listwise.market_calibration import MARKET_EVALUATION_VERSION
from boatrace_ai.runtime.promotion_candidates import (
    discover_market_evaluation_candidates,
)


def _write(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "evaluation_version": MARKET_EVALUATION_VERSION,
                "promotion_gate": {"pass": False},
                "deployment_configuration": {
                    "role": "next_day_refit_not_evaluation"
                },
                "source_model": "/models/source.joblib",
                "source_model_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )


def test_latest_job_id_wins_even_when_older_job_has_newer_mtime(
    tmp_path: Path,
) -> None:
    older = tmp_path / "job-00000009.json"
    latest = tmp_path / "job-00000010.json"
    _write(older)
    _write(latest)
    os.utime(older, (latest.stat().st_mtime + 60, latest.stat().st_mtime + 60))

    assert discover_market_evaluation_candidates(tmp_path) == [str(latest)]
