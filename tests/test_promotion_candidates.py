import json
from pathlib import Path

from boatrace_ai.listwise.market_calibration import MARKET_EVALUATION_VERSION
from boatrace_ai.runtime.promotion_candidates import (
    discover_market_evaluation_candidates,
)


def _write(path: Path, **overrides) -> None:
    payload = {
        "evaluation_version": MARKET_EVALUATION_VERSION,
        "promotion_gate": {"pass": False},
        "deployment_configuration": {"role": "next_day_refit_not_evaluation"},
        "source_model": "/models/source.joblib",
        "source_model_sha256": "a" * 64,
        **overrides,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_discovers_latest_complete_market_result_per_source(tmp_path: Path) -> None:
    _write(tmp_path / "job-00000001.json")
    _write(tmp_path / "job-00000002.json")
    _write(
        tmp_path / "job-00000003.json",
        source_model="/models/other.joblib",
        source_model_sha256="b" * 64,
    )
    _write(
        tmp_path / "job-00000004.json",
        evaluation_version=MARKET_EVALUATION_VERSION - 1,
    )
    (tmp_path / "job-00000005.json").write_text("bad", encoding="utf-8")

    candidates = discover_market_evaluation_candidates(tmp_path)

    assert candidates == [
        str(tmp_path / "job-00000003.json"),
        str(tmp_path / "job-00000002.json"),
    ]


def test_discovery_requires_market_deployment_contract(tmp_path: Path) -> None:
    _write(tmp_path / "job-00000001.json", promotion_gate=None)
    _write(tmp_path / "job-00000002.json", deployment_configuration=None)
    _write(tmp_path / "job-00000003.json", source_model_sha256="")

    assert discover_market_evaluation_candidates(tmp_path) == []
