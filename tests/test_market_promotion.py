from __future__ import annotations

import json
from pathlib import Path

from boatrace_ai.listwise.market_calibration import (
    MARKET_EVALUATION_VERSION,
    file_sha256,
)
from boatrace_ai.listwise.market_promotion import (
    promote_best_candidate,
    validate_candidate,
)


def _candidate(tmp_path: Path, name: str, *, profit_yen: int = 20_000) -> Path:
    source = tmp_path / f"{name}.joblib"
    source.write_bytes(f"model:{name}".encode())
    path = tmp_path / f"{name}.json"
    payload = {
        "evaluation_version": MARKET_EVALUATION_VERSION,
        "promotion_eligible": True,
        "promotion_gate": {
            "minimum_evaluation_races": 1000,
            "minimum_evaluation_days": 30,
            "sample_size_pass": True,
            "positive_profit_pass": True,
            "roi_pass": True,
            "fold_stability_pass": True,
            "calibration_pass": True,
            "market_confidence_pass": True,
            "no_lookahead_pass": True,
        },
        "evaluation_races": 1200,
        "evaluation_days": 35,
        "stake_yen": 100_000,
        "return_yen": 100_000 + profit_yen,
        "profit_yen": profit_yen,
        "roi": (100_000 + profit_yen) / 100_000,
        "probability_metrics": {
            "calibrated_trifecta_log_loss": 3.75,
            "calibrated_trifecta_top5_hit_rate": 0.34,
        },
        "source_model": str(source),
        "source_model_sha256": file_sha256(source),
        "source_model_trained_through": ["race", "2026-05-09", "24", 12],
        "from_date": "2026-07-18",
        "through_date": "2026-08-21",
        "deployment_configuration": {
            "role": "next_day_refit_not_evaluation",
            "calibrator_strategy": "newton_residual",
            "trained_through_date": "2026-08-21",
            "training_races": 1400,
            "calibrator": {"converged": True, "model_coefficient": 0.1},
            "selected_policy": {"name": "ev1.05", "no_bet": False},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_promotes_best_verified_candidate_atomically(tmp_path: Path) -> None:
    weaker = _candidate(tmp_path, "weaker", profit_yen=10_000)
    stronger = _candidate(tmp_path, "stronger", profit_yen=25_000)
    output = tmp_path / "active.json"

    result = promote_best_candidate([weaker, stronger], output_path=output)
    manifest = json.loads(output.read_text())

    assert result["status"] == "promoted"
    assert result["selected_candidate_id"] == "stronger"
    assert manifest["selected_candidate_id"] == "stronger"
    assert manifest["valid_from_date"] == "2026-08-22"
    assert manifest["evaluation_sha256"] == file_sha256(stronger)


def test_failed_candidate_preserves_existing_manifest(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path, "failed")
    payload = json.loads(candidate.read_text())
    payload["promotion_gate"]["market_confidence_pass"] = False
    payload["promotion_eligible"] = False
    candidate.write_text(json.dumps(payload))
    output = tmp_path / "active.json"
    output.write_text('{"sentinel":true}', encoding="utf-8")

    result = promote_best_candidate([candidate], output_path=output)

    assert result["status"] == "no_eligible_candidate"
    assert result["manifest_preserved"] is True
    assert output.read_text(encoding="utf-8") == '{"sentinel":true}'


def test_rejects_source_model_hash_mismatch(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path, "tampered")
    payload = json.loads(candidate.read_text())
    Path(payload["source_model"]).write_bytes(b"changed")

    result = validate_candidate(candidate)

    assert result["valid"] is False
    assert "source model SHA-256 mismatch" in result["errors"]


def test_rejects_stale_deployment_refit(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path, "stale")
    payload = json.loads(candidate.read_text())
    payload["deployment_configuration"]["trained_through_date"] = "2026-08-20"
    candidate.write_text(json.dumps(payload))

    result = validate_candidate(candidate)

    assert result["valid"] is False
    assert (
        "deployment configuration is not refit through evaluation end"
        in result["errors"]
    )
