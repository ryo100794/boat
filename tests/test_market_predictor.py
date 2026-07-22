from __future__ import annotations

import json
from datetime import date
from itertools import permutations
from pathlib import Path

import pytest

from boatrace_ai.listwise.market_calibration import file_sha256
from boatrace_ai.listwise.market_promotion import MANIFEST_VERSION
from boatrace_ai.runtime.market_predictor import (
    build_promoted_prediction_rows,
    load_active_manifest,
)


COMBINATIONS = tuple(
    "-".join(map(str, values)) for values in permutations(range(1, 7), 3)
)


def _manifest(tmp_path: Path) -> Path:
    source = tmp_path / "model.joblib"
    evaluation = tmp_path / "evaluation.json"
    source.write_bytes(b"model")
    evaluation.write_text("{}", encoding="utf-8")
    path = tmp_path / "active.json"
    path.write_text(
        json.dumps(
            {
                "manifest_version": MANIFEST_VERSION,
                "status": "active",
                "valid_from_date": "2026-08-22",
                "source_model_path": str(source),
                "source_model_sha256": file_sha256(source),
                "evaluation_path": str(evaluation),
                "evaluation_sha256": file_sha256(evaluation),
                "promotion_gate": {
                    "sample_size_pass": True,
                    "positive_profit_pass": True,
                    "roi_pass": True,
                    "fold_stability_pass": True,
                    "calibration_pass": True,
                    "market_confidence_pass": True,
                    "no_lookahead_pass": True,
                },
                "deployment_configuration": {
                    "trained_through_date": "2026-08-21"
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_manifest_is_inactive_before_valid_date_and_verified_after(tmp_path: Path) -> None:
    path = _manifest(tmp_path)

    assert load_active_manifest(path, race_date=date(2026, 8, 21)) is None
    assert load_active_manifest(path, race_date=date(2026, 8, 22)) is not None


def test_manifest_rejects_tampered_evaluation(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    manifest = json.loads(path.read_text())
    Path(manifest["evaluation_path"]).write_text("changed", encoding="utf-8")

    with pytest.raises(ValueError, match="evaluation hash mismatch"):
        load_active_manifest(path, race_date=date(2026, 8, 22))


def test_promoted_rows_use_market_calibration_and_keep_real_t5_odds() -> None:
    model = {combination: 1.0 / 120.0 for combination in COMBINATIONS}
    odds = {combination: 20.0 + index for index, combination in enumerate(COMBINATIONS)}
    snapshot = {
        "snapshot_id": 42,
        "captured_at": "2026-08-22T10:00:00+09:00",
        "odds_deadline_at": "2026-08-22T10:00:30+09:00",
        "odds": odds,
    }
    deployment = {
        "calibrator": {"model_weight": 0.0, "temperature": 1.0},
        "selected_policy": {"name": "ev1.05", "no_bet": False},
    }

    rows = build_promoted_prediction_rows(
        model,
        snapshot=snapshot,
        deployment=deployment,
    )

    assert len(rows) == 120
    assert sum(row["probability"] for row in rows) == pytest.approx(1.0)
    assert all(row["odds"] == odds[row["combination"]] for row in rows)
    assert all(row["feature_set"] == "promoted_market_t5_v1" for row in rows)
    assert all(row["selected_policy"] == "ev1.05" for row in rows)
