from __future__ import annotations

import json

from boatrace_ai.standard_evaluation import (
    MODEL_SOURCES,
    POLICY,
    PROTOCOL_ID,
    protocol_sha256,
)
from boatrace_ai.web.dashboard import (
    _load_standardized_v2_bundle,
    _merge_standardized_v2_status,
)


def _write_bundle(model_dir, *, protocol_hash_override: str | None = None):
    root = model_dir / "standardized_365d_v2"
    root.mkdir(parents=True)
    expected_ids = [source.model_id for source in MODEL_SOURCES]
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "holdout_start": "2025-07-20",
        "holdout_end": "2026-07-19",
        "prediction_races": 48_437,
        "bankroll_evaluable_races": 48_437,
        "policy": dict(POLICY),
        "comparison_ready": True,
        "comparison_model_ids": expected_ids,
        "valid_model_count": len(expected_ids),
        "failed_models": [],
        "promotion_decision": {
            "status": "retain_incumbent",
            "selected_model_id": "no_odds_v8",
        },
        "models": [
            {"model_id": model_id, "validation": {"passed": True}}
            for model_id in expected_ids
        ],
    }
    manifest["protocol_sha256"] = protocol_sha256(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    protocol = dict(manifest)
    protocol["protocol_sha256"] = (
        protocol_hash_override or manifest["protocol_sha256"]
    )
    (root / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    for model_id in expected_ids:
        (root / f"{model_id}.json").write_text(
            json.dumps(
                {
                    "model_id": model_id,
                    "protocol_id": PROTOCOL_ID,
                    "protocol_sha256": manifest["protocol_sha256"],
                    "policy": dict(POLICY),
                    "validation": {"passed": True},
                    "evaluated_races": 48_437,
                    "roi": 0.8,
                    "stake_yen": 1000,
                    "daily": [],
                }
            ),
            encoding="utf-8",
        )
    return manifest


def test_report_accepts_only_complete_current_protocol(tmp_path) -> None:
    manifest = _write_bundle(tmp_path)

    bundle = _load_standardized_v2_bundle(tmp_path)
    merged = _merge_standardized_v2_status({"jobs": []}, bundle)

    assert bundle["ready"] is True
    assert len(bundle["models"]) == len(MODEL_SOURCES)
    assert {
        row["name"]
        for row in merged["jobs"]
        if row["kind"] == "standardized_365d_v2_model"
    } == {
        f"standardized_365d_v2_{source.model_id}"
        for source in MODEL_SOURCES
    }
    assert merged["generated_at"] == manifest.get("generated_at")


def test_report_rejects_stale_manifest_for_new_protocol(tmp_path) -> None:
    _write_bundle(tmp_path, protocol_hash_override="new-protocol")

    bundle = _load_standardized_v2_bundle(tmp_path)
    merged = _merge_standardized_v2_status(
        {
            "jobs": [
                {
                    "kind": "standardized_365d_v2_model",
                    "name": "standardized_365d_v2_no_odds_v8",
                    "status": "完了",
                }
            ]
        },
        bundle,
    )

    assert bundle["ready"] is False
    assert "current protocol is not consolidated" in bundle["errors"]
    assert not any(
        row["kind"] == "standardized_365d_v2_model"
        for row in merged["jobs"]
    )
    assert any(
        row["kind"] == "standardized_365d_v2_queue"
        and row["status"] == "実行中"
        for row in merged["jobs"]
    )
