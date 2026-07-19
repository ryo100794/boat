from __future__ import annotations

from pathlib import Path

from boatrace_ai.legacy_model_aliases import load_model_bundle


def test_current_operational_artifact_loads_after_module_migration() -> None:
    path = Path("data/models/win_model_no_odds_v8.joblib")
    if not path.exists():
        return
    bundle = load_model_bundle(path)
    assert "pipeline" in bundle
    assert "metadata" in bundle
