from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType

import boatrace_ai.web.postgresql_dashboard as postgresql_dashboard
from boatrace_ai.historical_model import SparseIndex32
from boatrace_ai.legacy_model_aliases import (
    install_legacy_model_aliases,
    load_model_bundle,
)


def test_current_operational_artifact_loads_after_module_migration() -> None:
    path = Path("data/models/win_model_no_odds_v8.joblib")
    if not path.exists():
        return
    bundle = load_model_bundle(path)
    assert "pipeline" in bundle
    assert "metadata" in bundle


def test_installs_sparse_transformer_on_legacy_dashboard_module() -> None:
    install_legacy_model_aliases()

    assert postgresql_dashboard.SparseIndex32 is SparseIndex32


def test_installs_sparse_transformer_on_module_runtime_main(monkeypatch) -> None:
    runtime_main = ModuleType("__main__")
    runtime_main.__file__ = "/workspace/boat/src/boatrace_ai/web/postgresql_dashboard.py"
    monkeypatch.setitem(sys.modules, "__main__", runtime_main)

    install_legacy_model_aliases()

    assert runtime_main.SparseIndex32 is SparseIndex32
