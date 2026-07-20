from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import joblib


LEGACY_MODULE_ALIASES = {
    "boatrace_ai.modeling_no_odds_v6": "boatrace_ai.historical_model",
    "boatrace_ai.model_extended": "boatrace_ai.historical_model",
}
LEGACY_CLASS_ALIASES = {
    "boatrace_ai.web.postgresql_dashboard": {
        "SparseIndex32": ("boatrace_ai.historical_model", "SparseIndex32"),
    },
}


def install_legacy_model_aliases() -> None:
    """Expose renamed model classes while old joblib artifacts are in service."""
    for old_name, current_name in LEGACY_MODULE_ALIASES.items():
        if old_name not in sys.modules:
            sys.modules[old_name] = importlib.import_module(current_name)
    for module_name, aliases in LEGACY_CLASS_ALIASES.items():
        legacy_module = sys.modules.get(module_name)
        if legacy_module is None:
            legacy_module = importlib.import_module(module_name)
        for legacy_name, (source_module, source_name) in aliases.items():
            setattr(
                legacy_module,
                legacy_name,
                getattr(importlib.import_module(source_module), source_name),
            )


def load_model_bundle(path: str | Path) -> Any:
    install_legacy_model_aliases()
    return joblib.load(path)
