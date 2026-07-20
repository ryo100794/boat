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


def install_legacy_model_aliases() -> None:
    """Expose renamed model classes while old joblib artifacts are in service."""
    for old_name, current_name in LEGACY_MODULE_ALIASES.items():
        if old_name not in sys.modules:
            sys.modules[old_name] = importlib.import_module(current_name)


def load_model_bundle(path: str | Path) -> Any:
    install_legacy_model_aliases()
    return joblib.load(path)
