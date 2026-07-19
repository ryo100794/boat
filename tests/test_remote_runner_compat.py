from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "remote_best_roi_attribution_runner.py"
SPEC = importlib.util.spec_from_file_location("remote_runner_compat", SCRIPT)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_unsupported_policy_arguments_are_dropped() -> None:
    original = RUNNER.adaptive_bankroll_streaming

    def compatible(*, daily_budget_yen: int, ev_threshold: float) -> None:
        return None

    try:
        RUNNER.adaptive_bankroll_streaming = compatible
        kwargs, dropped = RUNNER.supported_policy_kwargs(
            {
                "daily_budget_yen": 10_000,
                "ev_threshold": 1.5,
                "require_real_odds": False,
                "allocation_mode": "normalized_kelly",
            }
        )
    finally:
        RUNNER.adaptive_bankroll_streaming = original

    assert kwargs == {"daily_budget_yen": 10_000, "ev_threshold": 1.5}
    assert dropped == ["allocation_mode", "require_real_odds"]
