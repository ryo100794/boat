from __future__ import annotations

import pytest

from boatrace_ai.listwise.paired_bootstrap import paired_mean_bootstrap


def test_paired_bootstrap_detects_consistent_improvement() -> None:
    result = paired_mean_bootstrap(
        [-0.2, -0.1, -0.3, -0.15] * 20,
        samples=2_000,
        seed=7,
    )
    assert result["mean_difference"] == pytest.approx(-0.1875)
    assert result["ci95_upper"] < 0.0
    assert result["probability_less_than_zero"] == 1.0


def test_paired_bootstrap_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        paired_mean_bootstrap([])
    with pytest.raises(ValueError, match="at least 100"):
        paired_mean_bootstrap([1.0], samples=10)
