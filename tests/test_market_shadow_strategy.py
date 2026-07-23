from argparse import Namespace
from pathlib import Path

from boatrace_ai.listwise.market_calibration import (
    blend_probabilities,
    waiting_walk_forward_result,
)
from boatrace_ai.listwise.market_residual import (
    fit_log_pool_newton,
    log_pool_probabilities,
)
from boatrace_ai.runtime.market_shadow_cycle import build_command


def test_newton_coefficients_map_to_existing_blend_contract() -> None:
    races = [
        {
            "actual_combination": "1-2-3",
            "model_probabilities": {"1-2-3": 0.7, "1-3-2": 0.3},
            "market_probabilities": {"1-2-3": 0.6, "1-3-2": 0.4},
        },
        {
            "actual_combination": "1-3-2",
            "model_probabilities": {"1-2-3": 0.3, "1-3-2": 0.7},
            "market_probabilities": {"1-2-3": 0.5, "1-3-2": 0.5},
        },
    ]
    calibrator = fit_log_pool_newton(races, regularization=1.0)

    direct = log_pool_probabilities(
        races[0]["model_probabilities"],
        races[0]["market_probabilities"],
        model_coefficient=calibrator["model_coefficient"],
        market_coefficient=calibrator["market_coefficient"],
    )
    compatible = blend_probabilities(
        races[0]["model_probabilities"],
        races[0]["market_probabilities"],
        model_weight=calibrator["model_weight"],
        temperature=calibrator["temperature"],
    )

    assert direct.keys() == compatible.keys()
    assert all(abs(direct[key] - compatible[key]) < 1e-12 for key in direct)


def test_waiting_result_and_cycle_preserve_calibrator_strategy() -> None:
    result = waiting_walk_forward_result(
        [],
        dates=[],
        daily_budget_yen=10_000,
        min_calibration_days=2,
        calibrator_strategy="newton_residual",
    )
    args = Namespace(
        db="db",
        model="model",
        output="output",
        from_date="2026-07-20",
        daily_budget_yen=10_000,
        min_calibration_days=2,
        calibrator_strategy="newton_residual",
        max_snapshot_age_seconds=60.0,
    )

    command = build_command(args, through_date="2026-07-22")

    assert result["calibrator_strategy"] == "newton_residual"
    assert command[command.index("--calibrator-strategy") + 1] == "newton_residual"


def test_stagewise_shadow_uses_newton_residual_calibration() -> None:
    script = (
        Path(__file__).parents[1]
        / "scripts"
        / "deployment"
        / "run-boatrace-market-blend-shadow.sh"
    ).read_text(encoding="utf-8")

    assert "--calibrator-strategy newton_residual" in script


def test_conditional_stagewise_shadow_is_isolated_and_strict_t5() -> None:
    root = Path(__file__).parents[1]
    script = (
        root
        / "scripts"
        / "deployment"
        / "run-boatrace-conditional-market-shadow.sh"
    ).read_text(encoding="utf-8")
    supervisor = (
        root
        / "scripts"
        / "deployment"
        / "supervisor-boatrace-conditional-market-shadow.ini"
    ).read_text(encoding="utf-8")

    assert "conditional_stagewise_holdout.joblib" in script
    assert "conditional_stagewise_market_shadow.json" in script
    assert "conditional_stagewise_market_residual.races.joblib" in script
    assert "--calibrator-strategy newton_residual" in script
    assert "--max-snapshot-age-seconds 65" in script
    assert "[program:boatrace-conditional-market-shadow]" in supervisor
    assert "autostart=true" in supervisor
