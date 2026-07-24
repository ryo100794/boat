import builtins

from boatrace_ai.feature_schema import (
    FEATURE_SCHEMA_VERSION,
    LEGACY_FEATURE_SCHEMA_VERSION,
    SPARSE_MISSING_FEATURE_SCHEMA_VERSION,
)
from boatrace_ai.listwise import backtest
from boatrace_ai.listwise.backtest import build_parser


def test_backtest_defaults_to_current_feature_schema() -> None:
    assert build_parser().parse_args([]).feature_schema_version == FEATURE_SCHEMA_VERSION


def test_backtest_accepts_legacy_schema_for_controlled_ablation() -> None:
    args = build_parser().parse_args(
        ["--feature-schema-version", LEGACY_FEATURE_SCHEMA_VERSION]
    )

    assert args.feature_schema_version == LEGACY_FEATURE_SCHEMA_VERSION


def test_backtest_output_disconnect_does_not_abort_evaluation(monkeypatch) -> None:
    calls = 0

    def broken_print(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise BrokenPipeError

    monkeypatch.setattr(builtins, "print", broken_print)
    monkeypatch.setattr(backtest, "_OUTPUT_AVAILABLE", True)
    backtest.emit_json({"fold": 1})
    backtest.emit_json({"fold": 2})

    assert calls == 1


def test_v3_schema_remains_selectable_for_reproducibility() -> None:
    args = build_parser().parse_args(
        ["--feature-schema-version", SPARSE_MISSING_FEATURE_SCHEMA_VERSION]
    )
    assert args.feature_schema_version == SPARSE_MISSING_FEATURE_SCHEMA_VERSION
