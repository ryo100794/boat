from boatrace_ai.feature_schema import (
    FEATURE_SCHEMA_VERSION,
    LEGACY_FEATURE_SCHEMA_VERSION,
)
from boatrace_ai.listwise.backtest import build_parser


def test_backtest_defaults_to_current_feature_schema() -> None:
    assert build_parser().parse_args([]).feature_schema_version == FEATURE_SCHEMA_VERSION


def test_backtest_accepts_legacy_schema_for_controlled_ablation() -> None:
    args = build_parser().parse_args(
        ["--feature-schema-version", LEGACY_FEATURE_SCHEMA_VERSION]
    )

    assert args.feature_schema_version == LEGACY_FEATURE_SCHEMA_VERSION
