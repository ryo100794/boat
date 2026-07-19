from __future__ import annotations

from pathlib import Path

from boatrace_ai.web.dashboard import _evaluation_scope, _fold_report_row


def test_standardized_result_is_labeled_separately() -> None:
    assert (
        _evaluation_scope(
            Path("standardized_365d_no_odds_v8_bankroll.json"),
            [],
        )
        == "standard_365d"
    )


def test_legacy_daily_range_is_explicit() -> None:
    assert _evaluation_scope(
        Path("listwise_newton_cg_v1.json"),
        [
            {"race_date": "2026-05-10"},
            {"race_date": "2026-07-18"},
        ],
    ) == "legacy:2026-05-10:2026-07-18:2"


def test_fold_row_keeps_evaluation_scope() -> None:
    row = _fold_report_row(
        "model",
        {"fold": 1, "train_races": 100, "test_races": 50},
        "standard_365d",
    )
    assert row["evaluation_scope"] == "standard_365d"
