from __future__ import annotations

from datetime import datetime, timezone

from boatrace_ai.runtime.market_shadow_cycle import (
    completed_through_date,
    evaluation_due,
)


def test_completed_through_date_uses_previous_jst_day() -> None:
    assert completed_through_date(
        datetime(2026, 7, 21, 14, 59, tzinfo=timezone.utc)
    ) == "2026-07-20"
    assert completed_through_date(
        datetime(2026, 7, 21, 15, 1, tzinfo=timezone.utc)
    ) == "2026-07-21"


def test_evaluation_due_only_for_new_completed_day() -> None:
    assert evaluation_due({}, through_date="2026-07-21", output_exists=False)
    assert evaluation_due(
        {"completed_through_date": "2026-07-20"},
        through_date="2026-07-21",
        output_exists=True,
    )
    assert not evaluation_due(
        {"completed_through_date": "2026-07-21"},
        through_date="2026-07-21",
        output_exists=True,
    )


def test_evaluation_due_when_model_artifact_changes_same_day() -> None:
    state = {
        "completed_through_date": "2026-07-21",
        "model_sha256": "a" * 64,
        "evaluation_version": 5,
    }
    assert not evaluation_due(
        state,
        through_date="2026-07-21",
        output_exists=True,
        model_sha256="a" * 64,
        evaluation_version=5,
    )
    assert evaluation_due(
        state,
        through_date="2026-07-21",
        output_exists=True,
        model_sha256="b" * 64,
        evaluation_version=5,
    )
    assert evaluation_due(
        state,
        through_date="2026-07-21",
        output_exists=True,
        model_sha256="a" * 64,
        evaluation_version=6,
    )
