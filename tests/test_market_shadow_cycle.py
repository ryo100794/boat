from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from boatrace_ai.runtime.market_shadow_cycle import (
    completed_through_date,
    evaluation_due,
    evaluation_slot,
)


def test_completed_through_date_uses_previous_jst_day() -> None:
    assert completed_through_date(
        datetime(2026, 7, 21, 14, 59, tzinfo=timezone.utc)
    ) == "2026-07-20"
    assert completed_through_date(
        datetime(2026, 7, 21, 15, 1, tzinfo=timezone.utc)
    ) == "2026-07-21"


def test_evaluation_due_only_for_new_completed_day() -> None:
    assert evaluation_due(
        {"status": "error", "completed_through_date": "2026-07-21"},
        through_date="2026-07-21",
        output_exists=True,
    )
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
        "odds_data_signature": {"snapshot_count": 1},
    }
    assert not evaluation_due(
        state,
        through_date="2026-07-21",
        output_exists=True,
        model_sha256="a" * 64,
        evaluation_version=5,
        odds_signature={"snapshot_count": 1},
    )
    assert evaluation_due(
        state,
        through_date="2026-07-21",
        output_exists=True,
        model_sha256="b" * 64,
        evaluation_version=5,
        odds_signature={"snapshot_count": 1},
    )
    assert evaluation_due(
        state,
        through_date="2026-07-21",
        output_exists=True,
        model_sha256="a" * 64,
        evaluation_version=6,
        odds_signature={"snapshot_count": 1},
    )
    assert evaluation_due(
        state,
        through_date="2026-07-21",
        output_exists=True,
        model_sha256="a" * 64,
        evaluation_version=5,
        odds_signature={"snapshot_count": 2},
    )


def test_evaluation_slots_bound_cross_process_work(tmp_path: Path) -> None:
    lock_dir = tmp_path / "slots"
    with evaluation_slot(lock_dir, 2) as first:
        with evaluation_slot(lock_dir, 2) as second:
            with evaluation_slot(lock_dir, 2) as unavailable:
                assert (first, second, unavailable) == (0, 1, None)
    with evaluation_slot(lock_dir, 2) as released:
        assert released == 0
