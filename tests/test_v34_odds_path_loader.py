from datetime import datetime, timedelta, timezone

from boatrace_ai.listwise.four_head_v22_evaluation import (
    COMBINATIONS,
    T5_ODDS_PATH_LEADS,
    _build_decision_odds_path,
)


def test_loader_builds_only_predecision_snapshots_in_ticket_order() -> None:
    t300 = datetime(2026, 8, 1, 3, 0, tzinfo=timezone.utc)
    snapshots = {}
    for offset, lead in T5_ODDS_PATH_LEADS:
        target = t300 - timedelta(seconds=offset - 300)
        snapshots[lead] = {
            "captured_at": (target - timedelta(seconds=5)).isoformat(),
            "odds_deadline_at": target.isoformat(),
            "odds": {
                combination: 20.0 + index / 100
                for index, combination in enumerate(COMBINATIONS)
            },
        }

    current = tuple(snapshots[5]["odds"][key] for key in COMBINATIONS)
    path = _build_decision_odds_path(
        snapshots, current_odds=current, max_snapshot_age_seconds=300.0
    )

    assert tuple(point.target_offset_seconds for point in path.snapshots) == (
        1800,
        1200,
        600,
        420,
        300,
    )
    assert path.snapshots[-1].odds == current
