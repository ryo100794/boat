from datetime import datetime, timedelta, timezone

import pytest

from boatrace_ai.listwise.four_head_nested_v22 import (
    DecisionOddsPath,
    DecisionOddsPoint,
    T5_ODDS_PATH_SCHEMA,
    validate_decision_odds_path,
)


def _point(offset: int, value: float = 20.0) -> DecisionOddsPoint:
    t300 = datetime(2026, 8, 1, 3, 0, tzinfo=timezone.utc)
    target = t300 - timedelta(seconds=offset - 300)
    return DecisionOddsPoint(
        target_offset_seconds=offset,
        captured_at=(target - timedelta(seconds=10)).isoformat(),
        target_at=target.isoformat(),
        odds=(value,) * 120,
    )


def test_t5_odds_path_accepts_canonical_predecision_points() -> None:
    path = DecisionOddsPath(
        schema=T5_ODDS_PATH_SCHEMA,
        snapshots=tuple(_point(offset) for offset in (1800, 1200, 600, 420, 300)),
    )

    indexed = validate_decision_odds_path(
        path, choices=120, expected_t300_odds=(20.0,) * 120
    )

    assert tuple(indexed) == (1800, 1200, 600, 420, 300)


@pytest.mark.parametrize("offset", [120, 60, 30, 10, 0, 300.5, True])
def test_t5_odds_path_rejects_postdecision_or_noninteger_offsets(offset) -> None:
    path = DecisionOddsPath(
        schema=T5_ODDS_PATH_SCHEMA,
        snapshots=(_point(1800), _point(300)),
    )
    invalid = DecisionOddsPoint(
        target_offset_seconds=offset,
        captured_at=path.snapshots[-1].captured_at,
        target_at=path.snapshots[-1].target_at,
        odds=(20.0,) * 120,
    )

    with pytest.raises(ValueError):
        validate_decision_odds_path(
            DecisionOddsPath(T5_ODDS_PATH_SCHEMA, (invalid,)), choices=120
        )


def test_t5_odds_path_requires_t300_and_exact_current_odds() -> None:
    with pytest.raises(ValueError, match="requires the T300"):
        validate_decision_odds_path(
            DecisionOddsPath(T5_ODDS_PATH_SCHEMA, (_point(420),)), choices=120
        )
    with pytest.raises(ValueError, match="equal decision current_odds"):
        validate_decision_odds_path(
            DecisionOddsPath(T5_ODDS_PATH_SCHEMA, (_point(300),)),
            choices=120,
            expected_t300_odds=(21.0,) * 120,
        )
