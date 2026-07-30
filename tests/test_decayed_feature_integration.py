from __future__ import annotations

from typing import Any

import pytest

from boatrace_ai import feature_tuning


def _race(
    race_id: str,
    race_date: str,
    rno: int,
    *,
    winner_lane: int,
) -> list[dict[str, Any]]:
    return [
        {
            "race_id": race_id,
            "race_date": race_date,
            "jcd": "01",
            "rno": rno,
            "lane": lane,
            "rank": 1 if lane == winner_lane else lane + 1,
            "racer_no": 1000 + lane,
            "motor_no": 10 + lane,
            "boat_no": 20 + lane,
            "result_start_timing": 0.10 + lane / 100.0,
        }
        for lane in range(1, 7)
    ]


def _collect(
    monkeypatch: pytest.MonkeyPatch,
    *,
    include_decayed_history: bool | None,
):
    races = [
        _race("20260101-01-01", "2026-01-01", 1, winner_lane=1),
        _race("20260101-01-02", "2026-01-01", 2, winner_lane=2),
        _race("20260102-01-01", "2026-01-02", 1, winner_lane=1),
    ]
    monkeypatch.setattr(
        feature_tuning,
        "iter_complete_races",
        lambda _conn, *, feature_schema_version: iter(races),
    )
    kwargs: dict[str, Any] = {
        "drop_feature_groups": feature_tuning.FEATURE_GROUPS,
    }
    if include_decayed_history is not None:
        kwargs["include_decayed_history"] = include_decayed_history
    return list(feature_tuning.iter_race_feature_rows(object(), **kwargs))


def test_disabled_decayed_history_preserves_existing_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    implicit = _collect(monkeypatch, include_decayed_history=None)
    explicit = _collect(monkeypatch, include_decayed_history=False)

    assert implicit == explicit
    assert all(
        not key.startswith("decayed_")
        for race in explicit
        for row in race
        for key in row["features"]
    )


def test_decayed_history_updates_only_after_the_whole_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _collect(monkeypatch, include_decayed_history=True)

    assert [race[0]["meta"]["race_id"] for race in rows] == [
        "20260101-01-01",
        "20260101-01-02",
        "20260102-01-01",
    ]
    assert [[row["meta"]["lane"] for row in race] for race in rows] == [
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6],
    ]

    first_race_lane1 = rows[0][0]["features"]
    same_day_later_lane1 = rows[1][0]["features"]
    next_day_lane1 = rows[2][0]["features"]

    assert first_race_lane1["decayed_racer_30d_effective_count"] == 0.0
    assert first_race_lane1["decayed_racer_30d_has_recent_history"] == 0.0
    assert same_day_later_lane1["decayed_racer_30d_effective_count"] == 0.0
    assert same_day_later_lane1["decayed_racer_30d_has_recent_history"] == 0.0

    expected_count = 2.0 * 2.0 ** (-1.0 / 30.0)
    assert next_day_lane1["decayed_racer_30d_effective_count"] == pytest.approx(
        expected_count
    )
    assert next_day_lane1["decayed_racer_30d_has_recent_history"] == 1.0
    assert next_day_lane1["decayed_racer_30d_win_rate_s"] > 1.0 / 6.0
