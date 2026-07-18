from __future__ import annotations

import sqlite3
from collections import defaultdict
from typing import Any

from .features import _num, entry_features
from .features_no_odds_v3 import _group_by_race, race_relative_features
from .features_no_odds_v9 import RollingState, _history_groups_before, _race_context_features, _race_sort_key, load_race_entries


LIVE_ONLY_FEATURES = {
    "before_weight_kg",
    "exhibition_time",
    "tilt",
    "adjusted_weight",
    "course",
    "start_timing",
    "wind_speed_m",
    "air_temp_c",
    "water_temp_c",
    "wave_cm",
    "weather",
    "wind_direction",
    "propeller",
    "parts_exchange",
    "has_weather",
    "has_wind_direction",
    "has_propeller",
    "has_parts_exchange",
    "has_beforeinfo",
}
LIVE_ONLY_ROOTS = (
    "before_weight_kg",
    "exhibition_time",
    "start_timing",
    "wind_speed_m",
    "wave_cm",
)
LIVE_ONLY_SUFFIXES = (
    "_rank",
    "_vs_mean",
    "_z",
    "_best_gap",
    "_scaled",
)


def load_training_examples(
    conn: sqlite3.Connection,
    *,
    through_date: str | None = None,
    from_date: str | None = None,
    include_odds: bool = False,
) -> tuple[list[dict[str, Any]], list[int], list[dict[str, Any]]]:
    if include_odds:
        raise ValueError("features_pastlog_v1 does not use odds")
    filters = ["rr.rank IS NOT NULL"]
    params: list[Any] = []
    if through_date:
        filters.append("r.race_date <= ?")
        params.append(through_date)
    if from_date:
        filters.append("r.race_date >= ?")
        params.append(from_date)
    rows = conn.execute(
        f"""
        SELECT
          r.race_id, r.race_date, r.jcd, r.rno, r.race_type, r.distance_m,
          e.lane, e.racer_no, e.racer_name, e.racer_class, e.branch, e.origin,
          e.age, e.weight_kg, e.f_count, e.l_count, e.avg_st,
          e.national_win_rate, e.national_2_rate, e.national_3_rate,
          e.local_win_rate, e.local_2_rate, e.local_3_rate,
          e.motor_no, e.motor_2_rate, e.motor_3_rate,
          e.boat_no, e.boat_2_rate, e.boat_3_rate,
          rr.rank, rr.course AS result_course, rr.start_timing AS result_start_timing
        FROM entries e
        JOIN races r ON r.race_id = e.race_id
        JOIN race_results rr ON rr.race_id = e.race_id AND rr.lane = e.lane
        WHERE {" AND ".join(filters)}
        ORDER BY r.race_date, r.jcd, r.rno, e.lane
        """,
        params,
    ).fetchall()
    grouped = _group_by_race(rows)
    state = RollingState()
    features: list[dict[str, Any]] = []
    labels: list[int] = []
    meta: list[dict[str, Any]] = []

    for race_id_value in sorted(grouped, key=lambda rid: _race_sort_key(grouped[rid][0])):
        race_rows = sorted(grouped[race_id_value], key=lambda row: int(row["lane"]))
        if len(race_rows) != 6:
            continue
        relatives = race_relative_features(race_rows, {lane: {} for lane in range(1, 7)})
        for row in race_rows:
            lane = int(row["lane"])
            item = base_pastlog_features(row, relatives[lane])
            item.update(state.features_for(row))
            features.append(item)
            labels.append(1 if int(row["rank"]) == 1 else 0)
            meta.append(
                {
                    "race_id": row["race_id"],
                    "race_date": row["race_date"],
                    "jcd": row["jcd"],
                    "rno": row["rno"],
                    "lane": row["lane"],
                    "rank": row["rank"],
                }
            )
        state.update_race(race_rows)
    return features, labels, meta


def prediction_features(
    conn: sqlite3.Connection,
    *,
    race_id: str,
    include_odds: bool = False,
) -> list[dict[str, Any]]:
    if include_odds:
        raise ValueError("features_pastlog_v1 does not use odds")
    rows = load_race_entries(conn, race_id=race_id)
    if len(rows) != 6:
        return []
    state = RollingState()
    for history_rows in _history_groups_before(conn, rows[0]):
        state.update_race(history_rows)
    relatives = race_relative_features(rows, {lane: {} for lane in range(1, 7)})
    result = []
    for row in rows:
        lane = int(row["lane"])
        item = base_pastlog_features(row, relatives[lane])
        item.update(state.features_for(row))
        result.append(item)
    return result


def base_pastlog_features(row: sqlite3.Row, relatives: dict[str, Any]) -> dict[str, Any]:
    item = entry_features(row, odds_features={})
    item.update(relatives)
    _drop_live_only(item)
    item.pop("motor_no", None)
    item.pop("boat_no", None)
    item["has_motor_no"] = int(_num(row["motor_no"]) >= 0)
    item["has_boat_no"] = int(_num(row["boat_no"]) >= 0)
    item.update(_race_context_features(row))
    return item


def _drop_live_only(item: dict[str, Any]) -> None:
    for key in list(item.keys()):
        if key in LIVE_ONLY_FEATURES or _is_live_relative(key):
            item.pop(key, None)


def _is_live_relative(key: str) -> bool:
    return any(
        key == f"{root}{suffix}"
        for root in LIVE_ONLY_ROOTS
        for suffix in LIVE_ONLY_SUFFIXES
    )
