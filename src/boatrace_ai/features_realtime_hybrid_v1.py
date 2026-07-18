from __future__ import annotations

import sqlite3
from typing import Any

from .features import _num, odds_lane_features
from .features_no_odds_v3 import _group_by_race, _latest_beforeinfo, before_features, race_relative_features
from .features_no_odds_v9 import RollingState, _history_groups_before, _race_sort_key, load_race_entries
from .features_pastlog_v1 import base_pastlog_features


HYBRID_RELATIVE_ROOTS = (
    "before_weight_kg",
    "exhibition_time",
    "start_timing",
    "wind_speed_m",
    "wave_cm",
)
HYBRID_RELATIVE_SUFFIXES = (
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
    include_odds: bool = True,
) -> tuple[list[dict[str, Any]], list[int], list[dict[str, Any]]]:
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
    beforeinfo = _latest_beforeinfo(conn)
    odds_by_race = {}
    if include_odds:
        for race_id_value in sorted(grouped):
            odds_by_race[race_id_value] = odds_lane_features(conn, race_id_value)

    state = RollingState()
    features: list[dict[str, Any]] = []
    labels: list[int] = []
    meta: list[dict[str, Any]] = []
    for race_id_value in sorted(grouped, key=lambda rid: _race_sort_key(grouped[rid][0])):
        race_rows = sorted(grouped[race_id_value], key=lambda row: int(row["lane"]))
        if len(race_rows) != 6:
            continue
        before_rows = {lane: beforeinfo.get((race_id_value, lane), {}) for lane in range(1, 7)}
        relatives = race_relative_features(race_rows, before_rows)
        odds = odds_by_race.get(race_id_value, {})
        for row in race_rows:
            lane = int(row["lane"])
            item = base_pastlog_features(row, relatives[lane])
            item.update(before_features(before_rows.get(lane, {})))
            item.update(_hybrid_relative_features(relatives[lane]))
            item.update(_odds_features(odds.get(lane, {})))
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
    include_odds: bool = True,
) -> list[dict[str, Any]]:
    rows = load_race_entries(conn, race_id=race_id)
    if len(rows) != 6:
        return []
    state = RollingState()
    for history_rows in _history_groups_before(conn, rows[0]):
        state.update_race(history_rows)
    before_rows = _latest_beforeinfo(conn, race_id=race_id)
    by_lane = {lane: before_rows.get((race_id, lane), {}) for lane in range(1, 7)}
    relatives = race_relative_features(rows, by_lane)
    odds = odds_lane_features(conn, race_id) if include_odds else {}
    result = []
    for row in rows:
        lane = int(row["lane"])
        item = base_pastlog_features(row, relatives[lane])
        item.update(before_features(by_lane.get(lane, {})))
        item.update(_hybrid_relative_features(relatives[lane]))
        item.update(_odds_features(odds.get(lane, {})))
        item.update(state.features_for(row))
        result.append(item)
    return result


def _hybrid_relative_features(relatives: dict[str, Any]) -> dict[str, Any]:
    wanted = {}
    for root in HYBRID_RELATIVE_ROOTS:
        for suffix in HYBRID_RELATIVE_SUFFIXES:
            key = f"{root}{suffix}"
            if key in relatives:
                wanted[key] = relatives[key]
    return wanted


def _odds_features(values: dict[str, Any]) -> dict[str, float]:
    return {f"odds_{key}": _num(value) for key, value in values.items()}
