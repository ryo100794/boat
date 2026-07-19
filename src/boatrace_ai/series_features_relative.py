from __future__ import annotations

import sqlite3
from collections import defaultdict
from typing import Any

from .base_features import _group_by_race, race_relative_features
from .contextual_features import RollingState, _race_sort_key, load_race_entries
from .series_features_base import base_pastlog_features


def load_training_examples(
    conn: sqlite3.Connection,
    *,
    through_date: str | None = None,
    from_date: str | None = None,
    include_odds: bool = False,
) -> tuple[list[dict[str, Any]], list[int], list[dict[str, Any]]]:
    if include_odds:
        raise ValueError("series_features_relative does not use odds")
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
    by_date: dict[str, list[str]] = defaultdict(list)
    for race_id_value, race_rows in grouped.items():
        if len(race_rows) == 6:
            by_date[str(race_rows[0]["race_date"])].append(race_id_value)

    state = RollingState()
    features: list[dict[str, Any]] = []
    labels: list[int] = []
    meta: list[dict[str, Any]] = []
    for race_date_value in sorted(by_date):
        day_races = sorted(by_date[race_date_value], key=lambda rid: _race_sort_key(grouped[rid][0]))
        day_updates = []
        for race_id_value in day_races:
            race_rows = sorted(grouped[race_id_value], key=lambda row: int(row["lane"]))
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
            day_updates.append(race_rows)
        for race_rows in day_updates:
            state.update_race(race_rows)
    return features, labels, meta


def prediction_features(
    conn: sqlite3.Connection,
    *,
    race_id: str,
    include_odds: bool = False,
) -> list[dict[str, Any]]:
    if include_odds:
        raise ValueError("series_features_relative does not use odds")
    rows = load_race_entries(conn, race_id=race_id)
    if len(rows) != 6:
        return []
    state = RollingState()
    for history_rows in history_groups_prior_dates(conn, rows[0]):
        state.update_race(history_rows)
    relatives = race_relative_features(rows, {lane: {} for lane in range(1, 7)})
    result = []
    for row in rows:
        lane = int(row["lane"])
        item = base_pastlog_features(row, relatives[lane])
        item.update(state.features_for(row))
        result.append(item)
    return result


def history_groups_prior_dates(conn: sqlite3.Connection, target: sqlite3.Row) -> list[list[sqlite3.Row]]:
    rows = conn.execute(
        """
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
        WHERE rr.rank IS NOT NULL
          AND r.race_date < ?
        ORDER BY r.race_date, r.jcd, r.rno, e.lane
        """,
        (target["race_date"],),
    ).fetchall()
    grouped = _group_by_race(rows)
    return [
        sorted(grouped[race_id_value], key=lambda row: int(row["lane"]))
        for race_id_value in sorted(grouped, key=lambda rid: _race_sort_key(grouped[rid][0]))
        if len(grouped[race_id_value]) == 6
    ]
