from __future__ import annotations

import os
import sqlite3
from collections import defaultdict
from typing import Any

from .constants import CLASS_RANK
from .features import _num, entry_features, odds_lane_features


HIGH_IS_GOOD = (
    "class_rank",
    "national_win_rate",
    "national_2_rate",
    "national_3_rate",
    "local_win_rate",
    "local_2_rate",
    "local_3_rate",
    "motor_2_rate",
    "motor_3_rate",
    "boat_2_rate",
    "boat_3_rate",
)
LOW_IS_GOOD = (
    "avg_st",
    "f_count",
    "l_count",
    "weight_kg",
    "before_weight_kg",
    "exhibition_time",
    "start_timing",
)
RELATIVE_FIELDS = HIGH_IS_GOOD + LOW_IS_GOOD + ("age", "wind_speed_m", "wave_cm")
BEFORE_NUMERIC = (
    "weight_kg",
    "exhibition_time",
    "tilt",
    "adjusted_weight",
    "course",
    "start_timing",
    "wind_speed_m",
    "air_temp_c",
    "water_temp_c",
    "wave_cm",
)
BEFORE_CATEGORICAL = ("weather", "wind_direction", "propeller", "parts_exchange")


def load_training_examples(
    conn: sqlite3.Connection,
    *,
    through_date: str | None = None,
    from_date: str | None = None,
    include_odds: bool = False,
) -> tuple[list[dict[str, Any]], list[int], list[dict[str, Any]]]:
    through_date = through_date or os.environ.get("BOATRACE_EVAL_MAX_RACE_DATE")
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
          rr.rank
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

    features: list[dict[str, Any]] = []
    labels: list[int] = []
    meta: list[dict[str, Any]] = []
    for race_id_value in sorted(grouped):
        race_rows = sorted(grouped[race_id_value], key=lambda row: int(row["lane"]))
        if len(race_rows) != 6:
            continue
        before_rows = {lane: beforeinfo.get((race_id_value, lane), {}) for lane in range(1, 7)}
        relatives = race_relative_features(race_rows, before_rows)
        for row in race_rows:
            lane = int(row["lane"])
            item = entry_features(row, odds_features=odds_by_race.get(race_id_value, {}).get(lane, {}))
            item.update(before_features(before_rows.get(lane, {})))
            item.update(relatives[lane])
            features.append(item)
            labels.append(1 if row["rank"] == 1 else 0)
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
    return features, labels, meta


def load_race_entries(conn: sqlite3.Connection, *, race_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT
          r.race_id, r.race_date, r.jcd, r.rno, r.race_type, r.distance_m,
          e.lane, e.racer_no, e.racer_name, e.racer_class, e.branch, e.origin,
          e.age, e.weight_kg, e.f_count, e.l_count, e.avg_st,
          e.national_win_rate, e.national_2_rate, e.national_3_rate,
          e.local_win_rate, e.local_2_rate, e.local_3_rate,
          e.motor_no, e.motor_2_rate, e.motor_3_rate,
          e.boat_no, e.boat_2_rate, e.boat_3_rate
        FROM entries e
        JOIN races r ON r.race_id = e.race_id
        WHERE e.race_id = ?
        ORDER BY e.lane
        """,
        (race_id,),
    ).fetchall()


def prediction_features(conn: sqlite3.Connection, *, race_id: str, include_odds: bool = False) -> list[dict[str, Any]]:
    rows = load_race_entries(conn, race_id=race_id)
    before_rows = _latest_beforeinfo(conn, race_id=race_id)
    by_lane = {lane: before_rows.get((race_id, lane), {}) for lane in range(1, 7)}
    relatives = race_relative_features(rows, by_lane)
    odds = odds_lane_features(conn, race_id) if include_odds else {}
    result = []
    for row in rows:
        lane = int(row["lane"])
        item = entry_features(row, odds_features=odds.get(lane, {}))
        item.update(before_features(by_lane.get(lane, {})))
        item.update(relatives[lane])
        result.append(item)
    return result


def before_features(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    get = row.get if isinstance(row, dict) else row.__getitem__
    item: dict[str, Any] = {}
    for field in BEFORE_NUMERIC:
        key = "before_weight_kg" if field == "weight_kg" else field
        try:
            item[key] = _num(get(field))
        except (KeyError, IndexError):
            item[key] = -1.0
    for field in BEFORE_CATEGORICAL:
        try:
            value = get(field)
        except (KeyError, IndexError):
            value = None
        item[field] = str(value or "")
        item[f"has_{field}"] = int(bool(value))
    item["has_beforeinfo"] = int(bool(row))
    return item


def race_relative_features(
    rows: list[sqlite3.Row],
    before_rows: dict[int, sqlite3.Row | dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    values_by_lane = {int(row["lane"]): _relative_values(row, before_rows.get(int(row["lane"]), {})) for row in rows}
    stats = {field: _stats([values.get(field, -1.0) for values in values_by_lane.values()]) for field in RELATIVE_FIELDS}
    ranks = {
        field: _ranks(
            {lane: values.get(field, -1.0) for lane, values in values_by_lane.items()},
            high_is_good=field not in LOW_IS_GOOD,
        )
        for field in RELATIVE_FIELDS
    }
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        lane = int(row["lane"])
        item: dict[str, Any] = {
            "field_size": len(rows),
            "lane_rno": f"{lane}:{int(row['rno'] or 0)}",
            "lane_jcd": f"{lane}:{row['jcd'] or ''}",
            "lane_class": f"{lane}:{row['racer_class'] or ''}",
        }
        for field in RELATIVE_FIELDS:
            value = values_by_lane[lane].get(field, -1.0)
            is_present = value >= 0
            field_stats = stats[field]
            sign = 1.0 if field not in LOW_IS_GOOD else -1.0
            best = field_stats["max"] if field not in LOW_IS_GOOD else field_stats["min"]
            worst = field_stats["min"] if field not in LOW_IS_GOOD else field_stats["max"]
            spread = max(1e-6, abs(field_stats["max"] - field_stats["min"]))
            item[f"has_{field}"] = int(is_present)
            item[f"{field}_rank"] = ranks[field][lane]
            item[f"{field}_vs_mean"] = value - field_stats["mean"] if is_present else 0.0
            item[f"{field}_z"] = (
                (value - field_stats["mean"]) / max(1e-6, field_stats["std"])
                if is_present
                else 0.0
            )
            item[f"{field}_best_gap"] = sign * (value - best) if is_present else 0.0
            item[f"{field}_scaled"] = sign * (value - worst) / spread if is_present else 0.0
        _composites(item, lane, values_by_lane[lane], ranks)
        result[lane] = item
    return result


def _group_by_race(rows: list[sqlite3.Row]) -> dict[str, list[sqlite3.Row]]:
    grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        grouped[row["race_id"]].append(row)
    return grouped


def _latest_beforeinfo(conn: sqlite3.Connection, race_id: str | None = None) -> dict[tuple[str, int], sqlite3.Row]:
    params: list[Any] = []
    filter_sql = ""
    latest_filter_sql = ""
    if race_id:
        latest_filter_sql = "WHERE race_id = ?"
        filter_sql = "WHERE b.race_id = ?"
        params.extend((race_id, race_id))
    rows = conn.execute(
        f"""
        SELECT b.*
        FROM beforeinfo b
        JOIN (
          SELECT race_id, lane, MAX(captured_at) AS captured_at
          FROM beforeinfo
          {latest_filter_sql}
          GROUP BY race_id, lane
        ) latest ON latest.race_id = b.race_id
          AND latest.lane = b.lane
          AND latest.captured_at = b.captured_at
        {filter_sql}
        """,
        params,
    ).fetchall()
    return {(row["race_id"], int(row["lane"])): row for row in rows}


def _relative_values(row: sqlite3.Row, before: sqlite3.Row | dict[str, Any]) -> dict[str, float]:
    values = {
        "class_rank": float(CLASS_RANK.get(str(row["racer_class"] or ""), -1)),
        "age": _num(row["age"]),
        "weight_kg": _num(row["weight_kg"]),
        "f_count": _num(row["f_count"]),
        "l_count": _num(row["l_count"]),
        "avg_st": _num(row["avg_st"]),
        "national_win_rate": _num(row["national_win_rate"]),
        "national_2_rate": _num(row["national_2_rate"]),
        "national_3_rate": _num(row["national_3_rate"]),
        "local_win_rate": _num(row["local_win_rate"]),
        "local_2_rate": _num(row["local_2_rate"]),
        "local_3_rate": _num(row["local_3_rate"]),
        "motor_2_rate": _num(row["motor_2_rate"]),
        "motor_3_rate": _num(row["motor_3_rate"]),
        "boat_2_rate": _num(row["boat_2_rate"]),
        "boat_3_rate": _num(row["boat_3_rate"]),
    }
    before_item = before_features(before)
    values.update(
        {
            "before_weight_kg": before_item["before_weight_kg"],
            "exhibition_time": before_item["exhibition_time"],
            "start_timing": before_item["start_timing"],
            "wind_speed_m": before_item["wind_speed_m"],
            "wave_cm": before_item["wave_cm"],
        }
    )
    return values


def _stats(values: list[float]) -> dict[str, float]:
    valid = [value for value in values if value >= 0]
    if not valid:
        return {"mean": -1.0, "std": 1.0, "min": -1.0, "max": -1.0}
    mean = sum(valid) / len(valid)
    variance = sum((value - mean) ** 2 for value in valid) / len(valid)
    return {"mean": mean, "std": variance ** 0.5, "min": min(valid), "max": max(valid)}


def _ranks(values: dict[int, float], *, high_is_good: bool) -> dict[int, int]:
    valid = [(lane, value) for lane, value in values.items() if value >= 0]
    ordered = sorted(valid, key=lambda item: -item[1] if high_is_good else item[1])
    result = {lane: 0 for lane in values}
    previous: float | None = None
    rank = 0
    for index, (lane, value) in enumerate(ordered, start=1):
        if previous is None or value != previous:
            rank = index
            previous = value
        result[lane] = rank
    return result


def _composites(
    item: dict[str, Any],
    lane: int,
    values: dict[str, float],
    ranks: dict[str, dict[int, int]],
) -> None:
    ability = _avg_valid(
        values["national_win_rate"],
        values["national_2_rate"],
        values["local_win_rate"],
        values["local_2_rate"],
        values["motor_2_rate"],
        values["boat_2_rate"],
    )
    item["ability_score"] = ability
    item["ability_lane_score"] = ability * max(0, 7 - lane)
    item["best_count"] = sum(
        int(ranks[field][lane] == 1)
        for field in (
            "national_win_rate",
            "local_win_rate",
            "motor_2_rate",
            "boat_2_rate",
            "avg_st",
            "class_rank",
            "exhibition_time",
        )
    )


def _avg_valid(*values: float) -> float:
    valid = [value for value in values if value >= 0]
    if not valid:
        return -1.0
    return sum(valid) / len(valid)
