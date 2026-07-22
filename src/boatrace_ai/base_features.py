from __future__ import annotations

import os
import sqlite3
from collections import defaultdict
from typing import Any, Iterable

from .constants import CLASS_RANK
from .features import (
    MODEL_FEATURE_CUTOFF_FROM_START_MINUTES,
    _num,
    entry_features,
    odds_lane_features,
    pre_t5_odds_count_sql,
    stored_jst_timestamp_sql,
)


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

# Official branch used to identify racers local to each venue.
VENUE_HOME_BRANCH = {
    "01": "群馬", "02": "埼玉", "03": "東京", "04": "東京",
    "05": "東京", "06": "静岡", "07": "愛知", "08": "愛知",
    "09": "三重", "10": "福井", "11": "滋賀", "12": "大阪",
    "13": "兵庫", "14": "徳島", "15": "香川", "16": "岡山",
    "17": "広島", "18": "山口", "19": "山口", "20": "福岡",
    "21": "福岡", "22": "福岡", "23": "佐賀", "24": "長崎",
}
RESEARCH_FEATURE_PREFIX = "research_"
RACE_DATE_CHUNK_SIZE = 31


def load_training_examples(
    conn: sqlite3.Connection,
    *,
    through_date: str | None = None,
    from_date: str | None = None,
    include_odds: bool = False,
    include_research: bool = True,
    min_odds_snapshots: int = 0,
    complete_results_only: bool = False,
) -> tuple[list[dict[str, Any]], list[int], list[dict[str, Any]]]:
    features: list[dict[str, Any]] = []
    labels: list[int] = []
    meta: list[dict[str, Any]] = []
    for item, label, row_meta in iter_training_examples(
        conn,
        through_date=through_date,
        from_date=from_date,
        include_odds=include_odds,
        include_research=include_research,
        min_odds_snapshots=min_odds_snapshots,
        complete_results_only=complete_results_only,
    ):
        features.append(item)
        labels.append(label)
        meta.append(row_meta)
    return features, labels, meta


def iter_training_examples(
    conn: sqlite3.Connection,
    *,
    through_date: str | None = None,
    from_date: str | None = None,
    include_odds: bool = False,
    include_research: bool = True,
    include_races: set[str] | None = None,
    min_odds_snapshots: int = 0,
    complete_results_only: bool = False,
) -> Iterable[tuple[dict[str, Any], int, dict[str, Any]]]:
    through_date = through_date or os.environ.get("BOATRACE_EVAL_MAX_RACE_DATE")
    filters: list[str] = []
    params: list[Any] = []
    if through_date:
        filters.append("r.race_date <= ?")
        params.append(through_date)
    if from_date:
        filters.append("r.race_date >= ?")
        params.append(from_date)
    odds_eligible_races: set[str] | None = None
    if min_odds_snapshots > 0:
        odds_filters = [*filters, pre_t5_odds_count_sql(conn)]
        odds_where = " AND ".join(odds_filters)
        odds_rows = conn.execute(
            f"SELECT r.race_id FROM races r WHERE {odds_where}",
            (*params, int(min_odds_snapshots)),
        ).fetchall()
        odds_eligible_races = {str(row["race_id"]) for row in odds_rows}
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    date_rows = conn.execute(
        f"""
        SELECT DISTINCT r.race_date
        FROM races r
        {where}
        ORDER BY r.race_date
        """,
        params,
    ).fetchall()
    race_dates = [str(row["race_date"]) for row in date_rows]

    for offset in range(0, len(race_dates), RACE_DATE_CHUNK_SIZE):
        date_chunk = race_dates[offset : offset + RACE_DATE_CHUNK_SIZE]
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
              rr.rank
            FROM entries e
            JOIN races r ON r.race_id = e.race_id
            JOIN race_results rr
              ON rr.race_id = e.race_id AND rr.lane = e.lane
            WHERE rr.rank IS NOT NULL
              AND r.race_date >= ?
              AND r.race_date <= ?
            ORDER BY r.race_date, r.jcd, r.rno, e.lane
            """,
            (date_chunk[0], date_chunk[-1]),
        )
        beforeinfo = _latest_beforeinfo_between(
            conn,
            from_date=date_chunk[0],
            through_date=date_chunk[-1],
        )
        for race_rows in _iter_complete_race_rows(rows):
            race_id_value = str(race_rows[0]["race_id"])
            if include_races is not None and race_id_value not in include_races:
                continue
            if odds_eligible_races is not None and race_id_value not in odds_eligible_races:
                continue
            before_rows = {
                lane: beforeinfo.get((race_id_value, lane), {})
                for lane in range(1, 7)
            }
            odds = odds_lane_features(conn, race_id_value) if include_odds else {}
            relatives = race_relative_features(
                race_rows,
                before_rows,
                include_research=include_research,
            )
            for row in race_rows:
                lane = int(row["lane"])
                item = entry_features(row, odds_features=odds.get(lane, {}))
                item.update(before_features(before_rows.get(lane, {})))
                item.update(relatives[lane])
                yield item, 1 if int(row["rank"]) == 1 else 0, {
                    "race_id": row["race_id"],
                    "race_date": row["race_date"],
                    "jcd": row["jcd"],
                    "rno": row["rno"],
                    "lane": row["lane"],
                    "rank": row["rank"],
                }


def _iter_complete_race_rows(rows: Iterable[Any]) -> Iterable[list[Any]]:
    current_id: str | None = None
    current_rows: list[Any] = []
    for row in rows:
        race_id_value = str(row["race_id"])
        if current_id is None:
            current_id = race_id_value
        if race_id_value != current_id:
            if len(current_rows) == 6:
                yield current_rows
            current_id = race_id_value
            current_rows = []
        current_rows.append(row)
    if len(current_rows) == 6:
        yield current_rows


def _latest_beforeinfo_between(
    conn: sqlite3.Connection,
    *,
    from_date: str,
    through_date: str,
) -> dict[tuple[str, int], sqlite3.Row]:
    cutoff = _beforeinfo_cutoff_sql(conn, before_alias="b2", race_alias="r2")
    rows = conn.execute(
        f"""
        SELECT b.*
        FROM beforeinfo b
        JOIN (
          SELECT b2.race_id, b2.lane, MAX(b2.captured_at) AS captured_at
          FROM beforeinfo b2
          JOIN races r2 ON r2.race_id = b2.race_id
          WHERE r2.race_date >= ? AND r2.race_date <= ?
            AND {cutoff}
          GROUP BY b2.race_id, b2.lane
        ) latest ON latest.race_id = b.race_id
          AND latest.lane = b.lane
          AND latest.captured_at = b.captured_at
        """,
        (from_date, through_date),
    ).fetchall()
    return {(str(row["race_id"]), int(row["lane"])): row for row in rows}


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


def prediction_features(
    conn: sqlite3.Connection,
    *,
    race_id: str,
    include_odds: bool = False,
    include_research: bool = True,
) -> list[dict[str, Any]]:
    rows = load_race_entries(conn, race_id=race_id)
    before_rows = _latest_beforeinfo(conn, race_id=race_id)
    by_lane = {lane: before_rows.get((race_id, lane), {}) for lane in range(1, 7)}
    relatives = race_relative_features(
        rows,
        by_lane,
        include_research=include_research,
    )
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
    *,
    include_research: bool = True,
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
    if include_research:
        _add_research_correlates(result, rows, before_rows, values_by_lane)
    return result


def is_home_branch(jcd: Any, branch: Any) -> bool:
    venue_branch = VENUE_HOME_BRANCH.get(str(jcd or "").zfill(2))
    return bool(venue_branch and str(branch or "").strip() == venue_branch)


def _add_research_correlates(
    result: dict[int, dict[str, Any]],
    rows: list[sqlite3.Row],
    before_rows: dict[int, sqlite3.Row | dict[str, Any]],
    values_by_lane: dict[int, dict[str, float]],
) -> None:
    racer_fields = (
        "class_rank", "national_win_rate", "national_2_rate",
        "local_win_rate", "local_2_rate",
    )
    equipment_fields = ("motor_2_rate", "boat_2_rate")
    racer_strength = {
        lane: _mean_present_scaled(result[lane], racer_fields) for lane in result
    }
    equipment_strength = {
        lane: _mean_present_scaled(result[lane], equipment_fields) for lane in result
    }
    strength_values = list(racer_strength.values())
    best_strength = max(strength_values, default=0.0)
    strength_spread = best_strength - min(strength_values, default=0.0)
    lane1_strength = racer_strength.get(1, 0.0)
    outer_strength = max(
        (value for lane, value in racer_strength.items() if lane >= 5),
        default=0.0,
    )
    strength_ranks = _ranks(racer_strength, high_is_good=True)
    before_by_lane = {
        lane: before_features(before_rows.get(lane, {})) for lane in result
    }
    courses = {
        lane: int(item["course"])
        for lane, item in before_by_lane.items()
        if 1 <= item["course"] <= 6
    }
    has_full_course = len(courses) == len(result) == 6
    waku_nari = has_full_course and all(courses.get(lane) == lane for lane in result)
    rows_by_lane = {int(row["lane"]): row for row in rows}

    for lane, item in result.items():
        row = rows_by_lane[lane]
        values = values_by_lane[lane]
        jcd = str(_row_value(row, "jcd") or "").zfill(2)
        branch = str(_row_value(row, "branch") or "").strip()
        home = is_home_branch(jcd, branch)
        local_win_delta = _valid_delta(
            values.get("local_win_rate", -1.0),
            values.get("national_win_rate", -1.0),
        )
        local_2_delta = _valid_delta(
            values.get("local_2_rate", -1.0),
            values.get("national_2_rate", -1.0),
        )
        item.update({
            "research_home_branch": int(home),
            "research_home_lane": f"{int(home)}:{lane}",
            "research_branch_venue": f"{branch}:{jcd}" if branch else "",
            "research_has_local_rates": int(
                values.get("local_win_rate", -1.0) >= 0
                and values.get("local_2_rate", -1.0) >= 0
            ),
            "research_local_vs_national_win": local_win_delta,
            "research_local_vs_national_2": local_2_delta,
            "research_home_local_win_delta": float(home) * local_win_delta,
            "research_home_local_2_delta": float(home) * local_2_delta,
            "research_racer_strength": racer_strength[lane],
            "research_racer_strength_rank": strength_ranks[lane],
            "research_racer_strength_vs_lane1": racer_strength[lane] - lane1_strength,
            "research_racer_strength_gap_best": racer_strength[lane] - best_strength,
            "research_field_racer_strength_spread": strength_spread,
            "research_lane1_racer_strength": lane1_strength,
            "research_outer_threat_vs_lane1": outer_strength - lane1_strength,
            "research_equipment_strength": equipment_strength[lane],
            "research_equipment_balanced_field": (
                equipment_strength[lane] * max(0.0, 1.0 - strength_spread)
            ),
        })

        before = before_by_lane[lane]
        course = courses.get(lane, 0)
        exhibition_rank = int(item.get("exhibition_time_rank", 0) or 0)
        exhibition_scaled = float(item.get("exhibition_time_scaled", 0.0) or 0.0)
        weather = str(before.get("weather") or "")
        wind_direction = str(before.get("wind_direction") or "")
        wind_bucket = _numeric_bucket(
            before.get("wind_speed_m", -1.0), (2.0, 5.0, 8.0)
        )
        wave_bucket = _numeric_bucket(
            before.get("wave_cm", -1.0), (3.0, 6.0, 10.0)
        )
        item.update({
            "research_has_live_context": int(bool(before.get("has_beforeinfo"))),
            "research_has_full_course": int(has_full_course),
            "research_waku_nari": int(waku_nari) if has_full_course else 0,
            "research_course_cat": str(course) if course else "",
            "research_lane_course": f"{lane}:{course}" if course else "",
            "research_course_changed": int(course != lane) if course else 0,
            "research_course_delta": course - lane if course else 0,
            "research_exhibition_top1": int(exhibition_rank == 1),
            "research_exhibition_top2": int(0 < exhibition_rank <= 2),
            "research_exhibition_rank_venue": (
                f"{exhibition_rank}:{jcd}" if exhibition_rank else ""
            ),
            "research_exhibition_rank_weather": (
                f"{exhibition_rank}:{weather}"
                if exhibition_rank and weather else ""
            ),
            "research_exhibition_rank_distance": (
                f"{exhibition_rank}:{int(_row_value(row, 'distance_m') or 0)}"
                if exhibition_rank else ""
            ),
            "research_exhibition_racer_strength": (
                exhibition_scaled * racer_strength[lane]
            ),
            "research_exhibition_rain": exhibition_scaled * int("雨" in weather),
            "research_venue_weather": f"{jcd}:{weather}" if weather else "",
            "research_lane_wind_direction": (
                f"{lane}:{wind_direction}" if wind_direction else ""
            ),
            "research_lane_wind_bucket": (
                f"{lane}:{wind_bucket}" if wind_bucket else ""
            ),
            "research_lane_wave_bucket": (
                f"{lane}:{wave_bucket}" if wave_bucket else ""
            ),
        })


def _row_value(row: sqlite3.Row | dict[str, Any], key: str) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    try:
        return row[key]
    except (KeyError, IndexError):
        return None


def _mean_present_scaled(item: dict[str, Any], fields: tuple[str, ...]) -> float:
    values = [
        float(item[f"{field}_scaled"])
        for field in fields
        if item.get(f"has_{field}")
    ]
    return sum(values) / len(values) if values else 0.0


def _valid_delta(left: float, right: float) -> float:
    return left - right if left >= 0 and right >= 0 else 0.0


def _numeric_bucket(value: Any, boundaries: tuple[float, ...]) -> str:
    number = _num(value)
    if number < 0:
        return ""
    for boundary in boundaries:
        if number < boundary:
            return f"lt{boundary:g}"
    return f"ge{boundaries[-1]:g}"


def _group_by_race(rows: list[sqlite3.Row]) -> dict[str, list[sqlite3.Row]]:
    grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        grouped[row["race_id"]].append(row)
    return grouped


def _latest_beforeinfo(conn: sqlite3.Connection, race_id: str | None = None) -> dict[tuple[str, int], sqlite3.Row]:
    params: list[Any] = []
    latest_filter_sql = ""
    if race_id:
        latest_filter_sql = "AND b2.race_id = ?"
        params.append(race_id)
    cutoff = _beforeinfo_cutoff_sql(conn, before_alias="b2", race_alias="r2")
    rows = conn.execute(
        f"""
        SELECT b.*
        FROM beforeinfo b
        JOIN (
          SELECT b2.race_id, b2.lane, MAX(b2.captured_at) AS captured_at
          FROM beforeinfo b2
          JOIN races r2 ON r2.race_id = b2.race_id
          WHERE {cutoff}
          {latest_filter_sql}
          GROUP BY b2.race_id, b2.lane
        ) latest ON latest.race_id = b.race_id
          AND latest.lane = b.lane
          AND latest.captured_at = b.captured_at
        """,
        params,
    ).fetchall()
    return {(row["race_id"], int(row["lane"])): row for row in rows}


def _beforeinfo_cutoff_sql(
    conn,
    *,
    before_alias: str,
    race_alias: str,
) -> str:
    captured = stored_jst_timestamp_sql(conn, f"{before_alias}.captured_at")
    start_at = stored_jst_timestamp_sql(conn, f"{race_alias}.deadline_at")
    if getattr(conn, "dialect", "sqlite") == "postgresql":
        cutoff = (
            f"{start_at} - INTERVAL "
            f"'{MODEL_FEATURE_CUTOFF_FROM_START_MINUTES} minutes'"
        )
    else:
        cutoff = (
            f"datetime({start_at}, "
            f"'-{MODEL_FEATURE_CUTOFF_FROM_START_MINUTES} minutes')"
        )
    return f"{captured} <= {cutoff}"


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
