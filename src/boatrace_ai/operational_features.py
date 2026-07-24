from __future__ import annotations

import math
import sqlite3
from collections import defaultdict
from typing import Any

from .cache_entry_series_features import CACHE_FIELDS, ensure_series_cache_table
from .features import _num
from .base_features import _group_by_race, race_relative_features
from .feature_schema import (
    FEATURE_SCHEMA_VERSION,
    MISSING_SAFE_FEATURE_SCHEMA_VERSION,
    uses_empirical_series_trend_direction,
    uses_missing_safe_series,
    uses_sparse_series_missing,
)
from .contextual_features import RollingState, _race_sort_key
from .series_features_form import base_pastlog_features


SERIES_RELATIVE_FIELDS = {
    "series_starts": True,
    "series_avg_finish": False,
    "series_latest_finish": False,
    "series_best_finish": False,
    "series_worst_finish": False,
    "series_win_rate": True,
    "series_top2_rate": True,
    "series_top3_rate": True,
    "series_finish_trend": True,
}


SERIES_SELECT = ", ".join(f"sf.{field} AS {field}" for field in CACHE_FIELDS)


def load_training_examples(
    conn: sqlite3.Connection,
    *,
    through_date: str | None = None,
    from_date: str | None = None,
    include_odds: bool = False,
) -> tuple[list[dict[str, Any]], list[int], list[dict[str, Any]]]:
    if include_odds:
        raise ValueError("operational_features does not use odds")
    ensure_series_cache_table(conn)
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
          {SERIES_SELECT},
          rr.rank, rr.course AS result_course, rr.start_timing AS result_start_timing
        FROM entries e
        JOIN races r ON r.race_id = e.race_id
        JOIN race_results rr ON rr.race_id = e.race_id AND rr.lane = e.lane
        LEFT JOIN entry_series_features sf ON sf.race_id = e.race_id AND sf.lane = e.lane
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
            series_relatives = series_relative_features(
                race_rows,
                feature_schema_version=MISSING_SAFE_FEATURE_SCHEMA_VERSION,
            )
            for row in race_rows:
                lane = int(row["lane"])
                item = base_pastlog_features(row, relatives[lane])
                item.update(
                    cached_series_features(
                        row,
                        feature_schema_version=MISSING_SAFE_FEATURE_SCHEMA_VERSION,
                    )
                )
                item.update(series_relatives[lane])
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
    feature_schema_version: str = MISSING_SAFE_FEATURE_SCHEMA_VERSION,
    drop_feature_groups: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    if include_odds:
        raise ValueError("operational_features does not use odds")
    dropped = {str(value).strip() for value in drop_feature_groups if str(value).strip()}
    allowed_groups = {
        "base_pastlog",
        "series_cached",
        "series_relative",
        "rolling_history",
        "research_correlates",
    }
    unknown = dropped - allowed_groups
    if unknown:
        raise ValueError(f"unknown feature groups: {', '.join(sorted(unknown))}")
    ensure_series_cache_table(conn)
    rows = load_race_entries(conn, race_id=race_id)
    if len(rows) != 6:
        return []
    state = RollingState()
    if "rolling_history" not in dropped:
        for history_rows in history_groups_prior_dates(conn, rows[0]):
            state.update_race(history_rows)
    relatives = (
        race_relative_features(
            rows,
            {lane: {} for lane in range(1, 7)},
            include_research="research_correlates" not in dropped,
        )
        if "base_pastlog" not in dropped
        else {}
    )
    series_relatives = (
        series_relative_features(
            rows,
            feature_schema_version=feature_schema_version,
        )
        if "series_relative" not in dropped
        else {}
    )
    result = []
    for row in rows:
        lane = int(row["lane"])
        item: dict[str, Any] = {}
        if "base_pastlog" not in dropped:
            item.update(base_pastlog_features(row, relatives[lane]))
        if "series_cached" not in dropped:
            item.update(
                cached_series_features(
                    row,
                    feature_schema_version=feature_schema_version,
                )
            )
        if "series_relative" not in dropped:
            item.update(series_relatives[lane])
        if "rolling_history" not in dropped:
            item.update(state.features_for(row))
        if "research_correlates" in dropped:
            item = {
                key: value
                for key, value in item.items()
                if not key.startswith("research_")
            }
        result.append(item)
    return result


def load_race_entries(conn: sqlite3.Connection, *, race_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        f"""
        SELECT
          r.race_id, r.race_date, r.jcd, r.rno, r.race_type, r.distance_m,
          e.lane, e.racer_no, e.racer_name, e.racer_class, e.branch, e.origin,
          e.age, e.weight_kg, e.f_count, e.l_count, e.avg_st,
          e.national_win_rate, e.national_2_rate, e.national_3_rate,
          e.local_win_rate, e.local_2_rate, e.local_3_rate,
          e.motor_no, e.motor_2_rate, e.motor_3_rate,
          e.boat_no, e.boat_2_rate, e.boat_3_rate,
          {SERIES_SELECT}
        FROM entries e
        JOIN races r ON r.race_id = e.race_id
        LEFT JOIN entry_series_features sf ON sf.race_id = e.race_id AND sf.lane = e.lane
        WHERE e.race_id = ?
        ORDER BY e.lane
        """,
        (race_id,),
    ).fetchall()


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


def cached_series_features(
    row: sqlite3.Row,
    *,
    feature_schema_version: str = FEATURE_SCHEMA_VERSION,
) -> dict[str, Any]:
    if not uses_sparse_series_missing(feature_schema_version):
        return {field: _num(row[field]) for field in CACHE_FIELDS}
    return {
        field: _num(row[field])
        for field in CACHE_FIELDS
        if row[field] is not None
    }


def series_relative_features(
    rows: list[sqlite3.Row],
    *,
    feature_schema_version: str = FEATURE_SCHEMA_VERSION,
) -> dict[int, dict[str, float]]:
    if uses_sparse_series_missing(feature_schema_version):
        return _sparse_series_relative_features(
            rows,
            feature_schema_version=feature_schema_version,
        )
    missing_safe = uses_missing_safe_series(feature_schema_version)
    by_lane = {
        int(row["lane"]): cached_series_features(
            row,
            feature_schema_version=feature_schema_version,
        )
        for row in rows
    }
    out = {lane: {} for lane in by_lane}
    for field, high_is_good in SERIES_RELATIVE_FIELDS.items():
        values = {lane: _num(features.get(field)) for lane, features in by_lane.items()}
        valid = [value for value in values.values() if value >= 0]
        mean = sum(valid) / len(valid) if valid else -1.0
        if valid:
            variance = sum((value - mean) ** 2 for value in valid) / len(valid)
            std = variance ** 0.5 or 1.0
        else:
            std = 1.0
        ranks = (
            _ranks(values, high_is_good=high_is_good)
            if missing_safe
            else _legacy_ranks(values, high_is_good=high_is_good)
        )
        for lane, value in values.items():
            is_present = value >= 0
            if missing_safe:
                out[lane][f"has_{field}"] = int(is_present)
            out[lane][f"{field}_rank"] = ranks[lane]
            out[lane][f"{field}_vs_mean"] = (
                value - mean
                if is_present and mean >= 0
                else 0.0
                if missing_safe
                else -1.0
            )
            out[lane][f"{field}_z"] = (
                (value - mean) / std
                if is_present and mean >= 0
                else 0.0
                if missing_safe
                else -1.0
            )
    return out


def _sparse_series_relative_features(
    rows: list[sqlite3.Row],
    *,
    feature_schema_version: str,
) -> dict[int, dict[str, float]]:
    by_lane = {int(row["lane"]): row for row in rows}
    out: dict[int, dict[str, float]] = {lane: {} for lane in by_lane}
    for field, high_is_good in SERIES_RELATIVE_FIELDS.items():
        if (
            field == "series_finish_trend"
            and uses_empirical_series_trend_direction(feature_schema_version)
        ):
            high_is_good = False
        present_values = {
            lane: value
            for lane, row in by_lane.items()
            if (value := _series_relative_value(row, field)) is not None
        }
        if not present_values:
            continue
        all_present = len(present_values) == len(by_lane)
        if all_present and len(set(present_values.values())) == 1:
            continue
        mean = sum(present_values.values()) / len(present_values)
        variance = sum(
            (value - mean) ** 2 for value in present_values.values()
        ) / len(present_values)
        std = variance ** 0.5 or 1.0
        values = {lane: present_values.get(lane, 0.0) for lane in by_lane}
        present = {lane: lane in present_values for lane in by_lane}
        ranks = _ranks(
            values,
            high_is_good=high_is_good,
            present=present,
        )
        for lane, value in present_values.items():
            if not all_present:
                out[lane][f"has_{field}"] = 1
            out[lane][f"{field}_rank"] = ranks[lane]
            relative = value - mean
            if relative:
                out[lane][f"{field}_vs_mean"] = relative
                out[lane][f"{field}_z"] = relative / std
    return out


def _series_relative_value(row: sqlite3.Row, field: str) -> float | None:
    raw = row[field]
    if raw is None:
        return None
    if field != "series_starts":
        has_results = row["series_has_results"]
        if has_results is None or _num(has_results) <= 0:
            return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _ranks(
    values: dict[int, float],
    *,
    high_is_good: bool,
    present: dict[int, bool] | None = None,
) -> dict[int, int]:
    valid = [
        (lane, value)
        for lane, value in values.items()
        if (present.get(lane, False) if present is not None else value >= 0)
    ]
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


def _legacy_ranks(values: dict[int, float], *, high_is_good: bool) -> dict[int, int]:
    ordered = sorted(
        values.items(),
        key=lambda item: (item[1] < 0, -item[1] if high_is_good else item[1], item[0]),
    )
    return {lane: index + 1 for index, (lane, _) in enumerate(ordered)}
