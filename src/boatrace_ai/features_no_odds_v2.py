from __future__ import annotations

import sqlite3
from collections import defaultdict
from typing import Any

from .constants import CLASS_RANK
from .features import NUMERIC_ENTRY_FIELDS, _num, entry_features, odds_lane_features


HIGH_IS_GOOD_FIELDS = (
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

LOW_IS_GOOD_FIELDS = (
    "avg_st",
    "f_count",
    "l_count",
    "weight_kg",
)

RELATIVE_FIELDS = HIGH_IS_GOOD_FIELDS + LOW_IS_GOOD_FIELDS + ("age",)


def load_training_examples(
    conn: sqlite3.Connection,
    *,
    through_date: str | None = None,
    from_date: str | None = None,
    include_odds: bool = False,
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
          rr.rank
        FROM entries e
        JOIN races r ON r.race_id = e.race_id
        JOIN race_results rr ON rr.race_id = e.race_id AND rr.lane = e.lane
        WHERE {" AND ".join(filters)}
        ORDER BY r.race_date, r.jcd, r.rno, e.lane
        """,
        params,
    ).fetchall()

    grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        grouped[row["race_id"]].append(row)

    odds_by_race = {}
    if include_odds:
        for race in sorted(grouped):
            odds_by_race[race] = odds_lane_features(conn, race)

    features: list[dict[str, Any]] = []
    labels: list[int] = []
    meta: list[dict[str, Any]] = []
    for race_id_value in sorted(grouped):
        race_rows = sorted(grouped[race_id_value], key=lambda row: int(row["lane"]))
        if len(race_rows) != 6:
            continue
        relative = race_relative_features(race_rows)
        for row in race_rows:
            lane = int(row["lane"])
            odds_features = odds_by_race.get(race_id_value, {}).get(lane, {})
            item = entry_features(row, odds_features=odds_features)
            item.update(relative[lane])
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


def race_features(row: sqlite3.Row, race_rows: list[sqlite3.Row], *, odds_features: dict[str, Any] | None = None) -> dict[str, Any]:
    features = entry_features(row, odds_features=odds_features)
    features.update(race_relative_features(race_rows)[int(row["lane"])])
    return features


def race_relative_features(rows: list[sqlite3.Row]) -> dict[int, dict[str, Any]]:
    raw_by_lane: dict[int, dict[str, float]] = {}
    for row in rows:
        lane = int(row["lane"])
        values = {field: _field_value(row, field) for field in RELATIVE_FIELDS}
        raw_by_lane[lane] = values

    field_stats = {field: _stats([values[field] for values in raw_by_lane.values()]) for field in RELATIVE_FIELDS}
    ranks = {
        field: _ranks({lane: values[field] for lane, values in raw_by_lane.items()}, high_is_good=field not in LOW_IS_GOOD_FIELDS)
        for field in RELATIVE_FIELDS
    }

    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        lane = int(row["lane"])
        values = raw_by_lane[lane]
        item: dict[str, Any] = {
            "field_size": len(rows),
            "lane_rno": f"{lane}:{int(row['rno'] or 0)}",
            "lane_jcd": f"{lane}:{row['jcd'] or ''}",
            "lane_class": f"{lane}:{row['racer_class'] or ''}",
        }
        for field in RELATIVE_FIELDS:
            value = values[field]
            stats = field_stats[field]
            mean = stats["mean"]
            best = stats["max"] if field not in LOW_IS_GOOD_FIELDS else stats["min"]
            worst = stats["min"] if field not in LOW_IS_GOOD_FIELDS else stats["max"]
            sign = 1.0 if field not in LOW_IS_GOOD_FIELDS else -1.0
            spread = max(1e-6, abs(stats["max"] - stats["min"]))
            item[f"{field}_field_mean"] = mean
            item[f"{field}_vs_field_mean"] = value - mean
            item[f"{field}_z"] = (value - mean) / max(1e-6, stats["std"])
            item[f"{field}_rank"] = ranks[field][lane]
            item[f"{field}_best_gap"] = sign * (value - best)
            item[f"{field}_worst_gap"] = sign * (value - worst)
            item[f"{field}_scaled_field"] = sign * (value - worst) / spread
        result[lane] = item
    _add_composite_features(result, raw_by_lane, ranks)
    return result


def _field_value(row: sqlite3.Row, field: str) -> float:
    if field == "class_rank":
        return float(CLASS_RANK.get(str(row["racer_class"] or ""), -1))
    return _num(row[field])


def _stats(values: list[float]) -> dict[str, float]:
    valid = [value for value in values if value >= 0]
    if not valid:
        return {"mean": -1.0, "std": 1.0, "min": -1.0, "max": -1.0}
    mean = sum(valid) / len(valid)
    variance = sum((value - mean) ** 2 for value in valid) / len(valid)
    return {
        "mean": mean,
        "std": variance ** 0.5,
        "min": min(valid),
        "max": max(valid),
    }


def _ranks(values: dict[int, float], *, high_is_good: bool) -> dict[int, int]:
    ordered = sorted(
        values.items(),
        key=lambda item: (item[1] < 0, -item[1] if high_is_good else item[1], item[0]),
    )
    return {lane: index + 1 for index, (lane, _) in enumerate(ordered)}


def _add_composite_features(
    result: dict[int, dict[str, Any]],
    raw_by_lane: dict[int, dict[str, float]],
    ranks: dict[str, dict[int, int]],
) -> None:
    for lane, item in result.items():
        values = raw_by_lane[lane]
        ability_score = _avg_positive(
            values["national_win_rate"],
            values["national_2_rate"],
            values["local_win_rate"],
            values["local_2_rate"],
            values["motor_2_rate"],
            values["boat_2_rate"],
        )
        stability_score = _avg_positive(
            -values["avg_st"] if values["avg_st"] >= 0 else -1,
            -values["f_count"] if values["f_count"] >= 0 else -1,
            -values["l_count"] if values["l_count"] >= 0 else -1,
        )
        item["ability_score"] = ability_score
        item["stability_score"] = stability_score
        item["ability_lane_score"] = ability_score * max(0, 7 - lane)
        item["is_best_national_win_rate"] = int(ranks["national_win_rate"][lane] == 1)
        item["is_best_local_win_rate"] = int(ranks["local_win_rate"][lane] == 1)
        item["is_best_motor_2_rate"] = int(ranks["motor_2_rate"][lane] == 1)
        item["is_best_boat_2_rate"] = int(ranks["boat_2_rate"][lane] == 1)
        item["is_best_avg_st"] = int(ranks["avg_st"][lane] == 1)
        item["best_count"] = sum(
            int(ranks[field][lane] == 1)
            for field in (
                "national_win_rate",
                "local_win_rate",
                "motor_2_rate",
                "boat_2_rate",
                "avg_st",
                "class_rank",
            )
        )


def _avg_positive(*values: float) -> float:
    valid = [value for value in values if value >= 0]
    if not valid:
        return -1.0
    return sum(valid) / len(valid)
