from __future__ import annotations

import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from .constants import CLASS_RANK, LANES
from .odds_quality import TRIFECTA_PARSER_VERSION, plausible_trifecta_odds


STORED_START_TO_BETTING_DEADLINE_MINUTES = 5
MODEL_DECISION_LEAD_MINUTES = 5
MODEL_FEATURE_CUTOFF_FROM_START_MINUTES = (
    STORED_START_TO_BETTING_DEADLINE_MINUTES + MODEL_DECISION_LEAD_MINUTES
)


def stored_jst_timestamp_sql(conn, expression: str) -> str:
    if getattr(conn, "dialect", "sqlite") == "postgresql":
        return (
            f"(CASE WHEN CAST({expression} AS text) ~ "
            "'(Z|[+-][0-9]{2}:[0-9]{2})$' "
            f"THEN CAST({expression} AS timestamptz) "
            f"ELSE CAST({expression} AS timestamp) AT TIME ZONE 'Asia/Tokyo' END)"
        )
    return (
        f"(CASE WHEN substr({expression}, -1) = 'Z' "
        f"OR substr({expression}, -6, 1) IN ('+', '-') "
        f"THEN datetime({expression}) "
        f"ELSE datetime({expression}, '-9 hours') END)"
    )


def pre_t5_odds_count_sql(conn, *, race_alias: str = "r") -> str:
    start_at = stored_jst_timestamp_sql(conn, f"{race_alias}.deadline_at")
    captured = stored_jst_timestamp_sql(conn, "os.captured_at")
    if getattr(conn, "dialect", "sqlite") == "postgresql":
        cutoff = (
            f"{start_at} - INTERVAL '{MODEL_FEATURE_CUTOFF_FROM_START_MINUTES} minutes'"
        )
    else:
        cutoff = (
            f"datetime({start_at}, '-{MODEL_FEATURE_CUTOFF_FROM_START_MINUTES} minutes')"
        )
    return (
        "(SELECT COUNT(*) FROM odds_snapshots os "
        f"WHERE os.race_id = {race_alias}.race_id "
        f"AND os.parser_version = '{TRIFECTA_PARSER_VERSION}' "
        f"AND {captured} <= {cutoff} "
        "AND (SELECT COUNT(*) FROM odds_trifecta ot "
        "WHERE ot.snapshot_id = os.snapshot_id AND ot.odds IS NOT NULL AND ot.odds > 0) = 120) >= ?"
    )


NUMERIC_ENTRY_FIELDS = (
    "age",
    "weight_kg",
    "f_count",
    "l_count",
    "avg_st",
    "national_win_rate",
    "national_2_rate",
    "national_3_rate",
    "local_win_rate",
    "local_2_rate",
    "local_3_rate",
    "motor_no",
    "motor_2_rate",
    "motor_3_rate",
    "boat_no",
    "boat_2_rate",
    "boat_3_rate",
)


def load_training_examples(
    conn: sqlite3.Connection,
    *,
    through_date: str | None = None,
    from_date: str | None = None,
    include_odds: bool = False,
    min_odds_snapshots: int = 0,
    complete_results_only: bool = False,
) -> tuple[list[dict[str, Any]], list[int], list[dict[str, Any]]]:
    filters = ["rr.rank IS NOT NULL"]
    params: list[Any] = []
    if through_date:
        filters.append("r.race_date <= ?")
        params.append(through_date)
    if from_date:
        filters.append("r.race_date >= ?")
        params.append(from_date)
    if min_odds_snapshots > 0:
        filters.append(pre_t5_odds_count_sql(conn))
        params.append(int(min_odds_snapshots))
    if complete_results_only:
        filters.append(
            "(SELECT COUNT(rr2.rank) FROM race_results rr2 "
            "WHERE rr2.race_id = r.race_id AND rr2.rank IS NOT NULL) = 6"
        )
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
    odds_by_race = {}
    if include_odds:
        for race in sorted({row["race_id"] for row in rows}):
            odds_by_race[race] = odds_lane_features(conn, race)

    features: list[dict[str, Any]] = []
    labels: list[int] = []
    meta: list[dict[str, Any]] = []
    for row in rows:
        odds_features = odds_by_race.get(row["race_id"], {}).get(row["lane"], {})
        features.append(entry_features(row, odds_features=odds_features))
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


def load_race_entries(
    conn: sqlite3.Connection,
    *,
    race_id: str,
) -> list[sqlite3.Row]:
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


def entry_features(
    row: sqlite3.Row | dict[str, Any],
    *,
    odds_features: dict[str, Any] | None = None,
) -> dict[str, Any]:
    get = row.get if isinstance(row, dict) else row.__getitem__
    lane = int(get("lane"))
    features: dict[str, Any] = {
        "lane": str(lane),
        "lane_num": lane,
        "jcd": str(get("jcd") or ""),
        "rno": int(get("rno") or 0),
        "race_type": str(get("race_type") or ""),
        "distance_m": _num(get("distance_m")),
        "racer_class": str(get("racer_class") or ""),
        "class_rank": CLASS_RANK.get(str(get("racer_class") or ""), -1),
        "branch": str(get("branch") or ""),
        "origin": str(get("origin") or ""),
    }
    for field in NUMERIC_ENTRY_FIELDS:
        features[field] = _num(get(field))
    if odds_features:
        features.update({f"odds_{key}": _num(value) for key, value in odds_features.items()})
    return features


def _num(value: Any) -> float:
    if value is None:
        return -1.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return -1.0
    if math.isnan(number) or math.isinf(number):
        return -1.0
    return number


def odds_lane_features(conn: sqlite3.Connection, race_id: str) -> dict[int, dict[str, float]]:
    race = conn.execute(
        "SELECT deadline_at FROM races WHERE race_id = ?",
        (race_id,),
    ).fetchone()
    if not race or not race["deadline_at"]:
        return {}
    deadline_at = _parse_race_time(str(race["deadline_at"]))
    if deadline_at is None:
        return {}
    cutoff_at = deadline_at - timedelta(minutes=MODEL_FEATURE_CUTOFF_FROM_START_MINUTES)
    rows = conn.execute(
        """
        SELECT os.snapshot_id, os.captured_at, ot.combination, ot.odds
        FROM odds_snapshots os
        JOIN odds_trifecta ot ON ot.snapshot_id = os.snapshot_id
        WHERE os.race_id = ?
          AND os.parser_version = ?
          AND ot.odds IS NOT NULL
        ORDER BY os.captured_at, os.snapshot_id
        """,
        (race_id, TRIFECTA_PARSER_VERSION),
    ).fetchall()
    rows = [
        row
        for row in rows
        if (
            (captured := _parse_snapshot_time(
                str(row["captured_at"]), default_tz=cutoff_at.tzinfo
            ))
            is not None
            and captured <= cutoff_at
        )
    ]
    if not rows:
        return {}

    by_snapshot: dict[int, list[sqlite3.Row]] = defaultdict(list)
    ordered_ids: list[int] = []
    for row in rows:
        sid = int(row["snapshot_id"])
        if sid not in by_snapshot:
            ordered_ids.append(sid)
        by_snapshot[sid].append(row)
    valid_ids = [
        snapshot_id
        for snapshot_id in ordered_ids
        if plausible_trifecta_odds(
            {
                str(row["combination"]): float(row["odds"])
                for row in by_snapshot[snapshot_id]
            }
        )
    ]
    if not valid_ids:
        return {}
    first_rows = by_snapshot[valid_ids[0]]
    latest_rows = by_snapshot[valid_ids[-1]]
    first = _aggregate_odds(first_rows)
    latest = _aggregate_odds(latest_rows)

    features: dict[int, dict[str, float]] = {}
    snapshot_count = float(len(valid_ids))
    for lane in LANES:
        latest_lane = latest.get(lane, {})
        first_lane = first.get(lane, {})
        features[lane] = {
            "snapshot_count": snapshot_count,
            "first_mean": first_lane.get("mean", -1.0),
            "first_min": first_lane.get("min", -1.0),
            "first_implied_sum": first_lane.get("implied_sum", -1.0),
            "latest_mean": latest_lane.get("mean", -1.0),
            "latest_min": latest_lane.get("min", -1.0),
            "latest_implied_sum": latest_lane.get("implied_sum", -1.0),
            "mean_delta": latest_lane.get("mean", 0.0)
            - first_lane.get("mean", 0.0),
            "min_delta": latest_lane.get("min", 0.0)
            - first_lane.get("min", 0.0),
            "implied_delta": latest_lane.get("implied_sum", 0.0)
            - first_lane.get("implied_sum", 0.0),
        }
    return features


def latest_trifecta_odds(conn: sqlite3.Connection, race_id: str) -> dict[str, float]:
    rows = conn.execute(
        """
        SELECT snapshot_id
        FROM odds_snapshots
        WHERE race_id = ? AND parser_version = ?
        ORDER BY captured_at DESC, snapshot_id DESC
        LIMIT 8
        """,
        (race_id, TRIFECTA_PARSER_VERSION),
    ).fetchall()
    for row in rows:
        odds = {
            item["combination"]: float(item["odds"])
            for item in conn.execute(
                """
                SELECT combination, odds
                FROM odds_trifecta
                WHERE snapshot_id = ? AND odds IS NOT NULL
                """,
                (row["snapshot_id"],),
            ).fetchall()
        }
        if plausible_trifecta_odds(odds):
            return odds
    return {}


def latest_trifecta_odds_before_deadline(
    conn: sqlite3.Connection,
    race_id: str,
    *,
    min_combinations: int = 120,
) -> dict[str, Any] | None:
    race = conn.execute(
        """
        SELECT deadline_at
        FROM races
        WHERE race_id = ?
        """,
        (race_id,),
    ).fetchone()
    if not race or not race["deadline_at"]:
        return None

    start_at = _parse_race_time(str(race["deadline_at"]))
    if start_at is None:
        return None
    odds_deadline_at = start_at - timedelta(minutes=5)

    snapshots = conn.execute(
        """
        SELECT
          os.snapshot_id,
          os.captured_at,
          os.source_update_time,
          COUNT(ot.odds) AS odds_count
        FROM odds_snapshots os
        JOIN odds_trifecta ot ON ot.snapshot_id = os.snapshot_id
        WHERE os.race_id = ?
          AND os.bet_type = 'trifecta'
          AND os.parser_version = ?
          AND ot.odds IS NOT NULL
          AND ot.odds > 0
        GROUP BY os.snapshot_id, os.captured_at, os.source_update_time
        HAVING COUNT(ot.odds) >= ?
        """,
        (race_id, TRIFECTA_PARSER_VERSION, min_combinations),
    ).fetchall()

    eligible = []
    for snapshot in snapshots:
        captured_at = _parse_snapshot_time(str(snapshot["captured_at"]), default_tz=odds_deadline_at.tzinfo)
        if captured_at is None:
            continue
        if captured_at <= odds_deadline_at:
            eligible.append((captured_at, int(snapshot["snapshot_id"]), snapshot))
    if not eligible:
        return None

    for _captured_at, snapshot_id, snapshot in sorted(
        eligible, key=lambda item: (item[0], item[1]), reverse=True
    ):
        odds_rows = conn.execute(
            """
            SELECT combination, odds
            FROM odds_trifecta
            WHERE snapshot_id = ?
              AND odds IS NOT NULL
              AND odds > 0
            """,
            (snapshot_id,),
        ).fetchall()
        odds = {row["combination"]: float(row["odds"]) for row in odds_rows}
        if len(odds) < min_combinations or not plausible_trifecta_odds(odds):
            continue
        return {
            "snapshot_id": snapshot_id,
            "captured_at": snapshot["captured_at"],
            "source_update_time": snapshot["source_update_time"],
            "odds_deadline_at": odds_deadline_at.isoformat(timespec="seconds"),
            "odds_count": len(odds),
            "odds": odds,
        }
    return None


def _parse_race_time(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone(timedelta(hours=9)))
    return parsed


def _parse_snapshot_time(value: str, *, default_tz: timezone | None) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=default_tz or timezone.utc)
    if default_tz is None:
        return parsed
    return parsed.astimezone(default_tz)


def _aggregate_odds(rows: list[sqlite3.Row]) -> dict[int, dict[str, float]]:
    lane_odds: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        combination = row["combination"]
        try:
            first_lane = int(str(combination).split("-")[0])
            value = float(row["odds"])
        except (IndexError, TypeError, ValueError):
            continue
        if value > 0:
            lane_odds[first_lane].append(value)
    result: dict[int, dict[str, float]] = {}
    for lane, odds_values in lane_odds.items():
        implied = [1.0 / value for value in odds_values if value > 0]
        result[lane] = {
            "mean": sum(odds_values) / len(odds_values),
            "min": min(odds_values),
            "implied_sum": sum(implied),
        }
    return result
