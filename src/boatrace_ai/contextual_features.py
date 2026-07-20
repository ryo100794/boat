from __future__ import annotations

import math
import sqlite3
from collections import defaultdict
from datetime import date
from typing import Any

from .features import _num, entry_features
from .base_features import (
    _group_by_race,
    _latest_beforeinfo,
    before_features,
    is_home_branch,
    race_relative_features,
)


def load_training_examples(
    conn: sqlite3.Connection,
    *,
    through_date: str | None = None,
    from_date: str | None = None,
    include_odds: bool = False,
) -> tuple[list[dict[str, Any]], list[int], list[dict[str, Any]]]:
    if include_odds:
        raise ValueError("contextual_features is intentionally no-odds for leakage-safe backtesting")
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
        for row in race_rows:
            lane = int(row["lane"])
            item = _base_features(row, before_rows.get(lane, {}), relatives[lane])
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
        raise ValueError("contextual_features is intentionally no-odds")
    rows = load_race_entries(conn, race_id=race_id)
    if len(rows) != 6:
        return []
    state = RollingState()
    for history_rows in _history_groups_before(conn, rows[0]):
        state.update_race(history_rows)
    before_rows = _latest_beforeinfo(conn, race_id=race_id)
    by_lane = {lane: before_rows.get((race_id, lane), {}) for lane in range(1, 7)}
    relatives = race_relative_features(rows, by_lane)
    result = []
    for row in rows:
        lane = int(row["lane"])
        item = _base_features(row, by_lane.get(lane, {}), relatives[lane])
        item.update(state.features_for(row))
        result.append(item)
    return result


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


def _history_groups_before(conn: sqlite3.Connection, target: sqlite3.Row) -> list[list[sqlite3.Row]]:
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
          AND (
            r.race_date < ?
            OR (r.race_date = ? AND r.jcd < ?)
            OR (r.race_date = ? AND r.jcd = ? AND r.rno < ?)
          )
        ORDER BY r.race_date, r.jcd, r.rno, e.lane
        """,
        (
            target["race_date"],
            target["race_date"],
            target["jcd"],
            target["race_date"],
            target["jcd"],
            target["rno"],
        ),
    ).fetchall()
    grouped = _group_by_race(rows)
    return [
        sorted(grouped[race_id_value], key=lambda row: int(row["lane"]))
        for race_id_value in sorted(grouped, key=lambda rid: _race_sort_key(grouped[rid][0]))
        if len(grouped[race_id_value]) == 6
    ]


def _base_features(
    row: sqlite3.Row,
    before_row: sqlite3.Row | dict[str, Any],
    relatives: dict[str, Any],
) -> dict[str, Any]:
    item = entry_features(row, odds_features={})
    item.update(before_features(before_row))
    item.update(relatives)
    item.pop("motor_no", None)
    item.pop("boat_no", None)
    item["has_motor_no"] = int(_num(row["motor_no"]) >= 0)
    item["has_boat_no"] = int(_num(row["boat_no"]) >= 0)
    item.update(_race_context_features(row))
    return item


def _race_context_features(row: sqlite3.Row) -> dict[str, Any]:
    try:
        race_day = date.fromisoformat(str(row["race_date"]))
    except ValueError:
        race_day = None
    rno = int(row["rno"] or 0)
    distance = int(row["distance_m"] or 0)
    return {
        "race_month": str(race_day.month) if race_day else "",
        "race_weekday": str(race_day.weekday()) if race_day else "",
        "race_rno_bucket": _rno_bucket(rno),
        "distance_bucket": str(distance) if distance else "",
        "lane_rno_bucket": f"{int(row['lane'])}:{_rno_bucket(rno)}",
    }


def _rno_bucket(rno: int) -> str:
    if rno <= 4:
        return "early"
    if rno <= 8:
        return "middle"
    return "late"


class RollingState:
    def __init__(self) -> None:
        self.lane = defaultdict(_empty_bucket)
        self.venue_lane = defaultdict(_empty_bucket)
        self.rno_lane = defaultdict(_empty_bucket)
        self.racer = defaultdict(_empty_bucket)
        self.racer_lane = defaultdict(_empty_bucket)
        self.racer_venue = defaultdict(_empty_bucket)
        self.motor = defaultdict(_empty_bucket)
        self.motor_lane = defaultdict(_empty_bucket)
        self.boat = defaultdict(_empty_bucket)
        self.boat_lane = defaultdict(_empty_bucket)

    def features_for(self, row: sqlite3.Row) -> dict[str, Any]:
        lane = int(row["lane"])
        jcd = str(row["jcd"] or "")
        rno = int(row["rno"] or 0)
        racer_no = int(row["racer_no"] or 0) if row["racer_no"] else 0
        motor_no = int(row["motor_no"] or 0) if row["motor_no"] else 0
        boat_no = int(row["boat_no"] or 0) if row["boat_no"] else 0
        lane_feat = _bucket_features("hist_lane", self.lane[lane], prior=80.0)
        venue_lane_feat = _bucket_features("hist_venue_lane", self.venue_lane[(jcd, lane)], prior=30.0)
        rno_lane_feat = _bucket_features("hist_rno_lane", self.rno_lane[(rno, lane)], prior=30.0)
        racer_feat = _bucket_features("hist_racer", self.racer[racer_no], prior=18.0)
        racer_lane_feat = _bucket_features("hist_racer_lane", self.racer_lane[(racer_no, lane)], prior=12.0)
        racer_venue_feat = _bucket_features("hist_racer_venue", self.racer_venue[(racer_no, jcd)], prior=12.0)
        motor_feat = _bucket_features("hist_motor", self.motor[(jcd, motor_no)], prior=18.0)
        motor_lane_feat = _bucket_features("hist_motor_lane", self.motor_lane[(jcd, motor_no, lane)], prior=12.0)
        boat_feat = _bucket_features("hist_boat", self.boat[(jcd, boat_no)], prior=18.0)
        boat_lane_feat = _bucket_features("hist_boat_lane", self.boat_lane[(jcd, boat_no, lane)], prior=12.0)
        item = {
            **lane_feat,
            **venue_lane_feat,
            **rno_lane_feat,
            **racer_feat,
            **racer_lane_feat,
            **racer_venue_feat,
            **motor_feat,
            **motor_lane_feat,
            **boat_feat,
            **boat_lane_feat,
        }
        item["hist_racer_lane_win_delta"] = (
            item["hist_racer_lane_win_rate_s"] - item["hist_racer_win_rate_s"]
        )
        item["hist_racer_venue_win_delta"] = (
            item["hist_racer_venue_win_rate_s"] - item["hist_racer_win_rate_s"]
        )
        item["research_hist_home_racer_venue_delta"] = (
            float(is_home_branch(jcd, row["branch"]))
            * item["hist_racer_venue_win_delta"]
        )
        item["hist_motor_venue_lane_win_delta"] = (
            item["hist_motor_win_rate_s"] - item["hist_venue_lane_win_rate_s"]
        )
        item["hist_boat_venue_lane_win_delta"] = (
            item["hist_boat_win_rate_s"] - item["hist_venue_lane_win_rate_s"]
        )
        item["hist_racer_start_vs_card"] = _delta(
            item["hist_racer_avg_start_timing"],
            _num(row["avg_st"]),
        )
        return item

    def update_race(self, rows: list[sqlite3.Row]) -> None:
        for row in rows:
            rank = int(row["rank"] or 99)
            lane = int(row["lane"])
            jcd = str(row["jcd"] or "")
            rno = int(row["rno"] or 0)
            racer_no = int(row["racer_no"] or 0) if row["racer_no"] else 0
            motor_no = int(row["motor_no"] or 0) if row["motor_no"] else 0
            boat_no = int(row["boat_no"] or 0) if row["boat_no"] else 0
            start_timing = _num(row["result_start_timing"])
            for bucket in (
                self.lane[lane],
                self.venue_lane[(jcd, lane)],
                self.rno_lane[(rno, lane)],
                self.racer[racer_no],
                self.racer_lane[(racer_no, lane)],
                self.racer_venue[(racer_no, jcd)],
                self.motor[(jcd, motor_no)],
                self.motor_lane[(jcd, motor_no, lane)],
                self.boat[(jcd, boat_no)],
                self.boat_lane[(jcd, boat_no, lane)],
            ):
                _update_bucket(bucket, rank=rank, start_timing=start_timing)


def _empty_bucket() -> dict[str, float]:
    return {
        "count": 0.0,
        "wins": 0.0,
        "top2": 0.0,
        "top3": 0.0,
        "rank_sum": 0.0,
        "start_sum": 0.0,
        "start_count": 0.0,
    }


def _update_bucket(bucket: dict[str, float], *, rank: int, start_timing: float) -> None:
    if rank <= 0 or rank >= 99:
        return
    bucket["count"] += 1.0
    bucket["wins"] += float(rank == 1)
    bucket["top2"] += float(rank <= 2)
    bucket["top3"] += float(rank <= 3)
    bucket["rank_sum"] += float(rank)
    if start_timing != -1.0:
        bucket["start_sum"] += start_timing
        bucket["start_count"] += 1.0


def _bucket_features(prefix: str, bucket: dict[str, float], *, prior: float) -> dict[str, float]:
    count = bucket["count"]
    avg_rank = (
        (bucket["rank_sum"] + prior * 3.5) / (count + prior)
        if count + prior
        else 3.5
    )
    return {
        f"{prefix}_count_log": math.log1p(count),
        f"{prefix}_win_rate": bucket["wins"] / count if count else -1.0,
        f"{prefix}_top2_rate": bucket["top2"] / count if count else -1.0,
        f"{prefix}_top3_rate": bucket["top3"] / count if count else -1.0,
        f"{prefix}_win_rate_s": _smooth(bucket["wins"], count, prior, 1.0 / 6.0),
        f"{prefix}_top2_rate_s": _smooth(bucket["top2"], count, prior, 2.0 / 6.0),
        f"{prefix}_top3_rate_s": _smooth(bucket["top3"], count, prior, 3.0 / 6.0),
        f"{prefix}_avg_rank_s": avg_rank,
        f"{prefix}_avg_start_timing": (
            bucket["start_sum"] / bucket["start_count"] if bucket["start_count"] else -1.0
        ),
    }


def _smooth(successes: float, count: float, prior: float, base: float) -> float:
    return (successes + prior * base) / (count + prior)


def _delta(left: float, right: float) -> float:
    if left == -1.0 or right == -1.0:
        return 0.0
    return left - right


def _race_sort_key(row: sqlite3.Row) -> tuple[str, str, int]:
    return (str(row["race_date"] or ""), str(row["jcd"] or ""), int(row["rno"] or 0))
