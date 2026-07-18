from __future__ import annotations

from pathlib import Path
from typing import Any

from . import webserver_operational39 as prev
from .db import connect


HTML = prev.HTML

_history_base = prev.base
_archive_base = _history_base._archive_base


def archive_history_fast(db_path: Path, query: dict[str, list[str]]) -> dict[str, Any]:
    kind = (_archive_base._one(query, "kind", "racer") or "racer").lower()
    if kind != "lane":
        return _history_base.archive_history_fast(db_path, query)
    days = _history_base.base._bounded_int(
        _archive_base._one(query, "days", str(_history_base._DEFAULT_HISTORY_DAYS)),
        _history_base._DEFAULT_HISTORY_DAYS,
        1,
        _history_base._MAX_DAYS,
    )
    lane = _archive_base._required(query, "lane")
    jcd = _archive_base._one(query, "jcd")
    rno = _archive_base._one(query, "rno")
    with connect(db_path) as conn:
        cutoff_date, latest_date = _history_base.base._recent_cutoff(conn, days)
        summary = _lane_summary(db_path, lane, days)
        rows = _lane_rows(conn, lane, cutoff_date, jcd, rno)
    summary["period_days"] = days
    summary["cutoff_date"] = cutoff_date
    return {
        "kind": "lane",
        "generated_at": _archive_base._now(),
        "summary": summary,
        "rows": rows,
        "period_days": days,
        "cutoff_date": cutoff_date,
        "latest_date": latest_date,
    }


def _lane_summary(db_path: Path, lane: str, days: int) -> dict[str, Any]:
    payload = _archive_base.archive_stats(db_path, {"scope": ["lane"], "days": [str(days)], "min_starts": ["1"]})
    lane_text = str(lane)
    for row in payload.get("rows", []):
        if str(row.get("key")) == lane_text:
            return {
                "lane": int(lane),
                "starts": row.get("starts"),
                "result_rows": row.get("starts"),
                "wins": row.get("wins"),
                "top3": row.get("top3"),
                "win_rate": row.get("win_rate"),
                "top3_rate": row.get("top3_rate"),
                "avg_rank": row.get("avg_rank"),
                "avg_start": row.get("avg_start"),
                "avg_national_win_rate": row.get("avg_national_win_rate"),
                "avg_local_win_rate": row.get("avg_local_win_rate"),
                "avg_motor_2_rate": row.get("avg_motor_2_rate"),
                "avg_boat_2_rate": row.get("avg_boat_2_rate"),
            }
    return {"lane": int(lane), "starts": 0, "result_rows": 0, "wins": 0, "top3": 0}


def _lane_rows(conn, lane: str, cutoff_date: str, jcd: str | None, rno: str | None) -> list[dict[str, Any]]:
    params: list[Any] = [int(lane), cutoff_date]
    filters = ["rr.lane = ?", "r.race_date >= ?", "rr.rank IS NOT NULL"]
    if jcd:
        filters.append("r.jcd = ?")
        params.append(jcd.zfill(2))
    if rno:
        filters.append("r.rno = ?")
        params.append(int(rno))
    return _archive_base._rows(
        conn,
        f"""
        SELECT
          r.race_id, r.race_date, r.jcd, r.venue_name, r.rno,
          rr.lane, e.racer_no, e.racer_name, e.racer_class,
          e.motor_no, e.boat_no, rr.rank, rr.course, rr.start_timing
        FROM race_results rr
        JOIN races r ON r.race_id = rr.race_id
        LEFT JOIN entries e ON e.race_id = rr.race_id AND e.lane = rr.lane
        WHERE {" AND ".join(filters)}
        ORDER BY r.race_date DESC, r.jcd DESC, r.rno DESC
        LIMIT 80
        """,
        tuple(params),
    )


archive_history = archive_history_fast
_history_base.archive_history = archive_history_fast
_archive_base.archive_history = archive_history_fast


def main(argv: list[str] | None = None) -> int:
    return prev.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
