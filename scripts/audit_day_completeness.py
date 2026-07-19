from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


JST = timezone(timedelta(hours=9))


def parse_start(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=JST) if parsed.tzinfo is None else parsed.astimezone(JST)


def parse_captured(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def count_by_race(
    conn: sqlite3.Connection,
    table: str,
    race_date: str,
    expression: str = "COUNT(*)",
) -> dict[str, int]:
    return {
        str(row[0]): int(row[1] or 0)
        for row in conn.execute(
            f"""
            SELECT t.race_id, {expression}
            FROM {table} t
            JOIN races r ON r.race_id = t.race_id
            WHERE r.race_date = ?
            GROUP BY t.race_id
            """,
            (race_date,),
        )
    }


def audit(conn: sqlite3.Connection, race_date: str) -> dict[str, Any]:
    conn.row_factory = sqlite3.Row
    races = [
        dict(row)
        for row in conn.execute(
            """
            SELECT race_id, jcd, rno, deadline_at, status
            FROM races
            WHERE race_date = ? AND deadline_at IS NOT NULL
            ORDER BY jcd, rno
            """,
            (race_date,),
        )
    ]
    race_ids = {str(row["race_id"]) for row in races}
    entries = count_by_race(conn, "entries", race_date, "COUNT(DISTINCT t.lane)")
    ranks = count_by_race(
        conn,
        "race_results",
        race_date,
        "COUNT(DISTINCT CASE WHEN t.rank IS NOT NULL THEN t.lane END)",
    )
    payouts = count_by_race(conn, "payouts", race_date)
    predictions = count_by_race(
        conn,
        "predictions",
        race_date,
        "COUNT(DISTINCT t.generated_at)",
    )
    statuses = {
        str(row["race_id"]): dict(row)
        for row in conn.execute(
            """
            SELECT rs.race_id, rs.status, rs.trifecta_evaluable, rs.reason, rs.finish_rows, rs.payout_rows
            FROM race_result_status rs
            JOIN races r ON r.race_id = rs.race_id
            WHERE r.race_date = ?
            """,
            (race_date,),
        )
    }

    snapshot_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in conn.execute(
        """
        SELECT os.race_id, os.snapshot_id, os.captured_at, COUNT(DISTINCT ot.combination) AS combinations
        FROM odds_snapshots os
        JOIN races r ON r.race_id = os.race_id
        LEFT JOIN odds_trifecta ot ON ot.snapshot_id = os.snapshot_id
        WHERE os.bet_type = 'trifecta' AND r.race_date = ?
        GROUP BY os.snapshot_id
        """,
        (race_date,),
    ):
        race_id = str(row["race_id"])
        if race_id in race_ids:
            snapshot_rows[race_id].append(dict(row))

    close_gap_buckets: Counter[str] = Counter()
    latest_pre_deadline_gap: dict[str, float] = {}
    complete_snapshot_races: set[str] = set()
    snapshot_races: set[str] = set()
    for race in races:
        race_id = str(race["race_id"])
        snapshots = snapshot_rows.get(race_id, [])
        if snapshots:
            snapshot_races.add(race_id)
        complete = [row for row in snapshots if int(row["combinations"] or 0) == 120]
        if complete:
            complete_snapshot_races.add(race_id)
        start = parse_start(race.get("deadline_at"))
        cutoff = start - timedelta(minutes=5) if start else None
        before = []
        for row in complete:
            captured = parse_captured(row.get("captured_at"))
            if captured and cutoff and captured.astimezone(JST) <= cutoff:
                before.append(captured.astimezone(JST))
        if not before or cutoff is None:
            close_gap_buckets["no_complete_pre_deadline"] += 1
            continue
        gap = max(0.0, (cutoff - max(before)).total_seconds())
        latest_pre_deadline_gap[race_id] = gap
        if gap <= 60:
            close_gap_buckets["0_60s"] += 1
        elif gap <= 120:
            close_gap_buckets["61_120s"] += 1
        elif gap <= 300:
            close_gap_buckets["121_300s"] += 1
        else:
            close_gap_buckets["over_300s"] += 1

    final_races = {
        race_id
        for race_id in race_ids
        if ranks.get(race_id, 0) >= 3
        or str(statuses.get(race_id, {}).get("status") or "").lower() == "final"
    }
    evaluable_races = {
        race_id for race_id in race_ids if statuses.get(race_id, {}).get("trifecta_evaluable") == 1
    }
    venue_rnos: dict[str, set[int]] = defaultdict(set)
    for race in races:
        venue_rnos[str(race["jcd"])].add(int(race["rno"]))
    incomplete_venues = {
        venue: sorted(set(range(1, 13)) - rnos)
        for venue, rnos in venue_rnos.items()
        if rnos != set(range(1, 13))
    }

    missing = {
        "entries_not_6": sorted(race_id for race_id in race_ids if entries.get(race_id, 0) != 6),
        "result_not_final": sorted(race_ids - final_races),
        "no_payout_rows": sorted(race_id for race_id in race_ids if payouts.get(race_id, 0) == 0),
        "no_odds_snapshot": sorted(race_ids - snapshot_races),
        "no_complete_odds_snapshot": sorted(race_ids - complete_snapshot_races),
        "no_complete_pre_deadline_snapshot": sorted(race_ids - set(latest_pre_deadline_gap)),
        "no_prediction": sorted(race_id for race_id in race_ids if predictions.get(race_id, 0) == 0),
    }
    return {
        "race_date": race_date,
        "races": len(races),
        "venues": len(venue_rnos),
        "venue_schedule_complete": not incomplete_venues,
        "incomplete_venues": incomplete_venues,
        "entries_complete_races": sum(entries.get(race_id, 0) == 6 for race_id in race_ids),
        "final_races": len(final_races),
        "trifecta_evaluable_races": len(evaluable_races),
        "payout_races": sum(payouts.get(race_id, 0) > 0 for race_id in race_ids),
        "odds_snapshot_races": len(snapshot_races),
        "complete_odds_snapshot_races": len(complete_snapshot_races),
        "close_snapshot_gap_buckets": dict(sorted(close_gap_buckets.items())),
        "prediction_races": sum(predictions.get(race_id, 0) > 0 for race_id in race_ids),
        "missing_counts": {key: len(value) for key, value in missing.items()},
        "missing_race_ids": missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit per-race collection completeness for one day")
    parser.add_argument("--db", default="data/boatrace.sqlite", type=Path)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with sqlite3.connect(args.db) as conn:
        result = audit(conn, args.date)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
