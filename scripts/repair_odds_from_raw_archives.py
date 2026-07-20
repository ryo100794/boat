#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from boatrace_ai.ingestion.parsers import parse_odds3t_html
from boatrace_ai.postgresql import connection


JST = timezone(timedelta(hours=9))
MEMBER_PATTERN = re.compile(
    r"(?:^|/)pages/(?P<date>\d{8})/(?P<jcd>\d{2})/(?P<rno>\d{2})/"
    r"odds3t-(?P<stamp>\d{8}T\d{6}Z)\.html$"
)


def repair_archives(
    *,
    dsn: str,
    archive_dir: Path,
    race_date: str,
) -> dict[str, Any]:
    target_ymd = race_date.replace("-", "")
    archives = sorted(archive_dir.glob("*.tar.zst"))
    with connection(dsn) as conn:
        race_rows = conn.execute(
            """
            SELECT race_id, jcd, rno, deadline_at
            FROM races
            WHERE race_date = ? AND deadline_at IS NOT NULL
            """,
            (race_date,),
        ).fetchall()
        cutoffs = {
            str(row["race_id"]): _race_cutoff(str(row["deadline_at"]))
            for row in race_rows
        }
        snapshots = _snapshot_inventory(conn, race_date)

        selected: dict[str, tuple[datetime, Path, str]] = {}
        listed_members = 0
        for archive in archives:
            members = _tar_members(archive)
            listed_members += len(members)
            for member in members:
                match = MEMBER_PATTERN.search(member)
                if not match or match.group("date") != target_ymd:
                    continue
                race_id = (
                    f"{race_date}-{match.group('jcd')}-"
                    f"{int(match.group('rno')):02d}"
                )
                cutoff = cutoffs.get(race_id)
                if cutoff is None:
                    continue
                captured_at = datetime.strptime(
                    match.group("stamp"), "%Y%m%dT%H%M%SZ"
                ).replace(tzinfo=timezone.utc)
                if captured_at > cutoff:
                    continue
                previous = selected.get(race_id)
                if previous is None or captured_at > previous[0]:
                    selected[race_id] = (captured_at, archive, member)

        repaired = 0
        parse_failed = 0
        snapshot_missing = 0
        details = []
        for race_id, (captured_at, archive, member) in sorted(selected.items()):
            html = _tar_member(archive, member).decode("utf-8", errors="replace")
            parsed = parse_odds3t_html(html)
            if (
                parsed.get("parser_version") != "odds3t_dom_v2"
                or int(parsed.get("parsed_count") or 0) != 120
            ):
                parse_failed += 1
                continue
            snapshot_id = _closest_snapshot(
                snapshots.get(race_id, []),
                captured_at,
            )
            if snapshot_id is None:
                snapshot_missing += 1
                continue
            conn.execute(
                """
                UPDATE odds_snapshots
                SET parser_version = ?, raw_json = ?
                WHERE snapshot_id = ?
                """,
                (
                    "odds3t_dom_v2",
                    json.dumps(parsed, ensure_ascii=False, sort_keys=True),
                    snapshot_id,
                ),
            )
            conn.executemany(
                """
                UPDATE odds_trifecta
                SET odds = ?
                WHERE snapshot_id = ? AND combination = ?
                """,
                [
                    (odds, snapshot_id, combination)
                    for combination, odds in parsed["odds"].items()
                ],
            )
            repaired += 1
            details.append(
                {
                    "race_id": race_id,
                    "snapshot_id": snapshot_id,
                    "captured_at": captured_at.isoformat(),
                    "archive": archive.name,
                }
            )
    return {
        "race_date": race_date,
        "archives": len(archives),
        "listed_members": listed_members,
        "selected_races": len(selected),
        "repaired_races": repaired,
        "parse_failed": parse_failed,
        "snapshot_missing": snapshot_missing,
        "details": details,
    }


def _race_cutoff(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=JST)
    return parsed.astimezone(timezone.utc) - timedelta(minutes=10)


def _snapshot_inventory(conn, race_date: str) -> dict[str, list[tuple[int, datetime]]]:
    rows = conn.execute(
        """
        SELECT os.race_id, os.snapshot_id, os.captured_at
        FROM odds_snapshots os
        JOIN races r ON r.race_id = os.race_id
        WHERE r.race_date = ? AND os.bet_type = 'trifecta'
        """,
        (race_date,),
    ).fetchall()
    result: dict[str, list[tuple[int, datetime]]] = defaultdict(list)
    for row in rows:
        captured = datetime.fromisoformat(str(row["captured_at"]))
        if captured.tzinfo is None:
            captured = captured.replace(tzinfo=timezone.utc)
        result[str(row["race_id"])].append(
            (int(row["snapshot_id"]), captured.astimezone(timezone.utc))
        )
    return result


def _closest_snapshot(
    snapshots: list[tuple[int, datetime]],
    captured_at: datetime,
    *,
    tolerance_seconds: int = 90,
) -> int | None:
    if not snapshots:
        return None
    snapshot_id, found_at = min(
        snapshots,
        key=lambda item: abs((item[1] - captured_at).total_seconds()),
    )
    if abs((found_at - captured_at).total_seconds()) > tolerance_seconds:
        return None
    return snapshot_id


def _tar_members(archive: Path) -> list[str]:
    result = subprocess.run(
        ["tar", "--zstd", "-tf", str(archive)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def _tar_member(archive: Path, member: str) -> bytes:
    result = subprocess.run(
        ["tar", "--zstd", "-xOf", str(archive), member],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repair one pre-cutoff odds snapshot per race from raw archives."
    )
    parser.add_argument("--postgres-dsn", required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = repair_archives(
        dsn=args.postgres_dsn,
        archive_dir=args.archive_dir,
        race_date=args.date,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if result["repaired_races"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
