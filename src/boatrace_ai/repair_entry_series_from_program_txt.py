from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

from .constants import VENUE_BY_CODE
from .db import connection, init_db, race_id
from .ingestion.archive_extended import normalize, parse_program_entry


def repair_series_from_txt(
    conn,
    *,
    raw_dir: Path,
    from_date: str | None = None,
    through_date: str | None = None,
    limit_files: int | None = None,
) -> dict[str, Any]:
    files = sorted((raw_dir / "program").glob("**/B*.TXT"), reverse=True)
    stats = {
        "direction": "newest_to_oldest",
        "files_seen": 0,
        "files_used": 0,
        "entries_seen": 0,
        "entries_updated": 0,
        "entries_unmatched": 0,
        "parse_miss": 0,
    }
    for path in files:
        race_date = _date_from_path(path)
        if race_date is None:
            continue
        if from_date and race_date.isoformat() < from_date:
            continue
        if through_date and race_date.isoformat() > through_date:
            continue
        if limit_files is not None and stats["files_used"] >= limit_files:
            break
        stats["files_seen"] += 1
        event = repair_file(conn, path=path, race_date=race_date)
        stats["files_used"] += 1
        for key in ("entries_seen", "entries_updated", "entries_unmatched", "parse_miss"):
            stats[key] += int(event[key])
        conn.commit()
        print(json.dumps({"file": str(path), "date": race_date.isoformat(), **event, "stats": stats}, ensure_ascii=False), flush=True)
    return stats


def repair_file(conn, *, path: Path, race_date: date) -> dict[str, int]:
    current_jcd: str | None = None
    current_race_id: str | None = None
    stats = {"entries_seen": 0, "entries_updated": 0, "entries_unmatched": 0, "parse_miss": 0}
    text = path.read_text(encoding="utf-8", errors="replace")
    for raw_line in text.splitlines():
        line = normalize(raw_line)
        section = re.match(r"^(?P<jcd>\d{2})BBGN", line)
        if section:
            current_jcd = section.group("jcd")
            current_race_id = None
            continue
        if not current_jcd or current_jcd not in VENUE_BY_CODE:
            continue
        race_match = re.match(
            r"^\s*(?P<rno>\d{1,2})R\s+(?P<title>.*?)\s+H(?P<distance>\d{3,4})m.*?(?P<deadline>\d{1,2}:\d{2})",
            line,
        )
        if race_match:
            current_race_id = race_id(race_date.isoformat(), current_jcd, int(race_match.group("rno")))
            continue
        if not current_race_id or not re.match(r"^\s*[1-6]\s*\d{4}", line):
            continue
        stats["entries_seen"] += 1
        entry = parse_program_entry(line)
        if not entry:
            stats["parse_miss"] += 1
            continue
        payload = json.dumps(entry, ensure_ascii=False, sort_keys=True)
        cur = conn.execute(
            """
            UPDATE entries
            SET raw_json = ?, updated_at = CURRENT_TIMESTAMP
            WHERE race_id = ? AND lane = ?
            """,
            (payload, current_race_id, int(entry["lane"])),
        )
        if cur.rowcount:
            stats["entries_updated"] += int(cur.rowcount)
        else:
            stats["entries_unmatched"] += 1
    return stats


def _date_from_path(path: Path) -> date | None:
    candidates = [path.parent.name, path.stem[1:] if path.stem.startswith("B") else path.stem]
    for candidate in candidates:
        if re.fullmatch(r"\d{8}", candidate):
            try:
                return date(int(candidate[:4]), int(candidate[4:6]), int(candidate[6:8]))
            except ValueError:
                continue
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Populate entries.raw_json with series form from extracted B*.TXT files.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--from-date")
    parser.add_argument("--through-date")
    parser.add_argument("--limit-files", type=int)
    args = parser.parse_args(argv)
    init_db(args.db)
    with connection(args.db) as conn:
        stats = repair_series_from_txt(
            conn,
            raw_dir=Path(args.raw_dir),
            from_date=args.from_date,
            through_date=args.through_date,
            limit_files=args.limit_files,
        )
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
