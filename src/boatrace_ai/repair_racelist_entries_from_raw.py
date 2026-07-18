from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path

from .db import connection, init_db, upsert_entry, upsert_race
from .official import race_page_url
from .racelist_parser_dom import parse_racelist_html


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Repair racelist entries by reparsing saved official HTML.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--date", default=date.today().isoformat())
    args = parser.parse_args(argv)

    init_db(args.db)
    target = date.fromisoformat(args.date)
    page_root = Path(args.raw_dir) / "pages" / target.strftime("%Y%m%d")
    fixed = 0
    parsed = 0
    failed = 0
    with connection(args.db) as conn:
        for path in latest_racelist_pages(page_root):
            match = re.search(r"/(\d{2})/(\d{2})/racelist-", path.as_posix())
            if not match:
                continue
            jcd, rno_text = match.group(1), match.group(2)
            rno = int(rno_text)
            html = path.read_text(encoding="utf-8", errors="ignore")
            try:
                meta, entries = parse_racelist_html(
                    html,
                    race_date=target,
                    jcd=jcd,
                    rno=rno,
                    source_url=race_page_url("racelist", target, jcd, rno),
                )
            except Exception:
                failed += 1
                continue
            if len(entries) != 6:
                failed += 1
                continue
            rid = upsert_race(conn, meta)
            for entry in entries:
                upsert_entry(conn, rid, entry)
                fixed += 1
            parsed += 1
        conn.commit()
    print(json.dumps({"date": args.date, "parsed_races": parsed, "fixed_entries": fixed, "failed_pages": failed}, ensure_ascii=False), flush=True)
    return 0


def latest_racelist_pages(page_root: Path) -> list[Path]:
    latest: dict[tuple[str, str], Path] = {}
    if not page_root.exists():
        return []
    for path in page_root.glob("*/??/racelist-*.html"):
        parts = path.parts
        key = (parts[-3], parts[-2])
        current = latest.get(key)
        if current is None or path.name > current.name:
            latest[key] = path
    return [latest[key] for key in sorted(latest)]


if __name__ == "__main__":
    raise SystemExit(main())

