from datetime import date
from pathlib import Path

from boatrace_ai.db import connect, init_db
from boatrace_ai.ingestion.backfill import parse_archive


def test_safe_backfill_uses_official_fixed_width_parser(tmp_path: Path) -> None:
    archive = Path("data/raw/result/2022/20220609.lzh")
    if not archive.exists():
        return
    db_path = tmp_path / "archive.sqlite"
    init_db(db_path)

    with connect(db_path) as conn:
        parsed = parse_archive(
            conn,
            path=archive,
            kind="result",
            race_date=date(2022, 6, 9),
        )
        conn.commit()
        race_count = conn.execute(
            "SELECT COUNT(*) FROM races WHERE race_date = '2022-06-09'"
        ).fetchone()[0]
        result_count = conn.execute(
            "SELECT COUNT(*) FROM race_results"
        ).fetchone()[0]

    assert parsed["races"] == 120
    assert parsed["results"] == 707
    assert parsed["payouts"] == 839
    assert race_count == 120
    assert result_count == 707
