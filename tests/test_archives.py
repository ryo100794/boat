from datetime import date
from pathlib import Path

from boatrace_ai.db import connection, init_db
from boatrace_ai.ingestion.archive import parse_result_text
from boatrace_ai.ingestion.archives import extract_lzh


def test_extract_lzh_reads_official_archive_without_context_manager() -> None:
    archive = Path("data/raw/result/2022/20220609.lzh")
    if not archive.exists():
        return

    members = extract_lzh(archive)

    assert len(members) == 1
    assert members[0][0] == "K220609.TXT"
    assert members[0][1].startswith(b"STARTK")


def test_result_lzh_retains_trifecta_dead_heat_continuation(tmp_path) -> None:
    db_path = tmp_path / "dead-heat.sqlite"
    init_db(db_path)
    text = """
09KBGN
12R  ツッキー選抜  H1800m
01  1 3959 坪 井  康 晴 50  11  6.74  1  0.07
02  3 4550 石 岡  将 太 35  37  6.79  3  0.23
03  5 4604 岩 瀬  裕 亮 16  53  6.79  5  0.15
03  6 5004 河 野  主 樹 44  57  6.86  6  0.15
3連単   1-3-5      590  人気     4
         1-3-6     3940  人気    19
3連複   1-3-5      250  人気     3
         1-3-6     2180  人気    10
09KEND
"""
    with connection(db_path) as conn:
        result = parse_result_text(conn, text=text, race_date=date(2026, 3, 27))
        payouts = conn.execute(
            "SELECT bet_type, combination, payout_yen, popularity "
            "FROM payouts ORDER BY CASE bet_type WHEN '3連単' THEN 1 ELSE 2 END, "
            "combination"
        ).fetchall()

    assert result["payouts"] == 4
    assert [tuple(row) for row in payouts] == [
        ("3連単", "1-3-5", 590, 4),
        ("3連単", "1-3-6", 3940, 19),
        ("3連複", "1-3-5", 250, 3),
        ("3連複", "1-3-6", 2180, 10),
    ]
