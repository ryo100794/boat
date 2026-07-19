from datetime import datetime

from boatrace_ai.db import connection, init_db, upsert_race
from boatrace_ai.web.dashboard import (
    JST,
    _PROGRESS_CACHE,
    _race_is_final,
    _venue_display_row,
    progress_active_fast,
    venue_cards_fast,
)


RACE_DATE = "2026-07-19"


def test_final_status_is_complete_without_three_rank_rows() -> None:
    row = {"status": "final", "result_status": "final", "result_rows": 0}

    assert _race_is_final(row)
    assert _venue_display_row(
        [{**row, "deadline_at": "2026-07-19T15:00:00+09:00"}],
        now=datetime(2026, 7, 19, 21, 0, tzinfo=JST),
    ) is None


def test_venue_and_progress_finish_refund_only_result(tmp_path) -> None:
    db_path = tmp_path / "race.sqlite"
    init_db(db_path)
    with connection(db_path) as conn:
        upsert_race(
            conn,
            {
                "race_date": RACE_DATE,
                "jcd": "11",
                "venue_name": "びわこ",
                "rno": 10,
                "deadline_at": "2026-07-19T15:19:00+09:00",
                "status": "final",
            },
        )

    cards = venue_cards_fast(db_path, {"date": [RACE_DATE]})["venues"]
    biwako = next(row for row in cards if row["code"] == "11")
    assert biwako["status"] == "終了"
    assert biwako["finals"] == 1
    assert biwako["next_rno"] is None

    _PROGRESS_CACHE.clear()
    progress = progress_active_fast(db_path, {"date": [RACE_DATE]})
    assert progress["today"]["finals"] == 1
    assert progress["today"]["final_remaining"] == 0
