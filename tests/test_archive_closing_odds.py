from __future__ import annotations

from datetime import date

import pytest

from boatrace_ai.archive_closing_odds import (
    SOURCE_KEY,
    archive_url,
    ensure_archive_schema,
    parse_archive_closing_odds_html,
    pending_races,
    store_archive_closing_odds,
    verify_winning_payout,
)
from boatrace_ai.db import connection, init_db, upsert_race
from boatrace_ai.odds_quality import TRIFECTA_COMBINATION_KEYS


def _html(*, drop_last: bool = False, unavailable: set[str] | None = None) -> str:
    combinations = TRIFECTA_COMBINATION_KEYS[:-1] if drop_last else TRIFECTA_COMBINATION_KEYS
    unavailable = unavailable or set()
    rows = []
    for index, combination in enumerate(combinations, start=1):
        lanes = "".join(
            f'<div class="r{lane}"><div class="rb">{lane}</div></div>'
            for lane in combination.split("-")
        )
        rows.append(
            '<tr><td><div class="rgs3">'
            + lanes
            + '</div></td><td class="od_text" align="right">'
            + ("---" if combination in unavailable else f"{1.0 + index / 10:.1f}")
            + "</td></tr>"
        )
    return (
        '<div id="brOddslist"><div>締切時オッズ</div><table>'
        '<tr><td class="mainTopHeadline3t">3連単</td></tr>'
        + "".join(rows)
        + "</table></div>"
    )


def _odds_bank_html() -> str:
    race_html = _html()
    table = race_html[race_html.index("<table>") : race_html.index("</table>") + 8]
    table = table.replace(
        '<tr><td class="mainTopHeadline3t">3連単</td></tr>', "", 1
    )
    return f'<div id="oddsData" class="oddsCard"><h3 class="h3_sp">3連単</h3>{table}</div>'


def test_archive_parser_requires_complete_plausible_trifecta() -> None:
    parsed = parse_archive_closing_odds_html(_html())
    assert parsed["odds_count"] == 120
    assert set(parsed["odds"]) == set(TRIFECTA_COMBINATION_KEYS)
    with pytest.raises(ValueError, match="incomplete or implausible"):
        parse_archive_closing_odds_html(_html(drop_last=True))


def test_archive_parser_supports_odds_bank_layout_without_closing_label() -> None:
    parsed = parse_archive_closing_odds_html(_odds_bank_html())
    assert parsed["odds_count"] == 120
    assert set(parsed["odds"]) == set(TRIFECTA_COMBINATION_KEYS)


def test_archive_parser_accounts_for_unavailable_closing_combinations() -> None:
    unavailable = {"1-2-3", "6-5-4"}
    parsed = parse_archive_closing_odds_html(_html(unavailable=unavailable))
    assert parsed["odds_count"] == 118
    assert parsed["unavailable_count"] == 2
    assert set(parsed["unavailable_combinations"]) == unavailable
    assert set(parsed["odds"]) | unavailable == set(TRIFECTA_COMBINATION_KEYS)


def test_archive_winner_must_match_official_payout() -> None:
    odds = parse_archive_closing_odds_html(_html())["odds"]
    winner = TRIFECTA_COMBINATION_KEYS[0]
    payout = int(round(odds[winner] * 100))
    assert verify_winning_payout(
        odds, combination=winner, payout_yen=payout
    )["status"] == "winner_only_match_unverified_market"
    with pytest.raises(ValueError, match="payout mismatch"):
        verify_winning_payout(odds, combination=winner, payout_yen=payout + 10)


def test_archive_storage_is_separate_and_pending_runs_newest_first(tmp_path) -> None:
    db_path = tmp_path / "archive.sqlite"
    init_db(db_path)
    with connection(db_path) as conn:
        for day, rno in (("2026-07-26", 1), ("2026-07-27", 2)):
            rid = upsert_race(
                conn,
                {
                    "race_date": day,
                    "jcd": "01",
                    "venue_name": "桐生",
                    "rno": rno,
                    "status": "final",
                },
            )
            conn.execute(
                "INSERT INTO payouts(race_id, bet_type, combination, payout_yen) VALUES (?, '3連単', ?, ?)",
                (rid, TRIFECTA_COMBINATION_KEYS[0], 110),
            )
        ensure_archive_schema(conn)
        pending = pending_races(
            conn, from_date="2026-07-26", through_date="2026-07-27"
        )
        assert [row["race_date"] for row in pending] == ["2026-07-27", "2026-07-26"]
        parsed = parse_archive_closing_odds_html(_html())
        verification = verify_winning_payout(
            parsed["odds"], combination=TRIFECTA_COMBINATION_KEYS[0], payout_yen=110
        )
        store_archive_closing_odds(
            conn,
            race_id=pending[0]["race_id"],
            source_url="https://example.invalid/archive",
            payload=b"fixture",
            parsed=parsed,
            verification=verification,
            fetched_at="2026-07-28T00:00:00+00:00",
        )
        assert conn.execute(
            "SELECT COUNT(*) FROM archive_closing_odds WHERE race_id = ? AND source_key = ?",
            (pending[0]["race_id"], SOURCE_KEY),
        ).fetchone()[0] == 120
        assert len(
            pending_races(
                conn, from_date="2026-07-26", through_date="2026-07-27"
            )
        ) == 1
        assert conn.execute("SELECT COUNT(*) FROM odds_snapshots").fetchone()[0] == 0


def test_archive_url_is_stable() -> None:
    assert archive_url(date(2026, 7, 27), "1", 2) == (
        "https://odds.kyotei24.jp/od-20260727-01-2.html"
    )
