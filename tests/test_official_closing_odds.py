from __future__ import annotations

from boatrace_ai.archive_closing_odds import OFFICIAL_SOURCE_KEY, SOURCE_KEY, pending_races
from boatrace_ai.db import connection, init_db, upsert_race
from boatrace_ai.evaluation_queue import build_command, summarize_result
from boatrace_ai.official_closing_odds import (
    IncompleteOfficialTrifectaOdds,
    backfill_official_closing_odds,
    official_closing_url,
    parse_official_closing_odds_html,
)


def _official_matrix_html() -> str:
    headers = "".join(
        f'<th class="is-boatColor{lane}">{lane}</th><th colspan="2">R{lane}</th>'
        for lane in range(1, 7)
    )
    rows = []
    for row_index in range(20):
        cells = []
        for first in range(1, 7):
            others = [lane for lane in range(1, 7) if lane != first]
            second = others[row_index // 4]
            thirds = [lane for lane in range(1, 7) if lane not in {first, second}]
            third = thirds[row_index % 4]
            if row_index % 4 == 0:
                cells.append(
                    f'<td class="is-boatColor{second}" rowspan="4">{second}</td>'
                )
            odds = first * 100 + second * 10 + third + 0.5
            cells.append(f'<td class="is-boatColor{third}">{third}</td>')
            cells.append(f'<td class="oddsPoint">{odds}</td>')
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return (
        "<html><body><h3>3連単オッズ</h3><table>"
        f"<thead><tr>{headers}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></body></html>"
    )


def test_official_closing_parser_and_url() -> None:
    parsed = parse_official_closing_odds_html(_official_matrix_html())
    assert parsed["source_key"] == OFFICIAL_SOURCE_KEY
    assert parsed["odds_count"] == 120
    assert parsed["odds"]["1-2-3"] == 123.5
    assert official_closing_url("2026-06-01", "1", 2).endswith(
        "/odds3t?rno=2&jcd=01&hd=20260601"
    )


def test_official_backfill_is_verified_and_isolated_by_source(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "official-closing.sqlite"
    init_db(db_path)
    html = _official_matrix_html()
    with connection(db_path) as conn:
        race_id = upsert_race(
            conn,
            {
                "race_date": "2026-06-01",
                "jcd": "01",
                "venue_name": "桐生",
                "rno": 1,
                "status": "final",
            },
        )
        conn.execute(
            "INSERT INTO payouts(race_id, bet_type, combination, payout_yen) "
            "VALUES (?, '3連単', '1-2-3', 12350)",
            (race_id,),
        )
        monkeypatch.setattr(
            "boatrace_ai.official_closing_odds.fetch_text",
            lambda *_args, **_kwargs: (200, html, html.encode()),
        )
        result = backfill_official_closing_odds(
            conn,
            from_date="2026-06-01",
            through_date="2026-06-01",
            sleep_seconds=0.0,
        )
        assert result["stored"] == 1
        stored = conn.execute(
            "SELECT verification_status, odds_count FROM archive_closing_odds_snapshots "
            "WHERE race_id = ? AND source_key = ?",
            (race_id, OFFICIAL_SOURCE_KEY),
        ).fetchone()
        assert stored["verification_status"] == "official_primary_winner_payout_match"
        assert stored["odds_count"] == 120
        assert pending_races(
            conn,
            from_date="2026-06-01",
            through_date="2026-06-01",
            source_key=OFFICIAL_SOURCE_KEY,
        ) == []
        assert len(
            pending_races(
                conn,
                from_date="2026-06-01",
                through_date="2026-06-01",
                source_key=SOURCE_KEY,
            )
        ) == 1


def test_confirmed_five_boat_odds_are_terminally_excluded(
    tmp_path, monkeypatch
) -> None:
    db_path = tmp_path / "official-five-boat.sqlite"
    init_db(db_path)
    html = "<html>five boat closing odds</html>"
    with connection(db_path) as conn:
        race_id = upsert_race(
            conn,
            {
                "race_date": "2026-06-01",
                "jcd": "01",
                "venue_name": "桐生",
                "rno": 1,
                "status": "final",
            },
        )
        conn.execute(
            "INSERT INTO payouts(race_id, bet_type, combination, payout_yen) "
            "VALUES (?, '3連単', '1-2-3', 12350)",
            (race_id,),
        )
        for lane in range(1, 6):
            conn.execute(
                "INSERT INTO race_results(race_id, lane, rank) VALUES (?, ?, ?)",
                (race_id, lane, lane),
            )
        monkeypatch.setattr(
            "boatrace_ai.official_closing_odds.fetch_text",
            lambda *_args, **_kwargs: (200, html, html.encode()),
        )
        monkeypatch.setattr(
            "boatrace_ai.official_closing_odds.parse_official_closing_odds_html",
            lambda _html: (_ for _ in ()).throw(
                IncompleteOfficialTrifectaOdds(60)
            ),
        )

        result = backfill_official_closing_odds(
            conn,
            from_date="2026-06-01",
            through_date="2026-06-01",
            sleep_seconds=0.0,
        )

        assert result["excluded_non_six_boat"] == 1
        assert result["invalid"] == 0
        assert result["remaining"] == 0
        attempt = conn.execute(
            "SELECT status, error FROM archive_closing_odds_attempts "
            "WHERE race_id = ? AND source_key = ?",
            (race_id, OFFICIAL_SOURCE_KEY),
        ).fetchone()
        assert attempt["status"] == "excluded_non_six_boat"
        assert "five-boat" in attempt["error"]

        conn.execute(
            "UPDATE archive_closing_odds_attempts "
            "SET status = 'invalid', error = ? WHERE race_id = ?",
            (
                "ValueError: official trifecta odds are incomplete: 60/120",
                race_id,
            ),
        )
        replay = backfill_official_closing_odds(
            conn,
            from_date="2026-06-01",
            through_date="2026-06-01",
            sleep_seconds=0.0,
        )
        assert replay["reclassified_non_six_boat"] == 1
        assert replay["targets"] == 0
        assert replay["remaining"] == 0


def test_archive_queue_selects_official_backfill_module(tmp_path) -> None:
    command, _output = build_command(
        {
            "job_id": 9,
            "status": "running",
            "task_type": "archive_closing_backfill",
            "model_key": "official-closing",
            "parameters": {
                "from_date": "2026-05-01",
                "through_date": "2026-05-31",
                "sleep_seconds": 0.5,
                "source": "official",
                "timeout_seconds": 86400,
            },
        },
        app_root=tmp_path,
        python=tmp_path / ".venv/bin/python",
        db="postgresql://test",
    )
    assert command[1:3] == ["-m", "boatrace_ai.official_closing_odds"]


def test_official_collection_counts_are_preserved_in_job_summary() -> None:
    summary = summarize_result(
        {
            "status": "completed",
            "source_role": "primary_official_historical_closing",
            "source_key": OFFICIAL_SOURCE_KEY,
            "targets": 20,
            "stored": 20,
            "invalid": 0,
            "fetch_failed": 0,
            "remaining": 116,
            "invalid_reason_counts": {
                "IncompleteOfficialTrifectaOdds": 3
            },
            "incomplete_odds_count_counts": {"72": 3},
            "failure_examples": [
                {
                    "race_id": "202606020203",
                    "status": "invalid",
                    "odds_count": 72,
                }
            ],
        }
    )
    assert summary["archive_stored"] == 20
    assert summary["archive_remaining"] == 116
    assert summary["archive_source_key"] == OFFICIAL_SOURCE_KEY
    assert summary["archive_invalid_reason_counts"] == {
        "IncompleteOfficialTrifectaOdds": 3
    }
    assert summary["archive_incomplete_odds_count_counts"] == {"72": 3}
    assert summary["archive_failure_examples"][0]["race_id"] == "202606020203"

def test_invalid_official_odds_publish_bounded_diagnostics(
    tmp_path, monkeypatch
) -> None:
    db_path = tmp_path / "official-invalid.sqlite"
    init_db(db_path)
    with connection(db_path) as conn:
        race_id = upsert_race(
            conn,
            {
                "race_date": "2026-06-02",
                "jcd": "02",
                "venue_name": "戸田",
                "rno": 3,
                "status": "final",
            },
        )
        conn.execute(
            "INSERT INTO payouts(race_id, bet_type, combination, payout_yen) "
            "VALUES (?, '3連単', '1-2-3', 12350)",
            (race_id,),
        )
        for lane in range(1, 7):
            conn.execute(
                "INSERT INTO race_results(race_id, lane, rank) VALUES (?, ?, ?)",
                (race_id, lane, lane),
            )
        monkeypatch.setattr(
            "boatrace_ai.official_closing_odds.fetch_text",
            lambda *_args, **_kwargs: (200, "incomplete", b"incomplete"),
        )
        monkeypatch.setattr(
            "boatrace_ai.official_closing_odds.parse_official_closing_odds_html",
            lambda _html: (_ for _ in ()).throw(
                IncompleteOfficialTrifectaOdds(72)
            ),
        )

        result = backfill_official_closing_odds(
            conn,
            from_date="2026-06-02",
            through_date="2026-06-02",
            sleep_seconds=0.0,
        )

    assert result["invalid"] == 1
    assert result["invalid_reason_counts"] == {
        "IncompleteOfficialTrifectaOdds": 1
    }
    assert result["incomplete_odds_count_counts"] == {"72": 1}
    assert result["invalid_confirmed_result_boats_counts"] == {"6": 1}
    assert result["failure_examples"] == [
        {
            "race_id": race_id,
            "race_date": "2026-06-02",
            "jcd": "02",
            "rno": 3,
            "status": "invalid",
            "error": (
                "IncompleteOfficialTrifectaOdds: official trifecta odds are "
                "incomplete: 72/120"
            ),
            "odds_count": 72,
            "confirmed_result_boats": 6,
        }
    ]
