from datetime import date, timedelta
from pathlib import Path

from boatrace_ai.db import connection, init_db
from boatrace_ai.web import dashboard


def _seed(conn, race_id: str, race_date: str, jcd: str, lane: int) -> None:
    conn.execute(
        "INSERT INTO races(race_id, race_date, jcd, venue_name, rno) "
        "VALUES (?, ?, ?, ?, ?)",
        (race_id, race_date, jcd, "test", lane),
    )
    conn.execute(
        """
        INSERT INTO entries(
          race_id, lane, racer_no, racer_name, racer_class, motor_no, boat_no,
          motor_2_rate, motor_3_rate, boat_2_rate, boat_3_rate
        ) VALUES (?, ?, 4072, ?, "A1", 15, 18, 0.4, 0.6, 0.3, 0.5)
        """,
        (race_id, lane, "森永淳"),
    )
    conn.execute(
        "INSERT INTO race_results(race_id, lane, rank, start_timing) VALUES (?, ?, 1, 0.12)",
        (race_id, lane),
    )


def test_targeted_archive_queries_filter_before_aggregation(tmp_path) -> None:
    db_path = tmp_path / "archive.sqlite"
    init_db(db_path)
    with connection(db_path) as conn:
        _seed(conn, "2026-07-20-23-01", "2026-07-20", "23", 1)
        _seed(conn, "2026-07-20-24-01", "2026-07-20", "24", 2)
        racer = dashboard._history_racer(conn, "4072", "2026-07-01")
        motor = dashboard._history_equipment(
            conn, "motor", "15", "23", "2026-07-01"
        )
        boat = dashboard._history_equipment(
            conn, "boat", "18", "23", "2026-07-01"
        )

    assert racer["summary"]["starts"] == 2
    assert racer["summary"]["racer_no"] == 4072
    assert motor["summary"]["starts"] == 1
    assert motor["summary"]["number"] == 15
    assert boat["summary"]["starts"] == 1
    assert boat["summary"]["number"] == 18


def test_archive_history_defaults_to_ten_years_and_pages_into_older_years(tmp_path) -> None:
    db_path = tmp_path / "archive-pages.sqlite"
    init_db(db_path)
    with connection(db_path) as conn:
        for index in range(81):
            _seed(
                conn,
                f"2026-07-20-23-{index + 1:03d}",
                (date(2026, 7, 20) - timedelta(days=index)).isoformat(),
                "23",
                index % 6 + 1,
            )
        _seed(conn, "2025-07-20-23-01", "2025-07-20", "23", 1)
        _seed(conn, "2018-07-20-23-01", "2018-07-20", "23", 1)

    first = dashboard.archive_history(
        db_path, {"kind": ["racer"], "racer_no": ["4072"]}
    )
    second = dashboard.archive_history(
        db_path,
        {"kind": ["racer"], "racer_no": ["4072"], "offset": ["80"]},
    )
    selected_2025 = dashboard.archive_history(
        db_path,
        {"kind": ["racer"], "racer_no": ["4072"], "year": ["2025"]},
    )

    assert first["period_days"] == 3650
    assert first["summary"]["starts"] == 83
    assert len(first["rows"]) == 80
    assert first["pagination"] == {
        "limit": 80,
        "offset": 0,
        "returned": 80,
        "has_previous": False,
        "has_more": True,
    }
    assert second["pagination"]["has_previous"] is True
    assert second["pagination"]["has_more"] is False
    assert {row["race_date"] for row in second["rows"]} >= {
        "2025-07-20",
        "2018-07-20",
    }
    assert selected_2025["selected_year"] == 2025
    assert selected_2025["cutoff_date"] == "2025-01-01"
    assert selected_2025["end_date"] == "2025-12-31"
    assert selected_2025["summary"]["starts"] == 1
    assert [row["race_date"] for row in selected_2025["rows"]] == ["2025-07-20"]
    assert selected_2025["available_years"] == list(range(2026, 2015, -1))


def test_archive_year_filter_applies_to_every_history_kind(tmp_path) -> None:
    db_path = tmp_path / "archive-year-kinds.sqlite"
    init_db(db_path)
    with connection(db_path) as conn:
        _seed(conn, "2026-02-01-23-01", "2026-02-01", "23", 1)
        _seed(conn, "2025-02-01-23-01", "2025-02-01", "23", 1)
        for race_id in ("2026-02-01-23-01", "2025-02-01-23-01"):
            conn.execute(
                "INSERT INTO payouts(race_id, bet_type, combination, payout_yen) "
                "VALUES (?, '3連単', '1-2-3', 1200)",
                (race_id,),
            )

    queries = [
        {"kind": ["venue"], "jcd": ["23"]},
        {"kind": ["motor"], "motor_no": ["15"], "jcd": ["23"]},
        {"kind": ["boat"], "boat_no": ["18"], "jcd": ["23"]},
        {"kind": ["lane"], "lane": ["1"], "jcd": ["23"]},
        {"kind": ["combo"], "combination": ["1-2-3"]},
    ]
    for query in queries:
        payload = dashboard.archive_history(db_path, {**query, "year": ["2025"]})
        assert payload["rows"]
        assert {row["race_date"] for row in payload["rows"]} == {"2025-02-01"}


def test_archive_response_cache_reuses_identical_query(monkeypatch, tmp_path) -> None:
    dashboard._ARCHIVE_API_CACHE.clear()
    calls = []

    def fake_history(db_path, query):
        calls.append((db_path, query))
        return {"kind": "racer", "rows": []}

    monkeypatch.setattr(dashboard, "archive_history", fake_history)
    query = {"kind": ["racer"], "racer_no": ["4072"]}
    first = dashboard.archive_history_cached(tmp_path / "db.sqlite", query)
    second = dashboard.archive_history_cached(tmp_path / "db.sqlite", query)

    assert first is second
    assert len(calls) == 1


def test_dashboard_uses_lazy_official_racer_photos() -> None:
    html = (
        Path(__file__).parents[1]
        / "src"
        / "boatrace_ai"
        / "templates"
        / "dashboard.html"
    ).read_text(encoding="utf-8")

    assert "https://www.boatrace.jp/racerphoto/" in html
    assert "class=\"entry-photo\"" in html
    assert "loading=\"lazy\"" in html
    assert "state.archiveController.abort()" in html
    assert "予測上位5組 オッズ推移" in html
    assert "drawTrend(data.series || [], state.combo)" in html
    assert 'ctx.fillText("オッズ",0,0)' in html
    assert 'ctx.fillText("取得時刻 (JST)"' in html
    assert 'id="archivePeriod"' in html
    assert 'id="archiveYear"' in html
    assert 'year:state.archiveYear' in html
    assert 'function loadArchivePage(offset)' in html
    assert 'localStorage.getItem("boat.archiveDays") || "3650"' in html


def test_archive_stats_sql_is_portable_across_sqlite_and_postgresql(tmp_path) -> None:
    db_path = tmp_path / "stats.sqlite"
    init_db(db_path)
    with connection(db_path) as conn:
        _seed(conn, "2026-07-20-23-01", "2026-07-20", "23", 1)
        _seed(conn, "2026-07-20-24-01", "2026-07-20", "24", 2)
        for scope in ("lane", "venue", "rno", "class", "motor", "boat"):
            rows = dashboard._stat_rows_fast(
                conn, scope, "2026-07-01", limit=20, min_starts=1
            )
            assert rows, scope
            select_sql = dashboard._scope_sql(scope)[0]
            assert "printf(" not in select_sql
            assert "%" not in select_sql


def test_archive_json_default_serializes_postgresql_values() -> None:
    import json
    from datetime import date
    from decimal import Decimal

    payload = json.dumps(
        {"ratio": Decimal("0.125"), "day": date(2026, 7, 21)},
        default=dashboard._json_default,
    )

    assert json.loads(payload) == {"ratio": 0.125, "day": "2026-07-21"}
