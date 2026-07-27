from __future__ import annotations

import sqlite3
from datetime import date
from urllib.parse import parse_qs, urlsplit

from boatrace_ai import db
from boatrace_ai import http
from boatrace_ai.db import (
    connection,
    ensure_odds_signature_schema,
    init_db,
    insert_odds_snapshot,
    trifecta_odds_signature,
)
from boatrace_ai.ingestion import live
from boatrace_ai.odds_quality import (
    TRIFECTA_COMBINATION_KEYS,
    TRIFECTA_PARSER_VERSION,
)


def _odds() -> dict[str, float]:
    return {
        combination: float(index + 10)
        for index, combination in enumerate(TRIFECTA_COMBINATION_KEYS)
    }


def test_signature_is_deterministic_and_order_independent() -> None:
    odds = _odds()
    reversed_odds = dict(reversed(list(odds.items())))

    first = trifecta_odds_signature(odds)
    second = trifecta_odds_signature(reversed_odds)

    assert first == second
    assert len(first) == 64
    assert first == trifecta_odds_signature(odds)


def test_signature_changes_when_one_value_changes() -> None:
    odds = _odds()
    changed = dict(odds)
    changed[TRIFECTA_COMBINATION_KEYS[0]] += 0.1

    assert trifecta_odds_signature(odds) != trifecta_odds_signature(changed)


def test_snapshots_keep_same_content_polls_and_identify_value_changes(tmp_path) -> None:
    database = tmp_path / "signature.sqlite"
    race_id = "2026-07-27-01-01"
    odds = _odds()
    reordered = dict(reversed(list(odds.items())))
    changed = dict(odds)
    changed[TRIFECTA_COMBINATION_KEYS[-1]] += 0.1
    init_db(database)

    with connection(database) as conn:
        conn.execute(
            "INSERT INTO races "
            "(race_id, race_date, jcd, venue_name, rno) "
            "VALUES (?, '2026-07-27', '01', '桐生', 1)",
            (race_id,),
        )
        for captured_at, values in (
            ("2026-07-27T03:00:00+00:00", odds),
            ("2026-07-27T03:00:05+00:00", reordered),
            ("2026-07-27T03:00:10+00:00", changed),
        ):
            insert_odds_snapshot(
                conn,
                race_id,
                captured_at,
                "11:59",
                values,
                "https://example.invalid/odds",
                {"parser_version": TRIFECTA_PARSER_VERSION},
            )
        rows = conn.execute(
            "SELECT odds_signature FROM odds_snapshots ORDER BY snapshot_id"
        ).fetchall()

    assert len(rows) == 3
    assert rows[0][0] == rows[1][0]
    assert rows[1][0] != rows[2][0]


def test_snapshot_insert_does_not_run_schema_migration(monkeypatch, tmp_path) -> None:
    database = tmp_path / "insert-without-migration.sqlite"
    race_id = "2026-07-27-01-01"
    init_db(database)

    def unexpected_migration(_conn):
        raise AssertionError("schema migration must not run during snapshot insert")

    monkeypatch.setattr(db, "ensure_odds_signature_schema", unexpected_migration)
    with connection(database) as conn:
        conn.execute(
            "INSERT INTO races "
            "(race_id, race_date, jcd, venue_name, rno) "
            "VALUES (?, '2026-07-27', '01', '桐生', 1)",
            (race_id,),
        )
        snapshot_id = insert_odds_snapshot(
            conn,
            race_id,
            "2026-07-27T03:00:00+00:00",
            "11:59",
            _odds(),
            "https://example.invalid/odds",
            {"parser_version": TRIFECTA_PARSER_VERSION},
        )

    assert snapshot_id > 0


def test_sqlite_schema_migration_is_idempotent(tmp_path) -> None:
    database = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(database)
    conn.execute(
        "CREATE TABLE odds_snapshots ("
        "snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "race_id TEXT NOT NULL, bet_type TEXT NOT NULL, "
        "captured_at TEXT NOT NULL, source_update_time TEXT, "
        "parser_version TEXT, raw_json TEXT, source_url TEXT)"
    )
    conn.commit()
    conn.close()

    init_db(database)
    init_db(database)

    conn = sqlite3.connect(database)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(odds_snapshots)")}
    indexes = {row[1] for row in conn.execute("PRAGMA index_list(odds_snapshots)")}
    conn.close()
    assert "odds_signature" in columns
    assert "idx_odds_race_signature" in indexes


def test_postgresql_schema_migration_is_idempotent_per_process(monkeypatch) -> None:
    monkeypatch.setattr(db, "_POSTGRES_ODDS_SIGNATURE_SCHEMA_READY", False)

    class FakePostgresqlConnection:
        dialect = "postgresql"

        def __init__(self) -> None:
            self.statements: list[str] = []

        def execute(self, statement: str):
            self.statements.append(statement)
            return self

    conn = FakePostgresqlConnection()
    ensure_odds_signature_schema(conn)
    ensure_odds_signature_schema(conn)

    assert len(conn.statements) == 2
    assert "ADD COLUMN IF NOT EXISTS odds_signature" in conn.statements[0]
    assert "idx_odds_race_signature" in conn.statements[1]


def test_only_cache_busted_fetch_adds_nonce_and_no_cache_headers(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, str]]] = []

    class Response:
        status_code = 200
        content = b"ok"

    class FakeRequests:
        RequestException = RuntimeError

        @staticmethod
        def get(url, *, headers, timeout):
            calls.append((url, headers))
            return Response()

    monkeypatch.setattr(http, "_requests", lambda: FakeRequests)
    url = "https://example.invalid/odds?jcd=01&rno=1"

    http.fetch_bytes(url)
    http.fetch_bytes(url, cache_bust=True)
    http.fetch_bytes(url, cache_bust=True)

    normal_url, normal_headers = calls[0]
    assert normal_url == url
    assert "Cache-Control" not in normal_headers
    closing_urls = [calls[1][0], calls[2][0]]
    assert closing_urls[0] != closing_urls[1]
    for request_url, headers in calls[1:]:
        query = parse_qs(urlsplit(request_url).query)
        assert query["jcd"] == ["01"]
        assert query["rno"] == ["1"]
        assert len(query["_boatrace_nonce"][0]) == 32
        assert headers["Cache-Control"] == "no-cache, no-store, max-age=0"


def test_collect_odds_cache_bust_is_opt_in(monkeypatch, tmp_path) -> None:
    observed: list[bool] = []
    odds = _odds()

    def fake_fetch_page(*args, **kwargs):
        observed.append(kwargs["cache_bust"])
        return "html"

    monkeypatch.setattr(live, "_fetch_page", fake_fetch_page)
    monkeypatch.setattr(
        live,
        "parse_odds3t_html",
        lambda _html: {
            "parser_version": TRIFECTA_PARSER_VERSION,
            "parsed_count": 120,
            "source_update_time": "11:59",
            "odds": odds,
        },
    )
    monkeypatch.setattr(live, "_ensure_minimal_race", lambda *args, **kwargs: None)
    monkeypatch.setattr(live, "insert_odds_snapshot", lambda *args, **kwargs: 1)

    common = {
        "race_date": date(2026, 7, 27),
        "jcd": "01",
        "rno": 1,
        "raw_dir": tmp_path,
    }
    assert live.collect_odds(object(), **common)
    assert live.collect_odds(object(), **common, cache_bust=True)
    assert observed == [False, True]
