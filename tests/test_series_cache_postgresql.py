from contextlib import contextmanager

import boatrace_ai.cache_entry_series_features as series_cache
from boatrace_ai.cache_entry_series_features import populate_series_cache


class FakePostgresql:
    dialect = "postgresql"

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.committed = False

    def execute(self, statement: str, params=None):
        self.calls.append((statement, params))
        return []

    def commit(self) -> None:
        self.committed = True


def test_populate_series_cache_uses_postgresql_json_filter() -> None:
    conn = FakePostgresql()

    result = populate_series_cache(conn)

    select, params = conn.calls[1]
    assert "jsonb_extract_path" in select
    assert "LIKE '%series_results%'" not in select
    assert "LEFT JOIN entry_series_features" in select
    assert "sf.updated_at" in select
    assert params == []
    assert result == {
        "cached": 0,
        "from_date": None,
        "refresh_all": False,
    }
    assert conn.committed


def test_cli_preserves_postgresql_url_dsn(monkeypatch) -> None:
    targets: list[str] = []

    @contextmanager
    def fake_connection(target):
        targets.append(target)
        yield object()

    monkeypatch.setattr(series_cache, "init_db", lambda _target: None)
    monkeypatch.setattr(series_cache, "connection", fake_connection)
    monkeypatch.setattr(
        series_cache,
        "populate_series_cache",
        lambda _conn, **kwargs: {
            "cached": 0,
            "from_date": kwargs["from_date"],
            "refresh_all": kwargs["refresh_all"],
        },
    )

    dsn = "postgresql://boatrace_app@127.0.0.1:5432/boatrace"
    assert series_cache.main(["--db", dsn]) == 0
    assert targets == [dsn]
