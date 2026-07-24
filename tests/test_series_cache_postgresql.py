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
    assert params == []
    assert result == {"cached": 0}
    assert conn.committed
