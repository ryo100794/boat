from boatrace_ai.web.prediction_summary import attach_latest_prediction_summaries


class Cursor:
    def fetchall(self):
        return [
            {
                "race_id": "r1",
                "combination": "1-2-3",
                "probability": 0.4,
                "odds": 3.0,
                "expected_value": 1.2,
                "generated_at": "2026-07-20T00:00:00+00:00",
            }
        ]


class PostgreSQLConnection:
    dialect = "postgresql"

    def __init__(self) -> None:
        self.statement = ""
        self.params = []

    def execute(self, statement, params):
        self.statement = statement
        self.params = list(params)
        return Cursor()


def test_postgresql_uses_indexed_lateral_latest_lookup() -> None:
    conn = PostgreSQLConnection()
    items = [{"race_id": "r1"}]

    attach_latest_prediction_summaries(conn, items)

    assert "CROSS JOIN LATERAL" in conn.statement
    assert "ORDER BY generated_at DESC" in conn.statement
    assert conn.params == ["r1"]
    assert items[0]["top_prediction"]["combination"] == "1-2-3"
