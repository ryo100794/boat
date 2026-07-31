from __future__ import annotations

from contextlib import contextmanager

from boatrace_ai import work_tickets


class _Result:
    def fetchone(self):
        return {"ticket_key": "TEST-CLI-COMMIT"}


class _Connection:
    def __init__(self) -> None:
        self.commits = 0
        self.statements: list[str] = []

    def execute(self, statement, _parameters=()):
        self.statements.append(str(statement))
        return _Result()

    def commit(self) -> None:
        self.commits += 1


def test_cli_commits_ticket_updates(monkeypatch) -> None:
    conn = _Connection()

    @contextmanager
    def fake_connection(_db):
        yield conn

    monkeypatch.setattr(work_tickets, "connection", fake_connection)
    monkeypatch.setattr(work_tickets, "ensure_schema", lambda _conn: None)

    assert work_tickets.main([
        "--db",
        "postgresql://test",
        "add",
        "--key",
        "TEST-CLI-COMMIT",
        "--title",
        "CLI commit",
        "--area",
        "test",
    ]) == 0
    assert conn.commits == 1

    assert work_tickets.main([
        "--db",
        "postgresql://test",
        "update",
        "--key",
        "TEST-CLI-COMMIT",
        "--status",
        "in_progress",
        "--progress",
        "40",
        "--note",
        "persisted",
    ]) == 0
    assert conn.commits == 2
    assert any("UPDATE work_tickets" in statement for statement in conn.statements)
    assert any("INSERT INTO work_ticket_events" in statement for statement in conn.statements)
