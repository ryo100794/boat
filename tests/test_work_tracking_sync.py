from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json

import pytest

from boatrace_ai import work_tracking_sync as sync


NOW = "2026-07-23T12:00:00Z"


class Cursor:
    def __init__(self, rows=()):
        self.rows = list(rows)

    def fetchall(self):
        return list(self.rows)

    def fetchone(self):
        return self.rows[0] if self.rows else None


class FakeConnection:
    dialect = "postgresql"

    def __init__(self, tickets, *, columns=None):
        self.tickets = tickets
        self.columns = set(columns if columns is not None else sync.SYNC_COLUMNS)
        self.calls = []

    def execute(self, statement, params=()):
        normalized = " ".join(statement.split())
        self.calls.append((normalized, tuple(params or ())))
        if "FROM information_schema.columns" in normalized:
            return Cursor([{"column_name": name} for name in sorted(self.columns)])
        if "FROM work_tickets ORDER BY ticket_key" in normalized:
            rows = []
            for source in self.tickets:
                row = dict(source)
                for column in sync.SYNC_COLUMNS:
                    row.setdefault(column, None)
                rows.append(row)
            return Cursor(rows)
        return Cursor()

    @property
    def writes(self):
        prefixes = ("ALTER ", "CREATE UNIQUE ", "UPDATE ", "INSERT ")
        return [call for call in self.calls if call[0].startswith(prefixes)]


class FakeTransport:
    def __init__(self, issues=()):
        self.issues = {int(issue["number"]): dict(issue) for issue in issues}
        self.calls = []
        self.next_number = max(self.issues, default=0) + 1

    def request(self, method, path, payload=None):
        self.calls.append((method, path, dict(payload) if payload is not None else None))
        if method == "GET" and "?state=all" in path:
            return [dict(issue) for issue in self.issues.values()]
        if method == "GET":
            return dict(self.issues[int(path.rsplit("/", 1)[1])])
        if method == "POST":
            number = self.next_number
            self.next_number += 1
            issue = {
                "number": number,
                "title": payload["title"],
                "body": payload["body"],
                "state": "open",
                "html_url": f"https://github.test/issues/{number}",
                "updated_at": NOW,
            }
            self.issues[number] = issue
            return dict(issue)
        if method == "PATCH":
            number = int(path.rsplit("/", 1)[1])
            self.issues[number].update(payload)
            self.issues[number]["updated_at"] = NOW
            return dict(self.issues[number])
        raise AssertionError((method, path, payload))


def ticket_row(**overrides):
    row = {
        "ticket_key": "MODEL-OPT-001",
        "title": "収益ゲート収束",
        "area": "モデル",
        "description": "未使用holdoutで評価する",
        "acceptance_criteria": "ROIと確率指標の基準を満たす",
        "owner": "codex",
        "priority": 100,
        "status": "in_progress",
        "progress": 55,
        "related_job_id": 387,
        "source": "user",
        "updated_at": "2026-07-23T10:00:00Z",
        "repository_full_name": "",
        "github_issue_number": None,
        "github_issue_url": "",
        "github_issue_updated_at": None,
        "last_synced_at": None,
    }
    row.update(overrides)
    return row


def issue_for(ticket, *, number=2, body=None, state="open", updated_at=NOW):
    item = sync.Ticket(**ticket)
    return {
        "number": number,
        "title": sync.issue_title(item),
        "body": sync.render_managed_block(item) if body is None else body,
        "state": state,
        "html_url": f"https://github.test/issues/{number}",
        "updated_at": updated_at,
    }


def test_managed_block_replaces_legacy_marker_and_preserves_human_text():
    body = (
        "human introduction\n\n"
        f"{sync.MANAGED_START}\nlegacy table without json\n{sync.MANAGED_END}"
        "\n\nhuman follow-up"
    )
    block = sync.render_managed_block(sync.Ticket(**ticket_row()))

    updated = sync.replace_managed_block(body, block)

    assert updated.startswith("human introduction")
    assert updated.endswith("human follow-up")
    assert "legacy table without json" not in updated
    assert sync.parse_managed_metadata(updated)["ticket_key"] == "MODEL-OPT-001"


def test_prefix_title_adopts_pre_json_issue_and_duplicate_is_conflict():
    ticket = sync.Ticket(**ticket_row())
    first = issue_for(ticket_row(), number=2, body="legacy", state="open")
    first["title"] = "[MODEL-OPT-001] old wording"
    unrelated = issue_for(ticket_row(), number=3, body="unmanaged")
    unrelated["title"] = "prefix [MODEL-OPT-001] is not at start"

    assert sync._find_existing_issue(ticket, [first, unrelated]) == first
    second = dict(first, number=4)
    with pytest.raises(sync.IssueMatchConflict):
        sync._find_existing_issue(ticket, [first, second])


def test_dry_run_reports_adoption_and_update_without_any_write():
    row = ticket_row()
    issue = issue_for(row, body="human\n\n<!-- boatrace-work-ticket:start -->old<!-- boatrace-work-ticket:end -->")
    issue["title"] = "[MODEL-OPT-001] legacy title"
    conn = FakeConnection([row], columns=set())
    transport = FakeTransport([issue])

    result = sync.sync_work_tickets(
        conn,
        repository="ryo100794/boat",
        transport=transport,
        apply=False,
        direction="db-to-issue",
    )

    assert [action["action"] for action in result["actions"]] == [
        "link_existing",
        "update_issue",
    ]
    assert conn.writes == []
    assert {method for method, _, _ in transport.calls} == {"GET"}


def test_apply_updates_only_managed_body_and_records_audit_event():
    row = ticket_row(
        repository_full_name="ryo100794/boat",
        github_issue_number=2,
        github_issue_url="https://github.test/issues/2",
        last_synced_at="2026-07-23T12:00:00Z",
    )
    issue = issue_for(
        row,
        body="human\n\n<!-- boatrace-work-ticket:start -->legacy<!-- boatrace-work-ticket:end -->\n\nnotes",
        updated_at="2026-07-23T11:00:00Z",
    )
    conn = FakeConnection([row])
    transport = FakeTransport([issue])

    result = sync.sync_work_tickets(
        conn,
        repository="ryo100794/boat",
        transport=transport,
        apply=True,
        direction="db-to-issue",
    )

    patch = next(payload for method, _, payload in transport.calls if method == "PATCH")
    assert patch["body"].startswith("human\n\n")
    assert patch["body"].endswith("\n\nnotes")
    assert sync.parse_managed_metadata(patch["body"])["status"] == "in_progress"
    assert any("INSERT INTO work_ticket_events" in sql for sql, _ in conn.calls)
    assert any(action["action"] == "update_issue" for action in result["actions"])


def _completion_body(row, *, verified):
    data = sync._managed_metadata(sync.Ticket(**row))
    data.update(status="completed", progress=100, acceptance_verified=verified)
    encoded = json.dumps(data, separators=(",", ":"), sort_keys=True)
    return (
        f"{sync.MANAGED_START}\n{sync.MANAGED_JSON_PREFIX}{encoded} -->\n"
        f"{sync.MANAGED_END}"
    )


def test_closed_issue_alone_cannot_complete_performance_ticket():
    row = ticket_row(
        repository_full_name="ryo100794/boat",
        github_issue_number=2,
        github_issue_url="https://github.test/issues/2",
        last_synced_at="2026-07-23T11:00:00Z",
    )
    issue = issue_for(row, body=_completion_body(row, verified=False), state="closed")
    conn = FakeConnection([row])

    result = sync.sync_work_tickets(
        conn,
        repository="ryo100794/boat",
        transport=FakeTransport([issue]),
        apply=True,
        direction="issue-to-db",
    )

    assert result["actions"][0]["reason"] == "completion_not_verified"
    assert not any("SET status = ?" in sql for sql, _ in conn.calls)
    event = next(params for sql, params in conn.calls if "INSERT INTO work_ticket_events" in sql)
    assert "ignore_issue_change" in event[3]


def test_verified_managed_completion_applies_valid_transition():
    row = ticket_row(
        repository_full_name="ryo100794/boat",
        github_issue_number=2,
        github_issue_url="https://github.test/issues/2",
        last_synced_at="2026-07-23T11:00:00Z",
    )
    issue = issue_for(row, body=_completion_body(row, verified=True), state="closed")
    conn = FakeConnection([row])

    result = sync.sync_work_tickets(
        conn,
        repository="ryo100794/boat",
        transport=FakeTransport([issue]),
        apply=True,
        direction="issue-to-db",
    )

    action = result["actions"][0]
    assert (action["status"], action["progress"]) == ("completed", 100)
    update = next(params for sql, params in conn.calls if "SET status = ?" in sql)
    assert update[:3] == ("completed", 100, "completed")


def test_issue_edit_based_on_stale_db_revision_cannot_overwrite_db():
    row = ticket_row(
        updated_at="2026-07-23T12:30:00Z",
        repository_full_name="ryo100794/boat",
        github_issue_number=2,
        github_issue_url="https://github.test/issues/2",
        last_synced_at="2026-07-23T11:00:00Z",
    )
    stale = dict(row, updated_at="2026-07-23T10:00:00Z")
    issue = issue_for(
        row,
        body=_completion_body(stale, verified=True),
        state="closed",
        updated_at="2026-07-23T13:00:00Z",
    )
    conn = FakeConnection([row])

    result = sync.sync_work_tickets(
        conn,
        repository="ryo100794/boat",
        transport=FakeTransport([issue]),
        apply=True,
        direction="issue-to-db",
    )

    assert result["actions"][0]["reason"] == "stale_db_revision"
    assert not any("SET status = ?" in sql for sql, _ in conn.calls)


def test_apply_schema_is_idempotent_postgresql_ddl():
    conn = FakeConnection([])
    sync.ensure_sync_schema(conn)
    sync.ensure_sync_schema(conn)

    statements = [sql for sql, _ in conn.calls]
    assert sum("ADD COLUMN IF NOT EXISTS" in sql for sql in statements) == 10
    assert sum("CREATE UNIQUE INDEX IF NOT EXISTS" in sql for sql in statements) == 2


def test_interval_parser_accepts_once_or_minute_and_rejects_short_loop():
    parser = sync.build_parser()
    once = parser.parse_args(["--db", "host=test", "--repo", "o/r"])
    periodic = parser.parse_args(
        ["--db", "host=test", "--repo", "o/r", "--interval-seconds", "60"]
    )

    assert once.interval_seconds == 0
    assert periodic.interval_seconds == 60
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--db", "host=test", "--repo", "o/r", "--interval-seconds", "59"]
        )


def test_missing_token_is_normal_skip_and_periodic_wait(monkeypatch, capsys):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    args = argparse.Namespace(
        db="host=test",
        repo="o/r",
        direction="both",
        apply=True,
        interval_seconds=60,
    )
    sleeps = []

    assert sync.run_periodic(args, sleep=sleeps.append, max_cycles=2) == 0

    lines = capsys.readouterr().out.splitlines()
    assert [json.loads(line)["status"] for line in lines] == [
        "skipped_no_token",
        "skipped_no_token",
    ]
    assert sleeps == [60]


def test_periodic_cycles_open_fresh_database_connections(monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_TOKEN", "not-a-real-token")
    opened = []

    @contextmanager
    def fake_connection(_dsn):
        conn = object()
        opened.append(conn)
        yield conn

    monkeypatch.setattr(sync, "connection", fake_connection)
    monkeypatch.setattr(sync, "GitHubTransport", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        sync,
        "sync_work_tickets",
        lambda conn, **_kwargs: {"connection_id": id(conn), "actions": []},
    )
    args = argparse.Namespace(
        db="host=test",
        repo="o/r",
        direction="both",
        apply=False,
        interval_seconds=60,
    )

    assert sync.run_periodic(args, sleep=lambda _seconds: None, max_cycles=2) == 0

    assert len(opened) == 2
    assert opened[0] is not opened[1]
    assert capsys.readouterr().out.count('"status": "synchronized"') == 2


def test_cli_without_token_exits_successfully_with_skip(monkeypatch, capsys):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    assert sync.main(["--db", "host=test", "--repo", "o/r"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "skipped_no_token"
    assert payload["mode"] == "dry-run"
