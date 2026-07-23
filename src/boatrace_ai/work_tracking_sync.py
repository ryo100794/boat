from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
import re
import sys
from typing import Any, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .db import connection


MANAGED_START = "<!-- boatrace-work-ticket:start -->"
MANAGED_END = "<!-- boatrace-work-ticket:end -->"
MANAGED_JSON_PREFIX = "<!-- boatrace-work-ticket:json "
VALID_STATUSES = {"queued", "in_progress", "blocked", "completed", "cancelled"}
SYNC_COLUMNS = (
    "repository_full_name",
    "github_issue_number",
    "github_issue_url",
    "github_issue_updated_at",
    "last_synced_at",
)
REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
MANAGED_RE = re.compile(
    re.escape(MANAGED_START) + r"(?P<content>.*?)" + re.escape(MANAGED_END),
    re.DOTALL,
)


@dataclass(frozen=True)
class Ticket:
    ticket_key: str
    title: str
    area: str
    description: str
    acceptance_criteria: str
    owner: str
    priority: int
    status: str
    progress: int
    related_job_id: int | None
    source: str
    updated_at: Any
    repository_full_name: str = ""
    github_issue_number: int | None = None
    github_issue_url: str = ""
    github_issue_updated_at: Any = None
    last_synced_at: Any = None


class HttpTransport(Protocol):
    def request(
        self, method: str, path: str, payload: Mapping[str, Any] | None = None
    ) -> Any: ...


class GitHubApiError(RuntimeError):
    pass


class GitHubTransport:
    """Small GitHub REST client whose only credential source is the environment."""

    def __init__(
        self,
        repository: str,
        *,
        token: str | None = None,
        api_url: str = "https://api.github.com",
        timeout: float = 30.0,
    ) -> None:
        validate_repository(repository)
        self.repository = repository
        self.token = token if token is not None else os.environ.get("GITHUB_TOKEN", "")
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout

    def request(
        self, method: str, path: str, payload: Mapping[str, Any] | None = None
    ) -> Any:
        body = None
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "boatrace-work-ticket-sync/1",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if payload is not None:
            body = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(
            f"{self.api_url}{path}", data=body, headers=headers, method=method.upper()
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise GitHubApiError(f"GitHub API {exc.code} for {method} {path}: {detail}") from exc
        except URLError as exc:
            raise GitHubApiError(f"GitHub API unavailable for {method} {path}: {exc.reason}") from exc
        return json.loads(raw.decode("utf-8")) if raw else None


def validate_repository(repository: str) -> str:
    if not REPOSITORY_RE.fullmatch(repository):
        raise ValueError("repository must have the form owner/name")
    return repository


def ensure_sync_schema(conn: Any) -> None:
    if getattr(conn, "dialect", None) != "postgresql":
        raise RuntimeError("work ticket synchronization requires PostgreSQL")
    statements = (
        "ALTER TABLE work_tickets ADD COLUMN IF NOT EXISTS repository_full_name TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE work_tickets ADD COLUMN IF NOT EXISTS github_issue_number INTEGER",
        "ALTER TABLE work_tickets ADD COLUMN IF NOT EXISTS github_issue_url TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE work_tickets ADD COLUMN IF NOT EXISTS github_issue_updated_at TIMESTAMPTZ",
        "ALTER TABLE work_tickets ADD COLUMN IF NOT EXISTS last_synced_at TIMESTAMPTZ",
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_work_tickets_github_issue
        ON work_tickets(repository_full_name, github_issue_number)
        WHERE repository_full_name <> '' AND github_issue_number IS NOT NULL
        """,
    )
    for statement in statements:
        conn.execute(statement)


def _existing_sync_columns(conn: Any) -> set[str]:
    rows = conn.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = current_schema() AND table_name = 'work_tickets'
        """
    ).fetchall()
    return {str(row["column_name"]) for row in rows}


def load_tickets(conn: Any) -> list[Ticket]:
    existing = _existing_sync_columns(conn)
    optional = [
        column if column in existing else f"NULL AS {column}" for column in SYNC_COLUMNS
    ]
    rows = conn.execute(
        """
        SELECT ticket_key, title, area, description, acceptance_criteria,
               owner, priority, status, progress, related_job_id, source, updated_at,
               """
        + ", ".join(optional)
        + " FROM work_tickets ORDER BY ticket_key"
    ).fetchall()
    tickets: list[Ticket] = []
    for row in rows:
        values = {key: row[key] for key in row.keys()}
        values["repository_full_name"] = values.get("repository_full_name") or ""
        values["github_issue_url"] = values.get("github_issue_url") or ""
        tickets.append(Ticket(**values))
    return tickets


def _managed_metadata(ticket: Ticket) -> dict[str, Any]:
    terminal = ticket.status in {"completed", "cancelled"}
    return {
        "schema": 1,
        "ticket_key": ticket.ticket_key,
        "status": ticket.status,
        "progress": ticket.progress,
        "acceptance_verified": ticket.status == "completed" and ticket.progress == 100,
        "cancellation_confirmed": ticket.status == "cancelled",
        "db_updated_at": _isoformat(ticket.updated_at),
        "terminal": terminal,
    }


def render_managed_block(ticket: Ticket) -> str:
    metadata = json.dumps(
        _managed_metadata(ticket), ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    related_job = str(ticket.related_job_id) if ticket.related_job_id is not None else "-"
    return "\n".join(
        (
            MANAGED_START,
            f"{MANAGED_JSON_PREFIX}{metadata} -->",
            f"### DB work ticket `{ticket.ticket_key}`",
            "",
            f"- Status: `{ticket.status}` ({ticket.progress}%)",
            f"- Area: {ticket.area}",
            f"- Owner: `{ticket.owner}`",
            f"- Priority: {ticket.priority}",
            f"- Related job: `{related_job}`",
            "",
            "**Description**",
            "",
            ticket.description or "-",
            "",
            "**Acceptance criteria**",
            "",
            ticket.acceptance_criteria or "-",
            MANAGED_END,
        )
    )


def replace_managed_block(body: str | None, block: str) -> str:
    original = body or ""
    matches = list(MANAGED_RE.finditer(original))
    if len(matches) > 1:
        raise ValueError("issue body contains multiple managed work-ticket blocks")
    if matches:
        match = matches[0]
        return original[: match.start()] + block + original[match.end() :]
    separator = "\n\n" if original and not original.endswith("\n\n") else ""
    return f"{original}{separator}{block}"


def parse_managed_metadata(body: str | None) -> dict[str, Any] | None:
    matches = list(MANAGED_RE.finditer(body or ""))
    if len(matches) != 1:
        return None
    content = matches[0].group("content")
    match = re.search(
        re.escape(MANAGED_JSON_PREFIX) + r"(?P<data>\{.*?\})\s*-->", content, re.DOTALL
    )
    if match is None:
        return None
    try:
        metadata = json.loads(match.group("data"))
    except (TypeError, json.JSONDecodeError):
        return None
    return metadata if isinstance(metadata, dict) else None


def _isoformat(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return str(value)


def _parse_time(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _issue_is_newer(ticket: Ticket, issue: Mapping[str, Any]) -> bool:
    issue_time = _parse_time(issue.get("updated_at"))
    sync_time = _parse_time(ticket.last_synced_at)
    return issue_time is not None and (sync_time is None or issue_time > sync_time)


def issue_title(ticket: Ticket) -> str:
    return f"[{ticket.ticket_key}] {ticket.title}"


def _validate_pull(
    ticket: Ticket, issue: Mapping[str, Any]
) -> tuple[str, int] | tuple[None, str]:
    metadata = parse_managed_metadata(issue.get("body"))
    if metadata is None:
        return None, "missing_or_invalid_managed_block"
    if metadata.get("ticket_key") != ticket.ticket_key:
        return None, "ticket_key_mismatch"
    desired = metadata.get("status")
    progress = metadata.get("progress")
    if desired not in VALID_STATUSES or isinstance(progress, bool) or not isinstance(progress, int):
        return None, "invalid_status_or_progress"
    if not 0 <= progress <= 100:
        return None, "invalid_status_or_progress"
    if desired == ticket.status and progress == ticket.progress:
        return None, "unchanged"
    if ticket.status in {"completed", "cancelled"}:
        return None, "db_terminal_state"
    if progress < ticket.progress:
        return None, "progress_regression"

    state = issue.get("state")
    if desired == "completed":
        allowed_source = ticket.status in {"in_progress", "blocked"}
        verified = metadata.get("acceptance_verified") is True
        if not (state == "closed" and progress == 100 and allowed_source and verified):
            return None, "completion_not_verified"
        return desired, progress
    if desired == "cancelled":
        if not (state == "closed" and metadata.get("cancellation_confirmed") is True):
            return None, "cancellation_not_confirmed"
        return desired, progress
    if state != "open":
        return None, "closed_issue_nonterminal_status"

    allowed = {
        "queued": {"in_progress", "blocked"},
        "in_progress": {"in_progress", "blocked"},
        "blocked": {"in_progress", "blocked"},
    }
    if desired not in allowed.get(ticket.status, set()):
        return None, "invalid_state_transition"
    return desired, progress


def _event(
    conn: Any, ticket: Ticket, *, status: str, progress: int, action: str, detail: Any
) -> None:
    note = json.dumps(
        {"source": "github_sync", "action": action, "detail": detail},
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    conn.execute(
        "INSERT INTO work_ticket_events(ticket_key, status, progress, note) VALUES (?, ?, ?, ?)",
        (ticket.ticket_key, status, progress, note),
    )


def _update_link(conn: Any, ticket: Ticket, repository: str, issue: Mapping[str, Any]) -> None:
    conn.execute(
        """
        UPDATE work_tickets
        SET repository_full_name = ?, github_issue_number = ?, github_issue_url = ?,
            github_issue_updated_at = ?, last_synced_at = CURRENT_TIMESTAMP
        WHERE ticket_key = ?
        """,
        (
            repository,
            int(issue["number"]),
            str(issue.get("html_url") or ""),
            issue.get("updated_at"),
            ticket.ticket_key,
        ),
    )


def _update_after_sync(
    conn: Any, ticket: Ticket, repository: str, issue: Mapping[str, Any]
) -> None:
    conn.execute(
        """
        UPDATE work_tickets
        SET repository_full_name = ?, github_issue_number = ?, github_issue_url = ?,
            github_issue_updated_at = ?, last_synced_at = CURRENT_TIMESTAMP
        WHERE ticket_key = ?
        """,
        (
            repository,
            int(issue["number"]),
            str(issue.get("html_url") or ticket.github_issue_url),
            issue.get("updated_at"),
            ticket.ticket_key,
        ),
    )


def _apply_pull(
    conn: Any, ticket: Ticket, issue: Mapping[str, Any], status: str, progress: int
) -> None:
    conn.execute(
        """
        UPDATE work_tickets
        SET status = ?, progress = ?, updated_at = CURRENT_TIMESTAMP,
            completed_at = CASE WHEN ? = 'completed' THEN CURRENT_TIMESTAMP ELSE NULL END
        WHERE ticket_key = ?
        """,
        (status, progress, status, ticket.ticket_key),
    )
    _event(
        conn,
        ticket,
        status=status,
        progress=progress,
        action="issue_to_db",
        detail={"issue": issue["number"], "previous_status": ticket.status},
    )


def _list_repository_issues(transport: HttpTransport, repository: str) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for page in range(1, 101):
        batch = transport.request(
            "GET", f"/repos/{repository}/issues?state=all&per_page=100&page={page}"
        )
        if not isinstance(batch, list):
            raise GitHubApiError("GitHub issues response was not a list")
        issues.extend(issue for issue in batch if "pull_request" not in issue)
        if len(batch) < 100:
            break
    return issues


def _find_existing_issue(ticket: Ticket, issues: list[dict[str, Any]]) -> dict[str, Any] | None:
    expected_title = issue_title(ticket)
    managed_matches = []
    title_matches = []
    for issue in issues:
        metadata = parse_managed_metadata(issue.get("body"))
        if metadata and metadata.get("ticket_key") == ticket.ticket_key:
            managed_matches.append(issue)
        elif issue.get("title") == expected_title:
            title_matches.append(issue)
    candidates = managed_matches or title_matches
    if len(candidates) > 1:
        raise RuntimeError(f"multiple GitHub issues match {ticket.ticket_key}")
    return candidates[0] if candidates else None


def _fresh_ticket(ticket: Ticket, *, status: str, progress: int) -> Ticket:
    values = dict(ticket.__dict__)
    values.update(status=status, progress=progress)
    return Ticket(**values)


def sync_work_tickets(
    conn: Any,
    *,
    repository: str,
    transport: HttpTransport,
    apply: bool = False,
    direction: str = "both",
) -> dict[str, Any]:
    validate_repository(repository)
    if direction not in {"both", "db-to-issue", "issue-to-db"}:
        raise ValueError("invalid synchronization direction")
    if getattr(conn, "dialect", None) != "postgresql":
        raise RuntimeError("work ticket synchronization requires PostgreSQL")
    if apply:
        ensure_sync_schema(conn)

    tickets = load_tickets(conn)
    actions: list[dict[str, Any]] = []
    repository_issues: list[dict[str, Any]] | None = None
    for original in tickets:
        ticket = original
        issue: dict[str, Any] | None = None
        if ticket.github_issue_number and ticket.repository_full_name == repository:
            issue = transport.request(
                "GET", f"/repos/{repository}/issues/{ticket.github_issue_number}"
            )
        elif ticket.github_issue_number and ticket.repository_full_name != repository:
            actions.append(
                {
                    "ticket_key": ticket.ticket_key,
                    "action": "conflict",
                    "reason": "linked_to_different_repository",
                }
            )
            continue
        else:
            if repository_issues is None:
                repository_issues = _list_repository_issues(transport, repository)
            issue = _find_existing_issue(ticket, repository_issues)
            if issue is not None:
                actions.append(
                    {
                        "ticket_key": ticket.ticket_key,
                        "action": "link_existing",
                        "issue": issue["number"],
                    }
                )
                if apply:
                    _update_link(conn, ticket, repository, issue)
                    _event(
                        conn,
                        ticket,
                        status=ticket.status,
                        progress=ticket.progress,
                        action="link_existing",
                        detail={"issue": issue["number"], "repository": repository},
                    )

        if issue is not None and direction in {"both", "issue-to-db"} and _issue_is_newer(ticket, issue):
            decision = _validate_pull(ticket, issue)
            if decision[0] is None:
                reason = decision[1]
                if reason != "unchanged":
                    actions.append(
                        {
                            "ticket_key": ticket.ticket_key,
                            "action": "ignore_issue_change",
                            "issue": issue["number"],
                            "reason": reason,
                        }
                    )
                    if apply:
                        _event(
                            conn,
                            ticket,
                            status=ticket.status,
                            progress=ticket.progress,
                            action="ignore_issue_change",
                            detail={"issue": issue["number"], "reason": reason},
                        )
            else:
                status, progress = decision
                actions.append(
                    {
                        "ticket_key": ticket.ticket_key,
                        "action": "issue_to_db",
                        "issue": issue["number"],
                        "status": status,
                        "progress": progress,
                    }
                )
                if apply:
                    _apply_pull(conn, ticket, issue, status, progress)
                ticket = _fresh_ticket(ticket, status=status, progress=progress)

        if direction in {"both", "db-to-issue"}:
            desired_state = "closed" if ticket.status in {"completed", "cancelled"} else "open"
            desired_title = issue_title(ticket)
            desired_body = replace_managed_block(
                issue.get("body") if issue else "", render_managed_block(ticket)
            )
            if issue is None:
                actions.append(
                    {"ticket_key": ticket.ticket_key, "action": "create_issue"}
                )
                if apply:
                    issue = transport.request(
                        "POST",
                        f"/repos/{repository}/issues",
                        {"title": desired_title, "body": desired_body},
                    )
                    _update_link(conn, ticket, repository, issue)
                    _event(
                        conn,
                        ticket,
                        status=ticket.status,
                        progress=ticket.progress,
                        action="create_issue",
                        detail={"issue": issue["number"], "repository": repository},
                    )
            else:
                patch = {}
                if issue.get("title") != desired_title:
                    patch["title"] = desired_title
                if (issue.get("body") or "") != desired_body:
                    patch["body"] = desired_body
                if issue.get("state") != desired_state:
                    patch["state"] = desired_state
                if patch:
                    actions.append(
                        {
                            "ticket_key": ticket.ticket_key,
                            "action": "update_issue",
                            "issue": issue["number"],
                            "fields": sorted(patch),
                        }
                    )
                    if apply:
                        issue = transport.request(
                            "PATCH", f"/repos/{repository}/issues/{issue['number']}", patch
                        )
                        _event(
                            conn,
                            ticket,
                            status=ticket.status,
                            progress=ticket.progress,
                            action="db_to_issue",
                            detail={"issue": issue["number"], "fields": sorted(patch)},
                        )

        if apply and issue is not None:
            _update_after_sync(conn, ticket, repository, issue)

    return {
        "repository": repository,
        "mode": "apply" if apply else "dry-run",
        "direction": direction,
        "tickets": len(tickets),
        "actions": actions,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synchronize PostgreSQL work_tickets with GitHub Issues"
    )
    parser.add_argument("--db", required=True, help="PostgreSQL DSN")
    parser.add_argument("--repo", required=True, help="GitHub owner/repository")
    parser.add_argument(
        "--direction",
        choices=("both", "db-to-issue", "issue-to-db"),
        default="both",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="apply GitHub and database changes; the default only reports them",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_repository(args.repo)
        token = os.environ.get("GITHUB_TOKEN", "")
        if args.apply and not token:
            raise RuntimeError("GITHUB_TOKEN is required with --apply")
        transport = GitHubTransport(args.repo, token=token)
        with connection(args.db) as conn:
            result = sync_work_tickets(
                conn,
                repository=args.repo,
                transport=transport,
                apply=args.apply,
                direction=args.direction,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    except (GitHubApiError, RuntimeError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
