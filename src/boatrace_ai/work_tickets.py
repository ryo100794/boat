from __future__ import annotations

import argparse
import json
from typing import Any

from .db import connection
from .evaluation_queue import ensure_schema, seed_work_tickets, update_work_ticket


STATUSES = {"queued", "in_progress", "blocked", "completed", "cancelled"}


def add_ticket(
    conn: Any,
    *,
    ticket_key: str,
    title: str,
    area: str,
    description: str,
    acceptance_criteria: str,
    owner: str,
    priority: int,
    status: str,
    progress: int,
    source: str,
) -> None:
    if status not in STATUSES:
        raise ValueError("invalid ticket status")
    if not 0 <= progress <= 100:
        raise ValueError("progress must be between 0 and 100")
    conn.execute(
        """
        INSERT INTO work_tickets(
          ticket_key, title, area, description, acceptance_criteria,
          owner, priority, status, progress, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticket_key) DO UPDATE SET
          title = excluded.title, area = excluded.area,
          description = excluded.description,
          acceptance_criteria = excluded.acceptance_criteria,
          owner = excluded.owner, priority = excluded.priority,
          source = excluded.source, updated_at = CURRENT_TIMESTAMP
        """,
        (
            ticket_key,
            title,
            area,
            description,
            acceptance_criteria,
            owner,
            priority,
            status,
            progress,
            source,
        ),
    )


def list_tickets(conn: Any) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT ticket_key, title, area, owner, priority, status, progress,
               related_job_id, source, created_at, updated_at, completed_at,
               description, acceptance_criteria
        FROM work_tickets
        ORDER BY
          CASE status WHEN 'in_progress' THEN 0 WHEN 'queued' THEN 1
                      WHEN 'blocked' THEN 2 ELSE 3 END,
          priority DESC, updated_at DESC
        """
    ).fetchall()
    return [{key: row[key] for key in row.keys()} for row in rows]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DB-backed project work tickets")
    parser.add_argument("--db", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("seed")
    sub.add_parser("list")
    add = sub.add_parser("add")
    add.add_argument("--key", required=True)
    add.add_argument("--title", required=True)
    add.add_argument("--area", required=True)
    add.add_argument("--description", default="")
    add.add_argument("--acceptance", default="")
    add.add_argument("--owner", default="codex")
    add.add_argument("--priority", type=int, default=0)
    add.add_argument("--status", choices=sorted(STATUSES), default="queued")
    add.add_argument("--progress", type=int, default=0)
    add.add_argument("--source", default="user")
    update = sub.add_parser("update")
    update.add_argument("--key", required=True)
    update.add_argument("--status", choices=sorted(STATUSES), required=True)
    update.add_argument("--progress", type=int, required=True)
    update.add_argument("--note", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with connection(args.db) as conn:
        ensure_schema(conn)
        if args.command == "seed":
            print(json.dumps({"inserted": seed_work_tickets(conn)}))
        elif args.command == "list":
            print(json.dumps(list_tickets(conn), ensure_ascii=False, indent=2, default=str))
        elif args.command == "add":
            add_ticket(
                conn,
                ticket_key=args.key,
                title=args.title,
                area=args.area,
                description=args.description,
                acceptance_criteria=args.acceptance,
                owner=args.owner,
                priority=args.priority,
                status=args.status,
                progress=args.progress,
                source=args.source,
            )
        elif args.command == "update":
            update_work_ticket(
                conn,
                ticket_key=args.key,
                status=args.status,
                progress=args.progress,
                note=args.note,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
