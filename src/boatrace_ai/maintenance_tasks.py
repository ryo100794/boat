from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .db import connection


def _json_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def aggregate_evaluations(db: str, output: Path) -> dict[str, Any]:
    with connection(db) as conn:
        statuses = conn.execute(
            """
            SELECT category, task_type, status, COUNT(*) AS jobs,
                   MAX(completed_at) AS latest_completed_at
            FROM model_evaluation_jobs
            GROUP BY category, task_type, status
            ORDER BY category, task_type, status
            """
        ).fetchall()
        candidates = conn.execute(
            """
            SELECT c.job_id, c.model_key, c.task_type, c.decision,
                   c.metrics, c.parameters, c.result_path, c.created_at
            FROM model_improvement_candidates c
            JOIN model_evaluation_jobs j ON j.job_id = c.job_id
            WHERE j.status = 'completed'
            ORDER BY c.created_at DESC
            LIMIT 200
            """
        ).fetchall()
    payload = {
        "status": "completed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "job_groups": [
            {key: row[key] for key in row.keys()} for row in statuses
        ],
        "candidates": [
            {key: row[key] for key in row.keys()} for row in candidates
        ],
        "completed_candidates": len(candidates),
    }
    _json_file(output, payload)
    return payload


def backup_raw(app_root: Path, output: Path) -> dict[str, Any]:
    raw_dir = app_root / "data" / "raw"
    staging_dir = app_root / "data" / "archive-staging" / "raw"
    before_files = sum(1 for path in raw_dir.rglob("*") if path.is_file())
    before_bytes = sum(path.stat().st_size for path in raw_dir.rglob("*") if path.is_file())
    command = app_root / "scripts" / "deployment" / "run-boatrace-raw-archive.sh"
    env = dict(os.environ)
    env["BOATRACE_RAW_ARCHIVE_ONCE"] = "1"
    completed = subprocess.run(
        [str(command)],
        cwd=app_root,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"raw backup exited {completed.returncode}: {completed.stdout[-4000:]}"
        )
    after_files = sum(1 for path in raw_dir.rglob("*") if path.is_file())
    after_bytes = sum(path.stat().st_size for path in raw_dir.rglob("*") if path.is_file())
    staging_files = [path for path in staging_dir.glob("*") if path.is_file()]
    payload = {
        "status": "completed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_files_before": before_files,
        "source_files_after": after_files,
        "source_bytes_before": before_bytes,
        "source_bytes_after": after_bytes,
        "archived_files_removed": max(0, before_files - after_files),
        "archived_bytes_removed": max(0, before_bytes - after_bytes),
        "staging_files": len(staging_files),
        "log_tail": completed.stdout.splitlines()[-20:],
    }
    _json_file(output, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Allowlisted queued maintenance tasks")
    sub = parser.add_subparsers(dest="command", required=True)
    aggregate = sub.add_parser("aggregate-evaluations")
    aggregate.add_argument("--db", required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    backup = sub.add_parser("backup-raw")
    backup.add_argument("--app-root", type=Path, required=True)
    backup.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "aggregate-evaluations":
        aggregate_evaluations(args.db, args.output)
    elif args.command == "backup-raw":
        backup_raw(args.app_root.resolve(), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
