from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from .db import connection


CANONICAL_MARKDOWN = frozenset(
    {
        "README.md",
        "CONTRIBUTING.md",
        ".github/pull_request_template.md",
        "docs/WORKFLOW.md",
        "docs/ARCHITECTURE.md",
        "docs/PROJECT_STATUS.md",
        "docs/GPU_WORKSPACE.md",
        "docs/MODEL_FEATURE_RESEARCH.md",
        "docs/TELEBOAT_AGENT_AUDIT.md",
        "docs/TELEBOAT_API.md",
    }
)
DEFAULT_MAX_TRACKED_FILE_BYTES = 10 * 1024 * 1024
_MARKDOWN_LINK = re.compile(
    r"!?\[[^\]]*\]\(\s*(?:<(?P<angle>[^>]+)>|(?P<plain>[^\s)]+))"
)
_SECRET_FILE_NAMES = {
    ".env",
    "credentials.json",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
    "service-account.json",
}
_SECRET_FILE_SUFFIXES = {
    ".jks",
    ".key",
    ".keystore",
    ".p12",
    ".pem",
    ".pfx",
}


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
    env["BOATRACE_RAW_ARCHIVE_MAX_BATCHES"] = "1"
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


def _git_files(app_root: Path, *, include_untracked: bool) -> tuple[list[Path], bool]:
    command = ["git", "-C", str(app_root), "ls-files", "-z", "--cached"]
    if include_untracked:
        command.extend(["--others", "--exclude-standard"])
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        return [], False
    paths = [
        Path(os.fsdecode(value))
        for value in completed.stdout.split(b"\0")
        if value
    ]
    return sorted(set(paths), key=lambda path: path.as_posix()), True


def _looks_secret(path: Path) -> bool:
    name = path.name.lower()
    if name in _SECRET_FILE_NAMES or name.startswith(".env."):
        return name not in {".env.example", ".env.sample", ".env.template"}
    if path.suffix.lower() in _SECRET_FILE_SUFFIXES:
        return True
    stem = path.stem.lower().replace("_", "-")
    return (
        (stem.startswith("credential") or stem.startswith("secret"))
        and path.suffix.lower() in {".json", ".toml", ".yaml", ".yml"}
    )


def _relative_markdown_targets(markdown: Path) -> list[str]:
    try:
        content = markdown.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return []
    targets: list[str] = []
    for match in _MARKDOWN_LINK.finditer(content):
        target = (match.group("angle") or match.group("plain") or "").strip()
        parsed = urlsplit(target)
        if not target or target.startswith("#") or parsed.scheme or parsed.netloc:
            continue
        if parsed.path.startswith("/"):
            continue
        if parsed.path:
            targets.append(unquote(parsed.path))
    return targets


def repository_hygiene(
    app_root: Path,
    output: Path,
    *,
    max_file_bytes: int = DEFAULT_MAX_TRACKED_FILE_BYTES,
) -> dict[str, Any]:
    app_root = app_root.resolve()
    repository_files, repository_available = _git_files(
        app_root, include_untracked=True
    )
    tracked_files, tracked_available = _git_files(app_root, include_untracked=False)
    violations: list[dict[str, Any]] = []
    if not repository_available or not tracked_available:
        violations.append(
            {
                "kind": "repository_unavailable",
                "path": ".",
                "detail": "git ls-files could not inventory the application root",
            }
        )

    markdown_paths = sorted(
        (path for path in repository_files if path.suffix.lower() == ".md"),
        key=lambda path: path.as_posix(),
    )
    for relative in markdown_paths:
        if relative.as_posix() not in CANONICAL_MARKDOWN:
            violations.append(
                {
                    "kind": "unknown_markdown",
                    "path": relative.as_posix(),
                    "detail": "Markdown is outside the canonical documentation set",
                }
            )

    links_checked = 0
    for relative in markdown_paths:
        source = app_root / relative
        for target in _relative_markdown_targets(source):
            links_checked += 1
            resolved_target = (source.parent / target).resolve()
            try:
                resolved_target.relative_to(app_root)
            except ValueError:
                exists = False
            else:
                exists = resolved_target.exists()
            if not exists:
                violations.append(
                    {
                        "kind": "broken_markdown_link",
                        "path": relative.as_posix(),
                        "target": target,
                        "detail": "Relative Markdown link target does not exist",
                    }
                )

    for relative in tracked_files:
        absolute = app_root / relative
        if _looks_secret(relative):
            violations.append(
                {
                    "kind": "secret_like_tracked_file",
                    "path": relative.as_posix(),
                    "detail": "Tracked path resembles a credential or private key",
                }
            )
        try:
            size = absolute.stat().st_size
        except OSError:
            continue
        if size > max_file_bytes:
            violations.append(
                {
                    "kind": "oversized_tracked_file",
                    "path": relative.as_posix(),
                    "bytes": size,
                    "limit_bytes": max_file_bytes,
                    "detail": "Tracked file exceeds the repository size policy",
                }
            )

    violations.sort(
        key=lambda item: (
            str(item.get("kind", "")),
            str(item.get("path", "")),
            str(item.get("target", "")),
        )
    )
    payload = {
        "status": "requires_action" if violations else "completed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "app_root": str(app_root),
        "policy": {
            "canonical_markdown": sorted(CANONICAL_MARKDOWN),
            "max_tracked_file_bytes": max_file_bytes,
        },
        "summary": {
            "repository_files": len(repository_files),
            "tracked_files": len(tracked_files),
            "markdown_files": len(markdown_paths),
            "relative_links_checked": links_checked,
            "violations": len(violations),
        },
        "markdown_files": [path.as_posix() for path in markdown_paths],
        "violations": violations,
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
    hygiene = sub.add_parser("repository-hygiene")
    hygiene.add_argument("--app-root", type=Path, required=True)
    hygiene.add_argument("--output", type=Path, required=True)
    hygiene.add_argument(
        "--max-file-bytes",
        type=int,
        default=DEFAULT_MAX_TRACKED_FILE_BYTES,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "aggregate-evaluations":
        aggregate_evaluations(args.db, args.output)
    elif args.command == "backup-raw":
        backup_raw(args.app_root.resolve(), args.output)
    elif args.command == "repository-hygiene":
        repository_hygiene(
            args.app_root,
            args.output,
            max_file_bytes=args.max_file_bytes,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
