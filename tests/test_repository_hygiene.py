from __future__ import annotations

import json
import subprocess
from pathlib import Path

from boatrace_ai.maintenance_tasks import main, repository_hygiene


def _init_repository(root: Path) -> None:
    subprocess.run(["git", "init", "-q", str(root)], check=True)


def _track(root: Path, *paths: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), "add", "-f", "--", *paths],
        check=True,
    )


def test_repository_hygiene_accepts_canonical_docs_and_valid_links(
    tmp_path: Path,
) -> None:
    _init_repository(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (tmp_path / "README.md").write_text(
        "[workflow](docs/WORKFLOW.md#operations)\n"
        "[external](https://example.com/reference)\n",
        encoding="utf-8",
    )
    (docs / "WORKFLOW.md").write_text("# Operations\n", encoding="utf-8")
    _track(tmp_path, "README.md", "docs/WORKFLOW.md")

    output = tmp_path / "result.json"
    payload = repository_hygiene(tmp_path, output, max_file_bytes=1024)

    assert payload["status"] == "completed"
    assert payload["summary"]["relative_links_checked"] == 1
    assert payload["violations"] == []
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "completed"
    assert not output.with_suffix(".json.tmp").exists()


def test_repository_hygiene_reports_unknown_docs_broken_links_and_risky_files(
    tmp_path: Path,
) -> None:
    _init_repository(tmp_path)
    (tmp_path / "README.md").write_text(
        "[missing](docs/MISSING.md)\n", encoding="utf-8"
    )
    (tmp_path / "NOTES.md").write_text("temporary notes\n", encoding="utf-8")
    (tmp_path / ".env").write_text("TOKEN=not-a-real-secret\n", encoding="utf-8")
    (tmp_path / "large.bin").write_bytes(b"x" * 33)
    _track(tmp_path, "README.md", "NOTES.md", ".env", "large.bin")

    payload = repository_hygiene(
        tmp_path,
        tmp_path / "audit.json",
        max_file_bytes=32,
    )

    assert payload["status"] == "requires_action"
    violations = {(item["kind"], item["path"]) for item in payload["violations"]}
    assert ("unknown_markdown", "NOTES.md") in violations
    assert ("broken_markdown_link", "README.md") in violations
    assert ("secret_like_tracked_file", ".env") in violations
    assert ("oversized_tracked_file", "large.bin") in violations


def test_repository_hygiene_command_records_violations_and_returns_success(
    tmp_path: Path,
) -> None:
    _init_repository(tmp_path)
    (tmp_path / "scratch.md").write_text("scratch\n", encoding="utf-8")
    output = tmp_path / "state" / "repository-hygiene.json"

    result = main(
        [
            "repository-hygiene",
            "--app-root",
            str(tmp_path),
            "--output",
            str(output),
            "--max-file-bytes",
            "1024",
        ]
    )

    assert result == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "requires_action"
    assert payload["violations"][0]["kind"] == "unknown_markdown"


def test_real_repository_has_no_unknown_markdown(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]

    payload = repository_hygiene(
        repository_root,
        tmp_path / "real-repository-hygiene.json",
    )

    unknown = [
        violation
        for violation in payload["violations"]
        if violation["kind"] == "unknown_markdown"
    ]
    assert unknown == []
