from __future__ import annotations

import json
import subprocess
from contextlib import contextmanager
from pathlib import Path

from boatrace_ai import maintenance_tasks


def _git(*args: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _commit(root: Path, content: str, message: str) -> None:
    (root / "tracked.txt").write_text(content, encoding="utf-8")
    _git("add", "tracked.txt", cwd=root)
    _git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        message,
        cwd=root,
    )


def _repositories(tmp_path: Path) -> tuple[Path, Path]:
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    checkout = tmp_path / "checkout"
    _git("init", "--bare", str(remote))
    _git("init", str(seed))
    _commit(seed, "one\n", "initial")
    _git("remote", "add", "origin", str(remote), cwd=seed)
    _git("push", "-u", "origin", "master", cwd=seed)
    _git("clone", str(remote), str(checkout))
    return seed, checkout


def _connection(active: int):
    class RowResult:
        def fetchone(self):
            return {"count": active}

    class Connection:
        def execute(self, _statement):
            return RowResult()

    @contextmanager
    def connect(_db):
        yield Connection()

    return connect


def test_repository_sync_fetches_but_defers_during_evaluation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    seed, checkout = _repositories(tmp_path)
    before = _git("rev-parse", "HEAD", cwd=checkout)
    _commit(seed, "two\n", "upstream")
    _git("push", cwd=seed)
    monkeypatch.setattr(maintenance_tasks, "connection", _connection(1))

    output = tmp_path / "deferred.json"
    payload = maintenance_tasks.repository_sync(checkout, output, db="test")

    assert payload["action"] == "deferred_active_evaluation"
    assert payload["active_evaluations"] == 1
    assert payload["behind"] == 1
    assert payload["before_head"] == payload["after_head"] == before
    assert json.loads(output.read_text(encoding="utf-8"))["action"] == payload["action"]


def test_repository_sync_fast_forwards_clean_idle_checkout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    seed, checkout = _repositories(tmp_path)
    _commit(seed, "two\n", "upstream")
    _git("push", cwd=seed)
    expected = _git("rev-parse", "HEAD", cwd=seed)
    monkeypatch.setattr(maintenance_tasks, "connection", _connection(0))

    payload = maintenance_tasks.repository_sync(
        checkout, tmp_path / "updated.json", db="test"
    )

    assert payload["action"] == "fast_forwarded"
    assert payload["behind"] == 1
    assert payload["after_head"] == expected
    assert (checkout / "tracked.txt").read_text(encoding="utf-8") == "two\n"


def test_repository_sync_defers_dirty_checkout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    seed, checkout = _repositories(tmp_path)
    _commit(seed, "two\n", "upstream")
    _git("push", cwd=seed)
    (checkout / "tracked.txt").write_text("local\n", encoding="utf-8")
    before = _git("rev-parse", "HEAD", cwd=checkout)
    monkeypatch.setattr(maintenance_tasks, "connection", _connection(0))

    payload = maintenance_tasks.repository_sync(
        checkout, tmp_path / "dirty.json", db="test"
    )

    assert payload["action"] == "deferred_dirty_worktree"
    assert payload["dirty_paths"]
    assert payload["after_head"] == before
