from __future__ import annotations

import json
import re
import subprocess
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from boatrace_ai import maintenance_tasks


MIDDAY_UTC = datetime(2026, 8, 3, 11, 0, tzinfo=timezone.utc)
OVERNIGHT_UTC = datetime(2026, 8, 3, 16, 0, tzinfo=timezone.utc)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    payload = maintenance_tasks.repository_sync(
        checkout, output, db="test", now=MIDDAY_UTC
    )

    assert payload["action"] == "deferred_active_evaluation"
    assert payload["active_evaluations"] == 1
    assert payload["behind"] == 1
    assert payload["before_head"] == payload["after_head"] == before
    assert payload["service_refresh"]["action"] == "deferred_repository_not_ready"
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
        checkout, tmp_path / "updated.json", db="test", now=MIDDAY_UTC
    )

    assert payload["action"] == "fast_forwarded"
    assert payload["behind"] == 1
    assert payload["after_head"] == expected
    assert (checkout / "tracked.txt").read_text(encoding="utf-8") == "two\n"
    assert (
        payload["service_refresh"]["action"]
        == "deferred_outside_maintenance_window"
    )


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
        checkout, tmp_path / "dirty.json", db="test", now=MIDDAY_UTC
    )

    assert payload["action"] == "deferred_dirty_worktree"
    assert payload["dirty_paths"]
    assert payload["after_head"] == before


def test_repository_sync_schedules_overnight_refresh_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _seed, checkout = _repositories(tmp_path)
    monkeypatch.setattr(maintenance_tasks, "connection", _connection(0))
    scheduled: list[tuple[Path, str, str, datetime]] = []

    def schedule(app_root: Path, *, db: str, head: str, now: datetime):
        scheduled.append((app_root, db, head, now))
        return {"action": "scheduled", "head": head, "pid": 123}

    payload = maintenance_tasks.repository_sync(
        checkout,
        tmp_path / "scheduled.json",
        db="test",
        now=OVERNIGHT_UTC,
        schedule_refresh=schedule,
    )

    head = _git("rev-parse", "HEAD", cwd=checkout)
    assert payload["action"] == "up_to_date"
    assert payload["service_refresh"] == {
        "action": "scheduled",
        "head": head,
        "pid": 123,
    }
    assert scheduled == [(checkout.resolve(), "test", head, OVERNIGHT_UTC)]


def test_repository_sync_does_not_duplicate_recent_refresh(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _seed, checkout = _repositories(tmp_path)
    monkeypatch.setattr(maintenance_tasks, "connection", _connection(0))
    head = _git("rev-parse", "HEAD", cwd=checkout)
    monkeypatch.setattr(
        maintenance_tasks,
        "_deployment_state",
        lambda _root: {
            "status": "scheduled",
            "head": head,
            "scheduled_at": (OVERNIGHT_UTC - timedelta(minutes=30)).isoformat(),
        },
    )

    payload = maintenance_tasks.repository_sync(
        checkout,
        tmp_path / "already-scheduled.json",
        db="test",
        now=OVERNIGHT_UTC,
        schedule_refresh=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("refresh must not be scheduled twice")
        ),
    )

    assert payload["service_refresh"]["action"] == "already_scheduled"


def test_repository_sync_recognizes_deployed_head(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _seed, checkout = _repositories(tmp_path)
    monkeypatch.setattr(maintenance_tasks, "connection", _connection(0))
    head = _git("rev-parse", "HEAD", cwd=checkout)
    monkeypatch.setattr(
        maintenance_tasks,
        "_deployment_state",
        lambda _root: {"status": "completed", "head": head},
    )

    payload = maintenance_tasks.repository_sync(
        checkout,
        tmp_path / "deployed.json",
        db="test",
        now=OVERNIGHT_UTC,
    )

    assert payload["service_refresh"] == {"action": "up_to_date", "head": head}


def test_refresh_services_restarts_only_active_allowlisted_programs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    head = "a" * 40
    runner = "boatrace-evaluation-runner:boatrace-evaluation-runner_00"
    scheduler = "boatrace-evaluation-scheduler"
    calls: list[list[str]] = []
    status_text = "\n".join(
        [
            "boatrace-dashboard RUNNING pid 1, uptime 1:00:00",
            "boatrace-daily-shadow-bundle-update STOPPED Not started",
            f"{runner} RUNNING pid 2, uptime 1:00:00",
            f"{scheduler} RUNNING pid 3, uptime 1:00:00",
            "unrelated-service RUNNING pid 4, uptime 1:00:00",
        ]
    )

    monkeypatch.setattr(
        maintenance_tasks,
        "_git_value",
        lambda _root, *args: head if args == ("rev-parse", "HEAD") else "",
    )
    monkeypatch.setattr(maintenance_tasks, "connection", _connection(0))

    def run(command, **_kwargs):
        calls.append([str(value) for value in command])
        stdout = status_text if command[-1] == "status" else "ok"
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(maintenance_tasks.subprocess, "run", run)
    payload = maintenance_tasks.refresh_services(
        tmp_path, db="test", head=head, delay_seconds=0
    )

    assert payload["restarted"] == ["boatrace-dashboard", runner, scheduler]
    assert [command[-1] for command in calls] == [
        "status",
        "boatrace-dashboard",
        runner,
        scheduler,
    ]
    assert "boatrace-daily-shadow-bundle-update" in payload["skipped_stopped"]
    assert "unrelated-service" not in payload["restarted"]


def test_refresh_services_defers_if_evaluation_started_during_delay(
    tmp_path: Path,
    monkeypatch,
) -> None:
    head = "b" * 40
    monkeypatch.setattr(
        maintenance_tasks,
        "_git_value",
        lambda _root, *args: head if args == ("rev-parse", "HEAD") else "",
    )
    monkeypatch.setattr(maintenance_tasks, "connection", _connection(1))
    monkeypatch.setattr(
        maintenance_tasks.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("supervisor must not be touched during evaluation")
        ),
    )

    payload = maintenance_tasks.refresh_services(
        tmp_path, db="test", head=head, delay_seconds=0
    )

    assert payload["status"] == "deferred_active_evaluation"
    assert payload["active_evaluations"] == 1
    assert payload["restarted"] == []


def test_service_refresh_allowlist_covers_code_consuming_supervisor_programs(
) -> None:
    configs = [
        *PROJECT_ROOT.glob("scripts/deployment/supervisor-boatrace-*.ini"),
        *PROJECT_ROOT.glob("scripts/deployment/supervisor/boatrace-*.ini"),
    ]
    configured: set[str] = set()
    for path in configs:
        matches = re.finditer(
            r"^\[program:([^]]+)]",
            path.read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
        configured.update(match.group(1) for match in matches)
    deliberately_separate = {
        "boatrace-evaluation-runner",
        "boatrace-evaluation-scheduler",
        "boatrace-raw-archive",
        "boatrace-standard-evaluation",
    }

    assert configured - deliberately_separate == set(
        maintenance_tasks.SERVICE_REFRESH_PROGRAMS
    )
