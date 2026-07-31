from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/deployment/termux-teleboat-opening-preview.sh"


def _executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _run(
    tmp_path: Path,
    today: str,
    *,
    check_rc: int = 1,
    run_rc: int = 0,
    target_date: str | None = None,
):
    prefix = tmp_path / "termux"
    bin_dir = prefix / "bin"
    bin_dir.mkdir(parents=True)
    calls = tmp_path / "calls"
    _executable(bin_dir / "date", f"#!/bin/sh\nprintf '%s\\n' '{today}'\n")
    _executable(bin_dir / "flock", "#!/bin/sh\nexit 0\n")
    for command in ("termux-wake-lock", "termux-wake-unlock"):
        _executable(
            bin_dir / command,
            f"#!/bin/sh\nprintf '%s\\n' '{command}' >>'{calls}'\n",
        )
    _executable(
        bin_dir / "proot-distro",
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$*\" >>'{calls}'\n"
        "case \"$*\" in\n"
        f"  *' -c import json, pathlib, sys;'*) exit {check_rc} ;;\n"
        f"  *) exit {run_rc} ;;\n"
        "esac\n",
    )
    env = os.environ.copy()
    env["TERMUX_PREFIX"] = str(prefix)
    env["TERMUX_HOME"] = str(tmp_path / "home")
    if target_date is not None:
        env["TELEBOAT_PREVIEW_DATE"] = target_date
    result = subprocess.run(["bash", str(SCRIPT)], env=env, check=False)
    return result, calls.read_text(encoding="utf-8") if calls.exists() else ""


@pytest.mark.parametrize("today", ["2026-07-30", "2026-08-01"])
def test_outside_target_date_is_a_noop(tmp_path, today):
    result, calls = _run(tmp_path, today, target_date="2026-07-31")
    assert result.returncode == 0
    assert calls == ""


def test_existing_success_is_a_noop(tmp_path):
    result, calls = _run(tmp_path, "2026-07-31", check_rc=0)
    assert result.returncode == 0
    assert "termux-wake-lock" not in calls
    assert "teleboat_opening_preview.py" not in calls


def test_no_candidate_remains_retryable_and_unlocks(tmp_path):
    result, calls = _run(tmp_path, "2026-07-31", check_rc=1, run_rc=3)
    assert result.returncode == 3
    assert "termux-wake-lock\n" in calls
    assert calls.endswith("termux-wake-unlock\n")
    assert "/root/boat/scripts/teleboat_opening_preview.py" in calls
    assert "--date 2026-07-31" in calls
    assert "--poll-seconds 30" in calls
    assert "--timeout-seconds 840" in calls
    assert "--output /root/boat/data/teleboat-opening-preview-2026-07-31.json" in calls
    assert "--journal-path /root/boat/data/teleboat_vote_journal.jsonl" in calls
    assert "--secret-path /root/boat/.secrets/teleboat-login.json" in calls
    assert "05730047" not in calls
    assert "0911" not in calls


def test_current_jst_date_drives_daily_output_paths(tmp_path):
    result, calls = _run(tmp_path, "2026-08-01", check_rc=1, run_rc=3)
    assert result.returncode == 3
    assert "--date 2026-08-01" in calls
    assert "--output /root/boat/data/teleboat-opening-preview-2026-08-01.json" in calls


def test_shell_is_valid_and_never_requests_submission():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    source = SCRIPT.read_text(encoding="utf-8")
    assert "set -x" not in source
    assert "umask 077" in source
    assert '"$FLOCK" -n 9' in source
    assert "--execute" not in source
    assert "05730047" not in source
    assert "0911" not in source
