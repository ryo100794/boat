from __future__ import annotations

import threading
import time
from pathlib import Path

from boatrace_ai.evaluation_queue import build_command
from boatrace_ai.script_snapshot import run_snapshot


def test_snapshot_is_unchanged_when_source_is_replaced(tmp_path: Path) -> None:
    root = tmp_path / "boat"
    root.mkdir()
    script = root / "long-job.sh"
    script.write_text(
        '#!/usr/bin/env bash\nsleep 0.2\nprintf "original:%s:%s\\n" "$BOATRACE_APP_ROOT" "$1" > result.txt\n',
        encoding="utf-8",
    )

    def replace_source() -> None:
        time.sleep(0.05)
        replacement = root / "replacement.sh"
        replacement.write_text(
            '#!/usr/bin/env bash\nprintf "replaced\\n" > result.txt\n',
            encoding="utf-8",
        )
        replacement.replace(script)

    writer = threading.Thread(target=replace_source)
    writer.start()
    status = run_snapshot(script, app_root=root, arguments=["value"])
    writer.join()

    assert status == 0
    assert (root / "result.txt").read_text(encoding="utf-8") == (
        f"original:{root}:value\n"
    )


def test_standardized_command_uses_snapshot_runner(tmp_path: Path) -> None:
    root = tmp_path / "boat"
    python = root / ".venv/bin/python"
    command, output = build_command(
        {
            "job_id": 42,
            "task_type": "standardized_365d",
            "parameters": {"evaluation_date": "2026-07-25"},
        },
        app_root=root,
        python=python,
        db="postgresql://test",
    )

    assert command == [
        str(python),
        "-m",
        "boatrace_ai.script_snapshot",
        "--app-root",
        str(root),
        str(root / "scripts/run_standardized_365d_evaluations.sh"),
    ]
    assert output == root / "data/models/standardized_365d_v2/manifest.json"
