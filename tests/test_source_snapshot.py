from __future__ import annotations

import threading
import time
from pathlib import Path

from boatrace_ai.script_snapshot import run_snapshot


def test_snapshot_pins_python_source_and_helper_scripts(tmp_path: Path) -> None:
    root = tmp_path / "boat"
    source_dir = root / "src"
    scripts_dir = root / "scripts"
    source_dir.mkdir(parents=True)
    scripts_dir.mkdir()
    module = source_dir / "frozen_module.py"
    helper = scripts_dir / "helper.py"
    module.write_text('VALUE = "original"\n', encoding="utf-8")
    helper.write_text('print("original-helper")\n', encoding="utf-8")
    job = root / "long-job.sh"
    job.write_text(
        "#!/usr/bin/env bash\n"
        "sleep 0.2\n"
        "python3 -c 'import frozen_module; print(frozen_module.VALUE)' "
        "> module-result.txt\n"
        'python3 "$BOATRACE_SCRIPTS_DIR/helper.py" > helper-result.txt\n',
        encoding="utf-8",
    )

    def replace_sources() -> None:
        time.sleep(0.05)
        module.write_text('VALUE = "replaced"\n', encoding="utf-8")
        helper.write_text('print("replaced-helper")\n', encoding="utf-8")

    writer = threading.Thread(target=replace_sources)
    writer.start()
    status = run_snapshot(job, app_root=root, arguments=[])
    writer.join()

    assert status == 0
    assert (root / "module-result.txt").read_text(encoding="utf-8") == "original\n"
    assert (root / "helper-result.txt").read_text(encoding="utf-8") == (
        "original-helper\n"
    )
