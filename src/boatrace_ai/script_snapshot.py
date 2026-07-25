from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path


def run_snapshot(script: Path, *, app_root: Path, arguments: list[str]) -> int:
    source = script.resolve(strict=True)
    root = app_root.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="boatrace-script-") as temporary:
        snapshot = Path(temporary) / source.name
        snapshot.write_bytes(source.read_bytes())
        env = dict(os.environ)
        env["BOATRACE_APP_ROOT"] = str(root)
        completed = subprocess.run(
            ["bash", str(snapshot), *arguments],
            cwd=root,
            env=env,
            check=False,
        )
    return completed.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an immutable snapshot of a long-lived shell script."
    )
    parser.add_argument("--app-root", type=Path, required=True)
    parser.add_argument("script", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_snapshot(
        args.script,
        app_root=args.app_root,
        arguments=args.arguments,
    )


if __name__ == "__main__":
    raise SystemExit(main())
