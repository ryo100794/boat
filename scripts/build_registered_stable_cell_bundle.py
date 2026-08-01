from __future__ import annotations

import argparse
import json
from pathlib import Path

from boatrace_ai.runtime.stable_cell_bundle import (
    build_registered_stable_cell_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the preregistered Aug-2 stable-cell shadow bundle."
    )
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_registered_stable_cell_bundle(args.source_result, args.output),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
