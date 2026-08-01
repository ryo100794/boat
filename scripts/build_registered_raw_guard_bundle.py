#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from boatrace_ai.runtime.raw_guard_bundle import build_registered_raw_guard_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description="Build registered raw-EV guard bundle")
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--audit-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_registered_raw_guard_bundle(
                args.source_result, args.audit_result, args.output
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
