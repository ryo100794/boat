#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from boatrace_ai.runtime.quota_ceil_bundle import build_registered_quota_ceil_bundle


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the preregistered V21 quota-ceil shadow bundle."
    )
    parser.add_argument("source_result", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build_registered_quota_ceil_bundle(args.source_result, args.output),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
