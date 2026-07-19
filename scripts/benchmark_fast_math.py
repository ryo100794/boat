from __future__ import annotations

import argparse
import importlib.util
import json
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100_000)
    args = parser.parse_args()

    from boatrace_ai import fast_math

    values = (0.48, 0.19, 0.12, 0.09, 0.07, 0.05)
    started = time.perf_counter()
    checksum = 0.0
    for _ in range(args.iterations):
        checksum += sum(fast_math.plackett_luce_probabilities(values))
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "native": fast_math.native_available(),
                "iterations": args.iterations,
                "elapsed_seconds": elapsed,
                "races_per_second": args.iterations / max(elapsed, 1e-9),
                "checksum": checksum,
                "module": importlib.util.find_spec(
                    "boatrace_ai._fast_boat_math"
                ).origin
                if fast_math.native_available()
                else None,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
