#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict, deque
from pathlib import Path


PACKAGE = Path("src/boatrace_ai")
EXTERNAL_ROOTS = (Path("scripts"), Path("tests"))
OPERATIONAL_ROOTS = {
    "__init__",
    "cli",
    "web_dashboard",
    "realtime_collector",
    "realtime_predictor",
    "bankroll_optimizer",
    "feature_tuning",
    "feature_diagnostics_stream",
    "historical_safe",
}
VERSIONED = re.compile(r"\d")


def imported_modules(path: Path, *, package_local: bool) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level and package_local:
                if module:
                    found.add(module.split(".")[0])
                else:
                    found.update(alias.name.split(".")[0] for alias in node.names)
            elif module == "boatrace_ai":
                found.update(alias.name.split(".")[0] for alias in node.names)
            elif module.startswith("boatrace_ai."):
                found.add(module.split(".")[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("boatrace_ai."):
                    found.add(alias.name.split(".")[1])
    return found


def build_inventory(extra_roots: set[str]) -> dict[str, object]:
    files = {path.stem: path for path in PACKAGE.glob("*.py")}
    edges = {
        name: {target for target in imported_modules(path, package_local=True) if target in files}
        for name, path in files.items()
    }
    roots = set(OPERATIONAL_ROOTS) | extra_roots
    for root in EXTERNAL_ROOTS:
        for path in root.glob("*.py"):
            roots.update(target for target in imported_modules(path, package_local=False) if target in files)
    roots &= files.keys()

    reachable: set[str] = set()
    queue = deque(sorted(roots))
    while queue:
        current = queue.popleft()
        if current in reachable:
            continue
        reachable.add(current)
        queue.extend(sorted(edges.get(current, set()) - reachable))

    incoming: dict[str, set[str]] = defaultdict(set)
    for source, targets in edges.items():
        for target in targets:
            incoming[target].add(source)
    versioned = {name for name in files if VERSIONED.search(name)}
    return {
        "roots": sorted(roots),
        "module_count": len(files),
        "versioned_count": len(versioned),
        "reachable_versioned": sorted(versioned & reachable),
        "safe_to_remove": sorted(versioned - reachable),
        "incoming": {name: sorted(incoming[name]) for name in sorted(versioned)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inventory numbered boatrace_ai modules by operational reachability.")
    parser.add_argument("--root", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = build_inventory(set(args.root))
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
