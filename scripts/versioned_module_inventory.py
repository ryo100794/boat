#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict, deque
from pathlib import Path


PACKAGE = Path("src/boatrace_ai")
PACKAGE_NAME = "boatrace_ai"
EXTERNAL_ROOTS = (Path("scripts"), Path("tests"))
OPERATIONAL_ROOTS = {
    "boatrace_ai.cli",
    "boatrace_ai.web.dashboard",
    "boatrace_ai.runtime.collector",
    "boatrace_ai.runtime.predictor",
    "boatrace_ai.runtime.model_cycle",
    "boatrace_ai.bankroll_optimizer",
    "boatrace_ai.feature_tuning",
    "boatrace_ai.feature_diagnostics_stream",
    "boatrace_ai.ingestion.backfill",
}
VERSIONED = re.compile(r"\d")


def module_name(path: Path) -> str:
    relative = path.relative_to(PACKAGE).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join((PACKAGE_NAME, *parts)) if parts else PACKAGE_NAME


def source_modules() -> dict[str, Path]:
    return {module_name(path): path for path in PACKAGE.rglob("*.py")}


def imported_modules(
    path: Path,
    *,
    current_module: str | None,
    modules: set[str],
) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return set()

    found: set[str] = set()
    current_package = None
    if current_module:
        current_package = (
            current_module
            if path.name == "__init__.py"
            else current_module.rpartition(".")[0]
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in modules:
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = _import_base(node, current_package=current_package)
            if not base or not (base == PACKAGE_NAME or base.startswith(f"{PACKAGE_NAME}.")):
                continue
            if base in modules:
                found.add(base)
            for alias in node.names:
                candidate = f"{base}.{alias.name}"
                if candidate in modules:
                    found.add(candidate)
    return found


def _import_base(node: ast.ImportFrom, *, current_package: str | None) -> str | None:
    if not node.level:
        return node.module or None
    if not current_package:
        return None
    parts = current_package.split(".")
    keep = len(parts) - (node.level - 1)
    if keep < 1:
        return None
    prefix = ".".join(parts[:keep])
    return f"{prefix}.{node.module}" if node.module else prefix


def build_inventory(extra_roots: set[str]) -> dict[str, object]:
    files = source_modules()
    module_set = set(files)
    edges = {
        name: imported_modules(path, current_module=name, modules=module_set)
        for name, path in files.items()
    }
    roots = set(OPERATIONAL_ROOTS) | {
        root if root.startswith(f"{PACKAGE_NAME}.") else f"{PACKAGE_NAME}.{root}"
        for root in extra_roots
    }
    for root in EXTERNAL_ROOTS:
        for path in root.rglob("*.py"):
            roots.update(imported_modules(path, current_module=None, modules=module_set))
    roots &= module_set

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
    versioned = {
        name for name in files if VERSIONED.search(name.rpartition(".")[2])
    }
    return {
        "roots": sorted(roots),
        "module_count": len(files),
        "top_level_module_count": len(list(PACKAGE.glob("*.py"))),
        "versioned_count": len(versioned),
        "reachable_versioned": sorted(versioned & reachable),
        "safe_to_remove": sorted(versioned - reachable),
        "incoming": {name: sorted(incoming[name]) for name in sorted(versioned)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inventory numbered boatrace_ai modules by operational reachability."
    )
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
