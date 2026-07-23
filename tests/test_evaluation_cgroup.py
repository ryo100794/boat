from __future__ import annotations

from pathlib import Path

from boatrace_ai.evaluation_queue import _cgroup_memory


def test_cgroup_v1_memory_limit_and_usage(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "memory.limit_in_bytes").write_text("32000000000\n", encoding="utf-8")
    (memory / "memory.usage_in_bytes").write_text("18000000000\n", encoding="utf-8")

    assert _cgroup_memory(tmp_path) == (32_000_000_000, 18_000_000_000)


def test_cgroup_v2_memory_limit_and_usage(tmp_path: Path) -> None:
    (tmp_path / "memory.max").write_text("34359738368\n", encoding="utf-8")
    (tmp_path / "memory.current").write_text("17179869184\n", encoding="utf-8")

    assert _cgroup_memory(tmp_path) == (34_359_738_368, 17_179_869_184)


def test_unlimited_or_missing_cgroup_is_ignored(tmp_path: Path) -> None:
    (tmp_path / "memory.max").write_text("max\n", encoding="utf-8")
    (tmp_path / "memory.current").write_text("100\n", encoding="utf-8")

    assert _cgroup_memory(tmp_path) is None
