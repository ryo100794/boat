from __future__ import annotations

from pathlib import Path

from boatrace_ai.evaluation_queue import (
    _cgroup_memory,
    _cgroup_quota_available_mb,
    _cgroup_reclaimable_file_bytes,
)


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


def test_cgroup_v1_reclaimable_file_prefers_hierarchy_total(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "memory.stat").write_text(
        "inactive_file 1000\ntotal_inactive_file 9000\n",
        encoding="utf-8",
    )

    assert _cgroup_reclaimable_file_bytes(tmp_path) == 9000


def test_cgroup_v2_reclaimable_file(tmp_path: Path) -> None:
    (tmp_path / "memory.stat").write_text(
        "anon 5000\ninactive_file 3000\n",
        encoding="utf-8",
    )

    assert _cgroup_reclaimable_file_bytes(tmp_path) == 3000


def test_cgroup_quota_available_counts_reclaimable_cache_and_reserve() -> None:
    assert _cgroup_quota_available_mb(30_000, 22_000, 9_000) == 12_904
    assert _cgroup_quota_available_mb(30_000, 2_000, 99_000) == 25_904
    assert _cgroup_quota_available_mb(4_000, 4_000, 0) == 0


def test_unlimited_or_missing_cgroup_is_ignored(tmp_path: Path) -> None:
    (tmp_path / "memory.max").write_text("max\n", encoding="utf-8")
    (tmp_path / "memory.current").write_text("100\n", encoding="utf-8")

    assert _cgroup_memory(tmp_path) is None
