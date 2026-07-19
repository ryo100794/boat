from __future__ import annotations

from boatrace_ai.feature_tuning import FEATURE_GROUPS
from boatrace_ai.listwise.feature_search import day_boundary, feature_variants


def test_feature_search_covers_full_and_each_single_group_ablation() -> None:
    variants = feature_variants()
    assert variants[0] == ("full", ())
    assert {drops[0] for _name, drops in variants[1:]} == set(FEATURE_GROUPS)
    assert all(len(drops) == 1 for _name, drops in variants[1:])


def test_day_boundary_never_splits_a_race_day() -> None:
    keys = [
        (f"r{index}", f"2026-01-{index // 3 + 1:02d}", "01", index + 1)
        for index in range(12)
    ]
    boundary = day_boundary(keys, 4)
    assert boundary == 6
    assert keys[boundary - 1][1] != keys[boundary][1]
