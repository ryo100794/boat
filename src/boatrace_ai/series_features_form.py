from __future__ import annotations

from typing import Any

from . import series_features_relative as base
from .series_features_base import base_pastlog_features as base_v1


LOW_COVERAGE_ROOTS = (
    "avg_st",
    "f_count",
    "l_count",
    "national_3_rate",
    "local_3_rate",
    "motor_3_rate",
    "boat_3_rate",
)
BEST_COUNT_ROOTS = (
    "class_rank",
    "national_win_rate",
    "local_win_rate",
    "motor_2_rate",
    "boat_2_rate",
)
DERIVED_SUFFIXES = (
    "_rank",
    "_vs_mean",
    "_z",
    "_best_gap",
    "_scaled",
)


def base_pastlog_features(row: Any, relatives: dict[str, Any]) -> dict[str, Any]:
    item = base_v1(row, relatives)
    for key in list(item.keys()):
        if _drop_key(key):
            item.pop(key, None)
    item["best_count"] = sum(
        int(relatives.get(f"{root}_rank", 0) == 1)
        for root in BEST_COUNT_ROOTS
    )
    item["feature_pruning"] = "drop_low_coverage_card_fields_v2"
    return item


def _drop_key(key: str) -> bool:
    if key in LOW_COVERAGE_ROOTS:
        return True
    if any(key == f"has_{root}" for root in LOW_COVERAGE_ROOTS):
        return True
    return any(key == f"{root}{suffix}" for root in LOW_COVERAGE_ROOTS for suffix in DERIVED_SUFFIXES)


base.base_pastlog_features = base_pastlog_features

load_training_examples = base.load_training_examples
prediction_features = base.prediction_features
history_groups_prior_dates = base.history_groups_prior_dates
