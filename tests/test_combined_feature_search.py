from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from boatrace_ai.listwise.combined_feature_search import (
    COMBINED_FEATURE_VARIANTS,
    build_parser,
)
from boatrace_ai.listwise.feature_search import (
    _candidate_key,
    _checkpoint_payload,
    _checkpoint_signature,
    _load_checkpoint,
    _ordered_rows,
    feature_variants,
)


EXPECTED_COMBINED_VARIANTS = (
    (
        "drop_base_pastlog_research_correlates",
        ("base_pastlog", "research_correlates"),
    ),
    (
        "drop_base_pastlog_research_correlates_series_cached",
        ("base_pastlog", "research_correlates", "series_cached"),
    ),
    (
        "drop_base_pastlog_research_correlates_series_cached_series_relative",
        (
            "base_pastlog",
            "research_correlates",
            "series_cached",
            "series_relative",
        ),
    ),
)


def _signature(*, variants=None):
    return _checkpoint_signature(
        args=SimpleNamespace(
            as_of_date="2026-07-23",
            n_features=64,
            batch_races=2,
            epochs=1,
            learning_rate=0.02,
        ),
        race_keys=[
            ("race-a", "2026-07-22", "01", 1),
            ("race-b", "2026-07-23", "02", 2),
        ],
        train_end=1,
        selection_end=2,
        targets=("winner", "top3_pl"),
        alphas=(0.0001, 0.001),
        variants=variants,
    )


def _row(
    name: str,
    dropped: tuple[str, ...],
    target: str,
    alpha: float,
) -> dict:
    return {
        "feature_variant": name,
        "drop_feature_groups": list(dropped),
        "target": target,
        "alpha": alpha,
        "entry_log_loss": 0.3,
        "ranking_log_loss": 1.3,
        "winner_top1_accuracy": 0.5,
        "trifecta_top5_hit_rate": 0.25,
        "training_history": [],
    }


def test_combined_variants_are_fixed_and_default_variants_are_unchanged() -> None:
    defaults_before = feature_variants()
    parser = build_parser()
    args = parser.parse_args([])

    assert COMBINED_FEATURE_VARIANTS == EXPECTED_COMBINED_VARIANTS
    assert feature_variants() == defaults_before
    assert all(
        Path(name).name == name and ".." not in name
        for name, _drops in COMBINED_FEATURE_VARIANTS
    )
    assert "--variants" not in parser.format_help()
    assert args.variant_workers == 1
    assert args.candidate_workers == 1
    assert args.output.endswith("listwise_combined_feature_search_v1.json")
    assert args.cache_dir.endswith("listwise_combined_search_cache")


def test_combined_signature_is_separate_from_default_signature() -> None:
    default = _signature()
    combined = _signature(variants=COMBINED_FEATURE_VARIANTS)

    assert combined != default
    assert combined["feature_variants"] == [
        [name, list(dropped)] for name, dropped in COMBINED_FEATURE_VARIANTS
    ]
    assert default["feature_variants"] == [
        [name, list(dropped)] for name, dropped in feature_variants()
    ]


def test_combined_rows_use_canonical_variant_target_alpha_order() -> None:
    targets = ("winner", "top3_pl")
    alphas = (0.0001, 0.001)
    rows = [
        _row(name, dropped, target, alpha)
        for name, dropped in COMBINED_FEATURE_VARIANTS
        for target in targets
        for alpha in alphas
    ]
    completed = {
        _candidate_key(row["feature_variant"], row["target"], row["alpha"]): row
        for row in reversed(rows)
    }

    assert _ordered_rows(
        completed,
        targets=targets,
        alphas=alphas,
        variants=COMBINED_FEATURE_VARIANTS,
    ) == rows


def test_combined_checkpoint_resumes_without_accepting_default_signature(
    tmp_path: Path,
) -> None:
    rows = [
        _row(name, dropped, "winner", 0.0001)
        for name, dropped in COMBINED_FEATURE_VARIANTS[:2]
    ]
    completed = {
        _candidate_key(row["feature_variant"], row["target"], row["alpha"]): row
        for row in reversed(rows)
    }
    combined_signature = _signature(variants=COMBINED_FEATURE_VARIANTS)
    checkpoint = tmp_path / "combined-checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            _checkpoint_payload(
                combined_signature,
                completed,
                targets=("winner", "top3_pl"),
                alphas=(0.0001, 0.001),
                variants=COMBINED_FEATURE_VARIANTS,
            )
        ),
        encoding="utf-8",
    )

    resumed = _load_checkpoint(checkpoint, combined_signature)

    assert list(resumed.values()) == rows
    assert _load_checkpoint(checkpoint, _signature()) == {}
