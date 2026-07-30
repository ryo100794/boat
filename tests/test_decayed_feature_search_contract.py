from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from boatrace_ai.hashed_feature_dataset import FEATURE_SCHEMA_VERSION
from boatrace_ai.listwise import feature_search


RACE_KEYS = [
    ("20260101-01-01", "2026-01-01", "01", 1),
    ("20260102-01-01", "2026-01-02", "01", 1),
    ("20260103-01-01", "2026-01-03", "01", 1),
]
VARIANTS = (("full", ()),)


def _args(*, include_decayed_history: bool) -> Namespace:
    return Namespace(
        include_decayed_history=include_decayed_history,
        as_of_date="2026-01-03",
        n_features=128,
        batch_races=10,
        epochs=1,
        learning_rate=0.02,
        loss_blend=None,
    )


def _snapshot() -> dict[str, object]:
    value: dict[str, object] = {
        "snapshot_version": feature_search.SOURCE_DATA_SNAPSHOT_VERSION,
        "race_count": len(RACE_KEYS),
        "race_universe_sha256": feature_search.race_ids_sha256(RACE_KEYS),
        "source_watermark_sha256": "1" * 64,
        "trifecta_payouts_sha256": "2" * 64,
        "selected_cache_manifest_sha256": "3" * 64,
    }
    value["snapshot_sha256"] = feature_search._canonical_sha256(value)
    return value


def test_decayed_cli_is_opt_in() -> None:
    parser = feature_search.build_parser()

    assert parser.parse_args([]).include_decayed_history is False
    assert parser.parse_args(["--include-decayed-history"]).include_decayed_history is True


def test_decayed_cache_family_is_separate_without_renaming_v6(tmp_path: Path) -> None:
    baseline = feature_search.variant_cache_prefix(
        tmp_path, n_features=8192, name="full"
    )
    decayed = feature_search.variant_cache_prefix(
        tmp_path,
        n_features=8192,
        name="full",
        include_decayed_history=True,
    )

    assert baseline.name == "listwise_search_8192_full"
    assert decayed.name == "listwise_search_8192_full_decayed_history"
    assert baseline != decayed


def test_baseline_checkpoint_signature_remains_backward_compatible() -> None:
    baseline = feature_search._checkpoint_signature(
        args=_args(include_decayed_history=False),
        race_keys=RACE_KEYS,
        train_end=1,
        selection_end=2,
        targets=("winner",),
        alphas=(0.0001,),
        data_snapshot=_snapshot(),
        variants=VARIANTS,
    )
    decayed = feature_search._checkpoint_signature(
        args=_args(include_decayed_history=True),
        race_keys=RACE_KEYS,
        train_end=1,
        selection_end=2,
        targets=("winner",),
        alphas=(0.0001,),
        data_snapshot=_snapshot(),
        variants=VARIANTS,
    )

    assert baseline["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert "include_decayed_history" not in baseline
    assert decayed["include_decayed_history"] is True
    assert decayed["feature_schema_version"] == (
        feature_search.DECAYED_HISTORY_FEATURE_SCHEMA_VERSION
    )


def test_variant_builder_wires_decayed_generator_and_schema(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}
    fake_dataset = object()

    monkeypatch.setattr(feature_search, "load_hashed_dataset", lambda *a, **k: None)

    def fake_build(**kwargs):
        captured.update(kwargs)
        return fake_dataset, "built"

    monkeypatch.setattr(feature_search, "load_or_build_hashed_dataset", fake_build)
    monkeypatch.setattr(
        feature_search,
        "iter_race_feature_rows",
        lambda *args, **kwargs: captured.setdefault("generator_kwargs", kwargs),
    )

    dataset, source, prefix = feature_search.load_variant_dataset_with_cache(
        object(),
        race_keys=RACE_KEYS,
        cache_dir=tmp_path,
        name="full",
        dropped=(),
        n_features=128,
        batch_races=10,
        write_cache=False,
        include_decayed_history=True,
        feature_schema_version=(
            feature_search.DECAYED_HISTORY_FEATURE_SCHEMA_VERSION
        ),
    )

    assert dataset is fake_dataset
    assert source == "built"
    assert prefix is None
    assert captured["feature_schema_version"] == (
        feature_search.DECAYED_HISTORY_FEATURE_SCHEMA_VERSION
    )
    generator = captured["race_rows"]
    generator()
    assert captured["generator_kwargs"]["include_decayed_history"] is True
