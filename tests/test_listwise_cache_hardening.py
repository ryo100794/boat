from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from scipy import sparse
from sklearn.preprocessing import StandardScaler

from boatrace_ai.hashed_feature_dataset import (
    CACHE_VERSION,
    FEATURE_SCHEMA_VERSION,
    HashedRaceDataset,
    race_ids_sha256,
)
from boatrace_ai.listwise import feature_search
from boatrace_ai.listwise.feature_search import (
    _checkpoint_signature,
    _load_checkpoint,
    cleanup_selected_cache_family,
    load_variant_dataset_with_cache,
    selected_cache_candidates,
    variant_cache_prefix,
)
from boatrace_ai.listwise.model import ListwiseLinearModel
from boatrace_ai.listwise.newton_refine import (
    build_parser,
    dump_joblib_atomic,
    load_resume_model_artifact,
    validate_search_race_universe,
)
from boatrace_ai.standard_evaluation import race_set_sha256


def _race_keys():
    return [
        ("r1", "2026-01-01", "01", 1),
        ("r2", "2026-01-02", "01", 1),
    ]


def test_selected_cache_cleanup_is_limited_to_exact_feature_family(
    tmp_path: Path,
) -> None:
    selected = variant_cache_prefix(tmp_path, n_features=64, name="full")
    selected_manifest = Path(f"{selected}.manifest.json")
    selected_manifest.write_text("{}", encoding="utf-8")
    selected_temp = tmp_path / f".{selected.name}.abc.matrix.npz.tmp"
    selected_temp.write_text("partial", encoding="utf-8")
    other_width = tmp_path / "listwise_search_128_full.manifest.json"
    other_width.write_text("keep", encoding="utf-8")
    unrelated = tmp_path / "listwise_search_64_unrelated.manifest.json"
    unrelated.write_text("keep", encoding="utf-8")

    cleanup_selected_cache_family(tmp_path, n_features=64)

    assert not selected_manifest.exists()
    assert not selected_temp.exists()
    assert other_width.exists()
    assert unrelated.exists()


def test_selected_cache_candidates_only_returns_known_variants(tmp_path: Path) -> None:
    selected = variant_cache_prefix(tmp_path, n_features=64, name="full")
    Path(f"{selected}.manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "listwise_search_64_unknown.manifest.json").write_text(
        "{}", encoding="utf-8"
    )
    assert selected_cache_candidates(tmp_path, n_features=64) == [selected]


def test_variant_loader_uses_recorded_fallback_without_building(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fallback = tmp_path / "persistent" / "listwise_search_64_full"
    expected = HashedRaceDataset(
        matrix=sparse.csr_matrix((12, 64)),
        race_keys=_race_keys(),
        ranks=np.tile(np.arange(1, 7, dtype=np.int8), (2, 1)),
        n_features=64,
        drop_feature_groups=(),
    )
    checked: list[Path] = []

    def fake_load(prefix, **_kwargs):
        checked.append(prefix)
        return expected if prefix == fallback else None

    monkeypatch.setattr(feature_search, "load_hashed_dataset", fake_load)
    dataset, source, actual = load_variant_dataset_with_cache(
        None,
        race_keys=_race_keys(),
        cache_dir=tmp_path / "missing-tmp",
        name="full",
        dropped=(),
        n_features=64,
        batch_races=2,
        write_cache=False,
        fallback_cache_prefixes=(fallback,),
    )

    assert dataset is expected
    assert source == "disk"
    assert actual == fallback
    assert checked[-1] == fallback


def test_checkpoint_requires_exact_signature(tmp_path: Path) -> None:
    args = argparse.Namespace(
        n_features=64,
        batch_races=2,
        epochs=1,
        learning_rate=0.02,
    )
    signature = _checkpoint_signature(
        args=args,
        race_keys=_race_keys(),
        train_end=1,
        selection_end=2,
        targets=("winner",),
        alphas=(0.1,),
    )
    row = {
        "feature_variant": "full",
        "drop_feature_groups": [],
        "target": "winner",
        "alpha": 0.1,
        "entry_log_loss": 0.5,
        "ranking_log_loss": 1.2,
        "winner_top1_accuracy": 0.6,
        "trifecta_top5_hit_rate": 0.3,
        "training_history": [],
    }
    path = tmp_path / "checkpoint.json"
    path.write_text(
        json.dumps({"signature": signature, "search_results": [row]}),
        encoding="utf-8",
    )
    assert len(_load_checkpoint(path, signature)) == 1

    changed = {**signature, "race_universe_sha256": "0" * 64}
    assert _load_checkpoint(path, changed) == {}


def test_newton_rejects_stale_or_legacy_search_race_universe() -> None:
    keys = [
        (f"r{index}", f"2026-01-0{index}", "01", 1)
        for index in range(1, 5)
    ]
    valid = {
        "races": len(keys),
        "race_universe_sha256": race_ids_sha256(keys),
        "hashed_cache_version": CACHE_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "train_races": 2,
        "selection_races": 1,
        "holdout_races": 1,
        "evaluation_race_set_sha256": race_set_sha256(["r4"]),
    }
    validate_search_race_universe(valid, keys)

    with pytest.raises(ValueError, match="race count"):
        validate_search_race_universe({**valid, "races": 1}, keys)
    with pytest.raises(ValueError, match="race universe hash"):
        validate_search_race_universe({**valid, "race_universe_sha256": None}, keys)
    with pytest.raises(ValueError, match="cache/schema version"):
        validate_search_race_universe({**valid, "hashed_cache_version": 1}, keys)
    with pytest.raises(ValueError, match="cache/schema version"):
        validate_search_race_universe(
            {**valid, "hashed_cache_version": 1}, keys, allow_legacy=True
        )
    with pytest.raises(ValueError, match="cache/schema version"):
        validate_search_race_universe(
            {**valid, "feature_schema_version": 1}, keys, allow_legacy=True
        )

    legacy = {
        key: value
        for key, value in valid.items()
        if key not in {
            "race_universe_sha256",
            "hashed_cache_version",
            "feature_schema_version",
        }
    }
    validate_search_race_universe(legacy, keys, allow_legacy=True)
    with pytest.raises(ValueError, match="race universe hash"):
        validate_search_race_universe(legacy, keys)
    with pytest.raises(ValueError, match="evaluation holdout hash"):
        validate_search_race_universe(
            {**legacy, "evaluation_race_set_sha256": "0" * 64},
            keys,
            allow_legacy=True,
        )


def _resume_artifact_payload(*, include_race_hash: bool = True) -> dict:
    n_features = 4
    scaler = StandardScaler(with_mean=False).fit(np.eye(n_features))
    model = ListwiseLinearModel(
        weights=np.asarray([0.1, -0.2, 0.3, -0.4]),
        scaler=scaler,
        target="top3_pl",
        alpha=1e-4,
        learning_rate=0.02,
        epochs=2,
    )
    payload = {
        "model": model,
        "feature_variant": "drop_research_correlates",
        "drop_feature_groups": ("research_correlates",),
        "n_features": n_features,
        "trained_races": 3,
        "trained_through": ("r3", "2026-01-03", "01", 1),
        "target": "top3_pl",
        "alpha": 1e-4,
    }
    if include_race_hash:
        payload["race_universe_sha256"] = "a" * 64
    return payload


def _resume_expectations() -> dict:
    return {
        "feature_variant": "drop_research_correlates",
        "drop_feature_groups": ("research_correlates",),
        "n_features": 4,
        "trained_races": 3,
        "trained_through": ("r3", "2026-01-03", "01", 1),
        "target": "top3_pl",
        "alpha": 1e-4,
        "race_universe_sha256": "a" * 64,
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("feature_variant", "full", "feature_variant mismatch"),
        ("drop_feature_groups", (), "drop_feature_groups mismatch"),
        ("n_features", 8, "n_features mismatch"),
        ("trained_races", 2, "trained_races mismatch"),
        ("trained_through", ("r2", "2026-01-02", "01", 1), "trained_through mismatch"),
        ("target", "winner", "target mismatch"),
        ("alpha", 0.1, "alpha mismatch"),
        ("race_universe_sha256", "b" * 64, "race_universe_sha256 mismatch"),
    ),
)
def test_resume_artifact_rejects_metadata_mismatch(
    tmp_path: Path, field: str, value, message: str
) -> None:
    path = tmp_path / "resume.joblib"
    payload = _resume_artifact_payload()
    payload[field] = value
    joblib.dump(payload, path)

    with pytest.raises(ValueError, match=message):
        load_resume_model_artifact(path, **_resume_expectations())


@pytest.mark.parametrize("broken_part", ("weights", "scaler"))
def test_resume_artifact_rejects_model_dimension_mismatch(
    tmp_path: Path, broken_part: str
) -> None:
    path = tmp_path / "resume.joblib"
    payload = _resume_artifact_payload()
    if broken_part == "weights":
        payload["model"].weights = np.zeros(3)
        message = "weights length"
    else:
        payload["model"].scaler = StandardScaler(with_mean=False).fit(np.eye(3))
        message = "scaler n_features"
    joblib.dump(payload, path)

    with pytest.raises(ValueError, match=message):
        load_resume_model_artifact(path, **_resume_expectations())


@pytest.mark.parametrize("broken_part", ("nonfinite_weights", "centered_scaler"))
def test_resume_artifact_rejects_numerically_invalid_model(
    tmp_path: Path, broken_part: str
) -> None:
    path = tmp_path / "resume-invalid.joblib"
    payload = _resume_artifact_payload()
    if broken_part == "nonfinite_weights":
        payload["model"].weights[0] = np.nan
        message = "non-finite"
    else:
        payload["model"].scaler.with_mean = True
        message = "scaler is invalid"
    joblib.dump(payload, path)

    with pytest.raises(ValueError, match=message):
        load_resume_model_artifact(path, **_resume_expectations())



def test_valid_legacy_resume_artifact_requires_explicit_permission(tmp_path: Path) -> None:
    path = tmp_path / "legacy.joblib"
    joblib.dump(_resume_artifact_payload(include_race_hash=False), path)

    with pytest.raises(ValueError, match="allow-legacy-model-artifact"):
        load_resume_model_artifact(path, **_resume_expectations())
    model, artifact_sha256 = load_resume_model_artifact(
        path,
        **_resume_expectations(),
        allow_legacy=True,
    )

    assert len(model.weights) == 4
    assert len(artifact_sha256) == 64


def test_newton_resume_cli_defaults_and_flags() -> None:
    args = build_parser().parse_args([
        "--resume-model",
        "checkpoint.joblib",
        "--allow-legacy-model-artifact",
        "--promote-legacy-cache",
        "--allow-legacy-search-result",
    ])

    assert args.max_newton_iterations == 10
    assert args.max_cg_iterations == 50
    assert args.promote_legacy_cache is True
    assert args.resume_model == "checkpoint.joblib"
    assert args.allow_legacy_model_artifact is True
    assert args.allow_legacy_search_result is True


def test_joblib_artifact_is_atomically_replaced(tmp_path: Path, monkeypatch) -> None:
    import boatrace_ai.listwise.newton_refine as module

    path = tmp_path / "model.joblib"
    replacements: list[tuple[Path, Path]] = []
    real_replace = module.os.replace

    def recording_replace(source, destination):
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", recording_replace)
    dump_joblib_atomic(path, {"weights": np.asarray([1.0, 2.0])})

    assert joblib.load(path)["weights"].tolist() == [1.0, 2.0]
    assert replacements[0][1] == path
    assert replacements[0][0].parent == path.parent
    assert not list(tmp_path.glob(".model.joblib.*.tmp"))
