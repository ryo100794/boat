from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pytest

from boatrace_ai.runtime import v12_shadow_bundle as builder
from boatrace_ai.runtime.intraday_t300_shadow import V12RoleModelAdapter


class DummyEstimator:
    def predict(self, matrix: Any) -> np.ndarray:
        return np.zeros(len(matrix), dtype=np.float64)


def _closing_report() -> dict[str, Any]:
    return {
        "model_name": "closing_odds_t300_nonlinear_v12",
        "model_version": 12,
        "ready": True,
        "prediction_date": "2026-07-04",
        "trained_through_date": "2026-07-03",
        "checkpoint_label": "t300",
        "checkpoint_offset_seconds": 300,
        "selected_mode": "nonlinear_model",
        "challenger_adopted": True,
        "selection_reason": "strict_prior_mae_beats_current_odds_by_gate",
        "minimum_relative_mae_improvement": 0.01,
        "strict_prior_baseline_current_mae": 0.5,
        "strict_prior_challenger_mae": 0.4,
        "strict_prior_relative_mae_improvement": 0.2,
        "requested_engine": "test_engine",
        "actual_engine": "test_engine",
        "point_model": {
            "model_type": "nonlinear_t300_log_closing_to_current_ratio",
            "engine": "test_engine",
            "feature_indices": [1],
            "feature_names": ["log_horizon_seconds"],
            "training_examples": 3,
            "training_log_ratio_mae": 0.4,
            "random_state": 17,
        },
        "feature_names": ["log_horizon_seconds"],
        "forbidden_feature_tokens": ["result", "payout"],
        "teacher_provenance": {"selection_policy": "test"},
        "lower_quantile_model": {
            "ready": True,
            "model_type": "test_cluster_lower_bound",
            "quantile": 0.2,
            "effective_sample_clusters": 4,
        },
        "training_summary": {
            "training_dates": ["2026-07-01", "2026-07-02", "2026-07-03"],
            "training_days": 3,
            "training_races": 3,
            "training_examples": 3,
            "missing_t300_races": 0,
            "incomplete_t300_races": 0,
        },
        "boundary_audit": {
            "input_races": 3,
            "eligible_prior_races": 3,
            "prediction_date": "2026-07-04",
            "trained_through_date": "2026-07-03",
            "strict_training_boundary": True,
            "strict_outer_day_boundaries": True,
        },
    }


def _cache_payload() -> dict[str, Any]:
    return {
        "contract": {
            "version": 13,
            "model_sha256": "a" * 64,
            "trained_through": ["2025-07-01-01-01", "2025-07-01", "01", 1],
            "from_date": "2026-07-01",
            "through_date": "2026-07-03",
            "odds_data_signature": {"snapshot_count": 3, "max_snapshot_id": 9},
        },
        "races": [
            {
                "race_date": day,
                "race_id": f"{day.replace('-', '')}0101",
                "jcd": "01",
                "rno": 1,
            }
            for day in ("2026-07-01", "2026-07-02", "2026-07-03")
        ],
    }


def _evaluation(cache: Path) -> dict[str, Any]:
    return {
        "model": "odds_path_role_integrated_t300_nonlinear_v12",
        "calibrator_strategy": "odds_path_role_integrated_t300_nonlinear_v12",
        "evaluation_days": 1,
        "evaluated_races": 1,
        "evaluation_races": 1,
        "benchmark_evaluation_coverage": 1.0,
        "folds": [{"leakage_guard": {"pass": True}}],
        "from_date": "2026-07-01",
        "through_date": "2026-07-03",
        "source_model_sha256": "a" * 64,
        "source_model_trained_through": [
            "2025-07-01-01-01",
            "2025-07-01",
            "01",
            1,
        ],
        "odds_data_signature": {"snapshot_count": 3, "max_snapshot_id": 9},
        "scored_cache": str(cache),
        "deployment_configuration": {
            "role": "next_day_refit_not_evaluation",
            "prediction_date": "2026-07-04",
            "calibrator_strategy": "odds_path_role_integrated_t300_nonlinear_v12",
            "operational_model": {
                "model_type": "odds_path_market_offset_probability_v8",
                "trained_through_date": "2026-07-03",
                "coefficients": [0.1, 0.2],
            },
            "probability_lcb": {
                "ready": True,
                "method": "test_lcb",
                "trained_through_date": "2026-07-03",
                "factors": {"top5": 0.8},
            },
            "closing_t300_v12_model": _closing_report(),
            "closing_v11_fallback_model": {
                "model_name": "closing_odds_multihorizon_v11",
                "ready": True,
                "trained_through_date": "2026-07-03",
            },
            "closing_model_identity": {
                "requested_model": "closing_odds_t300_nonlinear_v12",
                "selected_model": "closing_odds_t300_nonlinear_v12",
                "v12_ready": True,
                "v12_adopted": True,
                "ready_for_purchase": True,
            },
            "selection_conformal": {
                "ready": False,
                "method": "selection_conformal_haircut",
                "trained_through_date": "2026-07-03",
            },
            "candidate_policy": {"name": "test"},
            "selected_policy": {"name": "no_bet", "no_bet": True},
        },
    }


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    cache = tmp_path / "scored.joblib"
    evaluation_path = tmp_path / "job-00007250.json"
    joblib.dump(_cache_payload(), cache)
    evaluation = _evaluation(cache)
    evaluation_path.write_text(
        json.dumps(evaluation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evaluation_path, cache, evaluation


def _install_refit(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_fit(races: list[dict[str, Any]], **options: Any) -> dict[str, Any]:
        calls.append({"races": copy.deepcopy(races), **options})
        model = _closing_report()
        model["point_model"]["estimator"] = DummyEstimator()
        return model

    monkeypatch.setattr(builder, "fit_closing_odds_t300_nonlinear_v12", fake_fit)
    return calls


def _contains_estimator_key(value: object) -> bool:
    if isinstance(value, dict):
        return "estimator" in value or any(
            _contains_estimator_key(child) for child in value.values()
        )
    if isinstance(value, list):
        return any(_contains_estimator_key(child) for child in value)
    return False


def test_builds_joblib_only_estimator_and_serializable_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation_path, cache, evaluation = _write_inputs(tmp_path)
    output = tmp_path / "artifacts" / "v12-20260704.joblib"
    calls = _install_refit(monkeypatch)

    manifest = builder.build_v12_shadow_bundle(
        evaluation_path,
        scored_cache=None,
        output=output,
        prediction_date="2026-07-04",
    )

    assert len(calls) == 1
    assert [race["race_date"] for race in calls[0]["races"]] == [
        "2026-07-01",
        "2026-07-02",
        "2026-07-03",
    ]
    assert calls[0]["prediction_date"] == "2026-07-04"
    assert calls[0]["engine"] == "test_engine"
    bundle = joblib.load(output)
    deployment = bundle["deployment"]
    assert isinstance(
        deployment["closing_t300_v12_model"]["point_model"]["estimator"],
        DummyEstimator,
    )
    for key in (
        "operational_model",
        "probability_lcb",
        "selection_conformal",
        "closing_v11_fallback_model",
    ):
        assert deployment[key] == evaluation["deployment_configuration"][key]

    manifest_path = output.with_suffix(".manifest.json")
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted == manifest
    assert not _contains_estimator_key(persisted)
    json.dumps(persisted, allow_nan=False)
    assert persisted["inputs"]["evaluation_json"]["sha256"] == hashlib.sha256(
        evaluation_path.read_bytes()
    ).hexdigest()
    assert persisted["inputs"]["scored_cache"]["sha256"] == hashlib.sha256(
        cache.read_bytes()
    ).hexdigest()
    assert persisted["output"]["bundle_sha256"] == hashlib.sha256(
        output.read_bytes()
    ).hexdigest()


def test_v12_adapter_loads_and_identifies_generated_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation_path, _, _ = _write_inputs(tmp_path)
    output = tmp_path / "v12.joblib"
    base = tmp_path / "base.joblib"
    joblib.dump({"feature_schema_version": 1}, base)
    _install_refit(monkeypatch)
    builder.build_v12_shadow_bundle(
        evaluation_path,
        scored_cache=None,
        output=output,
        prediction_date="2026-07-04",
    )

    adapter = V12RoleModelAdapter(
        model_key="v12-shadow-20260704",
        bundle_path=output,
        base_model_path=base,
    )

    assert adapter.identity.model_key == "v12-shadow-20260704"
    assert adapter.identity.strategy_name == "v12_role_t300"
    assert len(adapter.identity.model_hash) == 64
    closing = adapter._component("closing_t300_v12_model")
    assert closing["model_name"] == "closing_odds_t300_nonlinear_v12"
    assert hasattr(closing["point_model"]["estimator"], "predict")


def test_cache_may_include_race_excluded_by_the_closing_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation_path, cache_path, _ = _write_inputs(tmp_path)
    cache = joblib.load(cache_path)
    cache["races"].append({
        "race_date": "2026-07-03",
        "race_id": "202607290912",
        "jcd": "09",
        "rno": 12,
        "result_status": "cancelled",
        "official_closing_odds": [],
        "checkpoints": {},
    })
    joblib.dump(cache, cache_path)
    calls = _install_refit(monkeypatch)

    manifest = builder.build_v12_shadow_bundle(
        evaluation_path,
        scored_cache=cache_path,
        output=tmp_path / "job-7401.joblib",
        prediction_date="2026-07-04",
    )

    assert len(calls[0]["races"]) == 4
    assert manifest["training"]["cache_races"] == 4
    assert manifest["training"]["races"] == 3
    assert manifest["training"]["examples"] == 3


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("training_races", 2),
        ("training_dates", ["2026-07-01", "2026-07-03"]),
        ("training_examples", 2),
        ("missing_t300_races", 1),
    ],
)
def test_refit_training_summary_mismatch_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    evaluation_path, cache_path, _ = _write_inputs(tmp_path)

    def mismatched_fit(
        races: list[dict[str, Any]], **options: Any
    ) -> dict[str, Any]:
        model = _closing_report()
        model["training_summary"][field] = value
        model["point_model"]["estimator"] = DummyEstimator()
        return model

    monkeypatch.setattr(
        builder, "fit_closing_odds_t300_nonlinear_v12", mismatched_fit
    )
    with pytest.raises(ValueError, match="training summary does not match"):
        builder.build_v12_shadow_bundle(
            evaluation_path,
            scored_cache=cache_path,
            output=tmp_path / "mismatch.joblib",
            prediction_date="2026-07-04",
        )


def test_refit_estimator_excluded_canonical_identity_mismatch_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation_path, cache_path, _ = _write_inputs(tmp_path)

    def mismatched_fit(
        races: list[dict[str, Any]], **options: Any
    ) -> dict[str, Any]:
        model = _closing_report()
        model["selection_reason"] = "different_refit_identity"
        model["point_model"]["estimator"] = DummyEstimator()
        return model

    monkeypatch.setattr(
        builder, "fit_closing_odds_t300_nonlinear_v12", mismatched_fit
    )
    with pytest.raises(ValueError, match="model identity does not match"):
        builder.build_v12_shadow_bundle(
            evaluation_path,
            scored_cache=cache_path,
            output=tmp_path / "identity-mismatch.joblib",
            prediction_date="2026-07-04",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda evaluation, cache: cache["contract"].update(
                {"model_sha256": "b" * 64}
            ),
            "SHA256 mismatch",
        ),
        (
            lambda evaluation, cache: cache["contract"].update(
                {"through_date": "2026-07-02"}
            ),
            "through_date mismatch",
        ),
        (
            lambda evaluation, cache: cache["contract"].update(
                {"odds_data_signature": {"snapshot_count": 2}}
            ),
            "odds data signature mismatch",
        ),
    ],
)
def test_source_hash_and_date_mismatches_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Any,
    message: str,
) -> None:
    evaluation_path, cache_path, evaluation = _write_inputs(tmp_path)
    cache = joblib.load(cache_path)
    mutation(evaluation, cache)
    joblib.dump(cache, cache_path)
    _install_refit(monkeypatch)

    with pytest.raises(ValueError, match=message):
        builder.build_v12_shadow_bundle(
            evaluation_path,
            scored_cache=cache_path,
            output=tmp_path / "out.joblib",
            prediction_date="2026-07-04",
        )


def test_source_trained_through_accepts_json_list_joblib_tuple_equivalence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation_path, cache_path, _ = _write_inputs(tmp_path)
    cache = joblib.load(cache_path)
    cache["contract"]["trained_through"] = tuple(
        cache["contract"]["trained_through"]
    )
    joblib.dump(cache, cache_path)
    _install_refit(monkeypatch)

    builder.build_v12_shadow_bundle(
        evaluation_path,
        scored_cache=cache_path,
        output=tmp_path / "tuple-identity.joblib",
        prediction_date="2026-07-04",
    )


def test_future_dates_and_non_next_day_prediction_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation_path, cache_path, evaluation = _write_inputs(tmp_path)
    _install_refit(monkeypatch)

    evaluation["deployment_configuration"]["prediction_date"] = "2026-07-05"
    evaluation["deployment_configuration"]["closing_t300_v12_model"][
        "prediction_date"
    ] = "2026-07-05"
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one day"):
        builder.build_v12_shadow_bundle(
            evaluation_path,
            scored_cache=cache_path,
            output=tmp_path / "late.joblib",
            prediction_date="2026-07-05",
        )

    evaluation = _evaluation(cache_path)
    evaluation["deployment_configuration"]["selection_conformal"][
        "trained_through_date"
    ] = "2026-07-04"
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    with pytest.raises(ValueError, match="future-date training boundary"):
        builder.build_v12_shadow_bundle(
            evaluation_path,
            scored_cache=cache_path,
            output=tmp_path / "future.joblib",
            prediction_date="2026-07-04",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("folds", [], "incomplete"),
        ("closing_ready", False, "not ready"),
        ("closing_adopted", False, "not adopted"),
    ],
)
def test_incomplete_or_unadopted_evaluation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
    message: str,
) -> None:
    evaluation_path, cache_path, evaluation = _write_inputs(tmp_path)
    if field == "folds":
        evaluation["folds"] = value
    elif field == "closing_ready":
        evaluation["deployment_configuration"]["closing_t300_v12_model"][
            "ready"
        ] = value
    else:
        evaluation["deployment_configuration"]["closing_t300_v12_model"][
            "challenger_adopted"
        ] = value
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    _install_refit(monkeypatch)

    with pytest.raises(ValueError, match=message):
        builder.build_v12_shadow_bundle(
            evaluation_path,
            scored_cache=cache_path,
            output=tmp_path / "out.joblib",
            prediction_date="2026-07-04",
        )


def test_outputs_are_replaced_from_same_directory_temporaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation_path, cache_path, _ = _write_inputs(tmp_path)
    output = tmp_path / "nested" / "v12.joblib"
    _install_refit(monkeypatch)
    replacements: list[tuple[Path, Path]] = []
    real_replace = builder.os.replace

    def recording_replace(source: Path, destination: Path) -> None:
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(builder.os, "replace", recording_replace)
    builder.build_v12_shadow_bundle(
        evaluation_path,
        scored_cache=cache_path,
        output=output,
        prediction_date="2026-07-04",
    )

    assert [destination for _, destination in replacements] == [
        output.resolve(),
        output.resolve().with_suffix(".manifest.json"),
    ]
    assert all(
        source.parent == destination.parent for source, destination in replacements
    )
    assert all(not source.exists() for source, _ in replacements)


def test_failure_before_replace_preserves_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation_path, cache_path, _ = _write_inputs(tmp_path)
    output = tmp_path / "v12.joblib"
    manifest = output.with_suffix(".manifest.json")
    output.write_bytes(b"old-bundle")
    manifest.write_text("old-manifest\n", encoding="utf-8")
    _install_refit(monkeypatch)

    def fail_dump(payload: Any, path: Path) -> None:
        Path(path).write_bytes(b"partial")
        raise OSError("simulated dump failure")

    monkeypatch.setattr(builder.joblib, "dump", fail_dump)
    with pytest.raises(OSError, match="simulated dump failure"):
        builder.build_v12_shadow_bundle(
            evaluation_path,
            scored_cache=cache_path,
            output=output,
            prediction_date="2026-07-04",
        )

    assert output.read_bytes() == b"old-bundle"
    assert manifest.read_text(encoding="utf-8") == "old-manifest\n"
    assert not list(tmp_path.glob(".*.tmp"))
