from __future__ import annotations

from datetime import date
import os
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from boatrace_ai import calibrated_shadow_model as calibrated
from boatrace_ai import recency_mlp_evaluation as recency
from boatrace_ai.hashed_feature_dataset import HashedRaceDataset
from boatrace_ai.standard_evaluation import race_set_sha256


def make_dataset(dates: list[str]) -> HashedRaceDataset:
    rows: list[list[float]] = []
    ranks: list[list[int]] = []
    race_keys: list[tuple[str, str, str, int]] = []
    for index, race_date in enumerate(dates):
        race_keys.append((f"r{index}", race_date, "01", index + 1))
        ranks.append([1, 2, 3, 4, 5, 6])
        for lane in range(1, 7):
            rows.append([float(lane == 1), float(lane), float(index)])
    return HashedRaceDataset(
        matrix=sparse.csr_matrix(np.asarray(rows, dtype=np.float64)),
        race_keys=race_keys,
        ranks=np.asarray(ranks, dtype=np.int8),
        n_features=3,
        drop_feature_groups=(),
    )


def test_recency_weights_use_training_tail_date_and_repeat_for_six_lanes() -> None:
    dataset = make_dataset(["2026-01-01", "2026-01-02", "2026-01-03"])

    weights = calibrated.recency_sample_weights(
        dataset,
        train_race_count=3,
        recency_half_life_days=1,
    )

    assert weights is not None
    np.testing.assert_allclose(
        weights,
        np.repeat(np.asarray([0.25, 0.5, 1.0]), 6),
    )
    assert calibrated.recency_sample_weights(
        dataset,
        train_race_count=3,
        recency_half_life_days=None,
    ) is None
    with pytest.raises(ValueError, match="positive and finite"):
        calibrated.recency_sample_weights(
            dataset,
            train_race_count=3,
            recency_half_life_days=0,
        )


def test_weighted_training_passes_identical_weights_to_scaler_and_classifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = make_dataset(["2026-01-01", "2026-01-02", "2026-01-03"])

    class ScalerSpy:
        def __init__(self, **_kwargs: object) -> None:
            self.weights: list[np.ndarray | None] = []

        def partial_fit(self, _matrix: object, sample_weight=None):
            self.weights.append(
                None if sample_weight is None else np.asarray(sample_weight).copy()
            )
            return self

        def transform(self, matrix: object):
            return matrix

    class ClassifierSpy:
        def __init__(self) -> None:
            self.weights: list[np.ndarray | None] = []

        def partial_fit(self, _matrix: object, _labels: object, **kwargs: object):
            sample_weight = kwargs.get("sample_weight")
            self.weights.append(
                None if sample_weight is None else np.asarray(sample_weight).copy()
            )
            return self

    scaler = ScalerSpy()
    classifier = ClassifierSpy()
    monkeypatch.setattr(calibrated, "StandardScaler", lambda **_kwargs: scaler)
    monkeypatch.setattr(
        calibrated,
        "make_classifier",
        lambda *_args, **_kwargs: classifier,
    )

    bundle = calibrated.train_bundle_from_dataset(
        dataset,
        train_race_count=3,
        model_kind="mlp",
        batch_size=6,
        epochs=1,
        recency_half_life_days=1,
    )

    expected = [
        np.full(6, 0.25),
        np.full(6, 0.5),
        np.full(6, 1.0),
    ]
    assert len(scaler.weights) == len(classifier.weights) == 3
    for actual, wanted in zip(scaler.weights, expected):
        np.testing.assert_allclose(actual, wanted)
    for actual, wanted in zip(classifier.weights, expected):
        np.testing.assert_allclose(actual, wanted)
    assert bundle["recency_half_life_days"] == 1.0


def test_none_half_life_is_numerically_compatible_with_omitted_argument() -> None:
    dataset = make_dataset(
        [f"2026-01-{day:02d}" for day in range(1, 9)]
    )

    omitted = calibrated.train_bundle_from_dataset(
        dataset,
        train_race_count=6,
        model_kind="linear",
        batch_size=12,
        epochs=1,
    )
    explicit = calibrated.train_bundle_from_dataset(
        dataset,
        train_race_count=6,
        model_kind="linear",
        batch_size=12,
        epochs=1,
        recency_half_life_days=None,
    )

    np.testing.assert_array_equal(omitted["scaler"].scale_, explicit["scaler"].scale_)
    np.testing.assert_array_equal(
        omitted["classifier"].coef_, explicit["classifier"].coef_
    )
    np.testing.assert_array_equal(
        omitted["classifier"].intercept_, explicit["classifier"].intercept_
    )
    assert omitted["recency_half_life_days"] is None
    assert explicit["recency_half_life_days"] is None


def test_inner_calibration_boundary_uses_trailing_calendar_days() -> None:
    dataset = make_dataset(
        [
            "2026-01-01",
            "2026-01-09",
            "2026-01-10",
            "2026-01-11",
            "2026-01-12",
            "2026-01-13",
        ]
    )

    boundary, start, end = recency.inner_calibration_boundary(
        dataset.race_keys,
        outer_train_end=5,
        calibration_days=3,
    )

    assert boundary == 2
    assert start == "2026-01-10"
    assert end == "2026-01-12"


def test_selection_scores_only_inner_calibration_and_uses_fixed_tie_break(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = make_dataset(
        [
            "2026-01-01",
            "2026-01-02",
            "2026-01-03",
            "2026-01-10",
            "2026-01-11",
            "2026-01-12",
            "2026-01-13",
        ]
    )
    trained: list[tuple[int, float | None]] = []
    scored: list[tuple[int, int]] = []

    def fake_train(_dataset: HashedRaceDataset, **kwargs: object) -> dict[str, object]:
        half_life = kwargs["recency_half_life_days"]
        trained.append((int(kwargs["train_race_count"]), half_life))
        return {"half_life": half_life}

    def fake_score(
        _dataset: HashedRaceDataset,
        *,
        bundle: dict[str, object],
        race_start: int,
        race_end: int,
        batch_size: int,
    ) -> tuple[dict[str, float | int], dict[str, list[dict[str, object]]]]:
        del batch_size
        scored.append((race_start, race_end))
        loss = 0.4 if bundle["half_life"] in {None, 730.0} else 0.5
        return (
            {
                "entry_log_loss": loss,
                "entry_brier": 0.1,
                "winner_top1_accuracy": 0.2,
                "trifecta_top1_hit_rate": 0.01,
                "trifecta_top5_hit_rate": 0.05,
                "evaluated_races": race_end - race_start,
            },
            {
                "calibration": [
                    {"half_life": bundle["half_life"]}
                ]
            },
        )

    monkeypatch.setattr(recency, "train_bundle_from_dataset", fake_train)
    monkeypatch.setattr(recency, "score_range", fake_score)

    selected_predictions: dict[str, list[dict[str, object]]] = {}
    selected, candidates, split = recency.select_recency_half_life(
        dataset,
        outer_train_end=5,
        half_lives=(730.0, None, 365.0),
        calibration_days=2,
        prediction_output=selected_predictions,
    )

    assert selected is None
    assert trained == [(3, 730.0), (3, None), (3, 365.0)]
    assert scored == [(3, 5), (3, 5), (3, 5)]
    assert split["calibration_end"] == "2026-01-11"
    assert all(row["calibration_races"] == 2 for row in candidates)
    assert selected_predictions == {"calibration": [{"half_life": None}]}


def test_trifecta_probability_matrix_is_ordered_and_normalized() -> None:
    race_keys = [("r1", "2026-01-01", "01", 1)]
    predictions = {
        "r1": [
            {"lane": lane, "probability": float(7 - lane)}
            for lane in range(1, 7)
        ]
    }

    matrix = recency.trifecta_probability_matrix(predictions, race_keys)

    assert matrix.shape == (1, 120)
    assert matrix.sum() == pytest.approx(1.0)
    assert int(np.argmax(matrix[0])) == 0
    with pytest.raises(ValueError, match="incomplete race"):
        recency.trifecta_probability_matrix({"r1": predictions["r1"][:5]}, race_keys)


def test_conditional_payout_uses_pre_holdout_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    race_keys = [
        ("inner", "2025-12-30", "01", 1),
        ("cal", "2025-12-31", "01", 2),
        ("hold", "2026-01-01", "01", 1),
    ]
    lane_rows = lambda race_id: [
        {"race_id": race_id, "lane": lane, "probability": float(7 - lane)}
        for lane in range(1, 7)
    ]
    captured: dict[str, object] = {}

    def fake_simulate(probabilities, **kwargs):
        captured["probabilities"] = probabilities
        captured.update(kwargs)
        return {
            "evaluated_races": 1,
            "roi": 1.2,
            "profit_yen": 200,
            "daily": [{"race_date": "2026-01-01", "stake_yen": 100, "return_yen": 120}],
        }

    monkeypatch.setattr(recency, "_load_trifecta_payouts", lambda _conn: {})
    monkeypatch.setattr(recency, "simulate_conditional_payout_walk_forward", fake_simulate)
    monkeypatch.setattr(
        recency,
        "bootstrap_daily_bankroll",
        lambda *_args, **_kwargs: {
            "roi_ci95_lower": 1.01,
            "roi_delta_ci95_lower": 0.01,
        },
    )
    monkeypatch.setattr(
        recency,
        "bankroll_promotion_gate",
        lambda *_args, **_kwargs: {"pass": True},
    )

    result = recency.conditional_payout_summary(
        None,
        race_keys=race_keys,
        training_count=2,
        inner_train_count=1,
        calibration_predictions={"cal": lane_rows("cal")},
        holdout_predictions={"hold": lane_rows("hold")},
        baseline_bankroll={"roi": 0.8, "profit_yen": -100},
        baseline_daily=[{"race_date": "2026-01-01", "stake_yen": 100, "return_yen": 80}],
        protocol={"bankroll_evaluable_races": 1},
    )

    assert captured["race_keys"] == [race_keys[2]]
    assert captured["calibration_race_keys"] == [race_keys[1]]
    assert np.asarray(captured["probabilities"]).shape == (1, 120)
    assert np.asarray(captured["calibration_probabilities"]).shape == (1, 120)
    assert result["promotion_eligible"] is True


def test_protocol_race_validation_rejects_holdout_hash_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    race_keys = [
        ("train", "2025-01-01", "01", 1),
        ("holdout", "2026-01-01", "01", 1),
    ]
    monkeypatch.setattr(recency, "load_complete_race_ids", lambda _conn: race_keys)
    protocol = {
        "training_races": 1,
        "prediction_races": 1,
        "holdout_start": "2026-01-01",
        "holdout_end": "2026-01-01",
        "race_set_sha256": "0" * 64,
    }

    with pytest.raises(ValueError, match="holdout race set hash mismatch"):
        recency.validated_protocol_race_keys(None, protocol)


def test_final_evaluation_writes_atomic_training_only_selection_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = make_dataset(
        [
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2025-01-01",
            "2025-01-02",
        ]
    )
    training_count = 4
    holdout_ids = [row[0] for row in dataset.race_keys[training_count:]]
    holdout_hash = race_set_sha256(holdout_ids)
    protocol = {
        "calendar_days": 365,
        "training_races": training_count,
        "prediction_races": 2,
        "bankroll_evaluable_races": 2,
        "holdout_date_count": 2,
        "holdout_start": "2025-01-01",
        "holdout_end": "2025-01-02",
        "race_set_sha256": holdout_hash,
    }
    training_hash = race_set_sha256(
        row[0] for row in dataset.race_keys[:training_count]
    )
    final_score_calls: list[tuple[int, int]] = []

    def assert_frozen(*_args: object, **_kwargs: object) -> None:
        assert os.environ["BOATRACE_EVAL_MAX_RACE_DATE"] == "2025-01-02"

    monkeypatch.setattr(recency, "build_protocol", lambda *_args, **_kwargs: protocol)
    monkeypatch.setattr(recency, "verify_protocol_against_database", assert_frozen)
    monkeypatch.setattr(
        recency,
        "validated_protocol_race_keys",
        lambda *_args, **_kwargs: (dataset.race_keys, training_hash),
    )
    feature_contract: dict[str, tuple[str, ...]] = {}

    def fake_iter_rows(
        _conn: object,
        *,
        include_races: set[str],
        drop_feature_groups: tuple[str, ...],
    ):
        del include_races
        feature_contract["rows"] = drop_feature_groups
        return iter(())

    def fake_load_dataset(**kwargs: object):
        feature_contract["cache"] = kwargs["drop_feature_groups"]
        list(kwargs["race_rows"]())
        return dataset, "disk"

    monkeypatch.setattr(recency, "iter_race_feature_rows", fake_iter_rows)
    monkeypatch.setattr(recency, "load_or_build_hashed_dataset", fake_load_dataset)
    monkeypatch.setattr(recency, "validate_dataset_races", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        recency,
        "select_recency_half_life",
        lambda *_args, **_kwargs: (
            365.0,
            [{"recency_half_life_days": 365.0, "entry_log_loss": 0.3}],
            {
                "inner_train_races": 2,
                "calibration_races": 2,
                "calibration_start": "2024-01-03",
                "calibration_end": "2024-01-04",
            },
        ),
    )
    monkeypatch.setattr(
        recency,
        "train_bundle_from_dataset",
        lambda *_args, **kwargs: {"half_life": kwargs["recency_half_life_days"]},
    )

    def fake_final_score(
        _dataset: HashedRaceDataset,
        *,
        bundle: dict[str, object],
        race_start: int,
        race_end: int,
        batch_size: int,
    ) -> tuple[dict[str, float | int], dict[str, list[dict[str, object]]]]:
        del batch_size
        assert bundle["half_life"] == 365.0
        final_score_calls.append((race_start, race_end))
        predictions = {
            race_id: [{"race_id": race_id}] * 6 for race_id in holdout_ids
        }
        return (
            {
                "entry_log_loss": 0.25,
                "entry_brier": 0.08,
                "winner_top1_accuracy": 0.5,
                "trifecta_top1_hit_rate": 0.1,
                "trifecta_top5_hit_rate": 0.2,
                "evaluated_races": 2,
            },
            predictions,
        )

    monkeypatch.setattr(recency, "score_range", fake_final_score)
    policy = {"daily_budget_yen": 10_000, "model": recency.MODEL_NAME}
    bankroll = {
        "evaluated_races": 2,
        "stake_yen": 1_000,
        "return_yen": 1_200,
        "profit_yen": 200,
        "roi": 1.2,
        "evaluation_race_set_sha256": holdout_hash,
    }
    daily = [
        {"race_date": "2025-01-01", "profit_yen": 100},
        {"race_date": "2025-01-02", "profit_yen": 100},
    ]
    monkeypatch.setattr(
        recency,
        "bankroll_summary",
        lambda *_args, **_kwargs: (policy, bankroll, daily),
    )
    conditional = {
        "promotion_eligible": False,
        "bankroll": {"roi": 0.9, "daily": daily},
    }
    monkeypatch.setattr(
        recency,
        "conditional_payout_summary",
        lambda *_args, **_kwargs: conditional,
    )
    output = tmp_path / "result.json"

    result = recency.evaluate_recency_mlp(
        None,
        output_path=output,
        evaluation_date=date(2025, 1, 2),
        feature_cache=tmp_path / "features",
    )

    assert final_score_calls == [(training_count, dataset.race_count)]
    assert result["model"] == "calibrated_mlp_recency_selected"
    assert result["drop_feature_groups"] == ["research_correlates"]
    assert feature_contract == {"rows": ("research_correlates",), "cache": ("research_correlates",)}
    assert result["selected_recency_half_life_days"] == 365.0
    assert result["entry_log_loss"] == 0.25
    assert result["entry_brier"] == 0.08
    assert result["winner_top1_accuracy"] == 0.5
    assert result["trifecta_top1_hit_rate"] == 0.1
    assert result["trifecta_top5_hit_rate"] == 0.2
    assert result["evaluation_race_set_sha256"] == holdout_hash
    assert result["daily"] == daily
    assert result["conditional_payout_walk_forward"] == conditional
    assert result["promotion_eligible"] is False
    assert "outer training only" in result["selection"]["scope"]
    assert output.exists()
    assert not output.with_name(f".{output.name}.tmp").exists()
    assert "BOATRACE_EVAL_MAX_RACE_DATE" not in os.environ


def test_bankroll_summary_completes_and_validates_fixed_standard_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_policy: dict[str, object] = {}

    def fake_evaluate(**kwargs: object):
        captured_policy.update(kwargs["policy"])
        kwargs["daily_rows"].append({"race_date": "2025-01-01"})
        return (
            {"stake_yen": 0, "return_yen": 0, "profit_yen": 0, "roi": 0.0},
            (0, 0, 0),
        )

    monkeypatch.setattr(recency, "evaluate_bankroll_fold", fake_evaluate)
    monkeypatch.setattr(recency, "_load_trifecta_payouts", lambda _conn: {})
    protocol = {
        "bankroll_evaluable_races": 0,
        "holdout_date_count": 1,
        "holdout_start": "2025-01-01",
        "holdout_end": "2025-01-01",
        "race_set_sha256": "a" * 64,
    }

    policy, summary, daily = recency.bankroll_summary(
        None,
        predictions={},
        training_races=set(),
        test_dates={"2025-01-01"},
        protocol=protocol,
    )

    assert captured_policy["require_real_odds"] is False
    assert all(
        policy.get(key) == expected
        for key, expected in recency.STANDARD_POLICY.items()
    )
    assert summary["evaluated_races"] == 0
    assert daily == [{"race_date": "2025-01-01"}]


def test_cli_defaults_match_recency_protocol() -> None:
    args = recency.build_parser().parse_args(
        ["--db", "races.sqlite", "--output", "result.json", "--evaluation-date", "2026-01-31"]
    )

    assert args.feature_cache == recency.DEFAULT_FEATURE_CACHE
    assert args.model_output is None
    assert args.half_lives == (None, 180.0, 365.0, 730.0)
    assert args.calibration_days == 180
