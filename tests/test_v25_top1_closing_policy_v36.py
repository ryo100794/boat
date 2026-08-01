from __future__ import annotations

import numpy as np

import boatrace_ai.listwise.v25_top1_closing_policy_v36 as v36


def test_top1_teacher_does_not_depend_on_outcome_or_payout(monkeypatch) -> None:
    combinations = [f"c{index:03d}" for index in range(120)]
    current = {key: 10.0 + index for index, key in enumerate(combinations)}
    closing = {key: value * 0.9 for key, value in current.items()}
    probabilities = {key: 0.001 for key in combinations}
    probabilities["c007"] = 0.9
    monkeypatch.setattr(
        v36,
        "normalize_labeled_checkpoints",
        lambda *_args, **_kwargs: {"t300": {"odds": current}},
    )
    monkeypatch.setattr(v36, "_snapshot_odds", lambda row: dict(row["odds"]))
    monkeypatch.setattr(
        v36,
        "_teacher_selection",
        lambda _race: (closing, "official", False),
    )
    monkeypatch.setattr(
        v36,
        "direct_context_probabilities",
        lambda *_args, **_kwargs: probabilities,
    )
    monkeypatch.setattr(
        v36,
        "build_checkpoint_feature_vector",
        lambda *_args, **_kwargs: (
            np.asarray([1.0, 2.0]),
            {"used_checkpoint_offsets": [300], "future_checkpoint_offsets_used": []},
        ),
    )
    base = {
        "race_id": "r1",
        "race_date": "2026-07-20",
        "actual_combination": "c001",
        "actual_payout_yen": 1230,
    }
    changed = {**base, "actual_combination": "c119", "actual_payout_yen": 999999}
    first = v36._top1_example(base, {})
    second = v36._top1_example(changed, {})
    assert first is not None and second is not None
    assert first["combination"] == "c007"
    assert first["target_log_ratio"] == second["target_log_ratio"]
    np.testing.assert_array_equal(first["features"], second["features"])


def test_top1_teacher_rejects_future_checkpoint_features(monkeypatch) -> None:
    combinations = [f"c{index:03d}" for index in range(120)]
    odds = {key: 10.0 for key in combinations}
    monkeypatch.setattr(
        v36,
        "normalize_labeled_checkpoints",
        lambda *_args, **_kwargs: {"t300": {"odds": odds}},
    )
    monkeypatch.setattr(v36, "_snapshot_odds", lambda row: dict(row["odds"]))
    monkeypatch.setattr(v36, "_teacher_selection", lambda _race: (odds, "official", False))
    monkeypatch.setattr(
        v36,
        "direct_context_probabilities",
        lambda *_args, **_kwargs: {key: 1 / 120 for key in combinations},
    )
    monkeypatch.setattr(
        v36,
        "build_checkpoint_feature_vector",
        lambda *_args, **_kwargs: (
            np.asarray([1.0]),
            {"used_checkpoint_offsets": [300, 120], "future_checkpoint_offsets_used": [120]},
        ),
    )
    try:
        v36._top1_example({"race_id": "r1", "race_date": "2026-07-20"}, {})
    except ValueError as error:
        assert "post-T-5" in str(error)
    else:
        raise AssertionError("future checkpoint must be rejected")
