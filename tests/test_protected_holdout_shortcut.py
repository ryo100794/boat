import boatrace_ai.recency_mlp_evaluation as evaluation


def test_full_candidate_weight_skips_baseline_scoring(monkeypatch) -> None:
    predictions = {"race": [{"lane": lane} for lane in range(1, 7)]}
    metrics = {"evaluated_races": 1}

    def unexpected(*_args, **_kwargs):
        raise AssertionError("baseline scoring must be skipped")

    monkeypatch.setattr(
        evaluation,
        "cached_historical_baseline_range",
        unexpected,
    )

    actual_metrics, actual_predictions = evaluation.protected_holdout_predictions(
        None,
        [("train", "2026-01-01", "01", 1), ("race", "2026-01-02", "01", 2)],
        training_count=1,
        candidate_predictions=predictions,
        candidate_metrics=metrics,
        candidate_weight=1.0,
        cache_dir=evaluation.Path("unused"),
        baseline_model_path=evaluation.Path("unused.joblib"),
    )

    assert actual_metrics is metrics
    assert actual_predictions is predictions
