from pathlib import Path

import joblib

import boatrace_ai.protected_prediction_cache as cache


def _predictions(race_id: str) -> dict[str, list[dict[str, object]]]:
    return {
        race_id: [
            {
                "race_id": race_id,
                "lane": lane,
                "rank": lane,
                "probability": 1.0 / 6.0,
            }
            for lane in range(1, 7)
        ]
    }


def test_cached_historical_baseline_reuses_identical_boundary(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[int] = []
    predictions = _predictions("score")

    def score(*_args, **_kwargs):
        calls.append(1)
        return {"evaluated_races": 1}, predictions

    monkeypatch.setattr(cache, "score_historical_baseline_range", score)
    race_keys = [("train", "2026-01-01", "01", 1), ("score", "2026-01-02", "01", 2)]
    kwargs = {
        "train_end": 1,
        "score_start": 1,
        "score_end": 2,
        "cache_dir": tmp_path,
    }

    first = cache.cached_historical_baseline_range(None, race_keys, **kwargs)
    second = cache.cached_historical_baseline_range(None, race_keys, **kwargs)

    assert len(calls) == 1
    assert first[1] == second[1] == predictions
    assert second[0]["evaluated_races"] == 1


def test_cached_historical_baseline_rebuilds_invalid_payload(
    tmp_path: Path, monkeypatch
) -> None:
    predictions = _predictions("score")
    monkeypatch.setattr(
        cache,
        "score_historical_baseline_range",
        lambda *_args, **_kwargs: ({"evaluated_races": 1}, predictions),
    )
    race_keys = [("train", "2026-01-01", "01", 1), ("score", "2026-01-02", "01", 2)]
    kwargs = {
        "train_end": 1,
        "score_start": 1,
        "score_end": 2,
        "cache_dir": tmp_path,
    }
    cache.cached_historical_baseline_range(None, race_keys, **kwargs)
    path = next(tmp_path.glob("baseline-*.joblib"))
    joblib.dump({"metadata": {}, "predictions": predictions}, path)

    metrics, _ = cache.cached_historical_baseline_range(None, race_keys, **kwargs)

    assert metrics["evaluated_races"] == 1
