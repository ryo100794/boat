from pathlib import Path

import numpy as np
import pytest

from boatrace_ai import operational_features
from boatrace_ai import operational_model
from boatrace_ai.feature_schema import FEATURE_SCHEMA_VERSION
from boatrace_ai.listwise.conditional_order import ConditionalOrderModel


def test_operational_features_honor_schema_and_dropped_groups(monkeypatch) -> None:
    rows = [{"lane": lane, "race_date": "2026-07-24"} for lane in range(1, 7)]
    captured: dict[str, object] = {}

    class StateSpy:
        def __init__(self) -> None:
            self.updates = 0

        def update_race(self, _rows) -> None:
            self.updates += 1

        def features_for(self, row) -> dict[str, object]:
            return {"history": self.updates, "research_history": row["lane"]}

    monkeypatch.setattr(operational_features, "ensure_series_cache_table", lambda _conn: None)
    monkeypatch.setattr(
        operational_features,
        "load_race_entries",
        lambda _conn, *, race_id: rows,
    )
    monkeypatch.setattr(
        operational_features,
        "history_groups_prior_dates",
        lambda *_args: [[{"lane": 1}]],
    )
    monkeypatch.setattr(operational_features, "RollingState", StateSpy)

    def fake_relatives(_rows, _before, *, include_research):
        captured["include_research"] = include_research
        return {
            lane: {"base": lane, "research_relative": lane}
            for lane in range(1, 7)
        }

    monkeypatch.setattr(operational_features, "race_relative_features", fake_relatives)
    monkeypatch.setattr(
        operational_features,
        "base_pastlog_features",
        lambda row, relatives: {**relatives, "lane": row["lane"]},
    )
    monkeypatch.setattr(
        operational_features,
        "cached_series_features",
        lambda _row, *, feature_schema_version: {
            "series_schema": feature_schema_version
        },
    )
    monkeypatch.setattr(
        operational_features,
        "series_relative_features",
        lambda _rows, *, feature_schema_version: {
            lane: {"relative_schema": feature_schema_version}
            for lane in range(1, 7)
        },
    )

    features = operational_features.prediction_features(
        None,
        race_id="race",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        drop_feature_groups=("research_correlates",),
    )

    assert captured["include_research"] is False
    assert len(features) == 6
    assert all(row["series_schema"] == FEATURE_SCHEMA_VERSION for row in features)
    assert all(row["relative_schema"] == FEATURE_SCHEMA_VERSION for row in features)
    assert all(not any(key.startswith("research_") for key in row) for row in features)
    with pytest.raises(ValueError, match="unknown feature groups"):
        operational_features.prediction_features(
            None,
            race_id="race",
            drop_feature_groups=("unknown",),
        )


def test_operational_model_dispatches_calibrated_artifact(monkeypatch) -> None:
    bundle = {
        "hasher": object(),
        "scaler": object(),
        "classifier": object(),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "drop_feature_groups": ["research_correlates"],
        "metadata": {"feature_set": "calibrated_test"},
    }
    captured: dict[str, object] = {}
    monkeypatch.setattr(operational_model, "load_model_bundle", lambda _path: bundle)

    def fake_features(_conn, **kwargs):
        captured.update(kwargs)
        return [{"lane": lane} for lane in range(1, 7)]

    monkeypatch.setattr(operational_model, "calibrated_prediction_features", fake_features)
    monkeypatch.setattr(
        operational_model,
        "calibrated_predict_probabilities",
        lambda _bundle, _features: [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    )
    monkeypatch.setattr(operational_model, "latest_trifecta_odds", lambda *_args: {})
    monkeypatch.setattr(
        operational_model,
        "insert_prediction_rows",
        lambda *_args: pytest.fail("store=False must not write predictions"),
    )

    rows = operational_model.predict_race(
        None,
        model_path=Path("calibrated.joblib"),
        race_id_value="race",
        top_n=5,
        store=False,
    )

    assert captured["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert captured["drop_feature_groups"] == ("research_correlates",)
    assert len(rows) == 5
    assert all(row["feature_set"] == "calibrated_test_model_probability_rank" for row in rows)
    assert rows[0]["combination"] == "1-2-3"


def test_operational_model_uses_persisted_conditional_order(monkeypatch) -> None:
    second_bias = np.zeros((6, 6), dtype=float)
    second_bias[0, 5] = 5.0
    order_model = ConditionalOrderModel(
        scales=np.ones(3),
        second_bias=second_bias,
        third_first_bias=np.zeros((6, 6), dtype=float),
        third_second_bias=np.zeros((6, 6), dtype=float),
        regularization=0.1,
    )
    bundle = {
        "hasher": object(),
        "scaler": object(),
        "classifier": object(),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "conditional_order_model": order_model,
    }
    monkeypatch.setattr(operational_model, "load_model_bundle", lambda _path: bundle)
    monkeypatch.setattr(
        operational_model,
        "calibrated_prediction_features",
        lambda *_args, **_kwargs: [{"lane": lane} for lane in range(1, 7)],
    )
    monkeypatch.setattr(
        operational_model,
        "calibrated_predict_probabilities",
        lambda *_args, **_kwargs: [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    )
    monkeypatch.setattr(operational_model, "latest_trifecta_odds", lambda *_args: {})

    rows = operational_model.predict_race(
        None,
        model_path=Path("conditional.joblib"),
        race_id_value="race",
        top_n=1,
        store=False,
    )

    assert rows[0]["combination"].startswith("1-6-")
    assert rows[0]["rank_basis"] == "conditional_order_probability"


def test_operational_model_rejects_calibrated_artifact_without_schema(monkeypatch) -> None:
    monkeypatch.setattr(
        operational_model,
        "load_model_bundle",
        lambda _path: {"hasher": object(), "scaler": object(), "classifier": object()},
    )

    with pytest.raises(ValueError, match="lacks feature schema"):
        operational_model.predict_race(
            None,
            model_path=Path("calibrated.joblib"),
            race_id_value="race",
            store=False,
        )
