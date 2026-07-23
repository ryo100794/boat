from __future__ import annotations

from datetime import date

import pytest

from boatrace_ai import historical_model, model_core
from boatrace_ai.bankroll_optimizer import _validated_pretrained_bundle
from boatrace_ai.base_features import is_home_branch, race_relative_features
from boatrace_ai.cache_entry_series_features import ensure_series_cache_table
from boatrace_ai.contextual_features import RollingState
from boatrace_ai.feature_tuning import build_race_features
from boatrace_ai.standard_evaluation import race_set_sha256


def _entry(lane: int, *, branch: str = "東京", local_delta: float = 0.0) -> dict:
    return {
        "race_id": "202607200501",
        "race_date": "2026-07-20",
        "lane": lane,
        "rno": 1,
        "jcd": "05",
        "race_type": "一般",
        "distance_m": 1800,
        "racer_no": 4000 + lane,
        "racer_name": f"選手{lane}",
        "racer_class": "A1" if lane == 1 else "B1",
        "branch": branch,
        "origin": branch,
        "age": 30,
        "weight_kg": 52,
        "f_count": 0,
        "l_count": 0,
        "avg_st": 0.15,
        "national_win_rate": 6.0 if lane == 1 else 5.0,
        "national_2_rate": 40.0 if lane == 1 else 30.0,
        "national_3_rate": 55.0,
        "local_win_rate": (6.0 if lane == 1 else 5.0) + local_delta,
        "local_2_rate": (40.0 if lane == 1 else 30.0) + local_delta,
        "local_3_rate": 55.0,
        "motor_no": lane,
        "motor_2_rate": 35.0,
        "motor_3_rate": 50.0,
        "boat_no": lane,
        "boat_2_rate": 35.0,
        "boat_3_rate": 50.0,
        "rank": lane,
        "result_course": lane,
        "result_start_timing": 0.15,
    }


def _before(lane: int, *, course: int | None = None) -> dict:
    return {
        "weight_kg": 52.0,
        "exhibition_time": 6.70 + lane / 100,
        "tilt": 0.0,
        "adjusted_weight": 0.0,
        "course": lane if course is None else course,
        "start_timing": 0.10 + lane / 100,
        "weather": "雨",
        "wind_direction": "北",
        "wind_speed_m": 4.0,
        "air_temp_c": 25.0,
        "water_temp_c": 24.0,
        "wave_cm": 4.0,
        "propeller": "",
        "parts_exchange": "",
    }


def test_home_branch_maps_shared_venue_branches() -> None:
    assert is_home_branch("03", "東京")
    assert is_home_branch("05", "東京")
    assert is_home_branch("20", "福岡")
    assert not is_home_branch("05", "埼玉")


def test_research_features_separate_home_matchup_equipment_and_live_context() -> None:
    rows = [
        _entry(lane, local_delta=1.0 if lane == 1 else 0.0)
        for lane in range(1, 7)
    ]
    features = race_relative_features(
        rows,
        {lane: _before(lane) for lane in range(1, 7)},
    )

    assert features[1]["research_home_branch"] == 1
    assert features[1]["research_local_vs_national_win"] == 1.0
    assert features[1]["research_home_local_win_delta"] == 1.0
    assert (
        features[1]["research_racer_strength"]
        > features[2]["research_racer_strength"]
    )
    assert features[1]["research_racer_strength_rank"] == 1
    assert features[1]["research_waku_nari"] == 1
    assert features[1]["research_exhibition_top1"] == 1
    assert features[1]["research_exhibition_rank_weather"] == "1:雨"
    assert "research_equipment_strength" in features[1]


def test_course_change_and_research_group_ablation() -> None:
    rows = [_entry(lane) for lane in range(1, 7)]
    before = {lane: _before(lane) for lane in range(1, 7)}
    before[2] = _before(2, course=3)
    before[3] = _before(3, course=2)
    live_features = race_relative_features(rows, before)
    assert live_features[2]["research_waku_nari"] == 0
    assert live_features[2]["research_course_changed"] == 1
    assert live_features[2]["research_course_delta"] == 1

    full = build_race_features(
        rows,
        RollingState(),
        drop_feature_groups=("series_cached", "series_relative"),
    )
    dropped = build_race_features(
        rows,
        RollingState(),
        drop_feature_groups=(
            "research_correlates",
            "series_cached",
            "series_relative",
        ),
    )
    assert any(key.startswith("research_") for key in full[0]["features"])
    assert not any(key.startswith("research_") for key in dropped[0]["features"])


def test_postgresql_series_cache_check_is_read_only() -> None:
    class FakePostgresql:
        dialect = "postgresql"

        def __init__(self) -> None:
            self.statements: list[str] = []

        def execute(self, statement: str) -> None:
            self.statements.append(statement)

        def executescript(self, _statement: str) -> None:
            raise AssertionError("PostgreSQL schema must not be mutated by evaluation")

    conn = FakePostgresql()
    ensure_series_cache_table(conn)
    assert conn.statements == ["SELECT 1 FROM entry_series_features LIMIT 0"]


def test_pretrained_bankroll_model_validates_training_universe() -> None:
    train_races = {"r1", "r2"}
    bundle = {
        "drop_feature_groups": ["research_correlates"],
        "metadata": {
            "train_races": 2,
            "train_race_set_sha256": race_set_sha256(train_races),
            "drop_feature_groups": ["research_correlates"],
        },
    }
    assert _validated_pretrained_bundle(
        bundle,
        train_races=train_races,
        drop_feature_groups=("research_correlates",),
    ) is bundle

    with pytest.raises(ValueError, match="race set mismatch"):
        _validated_pretrained_bundle(
            bundle,
            train_races={"r1", "r3"},
            drop_feature_groups=("research_correlates",),
        )


class _RaceConnection:
    def execute(self, _sql, _params):
        return self

    def fetchall(self):
        return [{"race_id": "race-1"}]


def test_model_core_predict_race_keeps_beforeinfo_enabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    captured: list[bool] = []
    monkeypatch.setattr(
        model_core,
        "load_model_bundle",
        lambda _path: {"pipeline": object()},
    )

    def fake_features(_conn, **kwargs):
        captured.append(kwargs["include_beforeinfo"])
        return [{"lane": lane} for lane in range(1, 7)]

    monkeypatch.setattr(model_core, "prediction_features", fake_features)
    monkeypatch.setattr(
        model_core,
        "positive_probs",
        lambda _pipeline, _rows: [1.0] * 6,
    )
    monkeypatch.setattr(
        model_core,
        "latest_trifecta_odds",
        lambda _conn, _race_id: {},
    )
    monkeypatch.setattr(
        model_core,
        "trifecta_predictions",
        lambda _probs, **_kwargs: [],
    )

    model_core.predict_race(
        object(),
        model_path=tmp_path / "model.joblib",
        race_id_value="race-1",
        store=False,
    )

    assert captured == [True]


def test_model_core_predict_open_races_passes_beforeinfo_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    captured: list[bool] = []

    def fake_predict(_conn, **kwargs):
        captured.append(kwargs["include_beforeinfo"])
        return []

    monkeypatch.setattr(model_core, "predict_race", fake_predict)
    result = model_core.predict_open_races(
        _RaceConnection(),
        model_path=tmp_path / "model.joblib",
        race_date=date(2026, 7, 23),
        include_beforeinfo=False,
    )

    assert result == {"predicted": 1, "failed": 0}
    assert captured == [False]


def test_no_odds_v8_keeps_live_context_out_of_prediction_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    calls: list[tuple[str, bool, bool]] = []

    def fake_predict(conn, **kwargs):
        calls.append(
            ("predict", kwargs["include_research"], kwargs["include_beforeinfo"])
        )
        return []

    def fake_predict_open(conn, **kwargs):
        calls.append(
            (
                "predict_open",
                kwargs["include_research"],
                kwargs["include_beforeinfo"],
            )
        )
        return {"predicted": 0, "failed": 0}

    monkeypatch.setattr(historical_model.base, "predict_race", fake_predict)
    monkeypatch.setattr(historical_model.base, "predict_open_races", fake_predict_open)

    historical_model.predict_race(
        object(),
        model_path=tmp_path / "model.joblib",
        race_id_value="r1",
        include_beforeinfo=True,
    )
    historical_model.predict_open_races(
        object(),
        model_path=tmp_path / "model.joblib",
        race_date=None,
        include_beforeinfo=True,
    )

    assert calls == [
        ("predict", False, False),
        ("predict_open", False, False),
    ]


def test_no_odds_v8_streaming_fit_preserves_sparse_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        (
            {"lane": str(lane), "lane_num": lane, "strength": 7 - lane},
            1 if lane == 1 else 0,
            {
                "race_id": race_id,
                "race_date": "2026-01-01",
                "jcd": "01",
                "rno": race_number,
                "lane": lane,
                "rank": lane,
            },
        )
        for race_number, race_id in enumerate(("r1", "r2"), start=1)
        for lane in range(1, 7)
    ]

    def fake_iter(
        conn,
        *,
        include_odds,
        include_research,
        include_beforeinfo,
        include_races,
    ):
        assert include_odds is False
        assert include_research is False
        assert include_beforeinfo is False
        yield from (row for row in rows if row[2]["race_id"] in include_races)

    monkeypatch.setattr(historical_model, "iter_training_examples", fake_iter)
    bundle = historical_model.fit_streaming_pipeline(
        object(),
        train_races={"r1", "r2"},
        batch_size=3,
    )

    metadata = bundle["metadata"]
    assert metadata["examples"] == 12
    assert metadata["races"] == 2
    assert metadata["include_beforeinfo"] is False
    assert metadata["feature_set"] == (
        "no_odds_v8_historical_only_beforeinfo_excluded_"
        "sparse32_scaled_logreg_C0.20_unweighted"
    )
    assert metadata["train_race_set_sha256"] == race_set_sha256({"r1", "r2"})
    assert list(bundle["pipeline"].named_steps) == [
        "vectorizer",
        "sparse_index_32_a",
        "scaler",
        "sparse_index_32_b",
        "classifier",
    ]
    assert (
        bundle["pipeline"].named_steps["sparse_index_32_a"]
        .__class__.__module__
        == "boatrace_ai.model_core"
    )
