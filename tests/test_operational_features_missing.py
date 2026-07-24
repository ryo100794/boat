from boatrace_ai.cache_entry_series_features import CACHE_FIELDS
from boatrace_ai.feature_schema import (
    FEATURE_SCHEMA_VERSION,
    LEGACY_FEATURE_SCHEMA_VERSION,
    MISSING_SAFE_FEATURE_SCHEMA_VERSION,
)
from boatrace_ai.operational_features import (
    _ranks,
    cached_series_features,
    series_relative_features,
)


def _row(lane: int, **values: float | None) -> dict:
    row = {field: -1.0 for field in CACHE_FIELDS}
    row.update(values)
    row["lane"] = lane
    return row


def test_series_missing_values_are_not_ranked_by_lane() -> None:
    assert _ranks(
        {1: -1.0, 2: -1.0, 3: 5.0},
        high_is_good=True,
    ) == {1: 0, 2: 0, 3: 1}


def test_series_equal_values_share_competition_rank() -> None:
    assert _ranks(
        {1: 5.0, 2: 5.0, 3: 4.0},
        high_is_good=True,
    ) == {1: 1, 2: 1, 3: 3}


def test_series_missing_relative_value_has_neutral_derivatives() -> None:
    features = series_relative_features(
        [_row(lane) for lane in range(1, 7)],
        feature_schema_version=MISSING_SAFE_FEATURE_SCHEMA_VERSION,
    )

    for lane in range(1, 7):
        assert features[lane]["has_series_starts"] == 0
        assert features[lane]["series_starts_rank"] == 0
        assert features[lane]["series_starts_vs_mean"] == 0.0
        assert features[lane]["series_starts_z"] == 0.0


def test_series_presence_is_separate_from_competition_rank() -> None:
    features = series_relative_features(
        [
            _row(1, series_starts=5.0),
            _row(2, series_starts=5.0),
            _row(3, series_starts=3.0),
            *[_row(lane) for lane in range(4, 7)],
        ],
        feature_schema_version=MISSING_SAFE_FEATURE_SCHEMA_VERSION,
    )

    assert features[1]["has_series_starts"] == 1
    assert features[1]["series_starts_rank"] == 1
    assert features[2]["series_starts_rank"] == 1
    assert features[3]["series_starts_rank"] == 3
    assert features[4]["has_series_starts"] == 0
    assert features[4]["series_starts_rank"] == 0


def test_legacy_schema_reproduces_old_missing_encoding() -> None:
    features = series_relative_features(
        [_row(lane) for lane in range(1, 7)],
        feature_schema_version=LEGACY_FEATURE_SCHEMA_VERSION,
    )

    for lane in range(1, 7):
        assert "has_series_starts" not in features[lane]
        assert features[lane]["series_starts_rank"] == lane
        assert features[lane]["series_starts_vs_mean"] == -1.0
        assert features[lane]["series_starts_z"] == -1.0


def test_sparse_schema_omits_series_features_when_cache_join_is_missing() -> None:
    rows = [
        _row(lane, **{field: None for field in CACHE_FIELDS})
        for lane in range(1, 7)
    ]

    features = series_relative_features(
        rows,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
    )

    assert all(features[lane] == {} for lane in range(1, 7))
    assert cached_series_features(
        rows[0],
        feature_schema_version=FEATURE_SCHEMA_VERSION,
    ) == {}


def test_sparse_schema_preserves_negative_finish_trend() -> None:
    trends = (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    rows = [
        _row(
            lane,
            series_finish_trend=trends[lane - 1],
            series_has_results=1.0,
        )
        for lane in range(1, 7)
    ]

    features = series_relative_features(
        rows,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
    )

    assert features[1]["series_finish_trend_rank"] == 6
    assert features[6]["series_finish_trend_rank"] == 1
    assert features[1]["series_finish_trend_vs_mean"] < 0
    assert cached_series_features(
        rows[0],
        feature_schema_version=FEATURE_SCHEMA_VERSION,
    )["series_finish_trend"] == -2.0
