LEGACY_FEATURE_SCHEMA_VERSION = "pastlog-listwise-hashed-v1"
MISSING_SAFE_FEATURE_SCHEMA_VERSION = "pastlog-listwise-hashed-v2-series-missing-safe"
SPARSE_MISSING_FEATURE_SCHEMA_VERSION = "pastlog-listwise-hashed-v3-series-sparse-missing"
FEATURE_SCHEMA_VERSION = "pastlog-listwise-hashed-v4-series-trend-direction"


def uses_missing_safe_series(version: str | None) -> bool:
    return str(version or LEGACY_FEATURE_SCHEMA_VERSION) != LEGACY_FEATURE_SCHEMA_VERSION


def uses_sparse_series_missing(version: str | None) -> bool:
    return str(version or LEGACY_FEATURE_SCHEMA_VERSION) in {
        SPARSE_MISSING_FEATURE_SCHEMA_VERSION,
        FEATURE_SCHEMA_VERSION,
    }


def uses_empirical_series_trend_direction(version: str | None) -> bool:
    return str(version or LEGACY_FEATURE_SCHEMA_VERSION) == FEATURE_SCHEMA_VERSION
