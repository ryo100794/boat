LEGACY_FEATURE_SCHEMA_VERSION = "pastlog-listwise-hashed-v1"
FEATURE_SCHEMA_VERSION = "pastlog-listwise-hashed-v2-series-missing-safe"


def uses_missing_safe_series(version: str | None) -> bool:
    return str(version or LEGACY_FEATURE_SCHEMA_VERSION) != LEGACY_FEATURE_SCHEMA_VERSION
