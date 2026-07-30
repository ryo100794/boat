from __future__ import annotations

from datetime import date
import json

from boatrace_ai.racer_period_features import (
    enrich_racer_period_rows,
    load_racer_period_lookup,
    racer_period_feature_values,
    select_available_period,
)


class FakeResult:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def fetchall(self) -> list[dict[str, object]]:
        return self._rows


class FakeConnection:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def execute(self, statement: str) -> FakeResult:
        assert "FROM racer_period_stats" in statement
        return FakeResult(self._rows)


def _period(
    *,
    year: int,
    half: int,
    calculation_to: str,
    avg_st: float,
) -> dict[str, object]:
    payload = {
        "racer_class": "A1",
        "branch": "東京",
        "origin": "千　葉",
        "avg_st": avg_st,
        "win_rate": 6.5,
        "place2_rate": 45.0,
        "place3_rate": 60.0,
        "f_count": 1,
        "l_count": 0,
        "ability_index": 70.0,
        "starts": 120,
        "calculation_to": calculation_to,
    }
    return {
        "year": year,
        "half": half,
        "racer_no": 3415,
        "racer_class": "A1",
        "raw_json": json.dumps(payload, ensure_ascii=False),
    }


def test_lookup_normalizes_periods_and_enforces_release_lag() -> None:
    lookup = load_racer_period_lookup(
        FakeConnection(
            [
                _period(
                    year=2025,
                    half=2,
                    calculation_to="20250430",
                    avg_st=0.17,
                ),
                _period(
                    year=2026,
                    half=1,
                    calculation_to="20251031",
                    avg_st=0.15,
                ),
            ]
        )
    )

    periods = lookup[3415]
    assert periods[0]["available_from"] == date(2025, 5, 30)
    assert periods[0]["origin"] == "千葉"
    assert select_available_period(
        periods, race_date=date(2025, 5, 29)
    ) is None
    assert select_available_period(
        periods, race_date=date(2025, 5, 30)
    )["avg_st"] == 0.17
    assert select_available_period(
        periods, race_date=date(2025, 11, 29)
    )["avg_st"] == 0.17
    assert select_available_period(
        periods, race_date=date(2025, 11, 30)
    )["avg_st"] == 0.15


def test_enrichment_and_feature_projection_share_the_same_period() -> None:
    lookup = load_racer_period_lookup(
        FakeConnection(
            [
                _period(
                    year=2026,
                    half=1,
                    calculation_to="20251031",
                    avg_st=0.15,
                )
            ]
        )
    )
    rows = enrich_racer_period_rows(
        [
            {
                "race_id": "r1",
                "race_date": "2025-12-15",
                "racer_no": 3415,
                "lane": 1,
            }
        ],
        lookup,
    )
    features = racer_period_feature_values(rows[0])

    assert features["has_racer_period_stats"] == 1
    assert features["period_stats_age_days"] == 15
    assert features["period_avg_st"] == 0.15
    assert features["period_place3_rate"] == 60.0
    assert features["has_period_place3_rate"] == 1
    assert features["period_ability_index"] == 70.0
    assert features["period_origin"] == "千葉"


def test_missing_period_is_explicit() -> None:
    assert racer_period_feature_values({}) == {"has_racer_period_stats": 0}
