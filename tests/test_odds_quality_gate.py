from __future__ import annotations

from datetime import date

from boatrace_ai.db import connection, init_db, insert_odds_snapshot
from boatrace_ai.features import (
    latest_trifecta_odds,
    latest_trifecta_odds_before_deadline,
    odds_lane_features,
)
from boatrace_ai.ingestion import live
from boatrace_ai.odds_quality import (
    TRIFECTA_COMBINATION_KEYS,
    plausible_trifecta_odds,
)


def _odds(value: float = 10.0) -> dict[str, float]:
    return {combination: value for combination in TRIFECTA_COMBINATION_KEYS}


def _lane_markers() -> dict[str, float]:
    return {
        combination: float(index % 6 + 1)
        for index, combination in enumerate(TRIFECTA_COMBINATION_KEYS)
    }


def test_plausibility_rejects_lane_headers_and_incomplete_tables() -> None:
    assert plausible_trifecta_odds(_odds())
    assert not plausible_trifecta_odds(_lane_markers())
    assert not plausible_trifecta_odds({"1-2-3": 10.0})


def test_readers_skip_newer_legacy_and_corrupt_dom_snapshots(tmp_path) -> None:
    database = tmp_path / "odds.sqlite"
    race_id = "2026-07-22-01-01"
    init_db(database)
    with connection(database) as conn:
        conn.execute(
            "INSERT INTO races(race_id, race_date, jcd, venue_name, rno, deadline_at) "
            "VALUES (?, '2026-07-22', '01', '桐生', 1, '2026-07-22T12:00:00+09:00')",
            (race_id,),
        )
        valid_id = insert_odds_snapshot(
            conn,
            race_id,
            "2026-07-22T02:54:00+00:00",
            "11:54",
            _odds(12.0),
            "valid",
            {"parser_version": "odds3t_dom_v2"},
        )
        insert_odds_snapshot(
            conn,
            race_id,
            "2026-07-22T02:54:20+00:00",
            "11:54",
            _lane_markers(),
            "corrupt-dom",
            {"parser_version": "odds3t_dom_v2"},
        )
        insert_odds_snapshot(
            conn,
            race_id,
            "2026-07-22T02:54:40+00:00",
            "11:54",
            _lane_markers(),
            "legacy",
            {"parser_version": "odds3t_legacy"},
        )

        latest = latest_trifecta_odds(conn, race_id)
        cutoff = latest_trifecta_odds_before_deadline(conn, race_id)
        lane_features = odds_lane_features(conn, race_id)

    assert set(latest.values()) == {12.0}
    assert cutoff is not None
    assert cutoff["snapshot_id"] == valid_id
    assert set(cutoff["odds"].values()) == {12.0}
    assert {row["snapshot_count"] for row in lane_features.values()} == {1.0}


def test_collector_does_not_persist_legacy_fallback(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(live, "_fetch_page", lambda *args, **kwargs: "html")
    monkeypatch.setattr(
        live,
        "parse_odds3t_html",
        lambda _html: {
            "parser_version": "odds3t_legacy",
            "parsed_count": 120,
            "odds": _lane_markers(),
        },
    )

    assert not live.collect_odds(
        object(),
        race_date=date(2026, 7, 22),
        jcd="01",
        rno=1,
        raw_dir=tmp_path,
    )
