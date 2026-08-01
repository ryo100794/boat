from __future__ import annotations

import json
from datetime import date

from boatrace_ai.db import connection, init_db
from boatrace_ai.ingestion import live
from boatrace_ai.odds_quality import (
    TRIFECTA_COMBINATION_KEYS,
    TRIFECTA_PARSER_VERSION,
    describe_trifecta_market,
    plausible_trifecta_capture,
    plausible_trifecta_odds,
)
from boatrace_ai.runtime import t5_spool


RACE_DATE = date(2026, 8, 1)


def _odds_without_lane(absent_lane: int) -> dict[str, float | None]:
    return {
        combination: (
            None
            if str(absent_lane) in combination.split("-")
            else float(index + 10)
        )
        for index, combination in enumerate(TRIFECTA_COMBINATION_KEYS)
    }


def _parsed_without_lane(absent_lane: int = 6) -> dict:
    return {
        "parser_version": TRIFECTA_PARSER_VERSION,
        "parsed_count": 120,
        "source_update_time": "12:34",
        "odds": _odds_without_lane(absent_lane),
    }


def test_absent_lane_market_requires_the_complete_active_lane_permutation() -> None:
    odds = _odds_without_lane(6)
    shape = describe_trifecta_market(odds)

    assert shape == {
        "active_lanes": [1, 2, 3, 4, 5],
        "absent_lanes": [6],
        "active_combination_count": 60,
        "total_combination_count": 120,
        "special_market": True,
        "model_supported": False,
    }
    assert not plausible_trifecta_odds(odds)
    assert not plausible_trifecta_capture(odds)

    corrupt = dict(odds)
    corrupt["1-2-3"] = None
    assert describe_trifecta_market(corrupt) is None


def test_t5_fetch_accepts_and_labels_an_official_absent_lane_market(
    monkeypatch,
) -> None:
    payload = b"<html>official absent-lane odds</html>"
    monkeypatch.setattr(
        t5_spool,
        "fetch_text",
        lambda *_args, **_kwargs: (200, payload.decode(), payload),
    )
    monkeypatch.setattr(t5_spool, "result_page_is_cancelled", lambda _html: False)
    monkeypatch.setattr(
        t5_spool, "parse_odds3t_html", lambda _html: _parsed_without_lane()
    )

    fetched = t5_spool.fetch_t5_capture(
        race_date=RACE_DATE, jcd="11", rno=7
    )

    assert fetched is not None
    event, raw_payload = fetched
    assert raw_payload == payload
    assert event["parsed"]["market_shape"]["special_market"] is True
    assert event["parsed"]["market_shape"]["active_combination_count"] == 60
    assert len(event["odds"]) == 120


def test_t5_persistence_keeps_raw_nulls_but_writes_only_active_combinations(
    tmp_path,
) -> None:
    database = tmp_path / "collector.sqlite"
    init_db(database)
    parsed = _parsed_without_lane()
    parsed["market_shape"] = describe_trifecta_market(parsed["odds"])
    event = t5_spool.build_capture(
        race_date=RACE_DATE,
        jcd="11",
        rno=7,
        captured_at="2026-08-01T03:34:00+00:00",
        source_url="https://www.boatrace.jp/owpc/pc/race/odds3t?rno=7&jcd=11&hd=20260801",
        parsed=parsed,
        raw_sha256="0" * 64,
        raw_bytes=10,
    )
    event["raw_local_path"] = str(tmp_path / "official.html")

    with connection(database) as conn:
        snapshot_id = t5_spool.persist_capture(conn, event)
        rows = conn.execute(
            "SELECT combination, odds FROM odds_trifecta "
            "WHERE snapshot_id = ? ORDER BY combination",
            (snapshot_id,),
        ).fetchall()
        snapshot = conn.execute(
            "SELECT raw_json FROM odds_snapshots WHERE snapshot_id = ?",
            (snapshot_id,),
        ).fetchone()

    assert len(rows) == 60
    assert all("6" not in row["combination"].split("-") for row in rows)
    raw = json.loads(snapshot["raw_json"])
    assert len(raw["odds"]) == 120
    assert sum(value is None for value in raw["odds"].values()) == 60
    assert raw["market_shape"]["model_supported"] is False


def test_live_collector_persists_special_market_without_enabling_inference(
    monkeypatch, tmp_path
) -> None:
    database = tmp_path / "collector.sqlite"
    init_db(database)
    monkeypatch.setattr(live, "_fetch_page", lambda *_args, **_kwargs: "html")
    monkeypatch.setattr(live, "result_page_is_cancelled", lambda _html: False)
    monkeypatch.setattr(
        live, "parse_odds3t_html", lambda _html: _parsed_without_lane()
    )

    with connection(database) as conn:
        assert live.collect_odds(
            conn,
            race_date=RACE_DATE,
            jcd="11",
            rno=7,
            raw_dir=tmp_path,
        )
        count = conn.execute("SELECT COUNT(*) FROM odds_trifecta").fetchone()[0]
        raw_json = conn.execute(
            "SELECT raw_json FROM odds_snapshots ORDER BY snapshot_id DESC LIMIT 1"
        ).fetchone()[0]

    assert count == 60
    assert json.loads(raw_json)["market_shape"]["model_supported"] is False
