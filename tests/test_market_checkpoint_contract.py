from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from boatrace_ai.archive_closing_odds import (
    OFFICIAL_SOURCE_KEY,
    SOURCE_KEY,
    ensure_archive_schema,
)
from boatrace_ai.db import (
    connection,
    init_db,
    insert_odds_snapshot,
)
from boatrace_ai.listwise.market_calibration import (
    MARKET_MAX_SNAPSHOT_AGE_SECONDS,
    ODDS_CHECKPOINT_OFFSETS_SECONDS,
    PREFETCH_CHECKPOINTS_KEY,
    available_odds_checkpoints,
    load_odds_checkpoints,
    load_scored_cache,
    normalize_odds_checkpoint,
    odds_data_signature,
    prefetch_official_closing_odds,
    prefetch_trifecta_snapshots,
    score_real_odds_races,
    scored_cache_contract,
    write_scored_cache,
)
from boatrace_ai.odds_quality import (
    TRIFECTA_COMBINATION_KEYS,
    TRIFECTA_PARSER_VERSION,
)


JST = timezone(timedelta(hours=9))
RACE_ID = "2026-07-22-01-01"
START_AT = datetime(2026, 7, 22, 12, 0, tzinfo=JST)
BETTING_DEADLINE = START_AT - timedelta(minutes=5)


def _odds(value: float) -> dict[str, float]:
    return {
        combination: float(value)
        for combination in TRIFECTA_COMBINATION_KEYS
    }


def _insert_completed_race(conn, *, with_payout: bool = True) -> None:
    conn.execute(
        """
        INSERT INTO races(
          race_id, race_date, jcd, venue_name, rno, deadline_at
        ) VALUES (?, '2026-07-22', '01', 'Kiryu', 1, ?)
        """,
        (RACE_ID, START_AT.isoformat()),
    )
    conn.executemany(
        "INSERT INTO race_results(race_id, lane, rank) VALUES (?, ?, ?)",
        [(RACE_ID, lane, lane) for lane in range(1, 7)],
    )
    conn.executemany(
        "INSERT INTO entries(race_id, lane, racer_no) VALUES (?, ?, ?)",
        [(RACE_ID, lane, 4000 + lane) for lane in range(1, 7)],
    )
    if with_payout:
        conn.execute(
            """
            INSERT INTO payouts(
              race_id, bet_type, combination, payout_yen
            ) VALUES (?, '3連単', '1-2-3', 1230)
            """,
            (RACE_ID,),
        )


def _insert_checkpoint(
    conn,
    *,
    offset: int,
    seconds_before_target: int = 5,
    value: float,
    explicit: bool = True,
    measured_age: float | None = None,
) -> int:
    target = BETTING_DEADLINE - timedelta(seconds=offset)
    captured = target - timedelta(seconds=seconds_before_target)
    raw: dict[str, object] = {"parser_version": TRIFECTA_PARSER_VERSION}
    if explicit:
        raw["_collection"] = {
            "event_id": f"{RACE_ID}-T{offset}",
            "target_offset_seconds": offset,
            "observation_label": f"T{offset}",
            "captured_age_seconds": (
                float(measured_age)
                if measured_age is not None
                else (BETTING_DEADLINE - captured).total_seconds()
            ),
            "source_update_time": (
                captured.astimezone(JST) - timedelta(seconds=7)
            ).strftime("%H:%M:%S"),
            "source_update_staleness_seconds": 7.0,
        }
    return insert_odds_snapshot(
        conn,
        RACE_ID,
        captured.astimezone(timezone.utc).isoformat(),
        (captured.astimezone(JST) - timedelta(seconds=11)).strftime(
            "%H:%M:%S"
        ),
        _odds(value),
        f"https://example.test/{offset}/{seconds_before_target}",
        raw,
    )


def _store_official_archive(
    conn, *, value: float = 30.0, source_key: str = SOURCE_KEY
) -> None:
    ensure_archive_schema(conn)
    conn.execute(
        """
        INSERT INTO archive_closing_odds_snapshots(
          race_id, source_key, fetched_at, source_url, payload_sha256,
          parser_version, odds_count, verification_status, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, 120, ?, ?)
        """,
        (
            RACE_ID,
            source_key,
            "2026-07-23T00:00:00+00:00",
            "https://example.test/archive",
            "a" * 64,
            "archive_closing_odds_dom_v3",
            "winner_only_match_unverified_market",
            json.dumps(
                {
                    "source_kind": (
                        "secondary_archive_of_official_closing_display"
                    )
                }
            ),
        ),
    )
    conn.executemany(
        """
        INSERT INTO archive_closing_odds(
          race_id, source_key, combination, odds
        ) VALUES (?, ?, ?, ?)
        """,
        [
            (RACE_ID, source_key, combination, float(value))
            for combination in TRIFECTA_COMBINATION_KEYS
        ],
    )


def test_sqlite_checkpoint_loader_is_strict_and_preserves_provenance(
    tmp_path,
) -> None:
    database = tmp_path / "checkpoint.sqlite"
    init_db(database)
    selected_ids: dict[int, int] = {}
    with connection(database) as conn:
        _insert_completed_race(conn)
        for index, offset in enumerate(ODDS_CHECKPOINT_OFFSETS_SECONDS):
            selected_ids[offset] = _insert_checkpoint(
                conn,
                offset=offset,
                value=20.0 + index,
                explicit=offset != 120,
            )

        target_300 = BETTING_DEADLINE - timedelta(seconds=300)
        future_id = insert_odds_snapshot(
            conn,
            RACE_ID,
            (target_300 + timedelta(seconds=1))
            .astimezone(timezone.utc)
            .isoformat(),
            "11:50:01",
            _odds(99.0),
            "https://example.test/future",
            {"parser_version": TRIFECTA_PARSER_VERSION},
        )
        insert_odds_snapshot(
            conn,
            RACE_ID,
            (target_300 - timedelta(seconds=66))
            .astimezone(timezone.utc)
            .isoformat(),
            "11:48:54",
            _odds(98.0),
            "https://example.test/stale",
            {"parser_version": TRIFECTA_PARSER_VERSION},
        )
        insert_odds_snapshot(
            conn,
            RACE_ID,
            (target_300 - timedelta(seconds=1))
            .astimezone(timezone.utc)
            .isoformat(),
            "11:49:59",
            {
                key: 97.0
                for key in TRIFECTA_COMBINATION_KEYS[:-1]
            },
            "https://example.test/incomplete",
            {"parser_version": TRIFECTA_PARSER_VERSION},
        )

        checkpoints = load_odds_checkpoints(
            conn,
            RACE_ID,
            max_snapshot_age_seconds=MARKET_MAX_SNAPSHOT_AGE_SECONDS,
        )

    assert set(checkpoints) == {
        str(offset) for offset in ODDS_CHECKPOINT_OFFSETS_SECONDS
    }
    assert checkpoints["300"]["snapshot_id"] == selected_ids[300]
    assert checkpoints["300"]["snapshot_id"] != future_id
    assert checkpoints["300"]["provenance"] == {
        "mode": "explicit_checkpoint",
        "observation_label": "T300",
        "event_id": f"{RACE_ID}-T300",
        "collection_target_offset_seconds": 300,
    }
    assert checkpoints["300"]["source_update_staleness_seconds"] == 7.0
    assert checkpoints["120"]["provenance"]["mode"] == (
        "timestamp_reconstructed"
    )
    assert checkpoints["120"]["source_update_staleness_seconds"] == 11.0
    required = {
        "odds",
        "market_probabilities",
        "snapshot_id",
        "captured_at",
        "target_offset_seconds",
        "captured_age_seconds",
        "source_update_time",
        "source_update_staleness_seconds",
        "provenance",
    }
    assert all(set(point) == required for point in checkpoints.values())


def test_as_of_helper_blocks_future_checkpoints_and_is_pure() -> None:
    checkpoints = {
        str(offset): {
            "target_offset_seconds": offset,
            "odds": {"1-2-3": float(offset)},
            "market_probabilities": {"1-2-3": 1.0},
            "provenance": {"mode": "test"},
        }
        for offset in ODDS_CHECKPOINT_OFFSETS_SECONDS
    }

    at_t300 = available_odds_checkpoints(
        checkpoints, as_of_offset_seconds=300
    )
    at_t120 = available_odds_checkpoints(
        checkpoints, as_of_offset_seconds=120
    )
    at_close = available_odds_checkpoints(
        checkpoints, as_of_offset_seconds=0
    )

    assert list(at_t300) == ["300"]
    assert list(at_t120) == ["300", "120"]
    assert list(at_close) == ["300", "120", "60", "30", "10"]
    at_t300["300"]["odds"]["1-2-3"] = -1.0
    assert checkpoints["300"]["odds"]["1-2-3"] == 300.0


def test_raw_age_conflict_is_excluded_and_wrong_label_is_reconstructed() -> None:
    captured = BETTING_DEADLINE - timedelta(seconds=305)
    base = {
        "snapshot_id": 1,
        "captured_at": captured.isoformat(),
        "betting_deadline_at": BETTING_DEADLINE.isoformat(),
        "source_update_time": "11:49:45",
        "odds": _odds(20.0),
    }
    diagnostics: dict[str, int] = {}
    conflict = {
        **base,
        "raw_json": {
            "_collection": {
                "target_offset_seconds": 300,
                "observation_label": "T300",
                "captured_age_seconds": 299.0,
            }
        },
    }
    assert normalize_odds_checkpoint(
        conflict,
        target_offset_seconds=300,
        diagnostics=diagnostics,
    ) is None
    assert diagnostics == {"metadata_conflict": 1}

    wrong_label = {
        **base,
        "raw_json": {
            "_collection": {
                "target_offset_seconds": 120,
                "observation_label": "T120",
                "captured_age_seconds": 120.0,
            }
        },
    }
    reconstructed = normalize_odds_checkpoint(
        wrong_label,
        target_offset_seconds=300,
        diagnostics=diagnostics,
    )
    assert reconstructed is not None
    assert reconstructed["captured_age_seconds"] == 305.0
    assert reconstructed["provenance"]["mode"] == "timestamp_reconstructed"


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakePostgresql:
    dialect = "postgresql"

    def __init__(self, rows):
        self.rows = rows
        self.query = ""
        self.params = []

    def execute(self, query, params):
        self.query = query
        self.params = list(params)
        return _Rows(self.rows)


def test_postgresql_prefetch_uses_second_targets_and_strict_window() -> None:
    rows = []
    for index, offset in enumerate(ODDS_CHECKPOINT_OFFSETS_SECONDS):
        target = BETTING_DEADLINE - timedelta(seconds=offset)
        captured = target - timedelta(seconds=5)
        for combination in TRIFECTA_COMBINATION_KEYS:
            rows.append(
                {
                    "race_id": RACE_ID,
                    "target_offset_seconds": offset,
                    "snapshot_id": 100 + index,
                    "captured_at": captured.isoformat(),
                    "source_update_time": "11:49:45",
                    "raw_json": {
                        "_collection": {
                            "target_offset_seconds": offset,
                            "observation_label": f"T{offset}",
                            "captured_age_seconds": (
                                BETTING_DEADLINE - captured
                            ).total_seconds(),
                            "source_update_staleness_seconds": 10.0,
                        }
                    },
                    "odds_deadline_at": target.isoformat(),
                    "betting_deadline_at": BETTING_DEADLINE.isoformat(),
                    "combination": combination,
                    "odds": 20.0 + index,
                }
            )
    conn = _FakePostgresql(rows)

    prefetched = prefetch_trifecta_snapshots(
        conn,
        target_ids={RACE_ID},
        max_snapshot_age_seconds=65.0,
    )

    assert prefetched is not None
    assert set(prefetched[RACE_ID][PREFETCH_CHECKPOINTS_KEY]) == {
        "300",
        "120",
        "60",
        "30",
        "10",
    }
    assert prefetched[RACE_ID][5]["snapshot_id"] == 100
    assert "targets.target_offset_seconds" in conn.query
    assert "* INTERVAL '1 second'" in conn.query
    assert "COUNT(*)" in conn.query
    assert ") = 120" in conn.query
    assert "<=" in conn.query
    assert ">=" in conn.query
    assert conn.params[1] == 65.0


def test_race_contract_keeps_local_and_official_closing_separate(
    monkeypatch, tmp_path
) -> None:
    database = tmp_path / "race-contract.sqlite"
    init_db(database)
    with connection(database) as conn:
        _insert_completed_race(conn)
        for index, offset in enumerate(ODDS_CHECKPOINT_OFFSETS_SECONDS):
            _insert_checkpoint(
                conn,
                offset=offset,
                value=11.0 + index,
            )
        _insert_checkpoint(
            conn,
            offset=0,
            value=20.0,
        )
        _store_official_archive(conn, value=30.0)

        uniform = {
            combination: 1.0 / 120.0
            for combination in TRIFECTA_COMBINATION_KEYS
        }
        feature_rows = [
            {
                "meta": {
                    "race_id": RACE_ID,
                    "race_date": "2026-07-22",
                    "jcd": "01",
                    "rno": 1,
                    "lane": lane,
                }
            }
            for lane in range(1, 7)
        ]
        monkeypatch.setattr(
            "boatrace_ai.listwise.market_calibration."
            "iter_scored_artifact_feature_rows",
            lambda *_args, **_kwargs: iter([(feature_rows, uniform)]),
        )
        races, dataset = score_real_odds_races(
            conn,
            artifact={
                "classifier": object(),
                "hasher": object(),
                "model_kind": "linear",
                "trained_through": (
                    "2026-07-17-24-12",
                    "2026-07-17",
                    "24",
                    12,
                ),
            },
            from_date="2026-07-22",
            through_date="2026-07-22",
        )

    assert len(races) == 1
    race = races[0]
    assert set(race["closing_odds"].values()) == {20.0}
    assert set(race["official_closing_odds"].values()) == {30.0}
    assert race["official_closing_source"] == SOURCE_KEY
    assert race["official_closing_provenance"]["payload_sha256"] == "a" * 64
    assert race["betting_deadline_at"] == BETTING_DEADLINE.isoformat()
    assert race["decision_lead_seconds"] == 300
    assert race["odds_deadline_at"] < race["betting_deadline_at"]
    assert len(race["odds_checkpoints"]) == 5
    assert dataset["official_closing_odds_races"] == 1
    assert dataset["primary_official_closing_odds_races"] == 0
    assert dataset["fallback_mirror_closing_odds_races"] == 1
    assert dataset["odds_checkpoint_metadata_conflicts"] == 0


def test_primary_official_closing_odds_override_mirror(tmp_path) -> None:
    database = tmp_path / "closing-priority.sqlite"
    init_db(database)
    with connection(database) as conn:
        _insert_completed_race(conn)
        _store_official_archive(conn, value=30.0, source_key=SOURCE_KEY)
        _store_official_archive(
            conn, value=40.0, source_key=OFFICIAL_SOURCE_KEY
        )

        prefetched = prefetch_official_closing_odds(
            conn, target_ids={RACE_ID}
        )

    race = prefetched[RACE_ID]
    assert race["official_closing_source"] == OFFICIAL_SOURCE_KEY
    assert set(race["official_closing_odds"].values()) == {40.0}
    assert race["official_closing_provenance"]["mode"] == (
        "primary_official_historical_closing"
    )


def test_checkpoint_and_archive_arrival_invalidate_scored_cache(
    tmp_path,
) -> None:
    database = tmp_path / "invalidation.sqlite"
    model_path = tmp_path / "model.joblib"
    cache_path = tmp_path / "races.joblib"
    model_path.write_bytes(b"model")
    init_db(database)

    with connection(database) as conn:
        _insert_completed_race(conn)
        _insert_checkpoint(conn, offset=300, value=11.0)
        initial_signature = odds_data_signature(
            conn,
            from_date="2026-07-22",
            through_date="2026-07-22",
        )
        initial_contract = scored_cache_contract(
            model_path=model_path,
            artifact={},
            from_date="2026-07-22",
            through_date="2026-07-22",
            max_snapshot_age_seconds=65.0,
            odds_signature=initial_signature,
        )
        write_scored_cache(
            cache_path,
            contract=initial_contract,
            races=[{"race_id": RACE_ID}],
            dataset={"eligible_real_odds_races": 1},
        )
        assert load_scored_cache(
            cache_path, contract=initial_contract
        ) is not None

        _insert_checkpoint(conn, offset=120, value=12.0)
        _insert_checkpoint(conn, offset=0, value=20.0)
        checkpoint_signature = odds_data_signature(
            conn,
            from_date="2026-07-22",
            through_date="2026-07-22",
        )
        checkpoint_contract = scored_cache_contract(
            model_path=model_path,
            artifact={},
            from_date="2026-07-22",
            through_date="2026-07-22",
            max_snapshot_age_seconds=65.0,
            odds_signature=checkpoint_signature,
        )
        assert load_scored_cache(
            cache_path, contract=checkpoint_contract
        ) is None
        assert checkpoint_signature["checkpoint_120_snapshot_count"] == 1
        assert checkpoint_signature["closing_snapshot_count"] == 1

        _store_official_archive(conn)
        archive_signature = odds_data_signature(
            conn,
            from_date="2026-07-22",
            through_date="2026-07-22",
        )
        assert archive_signature["archive_closing_count"] == 1
        assert archive_signature["archive_closing_odds_count"] == 120
        assert archive_signature != checkpoint_signature

        conn.execute(
            """
            UPDATE archive_closing_odds_snapshots
            SET fetched_at = ?, payload_sha256 = ?
            WHERE race_id = ? AND source_key = ?
            """,
            (
                "2026-07-24T00:00:00+00:00",
                "b" * 64,
                RACE_ID,
                SOURCE_KEY,
            ),
        )
        updated_signature = odds_data_signature(
            conn,
            from_date="2026-07-22",
            through_date="2026-07-22",
        )

    assert (
        updated_signature["archive_closing_update_fingerprint"]
        != archive_signature["archive_closing_update_fingerprint"]
    )
