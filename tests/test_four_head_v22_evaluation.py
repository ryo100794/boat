from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from sklearn.feature_extraction import FeatureHasher

import boatrace_ai.listwise.four_head_v22_evaluation as evaluation


def _odds(offset: float = 0.0) -> dict[str, float]:
    return {
        combination: 20.0 + offset + index / 10.0
        for index, combination in enumerate(evaluation.COMBINATIONS)
    }


def _probabilities() -> dict[str, float]:
    values = np.arange(1.0, 121.0)
    values /= values.sum()
    return {
        combination: float(values[index])
        for index, combination in enumerate(evaluation.COMBINATIONS)
    }


def _feature_rows(race_id: str, race_date: str):
    return [
        {
            "meta": {
                "race_id": race_id,
                "race_date": race_date,
                "lane": lane,
                "rank": lane,
            },
            "features": {
                "lane": lane,
                "local_win_rate": 4.0 + lane / 10.0,
                "racer_class": "A1" if lane < 4 else "A2",
            },
        }
        for lane in range(1, 7)
    ]


def _artifact():
    return {
        "trained_through": ("race", "2026-06-30"),
        "hasher": FeatureHasher(n_features=64, input_type="dict"),
    }


def _install_sources(monkeypatch, dates: list[str], *, missing_odds: set[str] = set()):
    race_ids = [f"{date}-01-01" for date in dates]
    monkeypatch.setattr(
        evaluation,
        "_load_target_complete_race_ids",
        lambda conn, **kwargs: [
            (race_id, date, "01", 1) for race_id, date in zip(race_ids, dates)
        ],
    )
    target = datetime(2026, 7, 1, 3, 0, tzinfo=timezone.utc)
    snapshots = {}
    for index, race_id in enumerate(race_ids):
        if race_id in missing_odds:
            continue
        race_target = target + timedelta(days=index)
        snapshots[race_id] = {
            5: {
                "snapshot_id": index + 1,
                "captured_at": (race_target - timedelta(seconds=20)).isoformat(),
                "odds_deadline_at": race_target.isoformat(),
                "odds": _odds(index),
            }
        }
    monkeypatch.setattr(
        evaluation,
        "prefetch_trifecta_snapshots",
        lambda conn, **kwargs: snapshots,
    )
    monkeypatch.setattr(
        evaluation,
        "prefetch_official_closing_odds",
        lambda conn, **kwargs: {
            race_id: {"official_closing_odds": _odds(index + 0.5)}
            for index, race_id in enumerate(race_ids)
        },
    )

    def scored(conn, *, target_ids, artifact):
        for race_id, date in zip(race_ids, dates):
            if race_id in target_ids:
                yield _feature_rows(race_id, date), _probabilities()

    monkeypatch.setattr(evaluation, "iter_scored_artifact_feature_rows", scored)
    return race_ids, snapshots



def test_target_race_query_pushes_date_bounds_and_day_limit_into_sql():
    class Result:
        def fetchall(self):
            return [
                {
                    "race_id": "2026-07-01-01-01",
                    "race_date": "2026-07-01",
                    "jcd": "01",
                    "rno": 1,
                }
            ]

    class Connection:
        def __init__(self):
            self.sql = ""
            self.params = []

        def execute(self, sql, params):
            self.sql = sql
            self.params = list(params)
            return Result()

    conn = Connection()
    rows = evaluation._load_target_complete_race_ids(
        conn,
        training_from_date="2026-07-01",
        training_through_date="2026-07-03",
        outer_from_date="2026-07-04",
        outer_through_date="2026-07-04",
        max_races_per_day=4,
    )

    assert rows == [("2026-07-01-01-01", "2026-07-01", "01", 1)]
    assert "r.race_date BETWEEN ? AND ?" in conn.sql
    assert "WHERE day_sequence <= ?" in conn.sql
    assert conn.params == [
        "2026-07-01",
        "2026-07-03",
        "2026-07-04",
        "2026-07-04",
        4,
    ]

def test_loader_builds_120_choices_and_audits_snapshot_before_target(monkeypatch):
    race_ids, snapshots = _install_sources(
        monkeypatch, ["2026-07-01", "2026-07-02"]
    )
    loaded = evaluation.load_v22_evaluation_data(
        object(),
        source_artifact=_artifact(),
        training_from_date="2026-07-01",
        training_through_date="2026-07-01",
        outer_from_date="2026-07-02",
        outer_through_date="2026-07-02",
    )

    assert len(loaded.training_races) == len(loaded.outer_races) == 1
    for race in (*loaded.training_races, *loaded.outer_races):
        assert len(race.decision.features) == 120
        assert len(race.decision.current_odds) == 120
        assert len(race.outcome.closing_odds) == 120
        assert sorted(race.outcome.ranking_order) == list(range(120))
        assert race.outcome.ranking_order[0] == race.outcome.winner_index
    assert loaded.diagnostics["choice_count"] == 120
    assert loaded.diagnostics["eligible_coverage"] == 1.0
    for audit in loaded.decision_audit:
        assert datetime.fromisoformat(audit.captured_at) <= datetime.fromisoformat(
            audit.target_at
        )
        source = snapshots[audit.race_id][5]
        assert audit.captured_at == source["captured_at"]
        assert audit.target_at == source["odds_deadline_at"]
        assert audit.age_seconds == pytest.approx(20.0)
    assert {race.decision.race_id for race in loaded.training_races} == {race_ids[0]}
    assert {race.decision.race_id for race in loaded.outer_races} == {race_ids[1]}


def test_settlement_rank_never_enters_decision_features():
    rows = _feature_rows("2026-07-01-01-01", "2026-07-01")
    before = evaluation._choice_features(
        rows, _probabilities(), _artifact(), projection_dimensions=8
    )
    for row in rows:
        row["meta"]["rank"] = 7 - int(row["meta"]["lane"])
    after = evaluation._choice_features(
        rows, _probabilities(), _artifact(), projection_dimensions=8
    )

    assert before == after


def test_periods_and_source_artifact_fail_closed_on_outer_overlap(monkeypatch):
    _install_sources(monkeypatch, ["2026-07-01"])
    with pytest.raises(ValueError, match="strictly after"):
        evaluation.load_v22_evaluation_data(
            object(),
            source_artifact=_artifact(),
            training_from_date="2026-07-01",
            training_through_date="2026-07-02",
            outer_from_date="2026-07-02",
            outer_through_date="2026-07-03",
        )
    leaked = _artifact()
    leaked["trained_through"] = ("race", "2026-07-01")
    with pytest.raises(ValueError, match="overlaps V22"):
        evaluation.load_v22_evaluation_data(
            object(),
            source_artifact=leaked,
            training_from_date="2026-07-01",
            training_through_date="2026-07-01",
            outer_from_date="2026-07-02",
            outer_through_date="2026-07-02",
        )


def test_snapshot_after_target_is_excluded(monkeypatch):
    race_ids, snapshots = _install_sources(
        monkeypatch, ["2026-07-01", "2026-07-02"]
    )
    snapshot = snapshots[race_ids[0]][5]
    target = datetime.fromisoformat(snapshot["odds_deadline_at"])
    snapshot["captured_at"] = (target + timedelta(microseconds=1)).isoformat()

    loaded = evaluation.load_v22_evaluation_data(
        object(),
        source_artifact=_artifact(),
        training_from_date="2026-07-01",
        training_through_date="2026-07-01",
        outer_from_date="2026-07-02",
        outer_through_date="2026-07-02",
    )

    assert loaded.training_races == ()
    assert len(loaded.outer_races) == 1
    assert loaded.diagnostics["excluded_unsafe_t300"] == 1


def test_missing_t300_is_excluded_without_fallback(monkeypatch):
    race_ids, _snapshots = _install_sources(
        monkeypatch,
        ["2026-07-01", "2026-07-02"],
        missing_odds={"2026-07-02-01-01"},
    )
    fallback_calls = []
    monkeypatch.setattr(
        evaluation,
        "latest_trifecta_odds_before_deadline",
        lambda *args, **kwargs: fallback_calls.append((args, kwargs)),
    )
    loaded = evaluation.load_v22_evaluation_data(
        object(),
        source_artifact=_artifact(),
        training_from_date="2026-07-01",
        training_through_date="2026-07-01",
        outer_from_date="2026-07-02",
        outer_through_date="2026-07-02",
    )

    assert [race.decision.race_id for race in loaded.training_races] == [race_ids[0]]
    assert loaded.outer_races == ()
    assert loaded.diagnostics["excluded_missing_t300"] == 1
    assert loaded.diagnostics["eligible_coverage"] == 0.5
    assert fallback_calls == []


def test_db_backed_smoke_entrypoint_fits_then_evaluates_outer_only(monkeypatch):
    dates = ["2026-07-01", "2026-07-02", "2026-07-03", "2026-07-04"]
    race_ids, _snapshots = _install_sources(monkeypatch, dates)
    result = evaluation.run_v22_smoke_evaluation(
        object(),
        source_artifact=_artifact(),
        training_from_date="2026-07-01",
        training_through_date="2026-07-03",
        outer_from_date="2026-07-04",
        outer_through_date="2026-07-04",
        minimum_inner_training_dates=1,
        minimum_purchase_training_dates=1,
    )

    assert result["model_key"] == "four_head_nested_v22"
    assert result["coverage"]["training_races"] == 3
    assert result["coverage"]["outer_races"] == 1
    assert result["evaluation"]["races"] == 1
    assert result["evaluation"]["outer_outcomes_used_for_fit_or_selection"] is False
    assert result["evaluation"]["production_bankroll_evaluated"] is False
    assert race_ids[-1] == result["decision_audit"][-1]["race_id"]
