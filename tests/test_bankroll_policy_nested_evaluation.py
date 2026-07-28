from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, timedelta

import numpy as np
import pytest
from scipy import sparse

from boatrace_ai.feature_schema import FEATURE_SCHEMA_VERSION
from boatrace_ai.hashed_feature_dataset import CACHE_VERSION, HashedRaceDataset
from boatrace_ai.listwise import bankroll_policy_nested_evaluation as nested
from boatrace_ai.listwise.bankroll_policy_walk_forward import (
    build_annual_walk_forward_folds,
)
from boatrace_ai.packed_bankroll import pack_candidates


def _dates(start: str, count: int) -> list[str]:
    first = date.fromisoformat(start)
    return [(first + timedelta(days=index)).isoformat() for index in range(count)]


def _dataset(days: int = 10, races_per_day: int = 2) -> HashedRaceDataset:
    keys = []
    for day in _dates("2020-01-01", days):
        for rno in range(1, races_per_day + 1):
            keys.append((f"{day}-01-{rno:02d}", day, "01", rno))
    return HashedRaceDataset(
        matrix=sparse.csr_matrix((len(keys) * 6, 4), dtype=np.float64),
        race_keys=keys,
        ranks=np.tile(np.arange(1, 7, dtype=np.int8), (len(keys), 1)),
        n_features=4,
        drop_feature_groups=(),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
    )


def _policy() -> dict:
    return {
        "daily_budget_yen": 10_000,
        "ev_threshold": 1.2,
        "payout_prior_weight": 30.0,
    }


def _packed_for_rows(rows_by_race):
    candidates = {}
    evaluated = {}
    for rows in rows_by_race.values():
        race_date = rows[0]["race_date"]
        evaluated[race_date] = evaluated.get(race_date, 0) + 1
        candidates.setdefault(race_date, []).append({
            "race_id": rows[0]["race_id"],
            "estimated_odds": 4.0,
            "estimated_ev": 1.2,
            "probability": 0.3,
            "actual_payout_yen": 400,
            "hit": True,
        })
    return pack_candidates(candidates, evaluated)


def test_fold_builder_isolates_model_and_payout_training_boundaries(
    monkeypatch,
) -> None:
    dataset = _dataset()
    boundaries = build_annual_walk_forward_folds(
        _dates("2020-01-01", 10),
        selection_days=4,
        outer_days=1,
        folds=1,
        embargo_days=1,
    )
    selected_calls = []
    fit_calls = []
    evaluate_calls = []
    payout_calls = []

    def select(_dataset, **kwargs):
        selected_calls.append(kwargs["outer_train_end"])
        return ({
            "target": "top3_pl",
            "alpha": 1e-4,
            "inner_train_races": 6,
            "validation_races": 2,
        }, [{"target": "top3_pl", "alpha": 1e-4}])

    def fit(_dataset, **kwargs):
        fit_calls.append(kwargs["race_end"])
        return object(), [{"epoch": 1}]

    def evaluate(_dataset, _model, **kwargs):
        start, stop = kwargs["race_start"], kwargs["race_end"]
        evaluate_calls.append((start, stop))
        rows = {}
        for race_id, race_date, jcd, rno in _dataset.race_keys[start:stop]:
            rows[race_id] = [
                {
                    "race_id": race_id,
                    "race_date": race_date,
                    "jcd": jcd,
                    "rno": rno,
                    "lane": lane,
                    "rank": lane,
                    "probability": 1 / 6,
                }
                for lane in range(1, 7)
            ]
        return {"evaluated_races": stop - start}, rows

    def pack(rows, **kwargs):
        payout_calls.append(set(kwargs["train_races"]))
        return _packed_for_rows(rows)

    monkeypatch.setattr(nested, "nested_select_candidate", select)
    monkeypatch.setattr(nested, "_fit_selected_model", fit)
    monkeypatch.setattr(nested, "evaluate_range", evaluate)
    monkeypatch.setattr(nested, "packed_candidates_from_rows", pack)

    fold_inputs, audits = nested.build_fold_inputs(
        dataset,
        boundaries=boundaries,
        payouts={},
        targets=("winner", "top3_pl"),
        alphas=(1e-5, 1e-4),
        base_policy=_policy(),
        learning_rate=0.02,
        epochs=2,
        batch_races=4,
        validation_fraction=0.2,
        min_validation_races=2,
        provenance={"source_search_result_sha256": "a" * 64},
    )

    # Jan 9 is embargoed: no model or payout teacher may consume it.
    assert selected_calls == [8]
    assert fit_calls == [8, 16]
    assert evaluate_calls == [(8, 16), (18, 20)]
    assert payout_calls == [
        {row[0] for row in dataset.race_keys[:8]},
        {row[0] for row in dataset.race_keys[:16]},
    ]
    assert fold_inputs[0]["selection"].dates == (
        "2020-01-05", "2020-01-06", "2020-01-07", "2020-01-08"
    )
    assert fold_inputs[0]["holdout"].dates == ("2020-01-10",)
    assert audits[0]["indices"] == {
        "selection_train_end": 8,
        "selection_prediction_start": 8,
        "selection_prediction_end": 16,
        "holdout_train_end": 16,
        "holdout_prediction_start": 18,
        "holdout_prediction_end": 20,
    }
    assert audits[0]["payout_prior"]["selection_teacher_date_through"] == "2020-01-04"
    assert audits[0]["payout_prior"]["holdout_teacher_date_through"] == "2020-01-08"
    assert audits[0]["boundary_audit"]["holdout_training_stops_at_selection_end"] is True
    assert audits[0]["boundary_audit"]["passed"] is True


def test_fold_builder_rejects_prediction_rows_crossing_boundary(monkeypatch) -> None:
    dataset = _dataset(days=6, races_per_day=1)
    boundary = build_annual_walk_forward_folds(
        _dates("2020-01-01", 6), selection_days=4, outer_days=1, folds=1
    )
    monkeypatch.setattr(nested, "nested_select_candidate", lambda *args, **kwargs: ({
        "target": "winner", "alpha": 1e-4, "inner_train_races": 1,
        "validation_races": 1,
    }, []))
    monkeypatch.setattr(nested, "_fit_selected_model", lambda *args, **kwargs: (object(), []))

    def leaking_evaluate(_dataset, _model, **kwargs):
        race_id, _race_date, jcd, rno = _dataset.race_keys[kwargs["race_start"]]
        rows = [{
            "race_id": race_id, "race_date": "2099-01-01", "jcd": jcd,
            "rno": rno, "lane": lane, "rank": lane, "probability": 1 / 6,
        } for lane in range(1, 7)]
        return {}, {race_id: rows}

    monkeypatch.setattr(nested, "evaluate_range", leaking_evaluate)
    monkeypatch.setattr(
        nested, "packed_candidates_from_rows", lambda rows, **kwargs: _packed_for_rows(rows)
    )
    with pytest.raises(ValueError, match="leakage/boundary audit failed"):
        nested.build_fold_inputs(
            dataset,
            boundaries=boundary,
            payouts={},
            targets=("winner",),
            alphas=(1e-4,),
            base_policy=_policy(),
            learning_rate=0.02,
            epochs=1,
            batch_races=2,
            validation_fraction=0.2,
            min_validation_races=1,
            provenance={},
        )


def test_legacy_job_and_pre_v6_schema_are_rejected() -> None:
    with pytest.raises(ValueError, match="legacy job 3995"):
        nested._reject_legacy_source(
            {"provenance": {"source_job_id": 3995}}, explicit_source_job_id=None
        )
    with pytest.raises(ValueError, match="schema <6"):
        nested._require_schema_v6({
            "feature_schema_version": "pastlog-listwise-hashed-v5-low-coverage-guard",
            "hashed_cache_version": CACHE_VERSION,
        })
    with pytest.raises(ValueError, match="cache schema"):
        nested._require_schema_v6({
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "hashed_cache_version": 1,
        })


def test_five_fold_shortage_is_explicit_research_only() -> None:
    boundaries, mode = nested._build_available_boundaries(
        _dates("2020-01-01", 7),
        folds=5,
        selection_days=4,
        outer_days=1,
        embargo_days=0,
        allow_research_three_folds=True,
    )
    assert len(boundaries) == 3
    assert mode == "research_only_three_folds_insufficient_for_five"

    with pytest.raises(ValueError, match="at least 9 unique dates"):
        nested._build_available_boundaries(
            _dates("2020-01-01", 7),
            folds=5,
            selection_days=4,
            outer_days=1,
            embargo_days=0,
            allow_research_three_folds=False,
        )


def test_run_records_source_search_cache_and_candidate_hashes(
    tmp_path, monkeypatch,
) -> None:
    dataset = _dataset(days=9, races_per_day=1)
    search = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "hashed_cache_version": CACHE_VERSION,
        "race_date_through": "2020-01-09",
        "races": 9,
        "train_races": 5,
        "selection_races": 2,
        "holdout_races": 2,
        "n_features": 4,
        "selected": {"drop_feature_groups": []},
        "teacher_targets": ["winner"],
        "alphas": [0.0001],
    }
    search_path = tmp_path / "search.json"
    search_bytes = json.dumps(search, sort_keys=True).encode()
    search_path.write_bytes(search_bytes)
    output = tmp_path / "result.json"
    cache_hashes = {
        "cache_manifest_sha256": "b" * 64,
        "cache_bundle_sha256": "c" * 64,
    }
    candidate_hash = "d" * 64

    monkeypatch.setattr(nested, "load_complete_race_ids", lambda conn: dataset.race_keys)
    monkeypatch.setattr(nested, "validate_search_race_universe", lambda *args: None)
    monkeypatch.setattr(nested, "load_hashed_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(nested, "_cache_metadata", lambda prefix: cache_hashes)
    monkeypatch.setattr(nested, "_load_trifecta_payouts", lambda conn: {})
    monkeypatch.setattr(nested, "build_fold_inputs", lambda *args, **kwargs: (
        [{"fold": index, "selection": object(), "holdout": object()}
         for index in range(1, 6)],
        [{"fold": index, "boundary_audit": {"passed": True}}
         for index in range(1, 6)],
    ))
    monkeypatch.setattr(nested, "evaluate_annual_walk_forward", lambda *args, **kwargs: {
        "policy_candidates_sha256": candidate_hash,
        "promotion_eligible": True,
    })
    monkeypatch.setattr(nested, "_write_json_atomic", lambda path, result: path.write_text(
        json.dumps(result), encoding="utf-8"
    ))

    args = argparse.Namespace(
        search_result=str(search_path), cache_prefix=str(tmp_path / "cache"),
        output=str(output), source_job_id=5000, folds=5,
        selection_days=4, outer_days=1, embargo_days=0,
        allow_research_three_folds=False, targets=None, alphas=None,
        daily_budget_yen=10_000, ev_threshold=1.2, learning_rate=0.02,
        epochs=1, batch_races=2, validation_fraction=0.2,
        min_validation_races=1, candidate_count=2, finalists=1,
        selection_bootstrap_samples=10, aggregate_bootstrap_samples=10,
        seed=7,
    )
    result = nested.run(object(), args=args)

    provenance = result["provenance"]
    assert result["fold_count"] == 5
    assert result["research_only"] is False
    assert result["promotion_eligible"] is True
    assert provenance["source_job_id"] == 5000
    assert provenance["source_race_universe_sha256"]
    assert provenance["search_result_sha256"] == hashlib.sha256(search_bytes).hexdigest()
    assert provenance["cache_manifest_sha256"] == "b" * 64
    assert provenance["cache_bundle_sha256"] == "c" * 64
    assert provenance["policy_candidates_sha256"] == candidate_hash
    assert json.loads(output.read_text())["provenance"] == provenance


def test_three_fold_run_cannot_promote(tmp_path, monkeypatch) -> None:
    dataset = _dataset(days=7, races_per_day=1)
    search = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "hashed_cache_version": CACHE_VERSION,
        "race_date_through": "2020-01-07",
        "races": 7,
        "train_races": 3,
        "selection_races": 2,
        "holdout_races": 2,
        "n_features": 4,
        "selected": {"drop_feature_groups": []},
    }
    path = tmp_path / "search.json"
    path.write_text(json.dumps(search), encoding="utf-8")
    monkeypatch.setattr(nested, "load_complete_race_ids", lambda conn: dataset.race_keys)
    monkeypatch.setattr(nested, "validate_search_race_universe", lambda *args: None)
    monkeypatch.setattr(nested, "load_hashed_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(nested, "_cache_metadata", lambda prefix: {
        "cache_manifest_sha256": "b" * 64, "cache_bundle_sha256": "c" * 64,
    })
    monkeypatch.setattr(nested, "_load_trifecta_payouts", lambda conn: {})
    monkeypatch.setattr(nested, "build_fold_inputs", lambda *args, **kwargs: (
        [{"fold": index} for index in range(1, 4)], []
    ))
    monkeypatch.setattr(nested, "evaluate_annual_walk_forward", lambda *args, **kwargs: {
        "policy_candidates_sha256": "d" * 64, "promotion_eligible": True,
    })
    monkeypatch.setattr(nested, "_write_json_atomic", lambda *args: None)
    args = argparse.Namespace(
        search_result=str(path), cache_prefix=str(tmp_path / "cache"),
        output=str(tmp_path / "out.json"), source_job_id=5001, folds=5,
        selection_days=4, outer_days=1, embargo_days=0,
        allow_research_three_folds=True, targets=("winner",), alphas=(1e-4,),
        daily_budget_yen=10_000, ev_threshold=1.2, learning_rate=0.02,
        epochs=1, batch_races=2, validation_fraction=0.2,
        min_validation_races=1, candidate_count=2, finalists=1,
        selection_bootstrap_samples=10, aggregate_bootstrap_samples=10, seed=7,
    )
    result = nested.run(object(), args=args)
    assert result["fold_count"] == 3
    assert result["research_only"] is True
    assert result["promotion_eligible"] is False
    assert result["evaluation"]["promotion_eligible"] is False
