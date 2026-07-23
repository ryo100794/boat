from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

from boatrace_ai.db import connection, init_db
from boatrace_ai.feature_tuning import FEATURE_GROUPS
from boatrace_ai.listwise.feature_search import (
    _candidate_key,
    _checkpoint_signature,
    _ordered_rows,
    _evaluate_variant,
    build_parser,
    day_boundary,
    feature_variants,
    search,
    selected_cache_candidates,
)


def test_feature_search_covers_full_and_each_single_group_ablation() -> None:
    variants = feature_variants()
    assert variants[0] == ("full", ())
    assert {drops[0] for _name, drops in variants[1:]} == set(FEATURE_GROUPS)
    assert all(len(drops) == 1 for _name, drops in variants[1:])


def test_day_boundary_never_splits_a_race_day() -> None:
    keys = [
        (f"r{index}", f"2026-01-{index // 3 + 1:02d}", "01", index + 1)
        for index in range(12)
    ]
    boundary = day_boundary(keys, 4)
    assert boundary == 6
    assert keys[boundary - 1][1] != keys[boundary][1]


def test_worker_clis_are_documented_and_bounded() -> None:
    parser = build_parser()
    assert parser.parse_args([]).variant_workers == 1
    assert "--variant-workers" in parser.format_help()
    assert parser.parse_args([]).candidate_workers == 1
    assert "--candidate-workers" in parser.format_help()
    for value in ("0", "2"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--variant-workers", value])
    for value in ("0", "5"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--candidate-workers", value])


def test_completed_candidates_are_restored_to_canonical_search_order() -> None:
    targets = ("winner", "top3_pl")
    alphas = (0.01, 0.1)
    rows = [
        {
            "feature_variant": variant_name,
            "target": target,
            "alpha": alpha,
        }
        for variant_name, _dropped in feature_variants()
        for target in targets
        for alpha in alphas
    ]
    completed = {
        _candidate_key(row["feature_variant"], row["target"], row["alpha"]): row
        for row in reversed(rows)
    }

    assert _ordered_rows(
        completed,
        targets=targets,
        alphas=alphas,
    ) == rows


def _small_feature_db(path: Path) -> None:
    init_db(path)
    with connection(path) as conn:
        for index in range(10):
            race_id = f"fixture-{index:02d}"
            race_date = f"2026-01-{index + 1:02d}"
            conn.execute(
                """
                INSERT INTO races(
                  race_id, race_date, jcd, venue_name, rno, status
                ) VALUES (?, ?, '01', 'fixture', 1, 'completed')
                """,
                (race_id, race_date),
            )
            for lane in range(1, 7):
                rank = (lane + index - 1) % 6 + 1
                conn.execute(
                    """
                    INSERT INTO entries(
                      race_id, lane, racer_no, racer_name, racer_class,
                      age, weight_kg, avg_st, national_win_rate,
                      national_2_rate, national_3_rate, local_win_rate,
                      local_2_rate, local_3_rate, motor_no, motor_2_rate,
                      motor_3_rate, boat_no, boat_2_rate, boat_3_rate
                    ) VALUES (
                      ?, ?, ?, ?, 'A1', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        race_id,
                        lane,
                        1000 + lane,
                        f"racer-{lane}",
                        20 + lane,
                        50.0 + lane,
                        0.10 + lane / 100,
                        5.0 + lane / 10,
                        40.0 + lane,
                        60.0 + lane,
                        4.0 + lane / 10,
                        35.0 + lane,
                        55.0 + lane,
                        lane,
                        30.0 + lane,
                        50.0 + lane,
                        lane,
                        31.0 + lane,
                        51.0 + lane,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO race_results(
                      race_id, lane, rank, course, start_timing
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (race_id, lane, rank, lane, 0.10 + lane / 100),
                )


def _search_args(root: Path, db: Path, *, candidate_workers: int):
    return build_parser().parse_args([
        "--db", str(db),
        "--output", str(root / "result.json"),
        "--cache-dir", str(root / "search-cache"),
        "--cache-write-mode", "never",
        "--selected-cache-dir", str(root / "selected-cache"),
        "--checkpoint", str(root / "checkpoint.json"),
        "--variant-workers", "1",
        "--candidate-workers", str(candidate_workers),
        "--n-features", "64",
        "--batch-races", "2",
        "--epochs", "1",
        "--targets", "winner,top3_pl",
        "--alphas", "0.0001",
        "--train-fraction", "0.5",
        "--selection-fraction", "0.8",
    ])


def _deterministic_result(result: dict) -> dict:
    return {
        "search_results": result["search_results"],
        "selected": result["selected"],
        "holdout": result["holdout"],
        "evaluation_race_set_sha256": result["evaluation_race_set_sha256"],
        "roi": result["roi"],
        "profit_yen": result["profit_yen"],
        "stake_yen": result["stake_yen"],
        "return_yen": result["return_yen"],
        "max_drawdown_yen": result["max_drawdown_yen"],
        "daily": result["daily"],
    }


def test_candidate_workers_are_deterministic_and_checkpoint_resumes(
    tmp_path: Path,
) -> None:
    db = tmp_path / "fixture.sqlite"
    _small_feature_db(db)
    args_one = _search_args(tmp_path / "workers-one", db, candidate_workers=1)
    args_two = _search_args(tmp_path / "workers-two", db, candidate_workers=2)
    with connection(db) as conn:
        result_one = search(conn, args=args_one)
    with connection(db) as conn:
        result_two = search(conn, args=args_two)

    assert _deterministic_result(result_one) == _deterministic_result(result_two)
    expected_order = [
        name for name, _dropped in feature_variants()
        for _target in ("winner", "top3_pl")
    ]
    assert [
        row["feature_variant"] for row in result_two["search_results"]
    ] == expected_order
    assert len(selected_cache_candidates(
        Path(args_two.selected_cache_dir),
        n_features=args_two.n_features,
    )) == 1

    resume_root = tmp_path / "resume"
    resume_args = _search_args(resume_root, db, candidate_workers=2)
    race_keys = [
        (f"fixture-{index:02d}", f"2026-01-{index + 1:02d}", "01", 1)
        for index in range(10)
    ]
    train_end = day_boundary(
        race_keys,
        int(len(race_keys) * resume_args.train_fraction),
    )
    selection_end = day_boundary(
        race_keys,
        int(len(race_keys) * resume_args.selection_fraction),
    )
    signature = _checkpoint_signature(
        args=args_one,
        race_keys=race_keys,
        train_end=train_end,
        selection_end=selection_end,
        targets=("winner", "top3_pl"),
        alphas=(0.0001,),
    )
    Path(resume_args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    Path(resume_args.checkpoint).write_text(
        json.dumps({
            "signature": signature,
            "search_results": [result_one["search_results"][0]],
        }),
        encoding="utf-8",
    )
    with connection(db) as conn:
        resumed = search(conn, args=resume_args)

    assert _deterministic_result(resumed) == _deterministic_result(result_one)
    assert not Path(resume_args.checkpoint).exists()


def test_variant_reuses_dataset_and_scaler_and_parallelizes_only_missing_candidates(
    monkeypatch,
    tmp_path: Path,
) -> None:
    dataset = SimpleNamespace(matrix=SimpleNamespace(nnz=321))
    load_count = 0
    scaler_count = 0
    trained: list[tuple[str, float]] = []
    shared_ids: list[tuple[int, int]] = []
    callback_rows: list[dict] = []
    active = 0
    maximum_active = 0
    both_started = threading.Event()
    lock = threading.Lock()
    scaler = object()

    def fake_load(*_args, **_kwargs):
        nonlocal load_count
        load_count += 1
        return dataset, "fixture"

    def fake_scaler(actual_dataset, **_kwargs):
        nonlocal scaler_count
        assert actual_dataset is dataset
        scaler_count += 1
        return scaler

    def fake_train(actual_dataset, *, target, alpha, scaler: object, **_kwargs):
        nonlocal active, maximum_active
        assert actual_dataset is dataset
        shared_ids.append((id(actual_dataset), id(scaler)))
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
            if active >= 2:
                both_started.set()
        assert both_started.wait(timeout=2)
        with lock:
            trained.append((target, alpha))
            active -= 1
        return SimpleNamespace(target=target, alpha=alpha), [{"epoch": 1.0}]

    def fake_evaluate(_dataset, model, **_kwargs):
        offset = 0.01 if model.target == "winner" else 0.02
        return {
            "evaluated_races": 2,
            "entry_log_loss": model.alpha + offset,
            "entry_brier": 0.1,
            "ranking_log_loss": model.alpha + offset,
            "winner_top1_accuracy": 0.5,
            "trifecta_top5_hit_rate": 0.25,
        }, {}

    monkeypatch.setattr(
        "boatrace_ai.listwise.feature_search.load_variant_dataset",
        fake_load,
    )
    monkeypatch.setattr(
        "boatrace_ai.listwise.feature_search.fit_scaler",
        fake_scaler,
    )
    monkeypatch.setattr(
        "boatrace_ai.listwise.feature_search.train_listwise_model",
        fake_train,
    )
    monkeypatch.setattr(
        "boatrace_ai.listwise.feature_search.evaluate_range",
        fake_evaluate,
    )
    resumed = {
        "feature_variant": "full",
        "drop_feature_groups": [],
        "target": "winner",
        "alpha": 0.1,
        "cache_source": "fixture",
        "matrix_nnz": 321,
        "training_history": [{"epoch": 1.0}],
        "entry_log_loss": 0.11,
        "ranking_log_loss": 0.11,
        "winner_top1_accuracy": 0.5,
        "trifecta_top5_hit_rate": 0.25,
    }
    request = {
        "race_keys": [],
        "cache_dir": str(tmp_path),
        "variant_name": "full",
        "dropped": (),
        "n_features": 64,
        "batch_races": 2,
        "write_cache": False,
        "train_end": 2,
        "selection_end": 4,
        "targets": ("winner", "top3_pl"),
        "alphas": (0.1, 0.2),
        "learning_rate": 0.02,
        "epochs": 1,
        "completed_rows": [resumed],
    }

    actual_dataset, payload = _evaluate_variant(
        None,
        request=request,
        candidate_workers=2,
        on_candidate_complete=callback_rows.append,
    )

    assert actual_dataset is dataset
    assert load_count == 1
    assert scaler_count == 1
    assert maximum_active == 2
    assert set(trained) == {
        ("winner", 0.2),
        ("top3_pl", 0.1),
        ("top3_pl", 0.2),
    }
    assert ("winner", 0.1) not in trained
    assert len(callback_rows) == 3
    assert set(shared_ids) == {(id(dataset), id(scaler))}
    assert [
        (row["target"], row["alpha"]) for row in payload["rows"]
    ] == [
        ("winner", 0.1),
        ("winner", 0.2),
        ("top3_pl", 0.1),
        ("top3_pl", 0.2),
    ]


def test_default_checkpoint_signature_remains_byte_for_byte_compatible() -> None:
    args = SimpleNamespace(
        as_of_date="2026-07-23",
        n_features=4096,
        batch_races=1000,
        epochs=2,
        learning_rate=0.02,
    )
    race_keys = [
        ("race-a", "2026-07-22", "01", 1),
        ("race-b", "2026-07-23", "02", 2),
    ]
    signature = _checkpoint_signature(
        args=args,
        race_keys=race_keys,
        train_end=1,
        selection_end=2,
        targets=("winner", "top3_pl"),
        alphas=(0.00001, 0.0001),
    )

    assert json.dumps(signature, separators=(",", ":")) == (
        '{"checkpoint_version":1,"cache_version":2,'
        '"feature_schema_version":"pastlog-listwise-hashed-v2-series-missing-safe",'
        '"as_of_date":"2026-07-23","race_count":2,'
        '"race_universe_sha256":'
        '"a5d59ddbba062a4884a2242737fca8bc14d2858a52d64cb1af19dddb3bd6bd23",'
        '"train_end":1,"selection_end":2,"n_features":4096,"batch_races":1000,'
        '"epochs":2,"learning_rate":0.02,"targets":["winner","top3_pl"],'
        '"alphas":[1e-05,0.0001],"feature_variants":[["full",[]],'
        '["drop_base_pastlog",["base_pastlog"]],'
        '["drop_research_correlates",["research_correlates"]],'
        '["drop_series_cached",["series_cached"]],'
        '["drop_series_relative",["series_relative"]],'
        '["drop_rolling_history",["rolling_history"]]]}'
    )
