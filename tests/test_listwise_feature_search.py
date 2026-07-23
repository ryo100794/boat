from __future__ import annotations

import json
from pathlib import Path

import pytest

from boatrace_ai.db import connection, init_db
from boatrace_ai.feature_tuning import FEATURE_GROUPS
from boatrace_ai.listwise.feature_search import (
    _candidate_key,
    _checkpoint_signature,
    _ordered_rows,
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


def test_variant_workers_cli_is_documented_and_bounded() -> None:
    parser = build_parser()
    assert parser.parse_args([]).variant_workers == 1
    assert "--variant-workers" in parser.format_help()
    assert "1-3" in parser.format_help()
    for value in ("0", "4"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--variant-workers", value])


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


def _search_args(root: Path, db: Path, *, workers: int):
    return build_parser().parse_args([
        "--db", str(db),
        "--output", str(root / "result.json"),
        "--cache-dir", str(root / "search-cache"),
        "--cache-write-mode", "never",
        "--selected-cache-dir", str(root / "selected-cache"),
        "--checkpoint", str(root / "checkpoint.json"),
        "--variant-workers", str(workers),
        "--n-features", "64",
        "--batch-races", "2",
        "--epochs", "1",
        "--targets", "winner",
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


def test_spawn_workers_are_deterministic_and_checkpoint_resumes(
    tmp_path: Path,
) -> None:
    db = tmp_path / "fixture.sqlite"
    _small_feature_db(db)
    args_one = _search_args(tmp_path / "workers-one", db, workers=1)
    args_two = _search_args(tmp_path / "workers-two", db, workers=2)
    with connection(db) as conn:
        result_one = search(conn, args=args_one)
    with connection(db) as conn:
        result_two = search(conn, args=args_two)

    assert _deterministic_result(result_one) == _deterministic_result(result_two)
    expected_order = [
        name for name, _dropped in feature_variants()
    ]
    assert [
        row["feature_variant"] for row in result_two["search_results"]
    ] == expected_order
    assert len(selected_cache_candidates(
        Path(args_two.selected_cache_dir),
        n_features=args_two.n_features,
    )) == 1

    resume_root = tmp_path / "resume"
    resume_args = _search_args(resume_root, db, workers=2)
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
        targets=("winner",),
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
