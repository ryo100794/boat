from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from boatrace_ai.db import connection, init_db
from boatrace_ai.hashed_feature_dataset import race_ids_sha256
from boatrace_ai.listwise import feature_search as feature_search_module
from boatrace_ai.listwise.combined_feature_search import (
    COMBINED_FEATURE_VARIANTS,
    RESEARCH_PARTITION_FEATURE_VARIANTS,
    build_parser,
    parse_combined_feature_variants,
)


def test_combined_search_can_keep_official_and_drop_speculative_research() -> None:
    assert ("drop_speculative_research", ("speculative_research",)) in (
        RESEARCH_PARTITION_FEATURE_VARIANTS
    )
    assert parse_combined_feature_variants("drop_speculative_research") == (
        ("drop_speculative_research", ("speculative_research",)),
    )
from boatrace_ai.listwise.feature_search import (
    _candidate_key,
    _canonical_sha256,
    _checkpoint_payload,
    _checkpoint_signature,
    _load_checkpoint,
    _ordered_rows,
    _persist_checkpoint_progress,
    _selected_row,
    feature_variants,
)


EXPECTED_COMBINED_VARIANTS = (
    ("drop_base_pastlog", ("base_pastlog",)),
    (
        "keep_card_identity_context",
        ("card_numeric", "card_relative", "research_correlates"),
    ),
    (
        "keep_card_numeric",
        ("card_identity_context", "card_relative", "research_correlates"),
    ),
    (
        "keep_card_numeric_without_raw_equipment_ids",
        (
            "card_identity_context",
            "card_relative",
            "raw_equipment_identifiers",
            "research_correlates",
        ),
    ),
    (
        "keep_card_relative",
        ("card_identity_context", "card_numeric", "research_correlates"),
    ),
    (
        "drop_base_pastlog_research_correlates",
        ("base_pastlog", "research_correlates"),
    ),
    (
        "drop_base_pastlog_speculative_research",
        ("base_pastlog", "speculative_research"),
    ),
    (
        "drop_research_correlates_rolling_history",
        ("research_correlates", "rolling_history"),
    ),
    (
        "drop_base_pastlog_rolling_history",
        ("base_pastlog", "rolling_history"),
    ),
)


def _source_snapshot(
    race_keys: list[tuple[str, str, str, int]],
    *,
    source_hash: str = "d" * 64,
) -> dict:
    identity = {
        "snapshot_version": 1,
        "race_count": len(race_keys),
        "race_universe_sha256": race_ids_sha256(race_keys),
        "source_watermark_sha256": source_hash,
        "trifecta_payouts_sha256": "e" * 64,
        "selected_cache_manifest_sha256": "f" * 64,
    }
    identity["snapshot_sha256"] = _canonical_sha256(identity)
    return identity


def _signature(*, variants=None, data_snapshot=None):
    race_keys = [
        ("race-a", "2026-07-22", "01", 1),
        ("race-b", "2026-07-23", "02", 2),
    ]
    return _checkpoint_signature(
        args=SimpleNamespace(
            as_of_date="2026-07-23",
            n_features=64,
            batch_races=2,
            epochs=1,
            learning_rate=0.02,
        ),
        race_keys=race_keys,
        train_end=1,
        selection_end=2,
        targets=("winner", "top3_pl"),
        alphas=(0.0001, 0.001),
        data_snapshot=data_snapshot or _source_snapshot(race_keys),
        variants=variants,
    )


def _row(
    name: str,
    dropped: tuple[str, ...],
    target: str,
    alpha: float,
) -> dict:
    return {
        "feature_variant": name,
        "drop_feature_groups": list(dropped),
        "target": target,
        "alpha": alpha,
        "entry_log_loss": 0.3,
        "ranking_log_loss": 1.3,
        "winner_top1_accuracy": 0.5,
        "trifecta_top5_hit_rate": 0.25,
        "training_history": [],
    }


def test_combined_variants_are_fixed_and_default_variants_are_unchanged() -> None:
    defaults_before = feature_variants()
    parser = build_parser()
    args = parser.parse_args([])

    assert COMBINED_FEATURE_VARIANTS == EXPECTED_COMBINED_VARIANTS
    assert len({drops for _name, drops in COMBINED_FEATURE_VARIANTS}) == 9
    assert ("base_pastlog", "research_correlates") in {
        drops for _name, drops in COMBINED_FEATURE_VARIANTS
    }
    assert ("base_pastlog", "speculative_research") in {
        drops for _name, drops in COMBINED_FEATURE_VARIANTS
    }
    assert ("research_correlates", "rolling_history") in {
        drops for _name, drops in COMBINED_FEATURE_VARIANTS
    }
    assert ("base_pastlog", "rolling_history") in {
        drops for _name, drops in COMBINED_FEATURE_VARIANTS
    }
    assert feature_variants() == defaults_before
    assert all(
        Path(name).name == name and ".." not in name
        for name, _drops in COMBINED_FEATURE_VARIANTS
    )
    assert "--variants" not in parser.format_help()
    assert args.variant_workers == 1
    assert args.candidate_workers == 1
    assert args.output.endswith("listwise_combined_feature_search_v1.json")
    assert args.cache_dir.endswith("listwise_combined_search_cache")


def test_combined_variant_subset_is_registered_and_ordered() -> None:
    selected = parse_combined_feature_variants(
        "keep_card_numeric,drop_base_pastlog,keep_card_numeric"
    )
    assert selected == (
        COMBINED_FEATURE_VARIANTS[2],
        COMBINED_FEATURE_VARIANTS[0],
    )


def test_combined_signature_is_separate_from_default_signature() -> None:
    default = _signature()
    combined = _signature(variants=COMBINED_FEATURE_VARIANTS)

    assert combined != default
    assert combined["feature_variants"] == [
        [name, list(dropped)] for name, dropped in COMBINED_FEATURE_VARIANTS
    ]
    assert default["feature_variants"] == [
        [name, list(dropped)] for name, dropped in feature_variants()
    ]


def test_combined_rows_use_canonical_variant_target_alpha_order() -> None:
    targets = ("winner", "top3_pl")
    alphas = (0.0001, 0.001)
    rows = [
        _row(name, dropped, target, alpha)
        for name, dropped in COMBINED_FEATURE_VARIANTS
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
        variants=COMBINED_FEATURE_VARIANTS,
    ) == rows


def test_combined_checkpoint_resumes_without_accepting_default_signature(
    tmp_path: Path,
) -> None:
    rows = [
        _row(name, dropped, "winner", 0.0001)
        for name, dropped in COMBINED_FEATURE_VARIANTS[:2]
    ]
    completed = {
        _candidate_key(row["feature_variant"], row["target"], row["alpha"]): row
        for row in reversed(rows)
    }
    combined_signature = _signature(variants=COMBINED_FEATURE_VARIANTS)
    checkpoint = tmp_path / "combined-checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            _checkpoint_payload(
                combined_signature,
                completed,
                targets=("winner", "top3_pl"),
                alphas=(0.0001, 0.001),
                variants=COMBINED_FEATURE_VARIANTS,
            )
        ),
        encoding="utf-8",
    )

    resumed = _load_checkpoint(checkpoint, combined_signature)

    assert list(resumed.values()) == rows
    assert _load_checkpoint(checkpoint, _signature()) == {}



def test_checkpoint_v1_without_source_identity_is_rejected(tmp_path: Path) -> None:
    signature = _signature(variants=COMBINED_FEATURE_VARIANTS)
    legacy = dict(signature)
    legacy["checkpoint_version"] = 1
    legacy.pop("source_data_snapshot")
    row = _row(*COMBINED_FEATURE_VARIANTS[0], "winner", 0.0001)
    checkpoint = tmp_path / "legacy-checkpoint.json"
    checkpoint.write_text(
        json.dumps({
            "signature": legacy,
            "progress": {
                "completed_candidates": 1,
                "total_candidates": 36,
                "completed_variants": 0,
                "total_variants": 9,
                "last_completed": row,
            },
            "search_results": [row],
        }),
        encoding="utf-8",
    )

    assert _load_checkpoint(checkpoint, signature) == {}


def test_same_race_ids_with_corrected_source_reject_old_candidates(
    tmp_path: Path,
) -> None:
    stored_signature = _signature(variants=COMBINED_FEATURE_VARIANTS)
    race_keys = [
        ("race-a", "2026-07-22", "01", 1),
        ("race-b", "2026-07-23", "02", 2),
    ]
    corrected_signature = _signature(
        variants=COMBINED_FEATURE_VARIANTS,
        data_snapshot=_source_snapshot(race_keys, source_hash="9" * 64),
    )
    row = _row(*COMBINED_FEATURE_VARIANTS[0], "winner", 0.0001)
    checkpoint = tmp_path / "corrected-source-checkpoint.json"
    checkpoint.write_text(
        json.dumps({
            "signature": stored_signature,
            "search_results": [row],
        }),
        encoding="utf-8",
    )

    assert len(_load_checkpoint(checkpoint, stored_signature)) == 1
    assert _load_checkpoint(checkpoint, corrected_signature) == {}


def test_combined_checkpoint_exposes_atomic_candidate_and_variant_progress(
    tmp_path: Path,
    capsys,
) -> None:
    targets = ("winner", "top3_pl")
    alphas = (0.0001, 0.001)
    signature = _signature(variants=COMBINED_FEATURE_VARIANTS)
    checkpoint = tmp_path / "combined-checkpoint.json"
    first = _row(*COMBINED_FEATURE_VARIANTS[0], "winner", 0.0001)
    completed = {
        _candidate_key("drop_base_pastlog", "winner", 0.0001): first,
    }

    _persist_checkpoint_progress(
        checkpoint,
        signature,
        completed,
        targets=targets,
        alphas=alphas,
        variants=COMBINED_FEATURE_VARIANTS,
        last_completed={
            "kind": "candidate",
            "feature_variant": "drop_base_pastlog",
            "target": "winner",
            "alpha": 0.0001,
        },
    )

    candidate_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert candidate_payload["progress"] == {
        "completed_candidates": 1,
        "total_candidates": 36,
        "completed_variants": 0,
        "total_variants": 9,
        "last_completed": {
            "kind": "candidate",
            "feature_variant": "drop_base_pastlog",
            "target": "winner",
            "alpha": 0.0001,
        },
    }
    assert not list(tmp_path.glob(".*.tmp"))

    for target in targets:
        for alpha in alphas:
            row = _row(*COMBINED_FEATURE_VARIANTS[0], target, alpha)
            completed[_candidate_key("drop_base_pastlog", target, alpha)] = row
    _persist_checkpoint_progress(
        checkpoint,
        signature,
        completed,
        targets=targets,
        alphas=alphas,
        variants=COMBINED_FEATURE_VARIANTS,
        last_completed={
            "kind": "variant",
            "feature_variant": "drop_base_pastlog",
        },
    )

    variant_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert variant_payload["progress"]["completed_candidates"] == 4
    assert variant_payload["progress"]["completed_variants"] == 1
    assert variant_payload["progress"]["last_completed"]["kind"] == "variant"
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["feature_search_progress"]["last_completed"]["kind"] for event in events] == [
        "candidate",
        "variant",
    ]


def test_combined_killed_run_resumes_to_uninterrupted_selection(tmp_path: Path) -> None:
    targets = ("winner", "top3_pl")
    alphas = (0.0001, 0.001)
    rows = [
        _row(name, dropped, target, alpha)
        for name, dropped in COMBINED_FEATURE_VARIANTS
        for target in targets
        for alpha in alphas
    ]
    uninterrupted = {
        _candidate_key(row["feature_variant"], row["target"], row["alpha"]): row
        for row in rows
    }
    signature = _signature(variants=COMBINED_FEATURE_VARIANTS)
    checkpoint = tmp_path / "combined-checkpoint.json"
    killed_after = 13
    partial = dict(list(uninterrupted.items())[:killed_after])
    _persist_checkpoint_progress(
        checkpoint,
        signature,
        partial,
        targets=targets,
        alphas=alphas,
        variants=COMBINED_FEATURE_VARIANTS,
        last_completed={"kind": "candidate", "sequence": killed_after},
    )

    resumed = _load_checkpoint(checkpoint, signature)
    assert len(resumed) == killed_after
    for key, row in list(uninterrupted.items())[killed_after:]:
        resumed[key] = row
        _persist_checkpoint_progress(
            checkpoint,
            signature,
            resumed,
            targets=targets,
            alphas=alphas,
            variants=COMBINED_FEATURE_VARIANTS,
            last_completed={"kind": "candidate", "key": key},
        )

    resumed_rows = _ordered_rows(
        _load_checkpoint(checkpoint, signature),
        targets=targets,
        alphas=alphas,
        variants=COMBINED_FEATURE_VARIANTS,
    )
    uninterrupted_rows = _ordered_rows(
        uninterrupted,
        targets=targets,
        alphas=alphas,
        variants=COMBINED_FEATURE_VARIANTS,
    )
    assert resumed_rows == uninterrupted_rows
    assert _selected_row(resumed_rows) == _selected_row(uninterrupted_rows)


def _small_combined_feature_db(path: Path) -> None:
    init_db(path)
    with connection(path) as conn:
        for index in range(10):
            race_id = f"combined-{index:02d}"
            race_date = f"2026-02-{index + 1:02d}"
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
                        2000 + lane,
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


def _combined_search_args(root: Path, db: Path):
    return build_parser().parse_args([
        "--db", str(db),
        "--output", str(root / "result.json"),
        "--cache-dir", str(root / "cache"),
        "--cache-write-mode", "never",
        "--checkpoint", str(root / "checkpoint.json"),
        "--candidate-workers", "1",
        "--n-features", "64",
        "--batch-races", "2",
        "--epochs", "1",
        "--targets", "winner,top3_pl",
        "--alphas", "0.0001",
        "--train-fraction", "0.5",
        "--selection-fraction", "0.8",
    ])


def _stable_result(result: dict) -> dict:
    return {
        key: result[key]
        for key in (
            "search_results",
            "selected",
            "holdout",
            "evaluation_race_set_sha256",
            "roi",
            "profit_yen",
            "stake_yen",
            "return_yen",
            "max_drawdown_yen",
            "daily",
        )
    }


def test_actual_combined_search_recovers_after_candidate_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db = tmp_path / "fixture.sqlite"
    _small_combined_feature_db(db)
    uninterrupted_args = _combined_search_args(tmp_path / "full", db)
    resumed_args = _combined_search_args(tmp_path / "resumed", db)
    with connection(db) as conn:
        uninterrupted = feature_search_module.search(
            conn,
            args=uninterrupted_args,
            variants=COMBINED_FEATURE_VARIANTS,
        )

    original = feature_search_module._persist_checkpoint_progress
    writes = 0

    def kill_after_atomic_write(*args, **kwargs):
        nonlocal writes
        original(*args, **kwargs)
        writes += 1
        if writes == 4:
            raise RuntimeError("simulated kill after durable candidate checkpoint")

    monkeypatch.setattr(
        feature_search_module,
        "_persist_checkpoint_progress",
        kill_after_atomic_write,
    )
    with connection(db) as conn, pytest.raises(RuntimeError, match="simulated kill"):
        feature_search_module.search(
            conn,
            args=resumed_args,
            variants=COMBINED_FEATURE_VARIANTS,
        )
    checkpoint = Path(resumed_args.checkpoint)
    durable = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert durable["progress"]["completed_candidates"] == 3
    assert durable["progress"]["completed_variants"] == 1

    monkeypatch.setattr(
        feature_search_module,
        "_persist_checkpoint_progress",
        original,
    )
    with connection(db) as conn:
        resumed = feature_search_module.search(
            conn,
            args=resumed_args,
            variants=COMBINED_FEATURE_VARIANTS,
        )

    assert _stable_result(resumed) == _stable_result(uninterrupted)
    assert not checkpoint.exists()
