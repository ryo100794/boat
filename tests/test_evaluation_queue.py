from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

import boatrace_ai.evaluation_queue as evaluation_queue

from boatrace_ai.evaluation_queue import (
    DEFAULT_WORK_TICKETS,
    JobDependencyUnavailable,
    ResourceSnapshot,
    TASK_PROFILES,
    build_command,
    claim_job,
    dedupe_key,
    defer_job,
    enqueue_job,
    ensure_schema,
    fail_job,
    prepare_standardized_workspace,
    result_decision,
    seed_default_jobs,
    seed_periodic_jobs,
    seed_work_tickets,
    summarize_result,
)
from boatrace_ai.listwise.conditional_order import (
    build_parser as conditional_parser,
)
from boatrace_ai.listwise.venue_conditional_order import (
    build_parser as venue_conditional_parser,
)


def _job(task_type: str, parameters: dict, *, job_id: int = 7) -> dict:
    return {
        "job_id": job_id,
        "status": "running",
        "task_type": task_type,
        "model_key": "candidate",
        "parameters": parameters,
    }


def test_dedupe_key_is_parameter_order_independent() -> None:
    assert dedupe_key("probe", "model", {"a": 1, "b": 2}) == dedupe_key(
        "probe", "model", {"b": 2, "a": 1}
    )


def test_market_curvature_command_uses_fixed_script_and_output(tmp_path) -> None:
    root = tmp_path / "boat"
    command, output = build_command(
        _job(
            "market_curvature",
            {"evaluation_date": "2026-07-22", "disagreement_clip": 2.0},
        ),
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )

    assert command[1] == str(root / "scripts/analyze_market_curvature.py")
    assert command[-1] == str(root / "data/models/evaluation_queue/job-00000007.json")
    assert output == root / "data/models/evaluation_queue/job-00000007.json"


def test_calibrated_mlp_recency_search_profile() -> None:
    assert TASK_PROFILES["calibrated_mlp_recency_search"] == {
        "category": "evaluation",
        "memory_mb": 16384,
        "disk_mb": 4096,
        "idle_cpu": 15.0,
        "max_parallel": 1,
    }


def test_repository_hygiene_profile_is_low_resource_and_serial() -> None:
    assert TASK_PROFILES["repository_hygiene"] == {
        "category": "maintenance",
        "memory_mb": 256,
        "disk_mb": 256,
        "idle_cpu": 3.0,
        "max_parallel": 1,
    }


def _write_standard_feature_artifact(
    root: Path,
    cache_dir: Path,
    *,
    variant: str = "drop_research_correlates",
    n_features: int = 4096,
    create_manifest: bool = True,
) -> Path:
    artifact = (
        root / "data/models/standardized_365d_v2/raw"
        / "listwise_feature_teacher.json"
    )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps({
        "selected": {"feature_variant": variant},
        "selected_cache_dir": str(cache_dir),
        "n_features": n_features,
    }), encoding="utf-8")
    cache_prefix = cache_dir / f"listwise_search_{n_features}_{variant}"
    if create_manifest:
        cache_dir.mkdir(parents=True, exist_ok=True)
        Path(str(cache_prefix) + ".manifest.json").write_text(
            "{}", encoding="utf-8"
        )
    return cache_prefix


def test_standardized_selected_cache_root_is_fixed() -> None:
    assert evaluation_queue.STANDARDIZED_SELECTED_CACHE_DIR == Path(
        "/tmp/boatrace-standardized-365d-v2"
    )


def test_conditional_payout_tail_profile_and_command_are_fixed(
    tmp_path,
    monkeypatch,
) -> None:
    root = tmp_path / "boat"
    python = root / ".venv/bin/python"
    cache_dir = tmp_path / "selected-standard-cache"
    monkeypatch.setattr(
        evaluation_queue,
        "STANDARDIZED_SELECTED_CACHE_DIR",
        cache_dir,
    )
    cache_prefix = _write_standard_feature_artifact(root, cache_dir)
    command, output = build_command(
        _job(
            "conditional_payout_tail",
            {
                "training_through": "2025-07-19",
                "evaluation_from": "2025-07-20",
                "evaluation_through": "2026-07-19",
                "timeout_seconds": 3600,
            },
        ),
        app_root=root,
        python=python,
        db="postgresql://test",
    )

    result = root / "data/models/evaluation_queue/job-00000007.json"
    assert TASK_PROFILES["conditional_payout_tail"] == {
        "category": "evaluation",
        "memory_mb": 12288,
        "disk_mb": 2048,
        "idle_cpu": 15.0,
        "max_parallel": 1,
    }
    assert command == [
        str(python),
        "-m",
        "boatrace_ai.listwise.conditional_order",
        "--db",
        "postgresql://test",
        "--cache-prefix",
        str(cache_prefix),
        "--baseline-model",
        str(root / "data/models/standardized_365d_v2/listwise_newton.joblib"),
        "--training-through",
        "2025-07-19",
        "--evaluation-from",
        "2025-07-20",
        "--evaluation-through",
        "2026-07-19",
        "--model-output",
        str(result.with_suffix(".joblib")),
        "--output",
        str(result),
        "--validation-days",
        "365",
        "--batch-races",
        "4000",
        "--payout-mean-corrections",
        "0.0",
        "0.5",
        "1.0",
        "--promote-legacy-cache",
    ]
    assert output == result


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("malformed", "incomplete or invalid"),
        ("incomplete", "incomplete or invalid"),
        ("unknown_variant", "unknown feature variant"),
        ("feature_range", "out of range"),
        ("cache_traversal", "must exactly match"),
    ],
)
def test_claimed_conditional_payout_fails_on_invalid_standard_artifact(
    tmp_path,
    monkeypatch,
    case,
    message,
) -> None:
    root = tmp_path / "boat"
    cache_dir = tmp_path / "selected-standard-cache"
    monkeypatch.setattr(
        evaluation_queue,
        "STANDARDIZED_SELECTED_CACHE_DIR",
        cache_dir,
    )
    artifact = (
        root / "data/models/standardized_365d_v2/raw"
        / "listwise_feature_teacher.json"
    )
    if case != "missing":
        artifact.parent.mkdir(parents=True, exist_ok=True)
        if case == "malformed":
            artifact.write_text("{", encoding="utf-8")
        else:
            payload = {
                "selected": {"feature_variant": "full"},
                "selected_cache_dir": str(cache_dir),
                "n_features": 4096,
            }
            if case == "incomplete":
                payload.pop("selected")
            elif case == "unknown_variant":
                payload["selected"]["feature_variant"] = "../full"
            elif case == "feature_range":
                payload["n_features"] = 999
            elif case == "cache_traversal":
                payload["selected_cache_dir"] = str(cache_dir / ".." / "escape")
            artifact.write_text(json.dumps(payload), encoding="utf-8")
            if case != "missing_manifest":
                prefix = cache_dir / "listwise_search_4096_full"
                cache_dir.mkdir(parents=True, exist_ok=True)
                Path(str(prefix) + ".manifest.json").write_text(
                    "{}", encoding="utf-8"
                )

    with pytest.raises(ValueError, match=message):
        build_command(
            _job(
                "conditional_payout_tail",
                {
                    "training_through": "2025-07-19",
                    "evaluation_from": "2025-07-20",
                    "evaluation_through": "2026-07-19",
                },
            ),
            app_root=root,
            python=root / ".venv/bin/python",
            db="postgresql://test",
        )


@pytest.mark.parametrize("missing", ["artifact", "manifest"])
def test_conditional_payout_defers_until_selected_cache_exists(
    tmp_path,
    monkeypatch,
    missing,
) -> None:
    root = tmp_path / "boat"
    cache_dir = tmp_path / "selected-standard-cache"
    monkeypatch.setattr(
        evaluation_queue,
        "STANDARDIZED_SELECTED_CACHE_DIR",
        cache_dir,
    )
    if missing == "manifest":
        _write_standard_feature_artifact(
            root,
            cache_dir,
            create_manifest=False,
        )

    with pytest.raises(JobDependencyUnavailable, match="not available yet"):
        build_command(
            _job(
                "conditional_payout_tail",
                {
                    "training_through": "2025-07-19",
                    "evaluation_from": "2025-07-20",
                    "evaluation_through": "2026-07-19",
                },
            ),
            app_root=root,
            python=root / ".venv/bin/python",
            db="postgresql://test",
        )


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        (
            {
                "training_through": "2025-07-19",
                "evaluation_from": "2025-07-21",
                "evaluation_through": "2026-07-19",
            },
            "adjacent",
        ),
        (
            {
                "training_through": "2025-07-19",
                "evaluation_from": "2025-07-20",
                "evaluation_through": "2025-07-19",
            },
            "chronological",
        ),
        (
            {
                "training_through": "2025-07-19",
                "evaluation_from": "2025-07-20",
                "evaluation_through": "2026-07-18",
            },
            "exactly 365 days",
        ),
        (
            {
                "training_through": "2025-07-19",
                "evaluation_from": "2025-07-20",
                "evaluation_through": "2026-07-19",
                "timeout_seconds": 299,
            },
            "timeout_seconds",
        ),
        (
            {
                "training_through": "2025-07-19",
                "evaluation_from": "2025-07-20",
                "evaluation_through": "2026-07-19",
                "command": "rm -rf /",
            },
            "unsupported",
        ),
        (
            {
                "training_through": "2025-07-19",
                "evaluation_from": "2025-07-20",
                "evaluation_through": "2026-07-19",
                "cache_prefix": "/tmp/untrusted",
            },
            "unsupported",
        ),
    ],
)
def test_conditional_payout_tail_rejects_invalid_parameters(
    tmp_path, parameters, message
) -> None:
    with pytest.raises(ValueError, match=message):
        build_command(
            _job("conditional_payout_tail", parameters),
            app_root=tmp_path,
            python=tmp_path / "python",
            db="postgresql://test",
        )


def test_conditional_payout_mean_correction_defaults_disable_double_correction() -> None:
    conditional = conditional_parser().parse_args(
        [
            "--cache-prefix", "cache",
            "--baseline-model", "baseline.joblib",
            "--training-through", "2025-07-19",
            "--evaluation-from", "2025-07-20",
            "--evaluation-through", "2026-07-19",
            "--model-output", "model.joblib",
            "--output", "result.json",
        ]
    )
    venue = venue_conditional_parser().parse_args(
        [
            "--baseline-model", "baseline.joblib",
            "--training-through", "2025-07-19",
            "--evaluation-from", "2025-07-20",
            "--evaluation-through", "2026-07-19",
            "--model-output", "model.joblib",
            "--output", "result.json",
        ]
    )

    assert conditional.payout_mean_corrections == [0.0]
    assert venue.payout_mean_corrections == [0.0]


def test_calibrated_mlp_recency_search_command_is_fixed(tmp_path) -> None:
    root = tmp_path / "boat"
    python = root / ".venv/bin/python"
    command, output = build_command(
        _job(
            "calibrated_mlp_recency_search",
            {
                "evaluation_date": "2026-07-22",
                "half_lives": "none,180,180.0,365",
                "calibration_days": 120,
            },
        ),
        app_root=root,
        python=python,
        db="postgresql://test",
    )

    assert command == [
        str(python),
        "-m",
        "boatrace_ai.recency_mlp_evaluation",
        "--db",
        "postgresql://test",
        "--output",
        str(root / "data/models/evaluation_queue/job-00000007.json"),
        "--evaluation-date",
        "2026-07-22",
        "--feature-cache",
        str(root / "data/models/calibrated_shadow_features_16384"),
        "--half-lives",
        "none,180,365",
        "--calibration-days",
        "120",
    ]
    assert output == root / "data/models/evaluation_queue/job-00000007.json"

    default_command, _ = build_command(
        _job("calibrated_mlp_recency_search", {"evaluation_date": "2026-07-22"}),
        app_root=root,
        python=python,
        db="postgresql://test",
    )
    assert default_command[-4:] == [
        "--half-lives",
        "none,180,365,730",
        "--calibration-days",
        "180",
    ]


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({}, "evaluation_date is required"),
        ({"evaluation_date": "2026-07-22", "half_lives": "none"}, "at least 2"),
        ({"evaluation_date": "2026-07-22", "half_lives": "none,29"}, "finite numbers"),
        ({"evaluation_date": "2026-07-22", "half_lives": "none,nan"}, "finite numbers"),
        ({"evaluation_date": "2026-07-22", "calibration_days": 29}, "calibration_days"),
        ({"evaluation_date": "2026-07-22", "timeout_seconds": 299}, "timeout_seconds"),
        ({"evaluation_date": "2026-07-22", "command": "rm -rf /"}, "unsupported"),
        ({"evaluation_date": "2026-07-22", "feature_cache": "/tmp/cache"}, "unsupported"),
    ],
)
def test_calibrated_mlp_recency_search_rejects_invalid_parameters(
    tmp_path, parameters, message
) -> None:
    with pytest.raises(ValueError, match=message):
        build_command(
            _job("calibrated_mlp_recency_search", parameters),
            app_root=tmp_path,
            python=tmp_path / "python",
            db="postgresql://test",
        )


def test_task_parameters_cannot_select_arbitrary_command(tmp_path) -> None:
    with pytest.raises(ValueError, match="unsupported task_type"):
        build_command(
            _job("shell", {"command": "rm -rf /"}),
            app_root=tmp_path,
            python=tmp_path / "python",
            db="postgresql://test",
        )


def test_feature_search_rejects_unregistered_target(tmp_path) -> None:
    with pytest.raises(ValueError, match="unsupported targets"):
        build_command(
            _job(
                "listwise_feature_search",
                {"targets": "future_result", "evaluation_date": "2026-07-22"},
            ),
            app_root=tmp_path,
            python=tmp_path / "python",
            db="postgresql://test",
        )


@pytest.mark.parametrize("parameter", ["variant_workers", "candidate_workers", "cache_dir"])
def test_feature_search_rejects_injected_worker_or_path(tmp_path, parameter) -> None:
    with pytest.raises(
        ValueError,
        match="unsupported listwise_feature_search parameters",
    ):
        build_command(
            _job(
                "listwise_feature_search",
                {"evaluation_date": "2026-07-22", parameter: 3},
            ),
            app_root=tmp_path,
            python=tmp_path / "python",
            db="postgresql://test",
        )


def test_fresh_work_ticket_seed_registers_feature_search_parallelization(
    tmp_path: Path,
) -> None:
    expected = next(
        row for row in DEFAULT_WORK_TICKETS if row[0] == "OPS-EVAL-PERF-001"
    )
    conn = sqlite3.connect(tmp_path / "fresh.sqlite")
    conn.execute(
        """
        CREATE TABLE work_tickets (
          ticket_key TEXT PRIMARY KEY,
          title TEXT NOT NULL,
          area TEXT NOT NULL,
          description TEXT NOT NULL,
          acceptance_criteria TEXT NOT NULL,
          priority INTEGER NOT NULL,
          status TEXT NOT NULL,
          progress INTEGER NOT NULL,
          source TEXT NOT NULL
        )
        """
    )
    try:
        assert seed_work_tickets(conn) == len(DEFAULT_WORK_TICKETS)
        assert seed_work_tickets(conn) == 0
        actual = conn.execute(
            """
            SELECT ticket_key, title, area, description, acceptance_criteria,
                   priority, status, progress
            FROM work_tickets
            WHERE ticket_key = 'OPS-EVAL-PERF-001'
            """
        ).fetchone()
    finally:
        conn.close()
    assert actual == expected
    assert expected[1:5] == (
        "特徴探索の並列化と再現性保証",
        "モデル基盤",
        "特徴バリアント生成を資源制約付きで並列化し、評価待ち時間を短縮する。GitHub Issue: https://github.com/ryo100794/boat/issues/1",
        "workers=1/2で候補順・selected・holdout hash・資金評価が一致し、checkpoint再開可能。Git commit SHAとDBイベントを記録し、リモートが同SHAで稼働する",
    )
    assert expected[6:] == ("in_progress", 35)


def test_default_work_tickets_include_sync_hygiene_and_model_followups() -> None:
    keys = {row[0] for row in DEFAULT_WORK_TICKETS}
    assert {
        "OPS-EVAL-MEM-001",
        "OPS-GITHUB-SYNC-001",
        "DOCS-HIERARCHY-001",
        "MODEL-FEATURE-COMBINE-001",
        "MODEL-PAYOUT-001",
        "MODEL-RECENCY-001",
        "MODEL-VENUE-001",
        "MODEL-SEGMENT-001",
        "UI-MODEL-DAILY-001",
    } <= keys
    memory_ticket = next(
        row for row in DEFAULT_WORK_TICKETS
        if row[0] == "OPS-EVAL-MEM-001"
    )
    assert memory_ticket[5:] == (98, "in_progress", 15)

    combined_ticket = next(
        row for row in DEFAULT_WORK_TICKETS
        if row[0] == "MODEL-FEATURE-COMBINE-001"
    )
    assert combined_ticket == (
        "MODEL-FEATURE-COMBINE-001",
        "Combined feature ablation and retraining",
        "Model",
        "Run selection-only search combining base_pastlog+research_correlates with inert series_cached/series_relative ablations",
        "Compare against single ablations on the same fixed 365-day holdout and evaluation axes without holdout leakage",
        88,
        "queued",
        10,
    )


def test_result_summary_and_decision_use_nested_evaluation_metrics() -> None:
    payload = {
        "status": "candidate_requires_new_day_confirmation",
        "incremental_confidence_pass": True,
        "momentum_newton_residual": {
            "metrics": {
                "evaluated_races": 136,
                "trifecta_log_loss": 3.84,
                "trifecta_top5_hit_rate": 0.31,
            }
        },
    }

    summary = summarize_result(payload)

    assert summary["evaluated_races"] == 136
    assert summary["trifecta_log_loss"] == 3.84
    assert result_decision("market_curvature", summary) == "confirm_on_new_holdout"


def test_result_summary_preserves_paired_payout_feature_comparison() -> None:
    summary = summarize_result({
        "model": "venue",
        "venue_conditional_order": {
            "trifecta_log_loss": 3.79,
            "trifecta_top5_hit_rate": 0.35,
        },
        "payout_feature_comparison": {
            "candidate_schema": "conditional_payout_interactions_v2",
            "legacy_schema": "conditional_payout_additive_v1",
            "candidate_bankroll": {"roi": 1.03},
            "legacy_bankroll": {"roi": 0.90},
            "confidence": {
                "roi_delta": 0.13,
                "roi_delta_ci95_lower": 0.02,
                "roi_delta_ci95_upper": 0.24,
                "probability_roi_delta_above_zero": 0.99,
            },
            "gate": {
                "pass": True,
                "roi_ci95_lower": 1.01,
                "roi_delta_ci95_lower": 0.02,
                "roi_pass": True,
                "profit_pass": True,
                "baseline_improved": True,
            },
        },
    })

    assert summary["payout_feature_candidate_roi"] == 1.03
    assert summary["trifecta_log_loss"] == 3.79
    assert summary["trifecta_top5_hit_rate"] == 0.35
    assert summary["payout_feature_legacy_roi"] == 0.90
    assert summary["payout_feature_roi_delta_ci95_lower"] == 0.02
    assert summary["payout_feature_probability_roi_delta_above_zero"] == 0.99
    assert summary["payout_feature_promotion_eligible"] is True
    assert summary["payout_feature_gate_roi_ci95_lower"] == 1.01
    assert summary["payout_feature_candidate_schema"].endswith("v2")
    assert (
        result_decision("venue_conditional_order", summary)
        == "payout_feature_promotion_candidate"
    )


def test_conditional_payout_tail_summary_respects_explicit_non_promotion() -> None:
    summary = summarize_result({
        "promotion_eligible": True,
        "roi": 1.50,
        "profit_yen": 50_000,
        "conditional_payout_walk_forward": {
            "promotion_eligible": False,
            "bankroll": {
                "roi": 1.08,
                "profit_yen": 8_000,
                "stake_yen": 100_000,
                "policy": {
                    "payout_tail_schema": "conditional_payout_tail_v1",
                    "payout_feature_schema": "conditional_payout_interactions_v2",
                },
            },
            "bankroll_confidence": {
                "roi_ci95_lower": 1.01,
                "probability_roi_above_one": 0.98,
            },
            "diagnostic_gate": {
                "pass": True,
                "roi_pass": True,
            },
        },
    })

    assert summary["payout_feature_candidate_roi"] == 1.08
    assert summary["payout_feature_candidate_profit_yen"] == 8_000
    assert summary["payout_feature_candidate_stake_yen"] == 100_000
    assert (
        summary["payout_feature_candidate_schema"]
        == "conditional_payout_tail_v1"
    )
    assert summary["payout_feature_roi_ci95_lower"] == 1.01
    assert summary["payout_feature_gate_pass"] is True
    assert summary["payout_feature_promotion_eligible"] is False
    assert (
        result_decision("conditional_payout_tail", summary)
        == "reject_or_research_only"
    )


def test_default_seed_contains_parameter_sweep(monkeypatch) -> None:
    calls = []

    def fake_enqueue(_conn, **kwargs):
        calls.append(kwargs)
        return len(calls)

    monkeypatch.setattr("boatrace_ai.evaluation_queue.enqueue_job", fake_enqueue)

    inserted = seed_default_jobs(object(), evaluation_date="2026-07-22")

    assert len(inserted) == 12
    standardized = [
        row for row in calls if row["task_type"] == "standardized_365d"
    ]
    assert standardized[0]["parameters"]["timeout_seconds"] == 86400
    assert standardized[0]["max_attempts"] == 3
    assert sum(row["task_type"] == "market_curvature" for row in calls) == 6
    assert sum(row["task_type"] == "listwise_feature_search" for row in calls) == 4
    combined = [row for row in calls if row["task_type"] == "combined_feature_search"]
    assert len(combined) == 1
    assert combined[0]["priority"] == 85
    assert combined[0]["parameters"]["n_features"] == 4096
    assert not any(
        row["task_type"] == "calibrated_mlp_recency_search" for row in calls
    )
    assert all(
        row["parameters"]["evaluation_date"] == "2026-07-22" for row in calls
    )


class _PeriodicScheduleConnection:
    def __init__(self):
        self.keys: set[str] = set()

    def execute(self, statement, parameters=()):
        sql = " ".join(statement.split())
        assert "dedupe_key = ?" in sql
        key, _task_type = parameters
        return _QueryResult({"count": int(key in self.keys)})


def test_periodic_seed_uses_low_backup_priority_and_skips_completed_bucket(
    monkeypatch,
) -> None:
    conn = _PeriodicScheduleConnection()
    calls = []

    def fake_enqueue(_conn, **kwargs):
        calls.append(kwargs)
        conn.keys.add(dedupe_key(
            kwargs["task_type"], kwargs["model_key"], kwargs["parameters"]
        ))
        return len(calls)

    monkeypatch.setattr(evaluation_queue, "enqueue_job", fake_enqueue)
    now = datetime(2026, 7, 23, 12, 34, tzinfo=timezone.utc)

    assert seed_periodic_jobs(conn, now=now) == [1, 2, 3]
    assert seed_periodic_jobs(conn, now=now) == []
    assert len(calls) == 3
    backup = next(row for row in calls if row["task_type"] == "gdrive_raw_archive")
    assert backup["priority"] == 10


def test_periodic_enqueue_retains_atomic_dedupe_conflict_guard() -> None:
    class RecordingConnection:
        def __init__(self):
            self.sql = ""

        def execute(self, statement, parameters=()):
            self.sql = " ".join(statement.split())
            return _QueryResult()

    conn = RecordingConnection()

    assert enqueue_job(
        conn,
        task_type="gdrive_raw_archive",
        model_key="raw-data",
        parameters={"schedule_bucket": "2026-07-23T12:30:00+00:00"},
    ) is None
    assert "ON CONFLICT(dedupe_key) DO NOTHING" in conn.sql


def test_leader_commits_maintenance_before_claim(monkeypatch, tmp_path) -> None:
    events = []
    connection_count = 0

    class Scope:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            events.append(f"enter:{self.name}")
            return object()

        def __exit__(self, exc_type, exc, traceback):
            events.append(f"commit:{self.name}")

    def fake_connection(_db):
        nonlocal connection_count
        names = ("startup", "maintenance", "claim")
        name = names[connection_count]
        connection_count += 1
        return Scope(name)

    monkeypatch.setattr(evaluation_queue, "connection", fake_connection)
    monkeypatch.setattr(evaluation_queue, "ensure_schema", lambda _conn: None)
    monkeypatch.setattr(evaluation_queue, "recover_worker_job", lambda *_a, **_k: 0)
    monkeypatch.setattr(evaluation_queue, "seed_work_tickets", lambda _conn: 0)
    monkeypatch.setattr(
        evaluation_queue,
        "requeue_stale_jobs",
        lambda *_a, **_k: events.append("requeue"),
    )
    monkeypatch.setattr(
        evaluation_queue,
        "seed_default_jobs",
        lambda *_a, **_k: events.append("seed-defaults"),
    )
    monkeypatch.setattr(
        evaluation_queue,
        "seed_periodic_jobs",
        lambda *_a, **_k: events.append("seed-periodic"),
    )
    resources = ResourceSnapshot(32000, 10000, 100.0, 16, 0.0)
    monkeypatch.setattr(evaluation_queue, "system_resources", lambda: resources)
    monkeypatch.setattr(
        evaluation_queue,
        "claim_job",
        lambda *_a, **_k: events.append("claim") or None,
    )
    monkeypatch.setattr(evaluation_queue.time, "monotonic", lambda: 10000.0)
    args = evaluation_queue.build_parser().parse_args([
        "run",
        "--db", "postgresql://test",
        "--app-root", str(tmp_path),
        "--python", "python",
        "--worker-id", "evaluator-00",
        "--seed-defaults",
        "--schedule-periodic",
        "--once",
    ])

    assert evaluation_queue.run_worker(args) == 0
    assert events.index("commit:maintenance") < events.index("enter:claim")
    assert events.index("seed-periodic") < events.index("commit:maintenance")
    assert events.index("enter:claim") < events.index("claim")


def test_supervisor_runs_four_postgresql_queue_workers() -> None:
    config = Path(
        "scripts/deployment/supervisor-boatrace-evaluation-runner.ini"
    ).read_text(encoding="utf-8")

    assert "boatrace_ai.evaluation_queue run" in config
    assert "numprocs=4" in config
    assert "--seed-defaults" in config
    assert "--vm-limit-gib 0" in config


def test_standardized_workspace_rotates_stale_protocol_metadata(tmp_path) -> None:
    current = tmp_path / "data/models/standardized_365d_v2"
    current.mkdir(parents=True)
    (current / "protocol.json").write_text(
        '{"as_of_date_jst":"2026-07-20"}', encoding="utf-8"
    )
    (current / "manifest.json").write_text('{"ready":true}', encoding="utf-8")

    prepare_standardized_workspace(tmp_path, evaluation_date="2026-07-22")

    assert not (current / "protocol.json").exists()
    archive = tmp_path / "data/models/evaluation_queue/standardized_history/2026-07-20"
    assert (archive / "protocol.json").is_file()
    assert (archive / "manifest.json").is_file()


def test_feature_search_profiles_fit_the_32gb_quota_and_migrate_old_defaults() -> None:
    assert TASK_PROFILES["standardized_365d"]["memory_mb"] == 14336
    assert TASK_PROFILES["listwise_feature_search"]["memory_mb"] == 14336
    assert TASK_PROFILES["combined_feature_search"] == {
        "category": "evaluation",
        "memory_mb": 14336,
        "disk_mb": 4096,
        "idle_cpu": 15.0,
        "max_parallel": 1,
    }

    class RecordingPostgres:
        dialect = "postgresql"

        def __init__(self):
            self.calls = []

        def execute(self, statement, params=()):
            self.calls.append((statement, params))

        def executescript(self, statement):
            self.calls.append((statement, ()))

    conn = RecordingPostgres()
    ensure_schema(conn)
    migration = next(
        (statement, params)
        for statement, params in conn.calls
        if "status = 'queued'" in statement
        and "min_free_memory_mb = ?" in statement
    )
    assert migration[1] == (
        14336,
        "standardized_365d",
        "listwise_feature_search",
        16384,
    )

class _QueryResult:
    def __init__(self, row=None):
        self.row = row

    def fetchone(self):
        return self.row


class _ClaimConnection:
    def __init__(self, state):
        self.state = state
        self.saved_timeouts = []

    def execute(self, statement, parameters=()):
        sql = " ".join(statement.split())
        if "pg_advisory_xact_lock" in sql:
            return _QueryResult()
        if "SELECT jobs.*" in sql:
            return _QueryResult(dict(self.state))
        if "UPDATE model_evaluation_jobs" in sql and "RETURNING *" in sql:
            saved = json.loads(parameters[1])
            self.saved_timeouts.append(saved["timeout_seconds"])
            self.state.update({
                "status": "running",
                "worker_id": parameters[0],
                "attempt": int(self.state["attempt"]) + 1,
                "parameters": saved,
                "error": None,
            })
            return _QueryResult(dict(self.state))
        if "INSERT INTO model_evaluation_job_runs" in sql:
            return _QueryResult()
        raise AssertionError(f"unexpected SQL: {sql}")


class _LifecycleConnection:
    def __init__(self, parent):
        self.parent = dict(parent)
        self.run = {
            "status": "running",
            "error": None,
        }

    def execute(self, statement, parameters=()):
        sql = " ".join(statement.split())
        if "UPDATE model_evaluation_jobs" in sql:
            if "max_attempts = max_attempts + 1" in sql:
                self.parent.update({
                    "status": "queued",
                    "max_attempts": int(self.parent["max_attempts"]) + 1,
                    "error": parameters[0],
                })
            else:
                self.parent.update({
                    "status": parameters[0],
                    "error": parameters[2],
                })
            return _QueryResult()
        if "UPDATE model_evaluation_job_runs" in sql:
            self.run.update({
                "status": "failed",
                "error": parameters[0],
            })
            return _QueryResult()
        raise AssertionError(f"unexpected SQL: {sql}")


def test_timeout_retry_doubles_once_when_job_387_is_next_claimed() -> None:
    state = {
        "job_id": 387,
        "task_type": "listwise_feature_search",
        "category": "evaluation",
        "model_key": "feature-search",
        "parameters": {"timeout_seconds": 28800},
        "status": "queued",
        "attempt": 1,
        "max_attempts": 3,
        "error": "TimeoutExpired: command timed out after 21600 seconds",
    }
    conn = _ClaimConnection(state)
    resources = ResourceSnapshot(
        available_memory_mb=32000,
        available_disk_mb=10000,
        idle_cpu_percent=100.0,
        cpu_count=16,
        load_1m=0.0,
    )

    claimed = claim_job(conn, worker_id="evaluator-00", resources=resources)

    assert claimed is not None
    assert claimed["parameters"]["timeout_seconds"] == 43200
    assert conn.saved_timeouts == [43200]

    state.update({
        "status": "queued",
        "max_attempts": 4,
        "error": None,
    })
    claimed_again = claim_job(
        conn,
        worker_id="evaluator-00",
        resources=resources,
    )

    assert claimed_again is not None
    assert claimed_again["parameters"]["timeout_seconds"] == 43200
    assert conn.saved_timeouts == [43200, 43200]


def test_timeout_retry_never_shortens_a_larger_configured_limit() -> None:
    parameters = evaluation_queue._timeout_retry_parameters(
        {"timeout_seconds": 86400},
        task_type="standardized_365d",
        previous_error="TimeoutExpired: command timed out after 21600 seconds",
    )

    assert parameters["timeout_seconds"] == 86400


def test_dependency_defer_preserves_job_1069_remaining_attempt() -> None:
    job = {
        "job_id": 1069,
        "attempt": 2,
        "max_attempts": 2,
    }
    conn = _LifecycleConnection(job)

    defer_job(
        conn,
        job=job,
        reason="selected standardized feature cache is not available yet",
    )

    assert conn.parent["status"] == "queued"
    assert conn.parent["max_attempts"] == 3
    assert conn.parent["max_attempts"] - job["attempt"] == 1
    assert conn.run["status"] == "failed"
    assert conn.parent["error"].startswith("dependency deferred:")
    assert conn.run["error"] == conn.parent["error"]


def test_invalid_artifact_uses_normal_terminal_failure() -> None:
    job = {
        "job_id": 1069,
        "attempt": 2,
        "max_attempts": 2,
    }
    conn = _LifecycleConnection(job)

    fail_job(
        conn,
        job=job,
        error="ValueError: standardized feature artifact is incomplete or invalid",
    )

    assert conn.parent["status"] == "failed"
    assert conn.parent["max_attempts"] == 2
    assert conn.run["status"] == "failed"
    assert conn.parent["error"].startswith("ValueError:")


class _ReprioritizeConnection:
    def __init__(self, *, job_status: str = "queued", ticket_exists: bool = True):
        self.job = {"job_id": 1069, "priority": 70, "status": job_status}
        self.ticket = (
            {"ticket_key": "MODEL-PAYOUT-001", "progress": 65}
            if ticket_exists
            else None
        )
        self.events = []

    def execute(self, statement, parameters=()):
        sql = " ".join(statement.split())
        if "UPDATE model_evaluation_jobs" in sql:
            priority, job_id = parameters
            if int(job_id) != self.job["job_id"] or self.job["status"] != "queued":
                return _QueryResult()
            self.job["priority"] = int(priority)
            return _QueryResult(dict(self.job))
        if "UPDATE work_tickets" in sql:
            if self.ticket is None or parameters[0] != self.ticket["ticket_key"]:
                return _QueryResult()
            self.ticket["progress"] = max(int(self.ticket["progress"]), 70)
            return _QueryResult(dict(self.ticket))
        if "INSERT INTO work_ticket_events" in sql:
            self.events.append(parameters)
            return _QueryResult()
        raise AssertionError(f"unexpected SQL: {sql}")


def test_reprioritize_job_is_bounded_audited_and_parser_exposed() -> None:
    args = evaluation_queue.build_parser().parse_args([
        "reprioritize",
        "--job-id", "1069",
        "--priority", "90",
        "--reason", "Run payout policy before recency search",
        "--ticket-key", "MODEL-PAYOUT-001",
    ])
    conn = _ReprioritizeConnection()

    result = evaluation_queue.reprioritize_job(
        conn,
        job_id=args.job_id,
        priority=args.priority,
        reason=args.reason,
        ticket_key=args.ticket_key,
    )

    assert result == {"job_id": 1069, "priority": 90, "status": "queued"}
    assert conn.ticket["progress"] == 70
    assert conn.events == [
        (
            "MODEL-PAYOUT-001",
            70,
            "Run payout policy before recency search",
        )
    ]


@pytest.mark.parametrize(
    ("job_id", "priority", "reason", "message"),
    [
        (0, 90, "reason", "job_id"),
        (1069, -1, "reason", "priority"),
        (1069, 1001, "reason", "priority"),
        (1069, 90, " ", "reason"),
        (1069, 90, "x" * 501, "reason"),
    ],
)
def test_reprioritize_job_rejects_unbounded_input(
    job_id, priority, reason, message
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluation_queue.reprioritize_job(
            _ReprioritizeConnection(),
            job_id=job_id,
            priority=priority,
            reason=reason,
        )


def test_reprioritize_job_requires_queued_job_and_known_ticket() -> None:
    with pytest.raises(ValueError, match="queued"):
        evaluation_queue.reprioritize_job(
            _ReprioritizeConnection(job_status="running"),
            job_id=1069,
            priority=90,
            reason="reason",
        )
    with pytest.raises(ValueError, match="unknown ticket"):
        evaluation_queue.reprioritize_job(
            _ReprioritizeConnection(ticket_exists=False),
            job_id=1069,
            priority=90,
            reason="reason",
            ticket_key="MODEL-PAYOUT-001",
        )


def test_combined_feature_search_command_is_fixed_and_isolated(tmp_path) -> None:
    root = tmp_path / "boat"
    python = root / ".venv/bin/python"
    command, output = build_command(
        _job(
            "combined_feature_search",
            {
                "evaluation_date": "2026-07-23",
                "n_features": 4096,
                "targets": "winner,top3_pl",
                "alphas": "0.00001,0.0001",
                "timeout_seconds": 21600,
            },
            job_id=77,
        ),
        app_root=root,
        python=python,
        db="postgresql://test",
    )

    assert command[0:3] == [
        str(python),
        "-m",
        "boatrace_ai.listwise.combined_feature_search",
    ]
    assert command[command.index("--cache-dir") + 1] == (
        "/tmp/boatrace-evaluation/job-00000077/combined"
    )
    assert command[command.index("--selected-cache-dir") + 1] == str(
        root / "data/models/evaluation_cache/job-00000077-combined"
    )
    assert command[command.index("--variant-workers") + 1] == "1"
    assert command[command.index("--candidate-workers") + 1] == "2"
    assert command[command.index("--as-of-date") + 1] == "2026-07-23"
    assert output == root / "data/models/evaluation_queue/job-00000077.json"
    assert result_decision("combined_feature_search", {"roi": 0.8}) == (
        "refine_selected_candidate"
    )


@pytest.mark.parametrize("parameter", ["variant_workers", "candidate_workers", "cache_dir"])
def test_combined_feature_search_rejects_injected_worker_or_path(
    tmp_path, parameter
) -> None:
    with pytest.raises(
        ValueError,
        match="unsupported combined_feature_search parameters",
    ):
        build_command(
            _job(
                "combined_feature_search",
                {"evaluation_date": "2026-07-23", parameter: 2},
            ),
            app_root=tmp_path,
            python=tmp_path / "python",
            db="postgresql://test",
        )
