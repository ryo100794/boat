from __future__ import annotations

import json
from pathlib import Path

import pytest

from boatrace_ai import evaluation_queue
from boatrace_ai.evaluation_queue import (
    ObsoleteJob,
    TASK_PROFILES,
    build_command,
)
from boatrace_ai.feature_schema import FEATURE_SCHEMA_VERSION
from boatrace_ai.listwise import bankroll_policy_evaluation
from boatrace_ai.listwise.bankroll_policy_evaluation import (
    _train_model,
    build_parser,
    flat_top_k_diagnostic,
    prior_selection_key,
)


def test_flat_top5_diagnostic_converts_hit_rate_to_realized_roi() -> None:
    rows = {
        "hit": [
            {"lane": lane, "probability": probability}
            for lane, probability in enumerate((0.40, 0.25, 0.15, 0.10, 0.06, 0.04), 1)
        ],
        "miss": [
            {"lane": lane, "probability": probability}
            for lane, probability in enumerate((0.40, 0.25, 0.15, 0.10, 0.06, 0.04), 1)
        ],
    }
    payouts = {
        "hit": {"combination": "1-2-3", "payout_yen": 1200},
        "miss": {"combination": "6-5-4", "payout_yen": 5000},
    }

    result = flat_top_k_diagnostic(rows, payouts=payouts)

    assert result["evaluated_races"] == 2
    assert result["tickets"] == 10
    assert result["hit_races"] == 1
    assert result["hit_rate"] == 0.5
    assert result["stake_yen"] == 1000
    assert result["return_yen"] == 1200
    assert result["roi"] == 1.2
    assert result["breakeven_average_hit_payout_yen"] == 1000


def _job(parameters):
    return {
        "job_id": 99,
        "task_type": "bankroll_policy_search",
        "model_key": "policy",
        "parameters": parameters,
    }


def _source(root, *, cache_prefix=None):
    result = root / "data/models/evaluation_queue/job-00003565.json"
    result.parent.mkdir(parents=True)
    cache = cache_prefix or (
        root
        / "data/models/evaluation_cache/job-00003565"
        / "listwise_search_8192_drop_research_correlates"
    )
    result.write_text(
        json.dumps({
            "selected_cache_prefix": str(cache),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
        }),
        encoding="utf-8",
    )
    return result, cache


def _standard_source(root, cache_dir):
    artifact = (
        root / "data/models/standardized_365d_v2/raw"
        / "listwise_feature_teacher.json"
    )
    artifact.parent.mkdir(parents=True)
    cache_prefix = cache_dir / "listwise_search_8192_drop_base_pastlog"
    cache_dir.mkdir(parents=True)
    Path(str(cache_prefix) + ".manifest.json").write_text(
        json.dumps({"feature_schema_version": FEATURE_SCHEMA_VERSION}),
        encoding="utf-8",
    )
    artifact.write_text(
        json.dumps({
            "selected": {"feature_variant": "drop_base_pastlog"},
            "selected_cache_dir": str(cache_dir),
            "selected_cache_prefix": str(cache_prefix),
            "n_features": 8192,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
        }),
        encoding="utf-8",
    )
    return artifact, cache_prefix


def test_bankroll_policy_search_profile_and_command(tmp_path) -> None:
    root = tmp_path / "boat"
    source, cache = _source(root)
    python = root / ".venv/bin/python"
    command, output = build_command(
        _job({
            "source_job_id": 3565,
            "learning_rate": 0.0075,
            "epochs": 3,
            "candidate_count": 24,
            "finalists": 6,
            "bootstrap_samples": 20000,
            "payout_prior_weights": "10,30,100",
            "timeout_seconds": 43200,
            "research_only": True,
        }),
        app_root=root,
        python=python,
        db="postgresql://test",
    )
    assert TASK_PROFILES["bankroll_policy_search"] == {
        "category": "evaluation",
        "memory_mb": 9216,
        "idle_cpu": 15.0,
        "max_parallel": 1,
        "disk_mb": 1024,
    }
    assert output == root / "data/models/evaluation_queue/job-00000099.json"
    assert command[:3] == [
        str(python),
        "-m",
        "boatrace_ai.listwise.bankroll_policy_evaluation",
    ]
    assert command[command.index("--search-result") + 1] == str(source)
    assert command[command.index("--cache-prefix") + 1] == str(cache)
    assert command[command.index("--candidate-count") + 1] == "24"
    assert command[command.index("--payout-prior-weights") + 1] == "10,30,100"
    assert command[command.index("--evaluation-days") + 1] == "365"
    assert command[command.index("--research-only") + 1] == "true"
    assert command[command.index("--coefficient-optimizer") + 1] == "adam"


def test_bankroll_policy_search_uses_fixed_standardized_source(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "boat"
    cache_dir = tmp_path / "standardized-selected-cache"
    monkeypatch.setattr(
        evaluation_queue, "STANDARDIZED_SELECTED_CACHE_DIR", cache_dir
    )
    artifact, cache_prefix = _standard_source(root, cache_dir)

    command, _output = build_command(
        _job({"source_kind": "standardized_selected"}),
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )

    assert command[command.index("--search-result") + 1] == str(artifact)
    assert command[command.index("--cache-prefix") + 1] == str(cache_prefix)


def test_bankroll_policy_search_rejects_obsolete_source_schema(tmp_path) -> None:
    root = tmp_path / "boat"
    source, _cache = _source(root)
    source.write_text(
        json.dumps({
            "selected_cache_prefix": str(_cache),
            "feature_schema_version": "pastlog-listwise-hashed-v3",
        }),
        encoding="utf-8",
    )

    with pytest.raises(ObsoleteJob, match="feature schema is obsolete"):
        build_command(
            _job({"source_job_id": 3565}),
            app_root=root,
            python=root / ".venv/bin/python",
            db="postgresql://test",
        )


def test_bankroll_policy_evaluation_defaults_to_standard_365_days() -> None:
    parser = build_parser()
    args = parser.parse_args(["--db", "x", "--search-result", "x", "--cache-prefix", "x", "--output", "x"])
    assert args.evaluation_days == 365
    assert args.coefficient_optimizer == "adam"


def test_bankroll_policy_search_builds_newton_command(tmp_path) -> None:
    root = tmp_path / "boat"
    _source(root)
    command, _output = build_command(
        _job(
            {
                "source_job_id": 3565,
                "coefficient_optimizer": "newton_cg",
                "max_newton_iterations": 8,
                "max_cg_iterations": 60,
                "gradient_tolerance": 0.0002,
                "cg_tolerance": 0.002,
            }
        ),
        app_root=root,
        python=root / ".venv/bin/python",
        db="postgresql://test",
    )

    assert command[command.index("--coefficient-optimizer") + 1] == "newton_cg"
    assert command[command.index("--max-newton-iterations") + 1] == "8"
    assert command[command.index("--max-cg-iterations") + 1] == "60"
    assert command[command.index("--gradient-tolerance") + 1] == "0.0002"
    assert command[command.index("--cg-tolerance") + 1] == "0.002"


def test_train_model_can_refine_adam_with_newton(monkeypatch) -> None:
    calls = {}
    monkeypatch.setattr(
        bankroll_policy_evaluation,
        "fit_scaler",
        lambda *args, **kwargs: "scaler",
    )
    monkeypatch.setattr(
        bankroll_policy_evaluation,
        "train_listwise_model",
        lambda *args, **kwargs: ("adam-model", [{"epoch": 1}]),
    )

    def refine(dataset, model, **kwargs):
        calls.update(kwargs)
        return "newton-model", {"converged": True}

    monkeypatch.setattr(bankroll_policy_evaluation, "refine_newton_cg", refine)

    model, history = _train_model(
        object(),
        race_end=100,
        selected={"target": "top3_pl", "alpha": 0.001},
        learning_rate=0.02,
        epochs=2,
        batch_races=50,
        coefficient_optimizer="newton_cg",
        max_newton_iterations=8,
        max_cg_iterations=60,
        gradient_tolerance=0.0002,
        cg_tolerance=0.002,
    )

    assert model == "newton-model"
    assert history["adam_history"] == [{"epoch": 1}]
    assert history["newton_convergence"] == {"converged": True}
    assert calls == {
        "train_race_end": 100,
        "batch_races": 50,
        "max_newton_iterations": 8,
        "max_cg_iterations": 60,
        "gradient_tolerance": 0.0002,
        "cg_tolerance": 0.002,
    }


def test_prior_selection_prefers_temporal_stability_before_bootstrap_ci() -> None:
    def row(prior, *, minimum_roi, stable_score, ci_lower):
        return {
            "payout_prior_weight": prior,
            "selected": {
                "temporal_stability": {
                    "all_minimum_evidence": True,
                    "minimum_roi": minimum_roi,
                    "mean_roi_minus_std": stable_score,
                },
                "confidence": {
                    "roi_ci95_lower": ci_lower,
                    "probability_roi_above_one": 0.55,
                },
                "metrics": {"profit_yen": 1000, "hit_tickets": 80},
            },
        }

    unstable = row(
        5,
        minimum_roi=0.991,
        stable_score=0.989,
        ci_lower=0.803,
    )
    stable = row(
        100,
        minimum_roi=1.024,
        stable_score=1.024,
        ci_lower=0.795,
    )

    assert max([unstable, stable], key=prior_selection_key) is stable


@pytest.mark.parametrize(
    "parameters,message",
    [
        ({"source_job_id": 3565, "candidate_count": 7}, "candidate_count"),
        (
            {"source_job_id": 3565, "candidate_count": 8, "finalists": 9},
            "finalists",
        ),
        (
            {"source_job_id": 3565, "payout_prior_weights": "0,30"},
            "payout_prior_weights",
        ),
        ({"source_job_id": 3565, "command": "sh"}, "unsupported"),
    ],
)
def test_bankroll_policy_search_rejects_invalid_parameters(
    tmp_path, parameters, message
) -> None:
    root = tmp_path / "boat"
    _source(root)
    with pytest.raises(ValueError, match=message):
        build_command(
            _job(parameters),
            app_root=root,
            python=root / ".venv/bin/python",
            db="postgresql://test",
        )
