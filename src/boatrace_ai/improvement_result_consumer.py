from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from .db import connection


ADVISORY_LOCK_ID = 2_026_073_001
SUPPORTED_TASK_TYPES = frozenset({"bankroll_policy_search"})
ACTIONABLE_DECISIONS = frozenset(
    {"promotion_candidate", "refine_selected_candidate", "reject_or_research_only"}
)
CLEARLY_NON_ACTIONABLE_DECISIONS = frozenset({"invalid_data_source"})
DEFAULT_SETTLE_SECONDS = 3600

SCHEMA = """
ALTER TABLE model_evaluation_jobs
  ADD COLUMN IF NOT EXISTS improvement_audit JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE model_improvement_candidates
  ADD COLUMN IF NOT EXISTS review_decision TEXT;
ALTER TABLE model_improvement_candidates
  ADD COLUMN IF NOT EXISTS review_reason TEXT;
ALTER TABLE model_improvement_candidates
  ADD COLUMN IF NOT EXISTS comparison_group_key TEXT;
ALTER TABLE model_improvement_candidates
  ADD COLUMN IF NOT EXISTS comparison_rank INTEGER;
ALTER TABLE model_improvement_candidates
  ADD COLUMN IF NOT EXISTS downstream_job_id BIGINT;
CREATE INDEX IF NOT EXISTS idx_model_improvement_candidates_review
  ON model_improvement_candidates(reviewed_at, task_type, candidate_id);
CREATE TABLE IF NOT EXISTS model_improvement_candidate_reviews (
  comparison_group_key TEXT PRIMARY KEY,
  task_type TEXT NOT NULL,
  winner_candidate_id BIGINT NOT NULL,
  downstream_job_id BIGINT,
  decision TEXT NOT NULL,
  reason TEXT NOT NULL,
  formal_spec JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
ALTER TABLE model_improvement_candidate_reviews
  ADD COLUMN IF NOT EXISTS formal_spec JSONB NOT NULL DEFAULT '{}'::jsonb;
"""


class FormalMappingUnavailable(ValueError):
    """A candidate cannot yet be mapped to a verified formal evaluation."""


@dataclass(frozen=True)
class Candidate:
    candidate_id: int
    job_id: int
    model_key: str
    task_type: str
    producer_decision: str
    metrics: dict[str, Any]
    parameters: dict[str, Any]
    result_path: str | None
    created_at: Any
    result_identity: dict[str, Any] | None = None


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def dedupe_key(
    task_type: str,
    model_key: str,
    parameters: dict[str, Any],
    audit_parameters: dict[str, Any] | None = None,
) -> str:
    identity = {"executable": parameters, "audit": audit_parameters or {}}
    return hashlib.sha256(_json([task_type, model_key, identity]).encode()).hexdigest()


def enqueue_formal_job(
    conn: Any,
    *,
    task_type: str,
    model_key: str,
    parameters: dict[str, Any],
    audit_parameters: dict[str, Any],
    priority: int,
    max_attempts: int,
    parent_job_id: int,
) -> int | None:
    if task_type != "bankroll_policy_nested_annual":
        raise ValueError(f"unsupported formal task type: {task_type}")
    key = dedupe_key(task_type, model_key, parameters, audit_parameters)
    row = conn.execute(
        """
        INSERT INTO model_evaluation_jobs(
          task_type, category, model_key, parameters, improvement_audit,
          dedupe_key, priority, max_attempts, parent_job_id,
          min_free_memory_mb, min_free_disk_mb, min_idle_cpu_percent, max_parallel
        ) VALUES (?, 'evaluation', ?, CAST(? AS JSONB), CAST(? AS JSONB),
                  ?, ?, ?, ?, 21504, 4096, 15.0, 1)
        ON CONFLICT(dedupe_key) DO NOTHING
        RETURNING job_id
        """,
        (
            task_type,
            model_key,
            _json(parameters),
            _json(audit_parameters),
            key,
            int(priority),
            int(max_attempts),
            int(parent_job_id),
        ),
    ).fetchone()
    return int(row["job_id"]) if row is not None else None


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _result_metadata(result_path: str | None) -> dict[str, Any]:
    if not result_path:
        return {}
    path = Path(result_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    names = {
        "evaluation_scope", "evaluation_from", "evaluation_start",
        "evaluation_through", "evaluation_end", "holdout_from", "holdout_start",
        "holdout_through", "holdout_end", "as_of_date", "evaluation_date",
        "training_through", "selection_from", "selection_through",
        "comparison_cohort", "protocol_id", "protocol_version",
        "evaluation_protocol", "selection_protocol", "comparison_role",
        "source_artifact_sha256", "source_model_sha256",
        "feature_source_sha256", "dataset_sha256", "race_universe_sha256",
        "odds_coverage_definition", "odds_snapshot_policy", "odds_field",
        "minimum_day_coverage", "selected_policy", "best_policy",
    }
    # Top-level evaluation metadata is authoritative. Nested calibration or
    # selection windows are fallback evidence only.
    found: dict[str, Any] = {
        key: payload[key] for key in names if key in payload
    }

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in names and key not in found:
                    found[key] = item
                if isinstance(item, dict):
                    visit(item)

    for item in payload.values():
        if isinstance(item, dict):
            visit(item)
    return found


def candidate_from_row(row: Any) -> Candidate:
    result_path = str(row["result_path"]) if row["result_path"] is not None else None
    return Candidate(
        candidate_id=int(row["candidate_id"]),
        job_id=int(row["job_id"]),
        model_key=str(row["model_key"]),
        task_type=str(row["task_type"]),
        producer_decision=str(row["decision"]),
        metrics=_mapping(row["metrics"]),
        parameters=_mapping(row["parameters"]),
        result_path=result_path,
        created_at=row["created_at"],
        result_identity=_result_metadata(result_path),
    )


def _normalized(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalized(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalized(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _first(candidate: Candidate, *names: str) -> Any:
    sources = (candidate.parameters, candidate.metrics, candidate.result_identity or {})
    for name in names:
        for source in sources:
            value = source.get(name)
            if value is not None:
                return value
    return None


def comparison_identity(candidate: Candidate) -> dict[str, Any]:
    """Build a conservative identity for one exact evaluation population."""
    if candidate.task_type != "bankroll_policy_search":
        return {"task_type": candidate.task_type, "unsupported_job_id": candidate.job_id}
    params = candidate.parameters
    source_job_id = params.get("source_job_id")
    if source_job_id is None:
        source = {"source_kind": params.get("source_kind")}
    else:
        try:
            normalized_source_job_id: Any = int(source_job_id)
        except (TypeError, ValueError):
            # Preserve the invalid value for grouping and audit. Validation is
            # deferred to formal_follow_up; no source id is inferred.
            normalized_source_job_id = {"invalid_value": str(source_job_id)}
        source = {"source_job_id": normalized_source_job_id}
    temporal_boundary_names = (
        "evaluation_from", "evaluation_start", "evaluation_through",
        "evaluation_end", "holdout_from", "holdout_start", "holdout_through",
        "holdout_end", "as_of_date", "evaluation_date", "selection_from",
        "selection_through",
    )
    boundaries = {
        name: _first(candidate, name)
        for name in temporal_boundary_names
        if _first(candidate, name) is not None
    }
    # A scope label or equal row count is not an actual temporal boundary.
    if not boundaries:
        boundaries["missing_boundary_job_id"] = candidate.job_id
    evaluation_scope = _first(candidate, "evaluation_scope")
    if evaluation_scope is not None:
        boundaries["evaluation_scope"] = evaluation_scope
    protocol_names = (
        "comparison_cohort", "protocol_id", "protocol_version",
        "evaluation_protocol", "selection_protocol", "comparison_role",
    )
    artifact_names = (
        "source_artifact_sha256", "source_model_sha256",
        "feature_source_sha256", "dataset_sha256", "race_universe_sha256",
    )
    odds_names = (
        "odds_coverage_definition", "odds_snapshot_policy", "odds_field",
        "minimum_day_coverage",
    )
    return {
        "task_type": candidate.task_type,
        "source": source,
        "boundaries": boundaries,
        "protocol": {
            name: _first(candidate, name)
            for name in protocol_names
            if _first(candidate, name) is not None
        },
        "artifact": {
            name: _first(candidate, name)
            for name in artifact_names
            if _first(candidate, name) is not None
        },
        "odds_coverage": {
            name: _first(candidate, name)
            for name in odds_names
            if _first(candidate, name) is not None
        },
        "evaluation_days_requested": int(params.get("evaluation_days") or 365),
        "evaluation_days_observed": _first(candidate, "evaluation_days", "race_days"),
        "evaluated_races": _first(candidate, "evaluated_races", "evaluation_races"),
        "daily_budget_yen": int(params.get("daily_budget_yen") or 10_000),
    }


def comparison_group_key(candidate: Candidate) -> str:
    return hashlib.sha256(_json(_normalized(comparison_identity(candidate))).encode()).hexdigest()


def _finite_metric(candidate: Candidate, *names: str) -> float | None:
    for name in names:
        value = candidate.metrics.get(name)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def ranking_key(candidate: Candidate) -> tuple[float, ...] | None:
    roi = _finite_metric(candidate, "chronological_roi", "roi")
    if roi is None:
        return None
    robust = _finite_metric(
        candidate, "chronological_roi_without_largest_hit",
        "roi_without_largest_hit", "largest_hit_excluded_roi",
    )
    lower = _finite_metric(
        candidate, "chronological_roi_ci95_lower",
        "daily_cluster_bootstrap_roi_lower_95", "roi_ci95_lower",
    )
    hits = _finite_metric(candidate, "chronological_effective_hit_count", "effective_hit_count")
    profit = _finite_metric(candidate, "chronological_profit_yen", "profit_yen")
    drawdown = _finite_metric(candidate, "chronological_max_drawdown_yen", "max_drawdown_yen")
    return (
        robust if robust is not None else -math.inf,
        lower if lower is not None else -math.inf,
        roi,
        hits if hits is not None else -math.inf,
        profit if profit is not None else -math.inf,
        -(drawdown if drawdown is not None else math.inf),
        -float(candidate.candidate_id),
    )


def rank_candidates(candidates: Iterable[Candidate]) -> tuple[list[Candidate], list[Candidate]]:
    valid, invalid = [], []
    for candidate in candidates:
        (valid if ranking_key(candidate) is not None else invalid).append(candidate)
    valid.sort(key=lambda item: ranking_key(item) or (), reverse=True)
    invalid.sort(key=lambda item: item.candidate_id)
    return valid, invalid


def _file_sha256(path: str | None) -> str | None:
    if not path:
        return None
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def formal_follow_up(
    candidate: Candidate,
) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    if candidate.task_type != "bankroll_policy_search":
        raise FormalMappingUnavailable(
            f"unsupported formal follow-up source: {candidate.task_type}"
        )
    params = candidate.parameters
    if params.get("source_job_id") is None:
        raise FormalMappingUnavailable("bankroll candidate lacks source_job_id")
    try:
        source_job_id = int(params["source_job_id"])
    except (TypeError, ValueError) as exc:
        raise FormalMappingUnavailable(
            "bankroll candidate has invalid source_job_id"
        ) from exc
    executable = {
        "source_job_id": source_job_id,
        "learning_rate": float(params.get("learning_rate") or 0.02),
        "epochs": int(params.get("epochs") or 2),
        "batch_races": int(params.get("batch_races") or 1000),
        "targets": str(params.get("targets") or "winner,top3_pl"),
        "alphas": str(params.get("alphas") or "0.00001,0.0001,0.001"),
        "candidate_count": 64, "finalists": 8,
        "selection_bootstrap_samples": max(20_000, int(params.get("bootstrap_samples") or 20_000)),
        "aggregate_bootstrap_samples": max(20_000, int(params.get("bootstrap_samples") or 20_000)),
        "selection_days": 365, "outer_days": 365,
        "embargo_days": int(params.get("embargo_days") or 0),
        "validation_fraction": float(params.get("validation_fraction") or 0.2),
        "min_validation_races": int(params.get("min_validation_races") or 1000),
        "daily_budget_yen": int(params.get("daily_budget_yen") or 10_000),
        "ev_threshold": float(params.get("ev_threshold") or 1.2),
        "seed": int(params.get("seed") or 20_260_728),
        "timeout_seconds": 86_400,
    }
    audit = {
        "winner_candidate_id": candidate.candidate_id,
        "winner_job_id": candidate.job_id,
        "winner_model_key": candidate.model_key,
        "winner_result_path": candidate.result_path,
        "winner_result_sha256": _file_sha256(candidate.result_path),
        "winning_policy_configuration": _first(candidate, "selected_policy", "best_policy"),
        "comparison_identity": comparison_identity(candidate),
        "formal_method": "nested_annual_reselection",
        "reselection_reason": (
            "The research result selects policy settings on its research window. "
            "The annual follow-up intentionally reselects from the same source inside "
            "each selection fold so the outer holdout remains untouched."
        ),
        "executable_parameters": executable,
    }
    suffix = comparison_group_key(candidate)[:12]
    model_key = f"{candidate.model_key}:winner-{candidate.candidate_id}:formal-365d:{suffix}"
    return "bankroll_policy_nested_annual", model_key, executable, audit


def ensure_schema(conn: Any) -> None:
    conn.executescript(SCHEMA)


def _review_candidate(
    conn: Any, candidate: Candidate, *, decision: str, reason: str,
    group_key: str, rank: int | None = None, downstream_job_id: int | None = None,
) -> None:
    conn.execute(
        """
        UPDATE model_improvement_candidates
        SET reviewed_at = CURRENT_TIMESTAMP, review_decision = ?, review_reason = ?,
            comparison_group_key = ?, comparison_rank = ?, downstream_job_id = ?
        WHERE candidate_id = ? AND reviewed_at IS NULL
        """,
        (decision, reason, group_key, rank, downstream_job_id, candidate.candidate_id),
    )


def _existing_review(conn: Any, group_key: str) -> Any:
    return conn.execute(
        """SELECT winner_candidate_id, downstream_job_id, decision, reason
           FROM model_improvement_candidate_reviews WHERE comparison_group_key = ?""",
        (group_key,),
    ).fetchone()


def _existing_downstream_job(conn: Any, key: str) -> int | None:
    row = conn.execute("SELECT job_id FROM model_evaluation_jobs WHERE dedupe_key = ?", (key,)).fetchone()
    return int(row["job_id"]) if row is not None else None


def _created_at(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def cohort_stability(
    conn: Any,
    candidates: list[Candidate],
    *,
    settle_seconds: int,
    now: datetime | None = None,
) -> tuple[bool, str]:
    if len(candidates) < 2:
        return False, "waiting_for_comparable_candidate"
    now = now or datetime.now(timezone.utc)
    youngest_age = min((now - _created_at(item.created_at)).total_seconds() for item in candidates)
    if youngest_age < settle_seconds:
        return False, f"settling_for_siblings:{max(0, int(settle_seconds - youngest_age))}s"
    cohorts = {str(item.parameters.get("comparison_cohort")) for item in candidates if item.parameters.get("comparison_cohort")}
    if len(cohorts) > 1:
        return False, "mixed_comparison_cohort"
    if cohorts:
        row = conn.execute(
            """SELECT COUNT(*) AS count FROM model_evaluation_jobs
               WHERE task_type = ? AND status IN ('queued','running')
                 AND parameters->>'comparison_cohort' = ?""",
            (candidates[0].task_type, next(iter(cohorts))),
        ).fetchone()
    else:
        first = candidates[0]
        source_match = {}
        for name in ("source_job_id", "source_kind", "evaluation_days"):
            if first.parameters.get(name) is not None:
                source_match[name] = first.parameters[name]
        row = conn.execute(
            """SELECT COUNT(*) AS count FROM model_evaluation_jobs
               WHERE task_type = ? AND status IN ('queued','running')
                 AND parameters @> CAST(? AS JSONB)""",
            (first.task_type, _json(source_match)),
        ).fetchone()
    active = int(row["count"]) if row is not None else 0
    if active:
        return False, f"waiting_for_{active}_producer_jobs"
    return True, "cohort_terminal_and_settled"


def review_group(
    conn: Any,
    candidates: list[Candidate],
    *,
    settle_seconds: int = DEFAULT_SETTLE_SECONDS,
    now: datetime | None = None,
    enqueue: Callable[..., int | None] = enqueue_formal_job,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("candidate group must not be empty")
    group_key = comparison_group_key(candidates[0])
    if any(comparison_group_key(item) != group_key for item in candidates):
        raise ValueError("candidate group is not semantically comparable")
    stable, stability_reason = cohort_stability(
        conn, candidates, settle_seconds=settle_seconds, now=now
    )
    if not stable:
        return {"group_key": group_key, "status": "pending", "reason": stability_reason}
    existing = _existing_review(conn, group_key)
    if existing is not None:
        downstream = existing["downstream_job_id"]
        for candidate in candidates:
            _review_candidate(
                conn, candidate, decision="group_already_reviewed",
                reason=f"comparison group already selected candidate {int(existing['winner_candidate_id'])}",
                group_key=group_key,
                downstream_job_id=int(downstream) if downstream is not None else None,
            )
        return {"group_key": group_key, "status": "already_reviewed"}
    valid, invalid = rank_candidates(candidates)
    if not valid:
        for candidate in invalid:
            _review_candidate(
                conn, candidate, decision="invalid_metrics",
                reason="finite chronological ROI is required for comparison",
                group_key=group_key,
            )
        return {"group_key": group_key, "status": "no_valid_candidate"}
    if len(valid) < 2:
        return {
            "group_key": group_key,
            "status": "pending",
            "reason": "waiting_for_two_valid_candidates",
        }
    winner = valid[0]
    try:
        task_type, model_key, parameters, audit = formal_follow_up(winner)
    except FormalMappingUnavailable as exc:
        return {
            "group_key": group_key,
            "status": "follow_up_unavailable",
            "reason": str(exc),
            "candidate_ids": [item.candidate_id for item in candidates],
            "winner_candidate_id": winner.candidate_id,
        }
    downstream = enqueue(
        conn, task_type=task_type, model_key=model_key, parameters=parameters,
        audit_parameters=audit, priority=95, max_attempts=3, parent_job_id=winner.job_id,
    )
    key = dedupe_key(task_type, model_key, parameters, audit)
    if downstream is None:
        downstream = _existing_downstream_job(conn, key)
    if downstream is None:
        raise RuntimeError("formal follow-up enqueue did not return or resolve a job")
    reason = (
        "highest robust ROI, confidence-bound ROI, raw ROI, effective hits, "
        "profit, and inverse drawdown under one exact evaluation protocol"
    )
    inserted = conn.execute(
        """
        INSERT INTO model_improvement_candidate_reviews(
          comparison_group_key, task_type, winner_candidate_id, downstream_job_id,
          decision, reason, formal_spec
        ) VALUES (?, ?, ?, ?, 'formal_follow_up_queued', ?, CAST(? AS JSONB))
        ON CONFLICT(comparison_group_key) DO NOTHING RETURNING comparison_group_key
        """,
        (group_key, winner.task_type, winner.candidate_id, downstream, reason, _json(audit)),
    ).fetchone()
    if inserted is None:
        raise RuntimeError("comparison group was reviewed concurrently")
    for candidate in invalid:
        _review_candidate(
            conn, candidate, decision="invalid_metrics",
            reason="finite chronological ROI is required for comparison",
            group_key=group_key,
        )
    for rank, candidate in enumerate(valid, start=1):
        selected = candidate.candidate_id == winner.candidate_id
        _review_candidate(
            conn, candidate,
            decision="winner_formal_follow_up_queued" if selected else "not_selected",
            reason=reason if selected else f"ranked {rank} behind candidate {winner.candidate_id}",
            group_key=group_key, rank=rank,
            downstream_job_id=downstream if selected else None,
        )
    return {
        "group_key": group_key, "status": "formal_follow_up_queued",
        "winner_candidate_id": winner.candidate_id, "downstream_job_id": downstream,
    }


def _load_supported_candidates(conn: Any, *, batch_size: int) -> list[Candidate]:
    # Supported candidates are intentionally loaded without LIMIT. A discovered
    # group must never be finalized from a batch prefix.
    rows = conn.execute(
        """
        SELECT c.candidate_id, c.job_id, c.model_key, c.task_type, c.decision,
               c.metrics, c.parameters, c.result_path, c.created_at
        FROM model_improvement_candidates c
        JOIN model_evaluation_jobs j ON j.job_id = c.job_id
        WHERE c.reviewed_at IS NULL AND j.status = 'completed'
          AND c.task_type = 'bankroll_policy_search'
        ORDER BY c.candidate_id
        FOR UPDATE OF c SKIP LOCKED
        """
    ).fetchall()
    return [candidate_from_row(row) for row in rows]


def _review_clearly_non_actionable(conn: Any, *, batch_size: int) -> int:
    rows = conn.execute(
        """
        WITH pending AS (
          SELECT c.candidate_id FROM model_improvement_candidates c
          JOIN model_evaluation_jobs j ON j.job_id = c.job_id
          WHERE c.reviewed_at IS NULL AND j.status = 'completed'
            AND c.decision = 'invalid_data_source'
          ORDER BY c.candidate_id LIMIT ? FOR UPDATE OF c SKIP LOCKED
        )
        UPDATE model_improvement_candidates c
        SET reviewed_at = CURRENT_TIMESTAMP,
            review_decision = 'producer_decision_not_actionable',
            review_reason = 'producer rejected the candidate because its data source is invalid',
            comparison_group_key = c.task_type || '-invalid-source-job-' || c.job_id::text,
            comparison_rank = NULL, downstream_job_id = NULL
        FROM pending WHERE c.candidate_id = pending.candidate_id
        RETURNING c.candidate_id
        """,
        (batch_size,),
    ).fetchall()
    return len(rows)


def _unsupported_pending_count(conn: Any) -> int:
    row = conn.execute(
        """
        SELECT COUNT(*) AS count FROM model_improvement_candidates c
        JOIN model_evaluation_jobs j ON j.job_id = c.job_id
        WHERE c.reviewed_at IS NULL AND j.status = 'completed'
          AND c.task_type <> 'bankroll_policy_search'
          AND c.decision <> 'invalid_data_source'
        """
    ).fetchone()
    return int(row["count"]) if row is not None else 0


def _record_mapping_ticket(
    conn: Any,
    unsupported_count: int,
    mapping_unavailable: list[dict[str, Any]],
) -> None:
    if unsupported_count <= 0 and not mapping_unavailable:
        return
    reasons = "; ".join(
        f"{item['group_key'][:12]}: {item['reason']}"
        for item in mapping_unavailable
    )
    description = (
        f"{unsupported_count} actionable completed candidates await a verified "
        f"task mapping; {len(mapping_unavailable)} comparable groups cannot build "
        f"a formal follow-up"
    )
    if reasons:
        description += f": {reasons}"
    conn.execute(
        """
        INSERT INTO work_tickets(
          ticket_key, title, area, description, acceptance_criteria,
          owner, priority, status, progress
        ) VALUES (
          'MODEL-IMPROVEMENT-CONSUMER-MAPPINGS',
          '改善候補consumerの正式評価mapping追加', 'モデル', ?,
          '各task typeに意味的比較署名と正式評価mappingをテスト付きで追加する',
          'codex', 85, 'in_progress', 0
        )
        ON CONFLICT(ticket_key) DO UPDATE SET
          description = excluded.description, status = 'in_progress',
          updated_at = CURRENT_TIMESTAMP
        """,
        (description,),
    )


def process_once(
    conn: Any, *, batch_size: int = 500,
    settle_seconds: int = DEFAULT_SETTLE_SECONDS,
    enqueue: Callable[..., int | None] = enqueue_formal_job,
) -> dict[str, Any]:
    lock = conn.execute("SELECT pg_try_advisory_xact_lock(?) AS acquired", (ADVISORY_LOCK_ID,)).fetchone()
    if lock is None or not bool(lock["acquired"]):
        return {"status": "lock_busy", "reviewed": 0, "groups": []}
    candidates = _load_supported_candidates(conn, batch_size=batch_size)
    groups: dict[str, list[Candidate]] = defaultdict(list)
    pending_supported_decisions = 0
    for candidate in candidates:
        if candidate.producer_decision in ACTIONABLE_DECISIONS:
            groups[comparison_group_key(candidate)].append(candidate)
        elif candidate.producer_decision in CLEARLY_NON_ACTIONABLE_DECISIONS:
            _review_candidate(
                conn, candidate, decision="producer_decision_not_actionable",
                reason=f"producer decision {candidate.producer_decision!r} is not actionable",
                group_key=comparison_group_key(candidate),
            )
        else:
            pending_supported_decisions += 1
    results = [
        review_group(conn, rows, settle_seconds=settle_seconds, enqueue=enqueue)
        for rows in groups.values()
    ]
    non_actionable = _review_clearly_non_actionable(conn, batch_size=batch_size)
    unsupported = _unsupported_pending_count(conn) + pending_supported_decisions
    mapping_unavailable = [
        {
            "group_key": result["group_key"],
            "reason": result["reason"],
            "candidate_ids": result.get("candidate_ids", []),
            "winner_candidate_id": result.get("winner_candidate_id"),
        }
        for result in results
        if result["status"] == "follow_up_unavailable"
    ]
    _record_mapping_ticket(conn, unsupported, mapping_unavailable)
    final_statuses = {
        "formal_follow_up_queued", "already_reviewed", "no_valid_candidate"
    }
    reviewed_groups = sum(
        len(groups[result["group_key"]])
        for result in results
        if result["status"] in final_statuses
    )
    return {
        "status": "completed", "reviewed": reviewed_groups + non_actionable,
        "supported_candidates": len(candidates),
        "unsupported_pending": unsupported,
        "mapping_unavailable_count": len(mapping_unavailable),
        "mapping_unavailable_reasons": mapping_unavailable,
        "clearly_non_actionable_reviewed": non_actionable,
        "groups": results,
    }


def run(args: argparse.Namespace) -> int:
    while True:
        try:
            with connection(args.db) as conn:
                ensure_schema(conn)
                result = process_once(
                    conn, batch_size=args.batch_size, settle_seconds=args.settle_seconds
                )
            print(_json({**result, "generated_at": datetime.now(timezone.utc).isoformat()}), flush=True)
            if args.once:
                return 0
            time.sleep(args.poll_seconds)
        except KeyboardInterrupt:
            return 130
        except Exception as exc:
            print(f"improvement result consumer error: {type(exc).__name__}: {exc}", flush=True)
            if args.once:
                raise
            time.sleep(args.poll_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review completed model-improvement candidates")
    parser.add_argument("--db", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--settle-seconds", type=int, default=DEFAULT_SETTLE_SECONDS)
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.poll_seconds <= 0 or args.batch_size <= 0 or args.settle_seconds < 0:
        raise ValueError("poll-seconds and batch-size must be positive; settle-seconds non-negative")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
