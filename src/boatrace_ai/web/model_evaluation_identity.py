from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _contexts(data: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    contexts: list[Mapping[str, Any]] = []
    pending = [data]
    seen: set[int] = set()
    while pending:
        context = pending.pop(0)
        if id(context) in seen:
            continue
        seen.add(id(context))
        contexts.append(context)
        pending.extend(
            value for value in context.values() if isinstance(value, Mapping)
        )
    return contexts


def _first(data: Mapping[str, Any], *keys: str) -> Any:
    for context in _contexts(data):
        for key in keys:
            value = context.get(key)
            if value is not None and value != "":
                return value
    return None


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda item: str(item[0])) if item is not None}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, Path):
        return value.name
    return value


def _digest(prefix: str, value: Mapping[str, Any]) -> str:
    encoded = json.dumps(_canonical(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()}"


def _period(data: Mapping[str, Any]) -> tuple[Any, Any]:
    start = _first(data, "evaluation_from", "evaluation_start", "holdout_start", "outer_start", "start_date", "from_date", "evaluation_from_date")
    end = _first(data, "evaluation_through", "evaluation_end", "holdout_end", "outer_end", "end_date", "through_date", "evaluation_through_date")
    if start is not None and end is not None:
        return start, end
    daily = data.get("daily")
    if isinstance(daily, list):
        dates = sorted({str(row.get("race_date")) for row in daily if isinstance(row, Mapping) and row.get("race_date")})
        if dates:
            return dates[0], dates[-1]
    return start, end


def evaluation_cohort_payload(data: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return canonical comparison fields, or None when identity is unsafe."""
    start, end = _period(data)
    race_set = _first(data, "evaluation_race_set_sha256", "race_set_sha256", "cohort_sha256")
    protocol = _first(data, "protocol_sha256", "evaluation_protocol_sha256")
    policy = _first(data, "policy_sha256", "bankroll_policy_sha256", "selected_policy_sha256")
    if start is None or end is None or not any((race_set, protocol, policy)):
        return None

    odds_mode = _first(data, "decision_odds_mode", "odds_mode", "odds_snapshot_mode")
    decision_minutes = _first(data, "decision_minutes_before_deadline", "decision_minutes_before", "odds_cutoff_minutes", "t_minus_minutes")
    if decision_minutes is None and isinstance(odds_mode, str):
        normalized = odds_mode.lower().replace("-", "_")
        if "t5" in normalized or "t_5" in normalized:
            decision_minutes = 5
    fold = _first(data, "fold_definition", "fold_definitions", "folds_sha256", "fold_count", "outer_folds")
    if fold is None and isinstance(data.get("folds"), list):
        fold = [{key: row.get(key) for key in ("fold", "train_start", "train_end", "test_start", "test_end") if isinstance(row, Mapping) and row.get(key) is not None} for row in data["folds"]]

    return {
        "evaluation_from": str(start),
        "evaluation_through": str(end),
        "race_set_sha256": race_set,
        "protocol_sha256": protocol,
        "policy_sha256": policy,
        "odds_mode": odds_mode,
        "decision_minutes_before_deadline": decision_minutes,
        "daily_budget_yen": _first(data, "daily_budget_yen"),
        "allocation_mode": _first(data, "allocation_mode", "allocation_api"),
        "profit_reinvestment": _first(data, "profit_reinvestment", "reinvest_profit"),
        "fold": fold,
    }


def evaluation_identity(data: Mapping[str, Any], *, path: Path | str | None, label: str, result_kind: str, component: str | None = None, job: Mapping[str, Any] | None = None) -> dict[str, Any]:
    job = _mapping(job)
    source = Path(str(path)).name if path else None
    cohort_payload = evaluation_cohort_payload(data)
    cohort_id = str(_first(data, "evaluation_cohort_id") or "") or None
    if cohort_id is None and cohort_payload is not None:
        cohort_id = _digest("cohort", cohort_payload)

    job_id = job.get("db_job_id") or job.get("job_id") or _first(data, "db_job_id", "job_id")
    attempt = job.get("attempt") if job.get("attempt") is not None else _first(data, "attempt")
    explicit_run = _first(data, "evaluation_run_id")
    if explicit_run:
        run_id = str(explicit_run)
    elif job_id is not None:
        run_id = f"job_{job_id}_attempt_{attempt if attempt is not None else 0}"
    elif source:
        run_id = _digest("run", {"source": source, "generated_at": _first(data, "generated_at", "modified_at")})
    else:
        run_id = None

    explicit_bundle = _first(data, "evaluation_bundle_id", "parent_bundle_id")
    if explicit_bundle:
        bundle_id = str(explicit_bundle)
    elif source and cohort_id:
        bundle_id = _digest("bundle", {"source": source, "cohort": cohort_id, "component": component})
    elif job_id is not None and cohort_id:
        parent_job_id = job.get("parent_job_id") or _first(data, "parent_job_id")
        bundle_id = f"job_bundle_{parent_job_id or job_id}"
    else:
        bundle_id = None

    explicit_result = _first(data, "result_id")
    if explicit_result:
        result_id = str(explicit_result)
    elif run_id:
        result_id = _digest("result", {"run": run_id, "kind": result_kind, "component": component, "cohort": cohort_id, "source": source})
    else:
        result_id = None
    return {"evaluation_run_id": run_id, "result_id": result_id, "evaluation_cohort_id": cohort_id, "evaluation_bundle_id": bundle_id, "evaluation_cohort": cohort_payload, "db_job_id": job_id, "attempt": attempt}


def compatible_evaluation_results(prediction: Mapping[str, Any], bankroll: Mapping[str, Any]) -> bool:
    cohort = prediction.get("evaluation_cohort_id")
    bundle = prediction.get("evaluation_bundle_id")
    return bool(cohort and bundle and cohort == bankroll.get("evaluation_cohort_id") and bundle == bankroll.get("evaluation_bundle_id"))
