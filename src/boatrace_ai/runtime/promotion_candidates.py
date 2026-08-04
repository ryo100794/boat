from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..listwise.market_calibration import MARKET_EVALUATION_VERSION


def discover_market_evaluation_candidates(
    directory: str | Path,
    *,
    limit: int = 20,
) -> list[str]:
    """Return recent complete market-evaluation artifacts eligible for validation."""
    root = Path(directory).resolve()
    if not root.is_dir() or limit < 1:
        return []

    paths = sorted(
        (path for path in root.glob("job-*.json") if path.is_file()),
        key=lambda path: path.name,
        reverse=True,
    )
    selected: list[str] = []
    seen_sources: set[tuple[str, str, str]] = set()
    for path in paths:
        try:
            payload: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("evaluation_version") != MARKET_EVALUATION_VERSION:
            continue
        if not isinstance(payload.get("promotion_gate"), dict):
            continue
        if not isinstance(payload.get("deployment_configuration"), dict):
            continue
        source = str(payload.get("source_model") or "")
        source_hash = str(payload.get("source_model_sha256") or "")
        if not source or not source_hash:
            continue
        track = payload.get("trend_point_empirical_lcb_walk_forward")
        if not isinstance(track, dict):
            track = payload.get("prospective_normalized_ev_walk_forward")
        policy = track.get("policy") if isinstance(track, dict) else None
        policy_identity = json.dumps(
            policy if isinstance(policy, dict) else {},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        identity = (source, source_hash, policy_identity)
        if identity in seen_sources:
            continue
        seen_sources.add(identity)
        selected.append(str(path))
        if len(selected) >= limit:
            break
    return selected
