from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .v21_prospective_evidence import (
    V21ProspectiveEvidenceConfig,
    aggregate_v21_prospective_evidence,
    collect_v21_prospective_evidence,
    write_v21_prospective_evidence_atomic,
)


V23_STRATEGY_NAME = "v23_top5_narrow_t300"
V23_DIAGNOSTIC_KEY = "v23_top5_narrow"
V23_EVIDENCE_KIND = "v23_frozen_policy_fully_unseen_prospective"


@dataclass(frozen=True)
class V23ProspectiveEvidenceConfig(V21ProspectiveEvidenceConfig):
    expected_strategy_name: str = V23_STRATEGY_NAME
    diagnostic_key: str = V23_DIAGNOSTIC_KEY
    evidence_kind: str = V23_EVIDENCE_KIND


def aggregate_v23_prospective_evidence(
    *,
    config: V23ProspectiveEvidenceConfig,
    races,
    decisions,
    settlements,
    source_odds,
    payouts,
) -> dict[str, Any]:
    return aggregate_v21_prospective_evidence(
        config=config,
        races=races,
        decisions=decisions,
        settlements=settlements,
        source_odds=source_odds,
        payouts=payouts,
    )


def collect_v23_prospective_evidence(
    conn: Any,
    *,
    config: V23ProspectiveEvidenceConfig,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    return collect_v21_prospective_evidence(
        conn, config=config, output_path=output_path
    )


def write_v23_prospective_evidence_atomic(
    path: str | Path, payload: Mapping[str, Any]
) -> None:
    write_v21_prospective_evidence_atomic(path, payload)


__all__ = [
    "V23ProspectiveEvidenceConfig",
    "aggregate_v23_prospective_evidence",
    "collect_v23_prospective_evidence",
    "write_v23_prospective_evidence_atomic",
]
