from __future__ import annotations

from pathlib import Path

from .intraday_t300_shadow import (
    V21TripleHeadModelAdapter,
    main as shadow_main,
    register_adapter,
)
from .raw_guard_shadow_policy import (
    LEARNED_DAILY_TICKET_LIMIT,
    MIN_RAW_EV,
    REGISTERED_AFTER,
    SOURCE_EVALUATION_JOB_ID,
    STRATEGY_NAME,
    registration,
)


class RawGuardV21ModelAdapter(V21TripleHeadModelAdapter):
    """Run the preregistered V21 raw-EV guard in shadow only."""

    strategy_name = STRATEGY_NAME
    artifact_label = "V21 raw-EV guard"
    source_evaluation_job_id = SOURCE_EVALUATION_JOB_ID

    def __init__(
        self, *, model_key: str, bundle_path: Path, base_model_path: Path
    ) -> None:
        super().__init__(
            model_key=model_key,
            bundle_path=bundle_path,
            base_model_path=base_model_path,
        )
        if self._bundle.get("prospective_policy_registration") != registration():
            raise ValueError("raw-guard prospective registration is inconsistent")
        if str(self._bundle.get("trained_through_date") or "") != REGISTERED_AFTER:
            raise ValueError("raw-guard training boundary is inconsistent")
        if self._ticket_limit != LEARNED_DAILY_TICKET_LIMIT:
            raise ValueError("raw-guard daily ticket limit is inconsistent")
        if self._quota_rounding != "ceil" or self._opportunity_policy is not None:
            raise ValueError("raw-guard schedule policy is inconsistent")
        if float(self._policy.get("min_raw_ev") or 0.0) != MIN_RAW_EV:
            raise ValueError("raw-guard minimum raw EV is inconsistent")


def adapter_factory(
    model_key: str, bundle_path: Path, base_model_path: Path
) -> RawGuardV21ModelAdapter:
    return RawGuardV21ModelAdapter(
        model_key=model_key,
        bundle_path=bundle_path,
        base_model_path=base_model_path,
    )


def main() -> int:
    register_adapter(STRATEGY_NAME, adapter_factory)
    return int(shadow_main())


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["RawGuardV21ModelAdapter", "adapter_factory"]
