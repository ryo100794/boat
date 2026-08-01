from __future__ import annotations

from pathlib import Path
from typing import Any

from .intraday_t300_shadow import (
    V23Top5NarrowModelAdapter,
    main as shadow_main,
    register_adapter,
)
from .stable_cell_shadow_policy import (
    DIAGNOSTIC_KEY,
    POLICY_NAME,
    REGISTERED_AFTER,
    SOURCE_EVALUATION_JOB_ID,
    registration,
    select_stable_cell_candidates,
)


STRATEGY_NAME = "stable_cell_top5_lt20_t300"


class StableCellTop5ModelAdapter(V23Top5NarrowModelAdapter):
    """Run the post-development, preregistered stable cell in shadow only."""

    strategy_name = STRATEGY_NAME
    artifact_label = "stable-cell-top5-lt20"
    source_evaluation_job_id = SOURCE_EVALUATION_JOB_ID
    diagnostic_key = DIAGNOSTIC_KEY
    policy_name = POLICY_NAME
    registered_after = REGISTERED_AFTER
    candidate_selector = staticmethod(select_stable_cell_candidates)
    no_candidate_reason = "stable_cell_no_candidate"
    allowed_closing_forecasts = frozenset({"contextual"})

    def __init__(
        self,
        *,
        model_key: str,
        bundle_path: Path,
        base_model_path: Path,
    ) -> None:
        super().__init__(
            model_key=model_key,
            bundle_path=bundle_path,
            base_model_path=base_model_path,
        )
        if self._bundle.get("prospective_policy_registration") != registration():
            raise ValueError("stable-cell prospective registration is inconsistent")
        if str(self._bundle.get("trained_through_date") or "") != REGISTERED_AFTER:
            raise ValueError("stable-cell training boundary is inconsistent")


def adapter_factory(
    model_key: str, bundle_path: Path, base_model_path: Path
) -> StableCellTop5ModelAdapter:
    return StableCellTop5ModelAdapter(
        model_key=model_key,
        bundle_path=bundle_path,
        base_model_path=base_model_path,
    )


def main() -> int:
    register_adapter(STRATEGY_NAME, adapter_factory)
    return int(shadow_main())


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["STRATEGY_NAME", "StableCellTop5ModelAdapter", "adapter_factory"]
