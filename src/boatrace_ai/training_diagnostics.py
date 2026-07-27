from __future__ import annotations

import math
from typing import Any


def classifier_training_diagnostics(classifier: Any) -> dict[str, Any]:
    raw_curve = getattr(classifier, "loss_curve_", ()) or ()
    curve = [
        float(value)
        for value in raw_curve
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    result: dict[str, Any] = {
        "optimizer_updates": len(curve),
        "reported_iterations": int(getattr(classifier, "n_iter_", 0) or 0),
        "finite_loss_curve": len(curve) == len(raw_curve),
    }
    if not curve:
        result.update(
            {
                "loss_start": None,
                "loss_end": None,
                "loss_min": None,
                "loss_reduction_fraction": None,
                "recent_loss_change": None,
            }
        )
        return result

    start = curve[0]
    end = curve[-1]
    result.update(
        {
            "loss_start": start,
            "loss_end": end,
            "loss_min": min(curve),
            "loss_reduction_fraction": (
                (start - end) / abs(start) if start != 0.0 else None
            ),
            "recent_loss_change": (
                end - curve[-2] if len(curve) >= 2 else None
            ),
        }
    )
    return result
