from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

import boatrace_ai.listwise.contextual_empirical_ev_calibration as calibration
from boatrace_ai.listwise.contextual_empirical_ev_calibration import (
    ODDS_BANDS,
    RANK_GROUPS,
    fit_contextual_empirical_ev_calibration,
)


def _record(
    day: int,
    rank: int,
    odds: float,
    returned: float,
    *,
    raw_ev: float = 1.12,
) -> dict[str, object]:
    return {
        "race_date": (date(2026, 1, 1) + timedelta(days=day)).isoformat(),
        "raw_estimated_ev": raw_ev,
        "gross_return_per_yen": returned,
        "probability_rank": rank,
        "forecast_odds": odds,
    }


def _fit(records: list[dict[str, object]], **kwargs: object):
    options: dict[str, object] = {
        "prediction_date": "2026-02-01",
        "bootstrap_samples": 200,
        "seed": 741,
        "min_days": 1,
        "min_tickets": 1,
        "min_candidate_days": 1,
        "min_rank_days": 3,
        "min_rank_tickets": 12,
        "min_cell_days": 3,
        "min_cell_tickets": 12,
        "rank_prior_tickets": 5.0,
        "cell_prior_tickets": 5.0,
    }
    options.update(kwargs)
    return fit_contextual_empirical_ev_calibration(records, **options)


def test_top5_and_non_top5_contexts_can_calibrate_differently() -> None:
    records = [
        _record(day, rank, 12.0, returned)
        for day in range(6)
        for rank, returned in ((1, 2.0), (3, 2.0), (8, 0.2), (15, 0.2))
    ]

    artifact = _fit(records)
    top5 = artifact.predict(1.12, 1, 12.0)
    non_top5 = artifact.predict(1.12, 8, 12.0)

    assert top5["rank_group"] == "top5"
    assert non_top5["rank_group"] == "6-20"
    assert top5["calibration_level"] == "rank_odds_cell"
    assert non_top5["calibration_level"] == "rank_odds_cell"
    assert top5["positive_return_days"] == 6
    assert top5["return_hhi"] == pytest.approx(1.0 / 6.0)
    assert artifact.calibration_version == 3
    assert top5["empirical_ev"] > non_top5["empirical_ev"]


def test_sparse_cells_fall_back_to_parent_then_global_conservatively() -> None:
    records = [
        _record(day, 1, 12.0, 0.8)
        for day in range(12)
        for _ in range(2)
    ]
    records.append(_record(0, 1, 150.0, 20.0))

    artifact = _fit(records, min_cell_days=5, min_cell_tickets=10)
    sparse = artifact.predict(1.12, 1, 150.0)
    same_rank_empty = artifact.predict(1.12, 1, 35.0)
    unsupported_rank = artifact.predict(1.12, 21, 150.0)
    global_prediction = artifact.global_calibration.predict(1.12)

    assert sparse["cell_ready"] is False
    assert sparse["purchase_lcb95_available"] is False
    assert sparse["calibration_level"] == "rank_group"
    assert sparse["empirical_ev"] == same_rank_empty["empirical_ev"]
    assert sparse["empirical_ev_lcb95"] == same_rank_empty["empirical_ev_lcb95"]
    assert sparse["empirical_ev"] < 20.0
    assert unsupported_rank["calibration_level"] == "global"
    assert unsupported_rank["purchase_lcb95_available"] is False
    assert unsupported_rank["empirical_ev"] == global_prediction["empirical_ev"]
    assert (
        unsupported_rank["empirical_ev_lcb95"]
        == global_prediction["empirical_ev_lcb95"]
    )


def test_lcb_never_exceeds_point_estimate_in_any_context() -> None:
    records = [
        _record(
            day,
            rank,
            odds,
            float((day + rank + int(odds)) % 5),
            raw_ev=raw_ev,
        )
        for day in range(10)
        for rank in (1, 8, 21)
        for odds in (12.0, 30.0, 70.0, 120.0)
        for raw_ev in (1.02, 1.12)
    ]

    artifact = _fit(records)

    for cell in artifact.cells:
        for bin_ in cell.bins:
            if bin_.empirical_ev is not None:
                assert bin_.empirical_ev_lcb95 is not None
                assert bin_.empirical_ev_lcb95 <= bin_.empirical_ev


def test_context_prediction_propagates_global_local_range_gate() -> None:
    records = [
        _record(day, 1, 12.0, 1.2, raw_ev=1.12)
        for day in range(20)
        for _ in range(4)
    ]
    artifact = _fit(
        records,
        min_rank_days=1,
        min_rank_tickets=1,
        min_cell_days=1,
        min_cell_tickets=1,
    )

    supported = artifact.predict(1.12, 1, 12.0)
    outside = artifact.predict(1.50, 1, 12.0)

    assert supported["local_support_ready"] is True
    assert supported["input_in_local_block_range"] is True
    assert supported["purchase_lcb95_available"] is True
    assert outside["input_in_local_block_range"] is False
    assert outside["purchase_lcb95_available"] is False


def test_hierarchical_shrinkage_preserves_raw_ev_monotonicity() -> None:
    records = []
    for day in range(20):
        records.extend(
            _record(day, 1, 12.0, 10.0, raw_ev=1.02) for _ in range(10)
        )
        records.append(_record(day, 1, 12.0, 10.0, raw_ev=1.07))
        records.extend(
            _record(day, 8, 12.0, 0.0, raw_ev=1.02) for _ in range(200)
        )
        records.extend(
            _record(day, 8, 12.0, 1.0, raw_ev=1.07) for _ in range(200)
        )

    artifact = _fit(
        records,
        bootstrap_samples=500,
        min_rank_days=1,
        min_rank_tickets=1,
        min_cell_days=1,
        min_cell_tickets=1,
        rank_prior_tickets=100.0,
        cell_prior_tickets=100.0,
    )
    predictions = [artifact.predict(raw_ev, 1, 12.0) for raw_ev in (1.02, 1.07)]

    assert predictions[0]["empirical_ev"] <= predictions[1]["empirical_ev"]
    assert predictions[0]["empirical_ev_lcb95"] <= predictions[1]["empirical_ev_lcb95"]


def test_bandwise_shape_preserves_profitable_middle_band_without_tail_pooling() -> None:
    records = []
    for day in range(20):
        records.extend(
            _record(day, 1, 12.0, 1.2, raw_ev=1.02) for _ in range(20)
        )
        records.extend(
            _record(day, 1, 12.0, 0.2, raw_ev=1.15) for _ in range(20)
        )

    isotonic = _fit(records, shape_constraint="isotonic")
    bandwise = _fit(records, shape_constraint="bandwise")

    assert isotonic.predict(1.02, 1, 12.0)["empirical_ev"] == pytest.approx(0.7)
    assert bandwise.predict(1.02, 1, 12.0)["empirical_ev"] > 1.0
    assert bandwise.predict(1.15, 1, 12.0)["empirical_ev"] < 1.0
    assert bandwise.shape_constraint == "bandwise"
    assert bandwise.global_calibration.shape_constraint == "bandwise"


def test_day_bootstrap_recomputes_shrinkage_from_resampled_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FirstDayRng:
        def integers(
            self,
            low: int,
            high: int | None = None,
            size: int | None = None,
        ) -> np.ndarray:
            del low, high
            return np.zeros(size, dtype=np.int64)

    monkeypatch.setattr(
        calibration.np.random,
        "default_rng",
        lambda seed: _FirstDayRng(),
    )
    records = [
        *[_record(0, 1, 12.0, 0.0) for _ in range(10)],
        *[_record(0, 8, 12.0, 2.0) for _ in range(10)],
        *[_record(1, 8, 12.0, 2.0) for _ in range(10)],
    ]

    artifact = _fit(
        records,
        prediction_date="2026-01-03",
        bin_edges=(-float("inf"), float("inf")),
        bootstrap_samples=100,
        min_rank_days=1,
        min_rank_tickets=1,
        min_cell_days=1,
        min_cell_tickets=1,
        rank_prior_tickets=10.0,
        cell_prior_tickets=10.0,
    )

    # Resampling day zero twice gives 20 child tickets at each hierarchy.
    # Re-estimated weights are 20 / (20 + 10), yielding 1/3 then 1/9.
    assert artifact.predict(1.12, 1, 12.0)["empirical_ev_lcb95"] == pytest.approx(
        1.0 / 9.0
    )


def test_same_seed_and_records_produce_identical_artifact() -> None:
    records = [
        _record(day, rank, odds, float((day + rank) % 3))
        for day in range(8)
        for rank in (1, 7, 25)
        for odds in (10.0, 25.0, 75.0, 150.0)
    ]

    first = _fit(records)
    second = _fit(list(reversed(records)))

    assert first == second
    assert first.as_dict() == second.as_dict()


def test_prediction_date_strictly_isolates_future_results() -> None:
    past = [
        _record(day, rank, 15.0, returned)
        for day in range(6)
        for rank, returned in ((1, 1.5), (8, 0.5))
        for _ in range(2)
    ]
    future_low = _record(20, 1, 15.0, 0.0)
    future_high = _record(20, 1, 15.0, 10_000.0)

    low = _fit(past + [future_low], prediction_date="2026-01-15")
    high = _fit(past + [future_high], prediction_date="2026-01-15")

    assert low.predict(1.12, 1, 15.0) == high.predict(1.12, 1, 15.0)
    assert low.global_calibration == high.global_calibration
    assert low.trained_through_date == "2026-01-06"
    assert high.trained_through_date == "2026-01-06"
    assert low.prediction_date == "2026-01-15"
    assert low.excluded_non_past_records == 1
    assert high.excluded_non_past_records == 1


def test_readiness_metadata_and_segment_boundaries_are_auditable() -> None:
    artifact = fit_contextual_empirical_ev_calibration(
        [_record(0, 1, 20.0, 1.0)],
        prediction_date="2026-01-02",
        bootstrap_samples=100,
    )

    assert artifact.ready is False
    assert artifact.trained_through_date == "2026-01-01"
    assert artifact.training_days == 1
    assert artifact.tickets == 1
    assert artifact.context_ready_cells == 0
    assert set(artifact.ready_reasons) == {
        "insufficient_training_days",
        "insufficient_tickets",
        "insufficient_candidate_days",
    }
    assert artifact.predict(1.12, 5, 19.999)["rank_group"] == "top5"
    assert artifact.predict(1.12, 6, 20.0)["rank_group"] == "6-20"
    assert artifact.predict(1.12, 20, 50.0)["odds_band"] == "50-101"
    assert artifact.predict(1.12, 21, 101.0)["odds_band"] == ">=101"
    assert len(artifact.cells) == len(RANK_GROUPS) * len(ODDS_BANDS)


def test_invalid_context_inputs_are_rejected() -> None:
    artifact = _fit([_record(0, 1, 10.0, 1.0)])

    with pytest.raises(ValueError, match="positive integer"):
        artifact.predict(1.1, 0, 10.0)
    with pytest.raises(ValueError, match="forecast_odds"):
        artifact.predict(1.1, 1, -1.0)
    with pytest.raises(ValueError, match="finite"):
        _fit([_record(0, 1, 10.0, 1.0)], rank_prior_tickets=float("nan"))
