from __future__ import annotations

from datetime import date, timedelta

import pytest

from boatrace_ai.listwise.empirical_ev_calibration import (
    fit_empirical_ev_calibration,
)


def _record(
    day: int, raw_ev: float, returned: float, *, weight: float = 1.0
) -> dict[str, object]:
    race_date = date(2026, 1, 1) + timedelta(days=day)
    return {
        "race_date": race_date.isoformat(),
        "raw_estimated_ev": raw_ev,
        "gross_return_per_yen": returned,
        "sample_weight": weight,
    }


def test_empty_data_returns_auditable_not_ready_artifact() -> None:
    artifact = fit_empirical_ev_calibration([], bootstrap_samples=100)

    assert artifact.ready is False
    assert artifact.trained_through_date is None
    assert artifact.training_days == 0
    assert artifact.tickets == 0
    assert artifact.total_exposure_weight == 0.0
    assert artifact.training_raw_ev_min is None
    assert artifact.training_raw_ev_max is None
    assert set(artifact.ready_reasons) == {
        "insufficient_training_days",
        "insufficient_tickets",
        "insufficient_candidate_days",
    }
    prediction = artifact.predict(1.08)
    assert prediction["support"] == 0
    assert prediction["empirical_ev"] is None
    assert prediction["empirical_ev_lcb95"] is None
    assert prediction["input_in_training_range"] is False


def test_ready_gate_counts_days_tickets_and_candidate_days_separately() -> None:
    records = [
        _record(day, 1.02 if day < 2 else 0.95, 0.8)
        for day in range(3)
        for _ in range(4)
    ]

    artifact = fit_empirical_ev_calibration(
        records,
        bootstrap_samples=100,
        min_days=3,
        min_tickets=12,
        min_candidate_days=3,
    )

    assert artifact.ready is False
    assert artifact.training_days == 3
    assert artifact.tickets == 12
    assert artifact.candidate_days == 2
    assert artifact.ready_reasons == ("insufficient_candidate_days",)
    assert artifact.trained_through_date == "2026-01-03"
    assert artifact.as_dict()["ready"] is False
    assert artifact.predict(1.02)["empirical_ev"] is not None


def test_weighted_pava_makes_bin_predictions_monotone() -> None:
    records = []
    for day in range(4):
        records.extend(
            [
                _record(day, 0.98, 0.80),
                _record(day, 1.02, 1.40),
                _record(day, 1.07, 0.20),
                _record(day, 1.15, 1.10),
            ]
        )

    artifact = fit_empirical_ev_calibration(
        records,
        bootstrap_samples=200,
        min_days=1,
        min_tickets=1,
        min_candidate_days=1,
    )
    points = [
        artifact.predict(raw_ev)["empirical_ev"]
        for raw_ev in (0.98, 1.02, 1.07, 1.15)
    ]

    assert points == sorted(points)
    assert points[1] == pytest.approx(points[2])
    assert points[1] == pytest.approx(0.8)


def test_predict_reports_fixed_bin_bounds_and_support() -> None:
    artifact = fit_empirical_ev_calibration(
        [_record(0, 1.01, 0.5), _record(0, 1.04, 1.5)],
        bootstrap_samples=100,
        min_days=1,
        min_tickets=1,
        min_candidate_days=1,
    )

    prediction = artifact.predict(1.03)
    assert prediction["lower"] == 1.0
    assert prediction["upper"] == 1.05
    assert prediction["support"] == 2
    assert prediction["support_days"] == 1
    assert prediction["positive_return_days"] == 1
    assert prediction["return_hhi"] == pytest.approx(1.0)
    assert prediction["empirical_ev"] == pytest.approx(1.0)
    assert prediction["training_raw_ev_min"] == pytest.approx(1.01)
    assert prediction["training_raw_ev_max"] == pytest.approx(1.04)
    assert prediction["input_in_training_range"] is True
    assert artifact.predict(1.005)["input_in_training_range"] is False
    assert artifact.predict(1.05)["input_in_training_range"] is False
    audit = artifact.as_dict()
    assert audit["lcb_tail_probability"] == pytest.approx(0.05)
    assert audit["lcb_confidence_level"] == pytest.approx(0.95)
    assert audit["lcb_sidedness"] == "one_sided_lower"
    assert audit["bootstrap_cluster_unit"] == "race_date"
    assert audit["bootstrap_resample_cluster_count"] == 1
    assert audit["within_day_candidates_resampled_together"] is True
    assert audit["ticket_level_independence_assumed"] is False
    assert audit["lcb_capped_at_point_estimate"] is True


def test_sample_weight_estimates_total_return_over_total_exposure() -> None:
    artifact = fit_empirical_ev_calibration(
        [
            _record(0, 1.03, 0.0, weight=900.0),
            _record(0, 1.03, 10.0, weight=100.0),
        ],
        bootstrap_samples=100,
        min_days=1,
        min_tickets=1,
        min_candidate_days=1,
    )

    prediction = artifact.predict(1.03)
    assert prediction["empirical_ev"] == pytest.approx(1.0)
    assert prediction["support"] == 2
    assert prediction["exposure_weight"] == 1_000.0
    assert artifact.total_exposure_weight == 1_000.0
    assert artifact.as_dict()["weighting"] == "optional_sample_weight_default_1"


def test_single_extreme_payout_does_not_raise_daily_cluster_lcb() -> None:
    records = []
    for day in range(30):
        records.append(_record(day, 1.08, 60.0 if day == 0 else 0.0))

    artifact = fit_empirical_ev_calibration(
        records,
        bootstrap_samples=2_000,
        seed=31,
        min_days=1,
        min_tickets=1,
        min_candidate_days=1,
    )
    prediction = artifact.predict(1.08)

    assert prediction["empirical_ev"] == pytest.approx(2.0)
    assert prediction["empirical_ev_lcb95"] == pytest.approx(0.0)


def test_lcb_is_never_above_point_estimate() -> None:
    records = [
        _record(0, 1.08, 0.0),
        _record(1, 1.08, 10.0),
    ]
    artifact = fit_empirical_ev_calibration(
        records,
        bootstrap_samples=100,
        seed=2,
        min_days=1,
        min_tickets=1,
        min_candidate_days=1,
    )

    for bin_ in artifact.bins:
        if bin_.empirical_ev is not None:
            assert bin_.empirical_ev_lcb95 <= bin_.empirical_ev


def test_bootstrap_is_reproducible_for_the_same_seed() -> None:
    records = [
        _record(day, 1.02 if ticket % 2 else 1.12, float((day + ticket) % 4))
        for day in range(12)
        for ticket in range(3)
    ]
    kwargs = {
        "bootstrap_samples": 500,
        "seed": 991,
        "min_days": 1,
        "min_tickets": 1,
        "min_candidate_days": 1,
    }

    first = fit_empirical_ev_calibration(records, **kwargs)
    second = fit_empirical_ev_calibration(records, **kwargs)

    assert first == second
    assert first.as_dict() == second.as_dict()


def test_rejects_invalid_records_and_configuration() -> None:
    with pytest.raises(ValueError, match="gross_return_per_yen"):
        fit_empirical_ev_calibration(
            [_record(0, 1.1, -1.0)],
            bootstrap_samples=100,
        )
    with pytest.raises(ValueError, match="sample_weight"):
        fit_empirical_ev_calibration(
            [_record(0, 1.1, 1.0, weight=0.0)],
            bootstrap_samples=100,
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        fit_empirical_ev_calibration(
            [],
            bin_edges=(-float("inf"), 1.0, 1.0, float("inf")),
            bootstrap_samples=100,
        )
    with pytest.raises(ValueError, match="float64 range"):
        fit_empirical_ev_calibration(
            [_record(0, 1.02, 1e308), _record(0, 1.02, 1e308)],
            bootstrap_samples=100,
            min_days=1,
            min_tickets=1,
            min_candidate_days=1,
        )
