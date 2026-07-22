from boatrace_ai.listwise import market_calibration


def _snapshot(*, captured_at: str, deadline_at: str) -> dict:
    return {
        "snapshot_id": 10,
        "captured_at": captured_at,
        "odds_deadline_at": deadline_at,
        "odds": {f"{first}-{second}-{third}": 10.0 for first in range(1, 7) for second in range(1, 7) for third in range(1, 7) if len({first, second, third}) == 3},
    }


def test_earlier_market_fields_requires_fresh_ordered_snapshot(monkeypatch) -> None:
    earlier = _snapshot(
        captured_at="2026-07-22T10:00:00+09:00",
        deadline_at="2026-07-22T10:00:30+09:00",
    )
    current = _snapshot(
        captured_at="2026-07-22T10:05:00+09:00",
        deadline_at="2026-07-22T10:05:30+09:00",
    )
    monkeypatch.setattr(
        market_calibration,
        "latest_trifecta_odds_before_deadline",
        lambda *args, **kwargs: earlier,
    )

    fields, reason = market_calibration.earlier_market_fields(
        object(),
        "2026-07-22-01-01",
        current_snapshot=current,
        max_snapshot_age_seconds=60.0,
    )

    assert reason == "ok"
    assert len(fields["earlier_market_probabilities"]) == 120
    assert fields["momentum_interval_seconds"] == 300.0
    assert fields["momentum_scale"] == 1.0


def test_earlier_market_fields_rejects_stale_snapshot(monkeypatch) -> None:
    earlier = _snapshot(
        captured_at="2026-07-22T09:58:00+09:00",
        deadline_at="2026-07-22T10:00:30+09:00",
    )
    current = _snapshot(
        captured_at="2026-07-22T10:05:00+09:00",
        deadline_at="2026-07-22T10:05:30+09:00",
    )
    monkeypatch.setattr(
        market_calibration,
        "latest_trifecta_odds_before_deadline",
        lambda *args, **kwargs: earlier,
    )

    fields, reason = market_calibration.earlier_market_fields(
        object(),
        "2026-07-22-01-01",
        current_snapshot=current,
        max_snapshot_age_seconds=60.0,
    )

    assert fields is None
    assert reason == "stale"
