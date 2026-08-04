from __future__ import annotations

from datetime import date, timedelta

from boatrace_ai.listwise import nonlinear_context_search_v41 as subject


def _races(start: date, days: int, *, prefix: str) -> list[dict]:
    return [
        {
            "race_id": f"{prefix}-{index}",
            "race_date": (start + timedelta(days=index)).isoformat(),
        }
        for index in range(days)
    ]


def test_context_search_selects_on_inner_period_and_refits(monkeypatch) -> None:
    calibration = _races(date(2026, 1, 1), 10, prefix="cal")
    evaluation = _races(date(2026, 1, 11), 3, prefix="outer")
    fit_calls: list[tuple[list[str], tuple[str, ...], str]] = []

    def fake_fit(races, *, tree_preset, context_features, num_threads):
        fit_calls.append((
            [race["race_date"] for race in races],
            tuple(context_features),
            str(tree_preset["name"]),
        ))
        return {
            "context_features": list(context_features),
            "tree_preset": str(tree_preset["name"]),
        }

    def fake_metrics(races, artifact, *, shrinkage):
        # Full context wins the inner comparison. Outer contents cannot affect it.
        loss = 0.8 if artifact["context_features"] == ["a", "b"] else 0.9
        return {
            "trifecta_log_loss": loss + float(shrinkage),
            "evaluated_races": len(races),
        }

    monkeypatch.setattr(subject, "fit_nonlinear_market_residual", fake_fit)
    monkeypatch.setattr(subject, "nonlinear_residual_metrics", fake_metrics)
    result = subject.fit_temporal_nonlinear_context_search(
        calibration,
        evaluation,
        context_variants={"core": ("a",), "full": ("a", "b")},
        tree_presets=({
            "name": "tiny",
            "num_leaves": 3,
            "max_depth": 2,
            "min_child_samples": 2,
        },),
        shrinkages=(0.0, 0.5),
        num_threads=1,
    )

    assert result["selected_context_variant"] == "full"
    assert result["selected_shrinkage"] == 0.0
    assert result["outer_period_used_for_selection"] is False
    assert result["inner_fit_through"] < result["inner_validation_from"]
    assert result["inner_validation_from"] < evaluation[0]["race_date"]
    assert fit_calls[-1][0] == [race["race_date"] for race in calibration]
    assert fit_calls[-1][1] == ("a", "b")
