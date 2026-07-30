from __future__ import annotations

from boatrace_ai.listwise.odds_path_conservative_v7 import (
    _crossfit_probability_rows,
)


def test_probability_crossfit_holds_out_complete_days() -> None:
    races = [
        {"race_id": str(day), "race_date": f"2026-07-{day:02d}"}
        for day in range(1, 7)
    ]
    boundaries: list[tuple[str, str]] = []

    def fit(rows):
        return {"trained_through": max(row["race_date"] for row in rows)}

    def attach(rows, model):
        held_date = str(rows[0]["race_date"])
        assert {str(row["race_date"]) for row in rows} == {held_date}
        assert str(model["trained_through"]) < held_date
        boundaries.append((str(model["trained_through"]), held_date))
        return list(rows)

    rows = _crossfit_probability_rows(
        races,
        probability_fit=fit,
        probability_attach=attach,
    )

    assert len(rows) == 4
    assert boundaries == [
        ("2026-07-02", "2026-07-03"),
        ("2026-07-03", "2026-07-04"),
        ("2026-07-04", "2026-07-05"),
        ("2026-07-05", "2026-07-06"),
    ]
