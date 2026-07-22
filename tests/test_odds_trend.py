from pathlib import Path

from boatrace_ai.db import (
    connection,
    init_db,
    insert_odds_snapshot,
    insert_prediction_rows,
)
from boatrace_ai.web import dashboard


def test_odds_chart_draws_five_ranked_series_and_labeled_axes() -> None:
    html = (
        Path(__file__).parents[1]
        / "src"
        / "boatrace_ai"
        / "templates"
        / "dashboard.html"
    ).read_text(encoding="utf-8")

    assert "series.slice(0,5)" in html
    assert "`${index+1}位 ${item.combination" in html
    assert 'ctx.fillText("オッズ",0,0)' in html
    assert 'ctx.fillText("取得時刻 (JST)"' in html
    assert "ctx.moveTo(plot.l-4,py)" in html
    assert "ctx.lineTo(px,h-plot.b+4)" in html


def test_odds_trend_returns_latest_model_top_five_with_history(tmp_path) -> None:
    db_path = tmp_path / "odds.sqlite"
    race_id = "2026-07-22-01-01"
    init_db(db_path)
    dashboard._ODDS_API_CACHE.clear()

    with connection(db_path) as conn:
        conn.execute(
            "INSERT INTO races(race_id, race_date, jcd, venue_name, rno) "
            "VALUES (?, ?, ?, ?, ?)",
            (race_id, "2026-07-22", "01", "桐生", 1),
        )
        insert_prediction_rows(
            conn,
            race_id,
            "2026-07-22T01:00:00+00:00",
            "old.joblib",
            [{"combination": "6-5-4", "probability": 0.9}],
        )
        latest = [
            ("1-2-3", 0.30),
            ("1-3-2", 0.25),
            ("2-1-3", 0.20),
            ("2-3-1", 0.15),
            ("3-1-2", 0.07),
            ("3-2-1", 0.03),
        ]
        insert_prediction_rows(
            conn,
            race_id,
            "2026-07-22T01:05:00+00:00",
            "current.joblib",
            [
                {"combination": combination, "probability": probability}
                for combination, probability in latest
            ],
        )
        for captured_at, shift in (
            ("2026-07-22T01:06:00+00:00", 0.0),
            ("2026-07-22T01:07:00+00:00", 1.0),
        ):
            insert_odds_snapshot(
                conn,
                race_id,
                captured_at,
                captured_at,
                {
                    combination: 10.0 + index + shift
                    for index, (combination, _) in enumerate(latest)
                },
                f"https://example.test/{captured_at}",
                {"parser_version": "odds3t_dom_v2"},
            )

        insert_odds_snapshot(
            conn,
            race_id,
            "2026-07-22T01:08:00+00:00",
            "2026-07-22T01:08:00+00:00",
            {
                combination: float(index % 6 + 1)
                for index, (combination, _) in enumerate(latest)
            },
            "https://example.test/legacy",
            {"parser_version": "odds3t_legacy"},
        )

    payload = dashboard.odds(
        db_path,
        {"race_id": [race_id], "combination": ["1-2-3"]},
    )

    assert [item["combination"] for item in payload["series"]] == [
        "1-2-3",
        "1-3-2",
        "2-1-3",
        "2-3-1",
        "3-1-2",
    ]
    assert all(len(item["trend"]) == 2 for item in payload["series"])
    assert "3-2-1" not in {item["combination"] for item in payload["series"]}
    assert "6-5-4" not in {item["combination"] for item in payload["series"]}
