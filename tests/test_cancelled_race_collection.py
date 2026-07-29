from datetime import date
from pathlib import Path

from boatrace_ai import modeling
from boatrace_ai.db import connection, init_db
from boatrace_ai.ingestion import live
from boatrace_ai.runtime.collector import scheduled_races


def test_cancelled_result_is_persisted_and_excluded_from_prediction(
    monkeypatch, tmp_path
) -> None:
    database = tmp_path / "cancelled.sqlite"
    race_date = date(2026, 7, 29)
    race_id = "2026-07-29-09-01"
    init_db(database)
    monkeypatch.setattr(
        live,
        "_fetch_page",
        lambda *args, **kwargs: (
            "<html><body><p>該当レースは中止となりました。</p></body></html>"
        ),
    )

    with connection(database) as conn:
        assert live.collect_result(
            conn,
            race_date=race_date,
            jcd="09",
            rno=1,
            raw_dir=tmp_path,
        ) == 0

        # Later racelist/odds collection still reports scheduled; it must not
        # reopen a terminal race.
        live._ensure_minimal_race(
            conn, race_date=race_date, jcd="09", rno=1, status="scheduled"
        )
        conn.execute(
            "UPDATE races SET deadline_at = ? WHERE race_id = ?",
            ("2026-07-29T10:38:00+09:00", race_id),
        )
        for lane in range(1, 7):
            conn.execute(
                "INSERT INTO entries (race_id, lane, racer_no) VALUES (?, ?, ?)",
                (race_id, lane, 5000 + lane),
            )

        race = conn.execute(
            "SELECT status FROM races WHERE race_id = ?", (race_id,)
        ).fetchone()
        result = conn.execute(
            "SELECT status, trifecta_evaluable, reason "
            "FROM race_result_status WHERE race_id = ?",
            (race_id,),
        ).fetchone()
        assert race["status"] == "final"
        assert result["status"] == "final"
        assert result["trifecta_evaluable"] == 0
        assert result["reason"] == "race_cancelled"
        assert scheduled_races(conn, race_date) == []

        predicted: list[str] = []
        monkeypatch.setattr(
            modeling,
            "predict_race",
            lambda _conn, **kwargs: predicted.append(kwargs["race_id_value"]),
        )
        counts = modeling.predict_open_races(
            conn, model_path=Path("unused.joblib"), race_date=race_date
        )

    assert counts == {"predicted": 0, "failed": 0}
    assert predicted == []
