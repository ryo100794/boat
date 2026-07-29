from __future__ import annotations

from pathlib import Path
from typing import Any

from boatrace_ai.db import connection, init_db
from boatrace_ai.feature_tuning import iter_complete_races, iter_race_feature_rows


class _Cursor:
    def __init__(self, cursor: Any, events: list[str], label: str) -> None:
        self._cursor = cursor
        self._events = events
        self._label = label

    def fetchall(self):
        rows = self._cursor.fetchall()
        self._events.append(f"fetchall:{self._label}")
        return rows

    def __iter__(self):
        return iter(self._cursor)


class _RecordingConnection:
    def __init__(self, conn: Any, *, dialect: str) -> None:
        self._conn = conn
        self.dialect = dialect
        self.events: list[str] = []

    def execute(self, statement: str, params: Any = ()) -> _Cursor:
        normalized = " ".join(statement.split())
        if "SELECT DISTINCT race_date" in normalized:
            label = "dates"
        elif "FROM races r INDEXED BY" in normalized:
            label = "race_chunk"
        else:
            label = "other"
        self.events.append(f"execute:{label}")
        return _Cursor(self._conn.execute(statement, params), self.events, label)

    def executescript(self, statement: str) -> None:
        self._conn.executescript(statement)

    def commit(self) -> None:
        self.events.append("commit")
        self._conn.commit()


def _fixture(path: Path, *, days: int = 32) -> None:
    init_db(path)
    with connection(path) as conn:
        for day in range(days):
            race_id = f"race-{day:02d}"
            race_date = f"2026-{day // 28 + 1:02d}-{day % 28 + 1:02d}"
            conn.execute(
                """
                INSERT INTO races(race_id, race_date, jcd, venue_name, rno, status)
                VALUES (?, ?, '01', 'fixture', 1, 'final')
                """,
                (race_id, race_date),
            )
            for lane in range(1, 7):
                conn.execute(
                    """
                    INSERT INTO entries(race_id, lane, racer_no, racer_name)
                    VALUES (?, ?, ?, ?)
                    """,
                    (race_id, lane, 1000 + lane, f"racer-{lane}"),
                )
                conn.execute(
                    """
                    INSERT INTO race_results(race_id, lane, rank)
                    VALUES (?, ?, ?)
                    """,
                    (race_id, lane, lane),
                )


def _identity(rows: list[list[Any]]) -> list[list[tuple[Any, ...]]]:
    return [
        [
            (
                row["race_id"],
                row["race_date"],
                row["jcd"],
                int(row["rno"]),
                int(row["lane"]),
                int(row["rank"]),
            )
            for row in race
        ]
        for race in rows
    ]


def test_postgresql_materializes_and_commits_each_read_before_yield(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "features.sqlite"
    _fixture(db_path)

    with connection(db_path) as sqlite_conn:
        expected = _identity(list(iter_complete_races(sqlite_conn)))

    with connection(db_path) as sqlite_conn:
        conn = _RecordingConnection(sqlite_conn, dialect="postgresql")
        iterator = iter_complete_races(conn)
        first_race = next(iterator)

        assert conn.events[-1] == "commit"
        assert conn.events.count("fetchall:race_chunk") == 1
        assert conn.events.count("commit") == 2
        actual = _identity([first_race, *iterator])

    assert actual == expected
    assert conn.events.count("fetchall:race_chunk") == 2
    assert conn.events.count("commit") == 3


def test_sqlite_does_not_gain_chunk_commits(tmp_path: Path) -> None:
    db_path = tmp_path / "features.sqlite"
    _fixture(db_path, days=1)

    with connection(db_path) as sqlite_conn:
        conn = _RecordingConnection(sqlite_conn, dialect="sqlite")
        rows = list(iter_complete_races(conn))

    assert len(rows) == 1
    assert "commit" not in conn.events


def test_postgresql_chunk_commits_preserve_feature_vectors(tmp_path: Path) -> None:
    db_path = tmp_path / "features.sqlite"
    _fixture(db_path)

    with connection(db_path) as sqlite_conn:
        expected = list(iter_race_feature_rows(sqlite_conn))

    with connection(db_path) as sqlite_conn:
        conn = _RecordingConnection(sqlite_conn, dialect="postgresql")
        actual = list(iter_race_feature_rows(conn))

    assert actual == expected
