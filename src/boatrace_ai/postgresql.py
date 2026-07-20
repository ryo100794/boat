from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import psycopg


_NAMED_PARAMETER = re.compile(r":([A-Za-z_][A-Za-z0-9_]*)")


class CompatRow(Sequence[Any]):
    def __init__(self, values: Sequence[Any], names: Sequence[str]) -> None:
        self._values = tuple(values)
        self._positions = {name: index for index, name in enumerate(names)}

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._values[self._positions[key]]
        return self._values[key]

    def __len__(self) -> int:
        return len(self._values)

    def keys(self):
        return self._positions.keys()


class CompatCursor:
    def __init__(self, cursor, *, scalar: Any = None, has_scalar: bool = False) -> None:
        self._cursor = cursor
        self._scalar = scalar
        self._has_scalar = has_scalar

    def _names(self) -> list[str]:
        if self._cursor is None or self._cursor.description is None:
            return []
        return [column.name for column in self._cursor.description]

    def _row(self, value):
        return None if value is None else CompatRow(value, self._names())

    def fetchone(self):
        if self._has_scalar:
            self._has_scalar = False
            return CompatRow((self._scalar,), ("value",))
        return self._row(self._cursor.fetchone())

    def fetchall(self):
        return [self._row(row) for row in self._cursor.fetchall()]

    def __iter__(self):
        if self._has_scalar:
            row = self.fetchone()
            if row is not None:
                yield row
            return
        names = self._names()
        for row in self._cursor:
            yield CompatRow(row, names)


def convert_sql(statement: str) -> str:
    converted = statement.strip()
    converted = re.sub(r"\s+INDEXED\s+BY\s+[A-Za-z_][A-Za-z0-9_]*", "", converted, flags=re.IGNORECASE)
    converted = converted.replace('races.status = "final"', "races.status = 'final'")
    converted = converted.replace('rp.page_type = "racelist"', "rp.page_type = 'racelist'")
    converted = converted.replace("INSERT OR REPLACE INTO odds_trifecta", "INSERT INTO odds_trifecta")
    converted = converted.replace("INSERT OR REPLACE INTO beforeinfo", "INSERT INTO beforeinfo")
    converted = _NAMED_PARAMETER.sub(r"%(\1)s", converted)
    converted = converted.replace("?", "%s")
    if converted.startswith("INSERT INTO odds_trifecta") and "ON CONFLICT" not in converted:
        converted += (
            " ON CONFLICT (snapshot_id, combination) DO UPDATE SET "
            "race_id=excluded.race_id, odds=excluded.odds"
        )
    if converted.startswith("INSERT INTO beforeinfo") and "ON CONFLICT" not in converted:
        converted += (
            " ON CONFLICT (race_id, captured_at, lane) DO UPDATE SET "
            "weight_kg=excluded.weight_kg, exhibition_time=excluded.exhibition_time, "
            "tilt=excluded.tilt, adjusted_weight=excluded.adjusted_weight, "
            "propeller=excluded.propeller, parts_exchange=excluded.parts_exchange, "
            "course=excluded.course, start_timing=excluded.start_timing, "
            "weather=excluded.weather, wind_direction=excluded.wind_direction, "
            "wind_speed_m=excluded.wind_speed_m, air_temp_c=excluded.air_temp_c, "
            "water_temp_c=excluded.water_temp_c, wave_cm=excluded.wave_cm, "
            "raw_json=excluded.raw_json"
        )
    return converted


class Connection:
    dialect = "postgresql"
    def __init__(self, raw: psycopg.Connection) -> None:
        self._raw = raw
        self._last_insert_id: int | None = None

    def execute(self, statement: str, params: Any = None) -> CompatCursor:
        if statement.strip().upper() == "SELECT LAST_INSERT_ROWID()":
            return CompatCursor(None, scalar=self._last_insert_id, has_scalar=True)
        converted = convert_sql(statement)
        if converted.startswith("INSERT INTO odds_snapshots") and "RETURNING" not in converted:
            cursor = self._raw.execute(converted + " RETURNING snapshot_id", params)
            self._last_insert_id = int(cursor.fetchone()[0])
            return CompatCursor(cursor)
        return CompatCursor(self._raw.execute(converted, params))

    def executemany(self, statement: str, params_seq) -> CompatCursor:
        cursor = self._raw.cursor()
        cursor.executemany(convert_sql(statement), params_seq)
        return CompatCursor(cursor)

    def commit(self) -> None:
        self._raw.commit()

    def rollback(self) -> None:
        self._raw.rollback()

    def close(self) -> None:
        self._raw.close()


@contextmanager
def connection(dsn: str) -> Iterator[Connection]:
    raw = psycopg.connect(dsn, connect_timeout=30, application_name="boatrace_realtime_collector")
    wrapped = Connection(raw)
    try:
        yield wrapped
        wrapped.commit()
    except Exception:
        wrapped.rollback()
        raise
    finally:
        wrapped.close()
