from __future__ import annotations

import os
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import psycopg


_MEMORY_SETTING = re.compile(r"[1-9][0-9]*(?:kB|MB|GB)", re.IGNORECASE)


def _dollar_quote_delimiter(statement: str, index: int) -> str | None:
    if index > 0 and (
        statement[index - 1].isalnum() or statement[index - 1] in "_$"
    ):
        return None
    end = statement.find("$", index + 1)
    if end < 0:
        return None
    tag = statement[index + 1 : end]
    if tag and not (
        (tag[0].isalpha() or tag[0] == "_")
        and all(char.isalnum() or char == "_" for char in tag[1:])
    ):
        return None
    return statement[index : end + 1]


def _convert_placeholders(statement: str) -> str:
    converted: list[str] = []
    index = 0
    state = "sql"
    block_comment_depth = 0
    dollar_delimiter: str | None = None

    while index < len(statement):
        char = statement[index]
        following = statement[index + 1] if index + 1 < len(statement) else ""

        if state == "sql":
            if char == "$" and (
                delimiter := _dollar_quote_delimiter(statement, index)
            ) is not None:
                converted.append(delimiter)
                index += len(delimiter)
                dollar_delimiter = delimiter
                state = "dollar_quote"
                continue
            if char == "'":
                state = "single_quote"
            elif char == '"':
                state = "double_quote"
            elif char == "-" and following == "-":
                converted.append("--")
                index += 2
                state = "line_comment"
                continue
            elif char == "/" and following == "*":
                converted.append("/*")
                index += 2
                block_comment_depth = 1
                state = "block_comment"
                continue
            elif char == "?":
                converted.append("%s")
                index += 1
                continue
            elif (
                char == ":"
                and following != ":"
                and (index == 0 or statement[index - 1] != ":")
                and following.isascii()
                and (following.isalpha() or following == "_")
            ):
                end = index + 2
                while end < len(statement) and (
                    statement[end].isascii()
                    and (statement[end].isalnum() or statement[end] == "_")
                ):
                    end += 1
                converted.append(f"%({statement[index + 1:end]})s")
                index = end
                continue

            converted.append(char)
            index += 1
            continue

        if state == "dollar_quote":
            delimiter = dollar_delimiter
            if delimiter is not None and statement.startswith(delimiter, index):
                converted.append(delimiter)
                index += len(delimiter)
                dollar_delimiter = None
                state = "sql"
            else:
                converted.append(char)
                index += 1
            continue

        if state in {"single_quote", "double_quote"}:
            quote = "'" if state == "single_quote" else '"'
            converted.append(char)
            index += 1
            if char == "\\" and index < len(statement):
                converted.append(statement[index])
                index += 1
            elif char == quote:
                if index < len(statement) and statement[index] == quote:
                    converted.append(statement[index])
                    index += 1
                else:
                    state = "sql"
            continue

        if state == "line_comment":
            converted.append(char)
            index += 1
            if char in "\r\n":
                state = "sql"
            continue

        converted.append(char)
        index += 1
        if char == "/" and following == "*":
            converted.append(following)
            index += 1
            block_comment_depth += 1
        elif char == "*" and following == "/":
            converted.append(following)
            index += 1
            block_comment_depth -= 1
            if block_comment_depth == 0:
                state = "sql"

    return "".join(converted)


class CompatRow(Sequence[Any]):
    def __init__(
        self, values: Sequence[Any] | Mapping[str, Any], names: Sequence[str]
    ) -> None:
        self._values = (
            tuple(values[name] for name in names)
            if isinstance(values, Mapping)
            else tuple(values)
        )
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

    @property
    def rowcount(self) -> int:
        if self._has_scalar:
            return 1
        if self._cursor is None:
            return -1
        return int(self._cursor.rowcount)

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
    converted = _convert_placeholders(converted)
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

    def executescript(self, statement: str) -> None:
        self._raw.execute(statement, prepare=False)

    def commit(self) -> None:
        self._raw.commit()

    def rollback(self) -> None:
        self._raw.rollback()

    def close(self) -> None:
        self._raw.close()


def _connection_options() -> dict[str, Any]:
    evaluation_process = bool(os.environ.get("BOATRACE_EVAL_MAX_RACE_DATE"))
    default_work_mem = "128MB" if evaluation_process else ""
    work_mem = os.environ.get("BOATRACE_PG_WORK_MEM", default_work_mem).strip()
    if work_mem and _MEMORY_SETTING.fullmatch(work_mem) is None:
        raise ValueError("BOATRACE_PG_WORK_MEM must be a positive kB, MB, or GB value")
    options: dict[str, Any] = {
        "connect_timeout": 30,
        "application_name": os.environ.get(
            "BOATRACE_PG_APPLICATION_NAME",
            "boatrace_evaluator"
            if evaluation_process
            else "boatrace_realtime_collector",
        ),
    }
    if work_mem:
        options["options"] = f"-c work_mem={work_mem}"
    return options


@contextmanager
def connection(dsn: str) -> Iterator[Connection]:
    raw = psycopg.connect(dsn, **_connection_options())
    wrapped = Connection(raw)
    try:
        yield wrapped
        wrapped.commit()
    except Exception:
        wrapped.rollback()
        raise
    finally:
        wrapped.close()
