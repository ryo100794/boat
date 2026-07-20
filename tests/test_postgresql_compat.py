from boatrace_ai.postgresql import CompatRow, convert_sql


def test_compat_row_supports_index_and_column_name() -> None:
    row = CompatRow(("01", 4), ("jcd", "rno"))
    assert row[0] == "01"
    assert row["rno"] == 4
    assert tuple(row) == ("01", 4)
    assert list(row.keys()) == ["jcd", "rno"]


def test_qmark_and_named_parameters_are_converted() -> None:
    assert convert_sql("SELECT * FROM races WHERE race_date = ?").endswith(
        "race_date = %s"
    )
    assert "%(race_id)s" in convert_sql(
        "SELECT * FROM races WHERE race_id = :race_id"
    )


def test_sqlite_replace_forms_become_postgresql_upserts() -> None:
    odds = convert_sql(
        "INSERT OR REPLACE INTO odds_trifecta "
        "(snapshot_id, race_id, combination, odds) VALUES (?, ?, ?, ?)"
    )
    assert odds.startswith("INSERT INTO odds_trifecta")
    assert "ON CONFLICT (snapshot_id, combination)" in odds

    beforeinfo = convert_sql(
        "INSERT OR REPLACE INTO beforeinfo "
        "(race_id, captured_at, lane) VALUES (?, ?, ?)"
    )
    assert beforeinfo.startswith("INSERT INTO beforeinfo")
    assert "ON CONFLICT (race_id, captured_at, lane)" in beforeinfo
