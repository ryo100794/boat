import pytest

from boatrace_ai.postgresql import CompatRow, _connection_options, convert_sql


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
    cast_query = convert_sql(
        "SELECT value::numeric, captured_at::timestamptz FROM samples WHERE id = ?"
    )
    assert "value::numeric" in cast_query
    assert "captured_at::timestamptz" in cast_query
    assert cast_query.endswith("id = %s")


def test_placeholders_inside_quoted_strings_are_preserved() -> None:
    statement = (
        "SELECT ':invalid-source-job:', '?', 'it''s :literal?', "
        "\"column:name?\" FROM samples WHERE id = :sample_id AND lane = ?"
    )

    assert convert_sql(statement) == (
        "SELECT ':invalid-source-job:', '?', 'it''s :literal?', "
        "\"column:name?\" FROM samples WHERE id = %(sample_id)s AND lane = %s"
    )


def test_placeholders_inside_escaped_quoted_strings_are_preserved() -> None:
    statement = (
        "SELECT E'escaped\\':literal?', \"escaped\\\":identifier?\" "
        "FROM samples WHERE id = :sample_id"
    )

    assert convert_sql(statement) == (
        "SELECT E'escaped\\':literal?', \"escaped\\\":identifier?\" "
        "FROM samples WHERE id = %(sample_id)s"
    )


def test_placeholders_inside_sql_comments_are_preserved() -> None:
    statement = (
        "SELECT ? -- :line_parameter ?\n"
        "FROM samples /* :block_parameter ? /* :nested ? */ still ? */ "
        "WHERE id = :sample_id"
    )

    assert convert_sql(statement) == (
        "SELECT %s -- :line_parameter ?\n"
        "FROM samples /* :block_parameter ? /* :nested ? */ still ? */ "
        "WHERE id = %(sample_id)s"
    )


def test_placeholders_inside_anonymous_dollar_quote_are_preserved() -> None:
    statement = (
        "CREATE FUNCTION reject_mutation() RETURNS trigger AS $$\n"
        "BEGIN\n"
        "  RAISE EXCEPTION ':invalid-source-job: ?';\n"
        "END;\n"
        "$$ LANGUAGE plpgsql; SELECT ? WHERE id = :race_id"
    )

    assert convert_sql(statement) == statement.replace(
        "SELECT ? WHERE id = :race_id",
        "SELECT %s WHERE id = %(race_id)s",
    )


def test_placeholders_inside_tagged_dollar_quote_are_preserved() -> None:
    statement = (
        "$body$ :inside ? $$ :not_a_closer ? $$ $other$ ? $other$ $body$ "
        "SELECT :outside, ?"
    )

    assert convert_sql(statement) == (
        "$body$ :inside ? $$ :not_a_closer ? $$ $other$ ? $other$ $body$ "
        "SELECT %(outside)s, %s"
    )


def test_unterminated_dollar_quote_is_preserved() -> None:
    statement = "DO $body$ BEGIN PERFORM :inside; RAISE NOTICE '?';"

    assert convert_sql(statement) == statement


def test_postgresql_casts_are_preserved_next_to_named_parameters() -> None:
    statement = "SELECT :value::numeric, :_other::text, source:::invalid"

    assert convert_sql(statement) == (
        "SELECT %(value)s::numeric, %(_other)s::text, source:::invalid"
    )


def test_non_ascii_colon_tokens_preserve_existing_behavior() -> None:
    assert convert_sql("SELECT :名前, :valid_name") == (
        "SELECT :名前, %(valid_name)s"
    )


@pytest.mark.parametrize(
    "statement",
    [
        "SELECT ':unterminated ?",
        'SELECT "unterminated :name?',
        "SELECT 1 /* unterminated :name ?",
    ],
)
def test_unterminated_literals_and_comments_are_preserved(statement: str) -> None:
    assert convert_sql(statement) == statement


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


def test_evaluator_connection_can_receive_bounded_work_mem(monkeypatch) -> None:
    monkeypatch.setenv("BOATRACE_PG_APPLICATION_NAME", "boatrace_evaluator")
    monkeypatch.setenv("BOATRACE_PG_WORK_MEM", "128MB")

    assert _connection_options() == {
        "connect_timeout": 30,
        "application_name": "boatrace_evaluator",
        "options": "-c work_mem=128MB",
    }


def test_standardized_child_uses_evaluator_memory_defaults(monkeypatch) -> None:
    monkeypatch.delenv("BOATRACE_PG_APPLICATION_NAME", raising=False)
    monkeypatch.delenv("BOATRACE_PG_WORK_MEM", raising=False)
    monkeypatch.setenv("BOATRACE_EVAL_MAX_RACE_DATE", "2026-07-25")

    assert _connection_options() == {
        "connect_timeout": 30,
        "application_name": "boatrace_evaluator",
        "options": "-c work_mem=128MB",
    }


@pytest.mark.parametrize("value", ["0MB", "128", "128MB -c fsync=off", "-1GB"])
def test_work_mem_rejects_unsafe_or_ambiguous_values(monkeypatch, value) -> None:
    monkeypatch.setenv("BOATRACE_PG_WORK_MEM", value)

    with pytest.raises(ValueError, match="positive kB, MB, or GB"):
        _connection_options()
