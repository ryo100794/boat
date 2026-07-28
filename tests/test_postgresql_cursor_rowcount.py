from boatrace_ai.postgresql import CompatCursor


def test_compat_cursor_exposes_underlying_rowcount() -> None:
    class Cursor:
        rowcount = 7

    assert CompatCursor(Cursor()).rowcount == 7
    assert CompatCursor(None).rowcount == -1
    assert CompatCursor(None, scalar=1, has_scalar=True).rowcount == 1
