from datetime import datetime, timedelta

from boatrace_ai.web_dashboard import JST, time_fields_from_stored_start


START = datetime(2026, 7, 19, 12, 0, tzinfo=JST)


def status_at(at: datetime, *, result_rows: int = 0) -> str:
    return str(
        time_fields_from_stored_start(
            START.isoformat(),
            now=at,
            before_minutes=5,
            result_rows=result_rows,
        )["time_status"]
    )


def test_post_deadline_status_boundaries() -> None:
    deadline = START - timedelta(minutes=5)

    assert status_at(deadline - timedelta(seconds=1)) == "T-5超過"
    assert status_at(deadline) == "出走待"
    assert status_at(START - timedelta(seconds=1)) == "出走待"
    assert status_at(START) == "出走"
    assert status_at(START + timedelta(minutes=7) - timedelta(seconds=1)) == "出走"
    assert status_at(START + timedelta(minutes=7)) == "結果待"


def test_final_result_overrides_time_status() -> None:
    assert status_at(START + timedelta(minutes=10), result_rows=3) == "確定"
