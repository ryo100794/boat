from datetime import date

from boatrace_ai.official import race_page_url


def test_public_result_endpoint() -> None:
    assert race_page_url("raceresult", date(2026, 7, 20), "10", 1).endswith(
        "/owpc/pc/race/raceresult?rno=1&jcd=10&hd=20260720"
    )
