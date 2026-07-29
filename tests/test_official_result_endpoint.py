from datetime import date

from boatrace_ai.ingestion.parsers import parse_result_html
from boatrace_ai.ingestion.result import parse_result_html_v2
from boatrace_ai.official import race_page_url


def test_public_result_endpoint() -> None:
    assert race_page_url("raceresult", date(2026, 7, 20), "10", 1).endswith(
        "/owpc/pc/race/raceresult?rno=1&jcd=10&hd=20260720"
    )


def test_refund_page_waits_for_finish_rows_when_trifecta_payout_exists() -> None:
    parsed = parse_result_html_v2(
        """
        <table>
          <tr><th>勝式</th><th>組番</th><th>払戻金</th><th>人気</th></tr>
          <tr>
            <td>3連単</td>
            <td><span class="numberSet1_row">1-2-4</span></td>
            <td>¥1,750</td><td>5</td>
          </tr>
        </table>
        <table>
          <tr><th>返還</th></tr>
          <tr><td><span class="numberSet1_number">3</span></td></tr>
        </table>
        """
    )

    assert parsed["status"] == "unknown"
    assert parsed["rows"] == []
    assert parsed["trifecta_evaluable"] is True
    assert parsed["result_reason"] is None


def test_trifecta_not_established_remains_a_final_non_evaluable_result() -> None:
    parsed = parse_result_html_v2(
        """
        <table>
          <tr><th>勝式</th><th>組番</th><th>払戻金</th></tr>
          <tr><td>3連単</td><td>不成立</td><td>100円返還</td></tr>
        </table>
        """
    )

    assert parsed["status"] == "final"
    assert parsed["rows"] == []
    assert parsed["trifecta_evaluable"] is False
    assert parsed["result_reason"] == "trifecta_not_established"


def test_official_cancelled_race_is_a_final_non_evaluable_result() -> None:
    html = """
    <html><body>
      <main><p>該当レースは 中止となりました。</p></main>
    </body></html>
    """

    for parser in (parse_result_html_v2, parse_result_html):
        parsed = parser(html)
        assert parsed["status"] == "final"
        assert parsed["rows"] == []
        assert parsed["payouts"] == []
        assert parsed["trifecta_evaluable"] is False
        assert parsed["result_reason"] == "race_cancelled"


def test_unrelated_cancellation_text_does_not_cancel_race() -> None:
    parsed = parse_result_html_v2(
        "<html><body><p>悪天候時は開催を中止する場合があります。</p></body></html>"
    )

    assert parsed["status"] == "unknown"
