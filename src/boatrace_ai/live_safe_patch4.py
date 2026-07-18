from __future__ import annotations

from datetime import date

from .live_safe_patch2 import install as install_base
from .official import race_page_url as official_race_page_url
from .result_parser_v2 import parse_result_html_v2


def install() -> None:
    install_base()
    from . import live

    live.race_page_url = race_page_url_safe
    live.parse_result_html = parse_result_html_v2


def race_page_url_safe(page: str, race_date: date, jcd: str, rno: int) -> str:
    normalized = "raceresult" if page == "result" else page
    return official_race_page_url(normalized, race_date, jcd, rno)
