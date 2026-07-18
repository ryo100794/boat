from __future__ import annotations

from dataclasses import dataclass


OFFICIAL_BASE_URL = "https://www.boatrace.jp"
DOWNLOAD_BASE_URL = "https://www1.mbrace.or.jp/od2"
USER_AGENT = (
    "boatrace-ai/0.1 (+local research; respectful polling; "
    "contact: set BOATRACE_AI_CONTACT)"
)


@dataclass(frozen=True)
class Venue:
    code: str
    name: str


VENUES: tuple[Venue, ...] = (
    Venue("01", "桐生"),
    Venue("02", "戸田"),
    Venue("03", "江戸川"),
    Venue("04", "平和島"),
    Venue("05", "多摩川"),
    Venue("06", "浜名湖"),
    Venue("07", "蒲郡"),
    Venue("08", "常滑"),
    Venue("09", "津"),
    Venue("10", "三国"),
    Venue("11", "びわこ"),
    Venue("12", "住之江"),
    Venue("13", "尼崎"),
    Venue("14", "鳴門"),
    Venue("15", "丸亀"),
    Venue("16", "児島"),
    Venue("17", "宮島"),
    Venue("18", "徳山"),
    Venue("19", "下関"),
    Venue("20", "若松"),
    Venue("21", "芦屋"),
    Venue("22", "福岡"),
    Venue("23", "唐津"),
    Venue("24", "大村"),
)

VENUE_BY_CODE = {venue.code: venue for venue in VENUES}

RACES_PER_DAY = range(1, 13)
LANES = range(1, 7)

CLASS_RANK = {
    "B2": 0,
    "B1": 1,
    "A2": 2,
    "A1": 3,
}
