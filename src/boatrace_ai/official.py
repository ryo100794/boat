from __future__ import annotations

from datetime import date
from itertools import permutations

from .constants import DOWNLOAD_BASE_URL, OFFICIAL_BASE_URL


def ymd(value: date) -> str:
    return value.strftime("%Y%m%d")


def yymmdd(value: date) -> str:
    return value.strftime("%y%m%d")


def historical_download_url(kind: str, value: date) -> str:
    """Build official daily LZH URL.

    kind is "program" for 番組表 or "result" for 競走成績.
    """
    normalized = kind.lower()
    if normalized in {"program", "b", "racelist"}:
        directory = "B"
        prefix = "b"
    elif normalized in {"result", "k", "race_result"}:
        directory = "K"
        prefix = "k"
    else:
        raise ValueError(f"unsupported historical kind: {kind}")
    return (
        f"{DOWNLOAD_BASE_URL}/{directory}/{value:%Y%m}/"
        f"{prefix}{yymmdd(value)}.lzh"
    )


def race_page_url(page: str, race_date: date, jcd: str, rno: int) -> str:
    return (
        f"{OFFICIAL_BASE_URL}/owpc/pc/race/{page}"
        f"?rno={int(rno)}&jcd={jcd.zfill(2)}&hd={ymd(race_date)}"
    )


def race_index_url(race_date: date) -> str:
    return f"{OFFICIAL_BASE_URL}/owpc/pc/race/index?hd={ymd(race_date)}"


def racer_stats_url(year: int, half: int) -> str:
    if half not in {1, 2}:
        raise ValueError("half must be 1 or 2")
    # Official links currently use fan{year}{half}.lzh under the extra/data
    # area. The collector verifies status and stores failures explicitly.
    return f"{OFFICIAL_BASE_URL}/owpc/pc/extra/data/stadium/fan{year}{half}.lzh"


def trifecta_combinations() -> list[str]:
    return ["-".join(map(str, combo)) for combo in permutations(range(1, 7), 3)]
