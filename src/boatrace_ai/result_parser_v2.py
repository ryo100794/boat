from __future__ import annotations

import re
from typing import Any

from .parsers import _soup, normalize_text, to_float, to_int


BET_TYPES = ("3連単", "3連複", "2連単", "2連複", "拡連複", "単勝", "複勝")
ZEN_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")


def parse_result_html_v2(html: str) -> dict[str, Any]:
    text = normalize_text(_plain_text(html))
    if "データはありません" in text:
        return {"status": "no_data", "rows": [], "payouts": []}
    if "予期せぬエラーが発生しました" in text:
        return {"status": "error", "rows": [], "payouts": []}

    soup = _soup(html)
    if soup is None:
        return {"status": "unknown", "rows": [], "payouts": []}

    starts = _start_timing_by_lane(soup)
    rows = _finish_rows(soup, starts)
    payouts = _payout_rows(soup)
    return {"status": "final" if rows else "unknown", "rows": rows, "payouts": payouts}


def _finish_rows(soup: Any, starts: dict[int, float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for table in soup.find_all("table"):
        headers = [normalize_text(cell.get_text(" ", strip=True)) for cell in table.find_all("th")]
        if not {"着", "枠"}.issubset(set(headers)):
            continue
        for tr in table.find_all("tr"):
            cells = [normalize_text(td.get_text(" ", strip=True)) for td in tr.find_all("td")]
            if len(cells) < 2:
                continue
            rank = _small_int(cells[0])
            lane = _small_int(cells[1])
            if rank is None or lane is None:
                continue
            if not (1 <= rank <= 6 and 1 <= lane <= 6):
                continue
            rows.append(
                {
                    "rank": rank,
                    "lane": lane,
                    "start_timing": starts.get(lane),
                    "race_time": cells[3] if len(cells) >= 4 and cells[3].strip() else None,
                    "raw_text": " ".join(cells),
                }
            )
    return rows[:6]


def _start_timing_by_lane(soup: Any) -> dict[int, float]:
    result: dict[int, float] = {}
    for block in soup.select(".table1_boatImage1"):
        lane_el = block.select_one(".table1_boatImage1Number")
        time_el = block.select_one(".table1_boatImage1TimeInner")
        if lane_el is None or time_el is None:
            continue
        lane = _small_int(lane_el.get_text(" ", strip=True))
        match = re.search(r"(\.[0-9]{2}|0\.[0-9]{2})", time_el.get_text(" ", strip=True))
        if lane is None or not match:
            continue
        result[lane] = to_float(match.group(1))
    return result


def _payout_rows(soup: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for table in soup.find_all("table"):
        headers = [normalize_text(cell.get_text(" ", strip=True)) for cell in table.find_all("th")]
        if not {"勝式", "組番", "払戻金"}.issubset(set(headers)):
            continue
        current_type = None
        for tr in table.find_all("tr"):
            cells = tr.find_all("td")
            if not cells:
                continue
            cell_texts = [normalize_text(cell.get_text(" ", strip=True)) for cell in cells]
            first = cell_texts[0] if cell_texts else ""
            if first in BET_TYPES:
                current_type = first
            bet_type = current_type
            if not bet_type:
                continue
            combo = _combination_from_row(tr)
            payout = _payout_from_text(" ".join(cell_texts))
            popularity = _popularity_from_cells(cell_texts)
            if combo and payout is not None:
                rows.append(
                    {
                        "bet_type": bet_type,
                        "combination": combo,
                        "payout_yen": payout,
                        "popularity": popularity,
                    }
                )
    return rows


def _combination_from_row(tr: Any) -> str | None:
    number_row = tr.select_one(".numberSet1_row")
    if number_row is None:
        return None
    text = normalize_text(number_row.get_text("", strip=True)).translate(ZEN_DIGITS)
    text = text.replace("－", "-").replace("ー", "-")
    text = re.sub(r"\s+", "", text)
    match = re.search(r"([1-6](?:[-=][1-6]){0,2})", text)
    if not match:
        return None
    return match.group(1).replace("=", "-")


def _payout_from_text(text: str) -> int | None:
    match = re.search(r"[¥￥]\s*([0-9,]+)|([0-9,]+)\s*円", text.translate(ZEN_DIGITS))
    if not match:
        return None
    value = match.group(1) or match.group(2)
    return int(value.replace(",", ""))


def _popularity_from_cells(cells: list[str]) -> int | None:
    if not cells:
        return None
    return to_int(cells[-1].translate(ZEN_DIGITS))


def _small_int(value: str) -> int | None:
    match = re.search(r"[1-6]", value.translate(ZEN_DIGITS))
    return int(match.group(0)) if match else None


def _plain_text(html: str) -> str:
    soup = _soup(html)
    if soup is None:
        return html
    return soup.get_text("\n", strip=True)
