from __future__ import annotations

import re
from datetime import date
from typing import Any

from .constants import LANES
from . import parsers as base


def parse_racelist_html(
    html: str,
    *,
    race_date: date,
    jcd: str,
    rno: int,
    source_url: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta = base.parse_race_meta(
        html, race_date=race_date, jcd=jcd, rno=rno, source_url=source_url
    )
    entries = parse_entries_from_dom(html)
    if len(entries) == 6:
        return meta, entries
    return base.parse_racelist_html(
        html, race_date=race_date, jcd=jcd, rno=rno, source_url=source_url
    )


def parse_entries_from_dom(html: str) -> list[dict[str, Any]]:
    soup = base._soup(html)
    if soup is None:
        return []
    entries: list[dict[str, Any]] = []
    for lane in LANES:
        marker = _lane_cell(soup, lane)
        if marker is None:
            continue
        row = marker.find_parent("tr")
        if row is None:
            continue
        cells = row.find_all("td", recursive=False)
        if len(cells) < 8:
            continue
        profile = cells[2]
        links = profile.find_all("a", href=re.compile(r"toban=\d{4}"))
        href = str(links[0].get("href", "")) if links else ""
        no_match = re.search(r"toban=(\d{4})", href)
        if no_match is None:
            no_match = re.search(r"\b(\d{4})\b", _text(profile))
        if no_match is None:
            continue
        racer_no = int(no_match.group(1))
        name = _profile_name(links)
        profile_text = "\n".join(_lines(profile))
        class_match = re.search(rf"\b{racer_no}\s*/\s*([AB]\d)\b", profile_text)
        branch, origin = _branch_origin(profile)
        age, weight_kg = _age_weight(profile_text)
        fl_st = _lines(cells[3])
        national = _rate_triplet(cells[4])
        local = _rate_triplet(cells[5])
        motor = _rate_triplet(cells[6])
        boat = _rate_triplet(cells[7])
        entries.append(
            {
                "lane": lane,
                "racer_no": racer_no,
                "racer_name": name,
                "racer_class": class_match.group(1) if class_match else None,
                "branch": branch,
                "origin": origin,
                "age": age,
                "weight_kg": weight_kg,
                "f_count": _prefixed_int(fl_st, "F"),
                "l_count": _prefixed_int(fl_st, "L"),
                "avg_st": next((base.to_float(line) for line in fl_st if re.fullmatch(r"\d+\.\d+", line)), None),
                "national_win_rate": national[0],
                "national_2_rate": national[1],
                "national_3_rate": national[2],
                "local_win_rate": local[0],
                "local_2_rate": local[1],
                "local_3_rate": local[2],
                "motor_no": int(motor[0]) if motor[0] is not None else None,
                "motor_2_rate": motor[1],
                "motor_3_rate": motor[2],
                "boat_no": int(boat[0]) if boat[0] is not None else None,
                "boat_2_rate": boat[1],
                "boat_3_rate": boat[2],
                "source": "racelist_html_dom",
            }
        )
    return entries


def _lane_cell(soup: Any, lane: int) -> Any | None:
    pattern = re.compile(rf"\bis-boatColor{lane}\b")
    for cell in soup.find_all("td", class_=pattern):
        if _text(cell) == str(lane):
            return cell
    return None


def _profile_name(links: list[Any]) -> str | None:
    for link in reversed(links):
        text = re.sub(r"\s+", " ", _text(link)).strip()
        if text and not text.isdigit():
            return text
    return None


def _branch_origin(cell: Any) -> tuple[str | None, str | None]:
    for line in _lines(cell):
        if "/" not in line or "歳" in line or re.search(r"\d{4}\s*/\s*[AB]\d", line):
            continue
        left, right = [part.strip() for part in line.split("/", 1)]
        if left and right:
            return left, right
    return None, None


def _age_weight(text: str) -> tuple[int | None, float | None]:
    match = re.search(r"(?P<age>\d{1,2})歳\s*/\s*(?P<weight>\d{2,3}(?:\.\d)?)kg", text)
    if not match:
        return None, None
    return int(match.group("age")), float(match.group("weight"))


def _rate_triplet(cell: Any) -> tuple[float | None, float | None, float | None]:
    values = [base.to_float(line) for line in _lines(cell)]
    values = [value for value in values if value is not None]
    return (
        values[0] if len(values) > 0 else None,
        values[1] if len(values) > 1 else None,
        values[2] if len(values) > 2 else None,
    )


def _prefixed_int(lines: list[str], prefix: str) -> int | None:
    for line in lines:
        match = re.fullmatch(rf"{re.escape(prefix)}(\d+)", line)
        if match:
            return int(match.group(1))
    return None


def _lines(cell: Any) -> list[str]:
    return [base.normalize_text(line) for line in cell.get_text("\n", strip=True).splitlines() if base.normalize_text(line)]


def _text(cell: Any) -> str:
    return base.normalize_text(cell.get_text(" ", strip=True))

