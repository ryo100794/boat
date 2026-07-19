from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from .constants import CLASS_RANK, LANES, VENUE_BY_CODE
from .db import race_id

FULLWIDTH_TRANS = str.maketrans(
    "０１２３４５６７８９．：－　",
    "0123456789.:- ",
)


def normalize_text(value: str) -> str:
    return value.translate(FULLWIDTH_TRANS).replace("\xa0", " ").strip()


def to_float(value: str | None) -> float | None:
    if value is None:
        return None
    normalized = normalize_text(value)
    if normalized in {"", "-", "欠", "欠場"}:
        return None
    if normalized.startswith("."):
        normalized = "0" + normalized
    try:
        return float(normalized)
    except ValueError:
        return None


def to_int(value: str | None) -> int | None:
    number = to_float(value)
    if number is None:
        return None
    return int(number)


def _soup(html: str) -> Any | None:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return None
    return BeautifulSoup(html, "html.parser")


def text_lines_from_html(html: str) -> list[str]:
    soup = _soup(html)
    if soup is None:
        text = re.sub(r"<[^>]+>", "\n", html)
    else:
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text("\n")
    return [normalize_text(line) for line in text.splitlines() if normalize_text(line)]


def compact_text_from_html(html: str) -> str:
    return "\n".join(text_lines_from_html(html))


def parse_race_meta(
    html: str,
    *,
    race_date: date,
    jcd: str,
    rno: int,
    source_url: str,
) -> dict[str, Any]:
    lines = text_lines_from_html(html)
    text = "\n".join(lines)
    soup = _soup(html)
    title = None
    if soup is not None and soup.find("h2"):
        title = normalize_text(soup.find("h2").get_text(" ", strip=True))
    if not title:
        title = next((line for line in lines if line and "R" not in line[:3]), None)

    deadline_at = None
    deadline_match = re.search(r"締切予定時刻(?P<body>.*?)(?:###|出走表|枠)", text, re.S)
    if deadline_match:
        times = re.findall(r"\b\d{1,2}:\d{2}\b", deadline_match.group("body"))
        if len(times) >= int(rno):
            deadline_at = f"{race_date.isoformat()}T{times[int(rno) - 1]}:00"

    race_type = None
    distance_m = None
    for line in lines:
        match = re.search(r"(?P<type>[^0-9]{1,24})\s+(?P<distance>\d{3,4})m", line)
        if match:
            race_type = match.group("type").strip()
            distance_m = int(match.group("distance"))
            break

    venue = VENUE_BY_CODE.get(jcd.zfill(2))
    status = "no_data" if "データはありません" in text else "scheduled"
    return {
        "race_id": race_id(race_date.isoformat(), jcd, rno),
        "race_date": race_date.isoformat(),
        "jcd": jcd.zfill(2),
        "venue_name": venue.name if venue else jcd.zfill(2),
        "rno": int(rno),
        "title": title,
        "race_type": race_type,
        "distance_m": distance_m,
        "deadline_at": deadline_at,
        "status": status,
        "source_url": source_url,
    }


def _lane_marker(line: str) -> int | None:
    normalized = normalize_text(line)
    if re.fullmatch(r"[1-6]", normalized):
        return int(normalized)
    return None


def _parse_racelist_dom_entries(html: str) -> list[dict[str, Any]]:
    soup = _soup(html)
    if soup is None:
        return []
    entries: list[dict[str, Any]] = []
    for body in soup.find_all("tbody"):
        lane_cell = body.find(class_=re.compile(r"^is-boatColor[1-6]$"))
        if lane_cell is None:
            continue
        lane_class = next(
            (value for value in lane_cell.get("class", []) if re.fullmatch(r"is-boatColor[1-6]", value)),
            None,
        )
        if not lane_class:
            continue
        lane = int(lane_class[-1])
        values = [normalize_text(value) for value in body.stripped_strings if normalize_text(value)]
        block_text = "\n".join(values)
        racer_match = re.search(r"\b(?P<no>\d{4})\s*/", block_text)
        racer_class = next((value for value in values if re.fullmatch(r"[AB]\d", value)), None)
        if not racer_match or not racer_class:
            continue
        name_node = body.select_one(".is-fs18.is-fBold a")
        name = normalize_text(name_node.get_text(" ", strip=True)) if name_node else None
        branch_origin = next(
            (value for value in values if re.fullmatch(r"[^/\d]+/[^/\d]+", value)),
            None,
        )
        branch = origin = None
        if branch_origin:
            branch, origin = (normalize_text(value) for value in branch_origin.split("/", 1))
        age_match = re.search(r"(?P<age>\d{1,2})歳\s*/\s*(?P<weight>\d{2,3}(?:\.\d)?)kg", block_text)
        f_match = re.search(r"\bF(?P<count>\d+)", block_text)
        l_match = re.search(r"\bL(?P<count>\d+)", block_text)
        l_index = next((index for index, value in enumerate(values) if re.fullmatch(r"L\d+", value)), -1)
        stats: list[float] = []
        if l_index >= 0:
            for value in values[l_index + 1 :]:
                if not re.fullmatch(r"(?:\d+(?:\.\d+)?|\.\d+)", value):
                    continue
                number = to_float(value)
                if number is not None:
                    stats.append(number)
                if len(stats) >= 13:
                    break
        entries.append(
            {
                "lane": lane,
                "racer_no": int(racer_match.group("no")),
                "racer_name": name,
                "racer_class": racer_class,
                "branch": branch,
                "origin": origin,
                "age": int(age_match.group("age")) if age_match else None,
                "weight_kg": float(age_match.group("weight")) if age_match else None,
                "f_count": int(f_match.group("count")) if f_match else None,
                "l_count": int(l_match.group("count")) if l_match else None,
                "avg_st": stats[0] if len(stats) > 0 else None,
                "national_win_rate": stats[1] if len(stats) > 1 else None,
                "national_2_rate": stats[2] if len(stats) > 2 else None,
                "national_3_rate": stats[3] if len(stats) > 3 else None,
                "local_win_rate": stats[4] if len(stats) > 4 else None,
                "local_2_rate": stats[5] if len(stats) > 5 else None,
                "local_3_rate": stats[6] if len(stats) > 6 else None,
                "motor_no": int(stats[7]) if len(stats) > 7 else None,
                "motor_2_rate": stats[8] if len(stats) > 8 else None,
                "motor_3_rate": stats[9] if len(stats) > 9 else None,
                "boat_no": int(stats[10]) if len(stats) > 10 else None,
                "boat_2_rate": stats[11] if len(stats) > 11 else None,
                "boat_3_rate": stats[12] if len(stats) > 12 else None,
                "source": "racelist_html_dom",
            }
        )
    unique = {int(entry["lane"]): entry for entry in entries}
    return [unique[lane] for lane in sorted(unique)]


def parse_racelist_html(
    html: str,
    *,
    race_date: date,
    jcd: str,
    rno: int,
    source_url: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta = parse_race_meta(
        html, race_date=race_date, jcd=jcd, rno=rno, source_url=source_url
    )
    dom_entries = _parse_racelist_dom_entries(html)
    if len(dom_entries) == 6:
        return meta, dom_entries

    lines = text_lines_from_html(html)
    starts: list[tuple[int, int]] = []
    for index, line in enumerate(lines):
        lane = _lane_marker(line)
        if lane is None:
            continue
        lookahead = "\n".join(lines[index + 1 : index + 7])
        if re.search(r"\b\d{4}\s*/\s*[AB]\d\b", lookahead):
            starts.append((index, lane))

    entries: list[dict[str, Any]] = []
    for pos, (start, lane) in enumerate(starts[:6]):
        end = starts[pos + 1][0] if pos + 1 < len(starts) else len(lines)
        block = lines[start:end]
        block_text = "\n".join(block)
        racer_match = re.search(r"\b(?P<no>\d{4})\s*/\s*(?P<class>[AB]\d)\b", block_text)
        if not racer_match:
            continue

        racer_line_index = next(
            (
                idx
                for idx, value in enumerate(block)
                if re.search(r"\b\d{4}\s*/\s*[AB]\d\b", value)
            ),
            0,
        )
        name = None
        for candidate in block[racer_line_index + 1 : racer_line_index + 4]:
            if "/" not in candidate and not re.search(r"(歳|F\d|L\d)", candidate):
                name = candidate
                break

        branch = origin = None
        branch_match = re.search(r"([^\s/]+)/([^\s/]+)", block_text)
        if branch_match:
            branch, origin = branch_match.group(1), branch_match.group(2)

        age = weight_kg = None
        age_match = re.search(r"(?P<age>\d{1,2})歳\s*/\s*(?P<weight>\d{2,3}(?:\.\d)?)kg", block_text)
        if age_match:
            age = int(age_match.group("age"))
            weight_kg = float(age_match.group("weight"))

        f_count = to_int((re.search(r"\bF(?P<n>\d+)", block_text) or {}).group("n") if re.search(r"\bF(?P<n>\d+)", block_text) else None)
        l_count = to_int((re.search(r"\bL(?P<n>\d+)", block_text) or {}).group("n") if re.search(r"\bL(?P<n>\d+)", block_text) else None)

        tail = block_text
        l_match = re.search(r"\bL\d+\b", block_text)
        if l_match:
            tail = block_text[l_match.end() :]
        numbers = [
            to_float(token)
            for token in re.findall(r"(?<![A-Za-z])(?:\d+\.\d+|\d+)(?![A-Za-z])", tail)
        ]
        numbers = [value for value in numbers if value is not None]
        fields = {
            "avg_st": numbers[0] if len(numbers) > 0 else None,
            "national_win_rate": numbers[1] if len(numbers) > 1 else None,
            "national_2_rate": numbers[2] if len(numbers) > 2 else None,
            "national_3_rate": numbers[3] if len(numbers) > 3 else None,
            "local_win_rate": numbers[4] if len(numbers) > 4 else None,
            "local_2_rate": numbers[5] if len(numbers) > 5 else None,
            "local_3_rate": numbers[6] if len(numbers) > 6 else None,
            "motor_no": int(numbers[7]) if len(numbers) > 7 else None,
            "motor_2_rate": numbers[8] if len(numbers) > 8 else None,
            "motor_3_rate": numbers[9] if len(numbers) > 9 else None,
            "boat_no": int(numbers[10]) if len(numbers) > 10 else None,
            "boat_2_rate": numbers[11] if len(numbers) > 11 else None,
            "boat_3_rate": numbers[12] if len(numbers) > 12 else None,
        }
        entries.append(
            {
                "lane": lane,
                "racer_no": int(racer_match.group("no")),
                "racer_name": name,
                "racer_class": racer_match.group("class"),
                "branch": branch,
                "origin": origin,
                "age": age,
                "weight_kg": weight_kg,
                "f_count": f_count,
                "l_count": l_count,
                **fields,
                "source": "racelist_html",
            }
        )
    return meta, entries


def parse_odds_token(token: str) -> float | None:
    normalized = normalize_text(token)
    if normalized in {"-", "欠", "欠場"}:
        return None
    return to_float(normalized)


def _numeric_tokens(value: str) -> list[str]:
    return re.findall(r"欠場|欠|[-]|\d+\.\d+|\d+", normalize_text(value))


def _parse_single_first_place_table(tokens: list[str]) -> dict[str, float | None]:
    if not tokens:
        return {}
    first = to_int(tokens[0])
    if first not in LANES:
        return {}
    parsed: dict[str, float | None] = {}
    idx = 1
    others = [lane for lane in LANES if lane != first]
    for second in others:
        if idx < len(tokens) and to_int(tokens[idx]) == second:
            idx += 1
        third_candidates = [lane for lane in LANES if lane not in {first, second}]
        for third in third_candidates:
            if idx < len(tokens) and to_int(tokens[idx]) == third:
                idx += 1
            if idx >= len(tokens):
                return {}
            parsed[f"{first}-{second}-{third}"] = parse_odds_token(tokens[idx])
            idx += 1
    return parsed if len(parsed) == 20 else {}


def _parse_row_layout_lines(lines: list[str]) -> dict[str, float | None]:
    odds: dict[str, float | None] = {}
    row_candidates: list[list[str]] = []
    for line in lines:
        tokens = _numeric_tokens(line)
        if len(tokens) in {12, 18}:
            row_candidates.append(tokens)
    if len(row_candidates) < 20:
        return {}

    for row_index, tokens in enumerate(row_candidates[:20]):
        idx = 0
        group_start = row_index % 4 == 0
        for first in LANES:
            others = [lane for lane in LANES if lane != first]
            second = others[row_index // 4]
            third_candidates = [lane for lane in LANES if lane not in {first, second}]
            third = third_candidates[row_index % 4]
            if group_start:
                if idx < len(tokens) and to_int(tokens[idx]) == second:
                    idx += 1
            if idx < len(tokens) and to_int(tokens[idx]) == third:
                idx += 1
            if idx >= len(tokens):
                return {}
            odds[f"{first}-{second}-{third}"] = parse_odds_token(tokens[idx])
            idx += 1
    return odds


def parse_odds3t_html(html: str) -> dict[str, Any]:
    lines = text_lines_from_html(html)
    text = "\n".join(lines)
    source_update_time = None
    match = re.search(r"オッズ更新時間\s*([0-9]{1,2}:[0-9]{2})", text)
    if match:
        source_update_time = match.group(1)

    odds: dict[str, float | None] = {}
    soup = _soup(html)
    if soup is not None:
        for table in soup.find_all("table"):
            tokens = _numeric_tokens(table.get_text(" ", strip=True))
            parsed = _parse_single_first_place_table(tokens)
            odds.update(parsed)
        if len(odds) < 100:
            row_lines = [
                " ".join(cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"]))
                for row in soup.find_all("tr")
            ]
            odds.update(_parse_row_layout_lines(row_lines))

    if len(odds) < 100:
        try:
            start = next(i for i, line in enumerate(lines) if "3連単オッズ" in line)
            odds.update(_parse_row_layout_lines(lines[start + 1 :]))
        except StopIteration:
            pass

    return {
        "source_update_time": source_update_time,
        "odds": odds,
        "parsed_count": len(odds),
    }


def parse_beforeinfo_html(html: str) -> dict[str, Any]:
    lines = text_lines_from_html(html)
    text = "\n".join(lines)
    weather: dict[str, Any] = {}
    patterns = {
        "air_temp_c": r"気温\s*([0-9.]+)",
        "wind_speed_m": r"風速\s*([0-9.]+)",
        "water_temp_c": r"水温\s*([0-9.]+)",
        "wave_cm": r"波高\s*([0-9.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            weather[key] = to_float(match.group(1))

    rows: list[dict[str, Any]] = []
    starts: list[tuple[int, int]] = []
    for index, line in enumerate(lines):
        lane = _lane_marker(line)
        if lane is None:
            continue
        lookahead = "\n".join(lines[index + 1 : index + 8])
        if "R" in lookahead or re.search(r"\d{2,3}(?:\.\d)?kg", lookahead):
            starts.append((index, lane))
    for pos, (start, lane) in enumerate(starts[:6]):
        end = starts[pos + 1][0] if pos + 1 < len(starts) else len(lines)
        block_text = "\n".join(lines[start:end])
        nums = [to_float(token) for token in re.findall(r"\d+\.\d+|\d+", block_text)]
        nums = [value for value in nums if value is not None]
        rows.append(
            {
                "lane": lane,
                "weight_kg": next((value for value in nums if 40 <= value <= 70), None),
                "exhibition_time": next((value for value in nums if 5.0 <= value <= 8.5), None),
                "tilt": next((value for value in nums if -2.0 <= value <= 3.0), None),
                "raw_text": block_text,
                **weather,
            }
        )
    return {"rows": rows, "weather": weather, "raw_text": text}


def parse_result_html(html: str) -> dict[str, Any]:
    lines = text_lines_from_html(html)
    text = "\n".join(lines)
    if "データはありません" in text:
        return {"status": "no_data", "rows": [], "payouts": []}
    rows: list[dict[str, Any]] = []
    for line in lines:
        match = re.match(
            r"^(?P<rank>[1-6])\s+(?P<lane>[1-6])\s+.*?(?P<st>\.[0-9]{2}|0\.[0-9]{2})?",
            line,
        )
        if match:
            rows.append(
                {
                    "rank": int(match.group("rank")),
                    "lane": int(match.group("lane")),
                    "start_timing": to_float(match.group("st")),
                    "raw_text": line,
                }
            )

    payouts: list[dict[str, Any]] = []
    payout_pattern = re.compile(
        r"(3連単|3連複|2連単|2連複|拡連複|単勝|複勝)\s+([1-6=-]+)\s+([0-9,]+)円?(?:\s+(\d+)人気)?"
    )
    for match in payout_pattern.finditer(text):
        payouts.append(
            {
                "bet_type": match.group(1),
                "combination": match.group(2).replace("=", "-"),
                "payout_yen": int(match.group(3).replace(",", "")),
                "popularity": to_int(match.group(4)),
            }
        )
    return {"status": "final" if rows else "unknown", "rows": rows, "payouts": payouts}


def parse_racer_stats_bytes(payload: bytes, *, year: int, half: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in payload.splitlines():
        if len(raw_line) < 80 or not raw_line[:4].isdigit():
            continue

        def field(start: int, end: int) -> str:
            return raw_line[start:end].decode("cp932", errors="ignore").strip()

        racer_no = to_int(field(0, 4))
        if racer_no is None:
            continue
        row = {
            "year": year,
            "half": half,
            "racer_no": racer_no,
            "racer_name": field(4, 20),
            "racer_name_kana": field(20, 35),
            "branch": field(35, 39),
            "racer_class": field(39, 41),
            "era": field(41, 42),
            "birth_ymd_raw": field(42, 48),
            "gender": field(48, 49),
            "age": to_int(field(49, 51)),
            "height_cm": to_int(field(51, 54)),
            "weight_kg": to_int(field(54, 56)),
            "blood_type": field(56, 58),
            "win_rate": _scaled_int(field(58, 62), 100),
            "place2_rate": _scaled_int(field(62, 66), 10),
            "first_count": to_int(field(66, 69)),
            "second_count": to_int(field(69, 72)),
            "starts": to_int(field(72, 75)),
            "final_count": to_int(field(75, 77)),
            "champion_count": to_int(field(77, 79)),
            "avg_st": _scaled_int(field(79, 82), 100),
            "class_rank": CLASS_RANK.get(field(39, 41)),
        }
        rows.append(row)
    return rows


def _scaled_int(value: str, scale: int) -> float | None:
    parsed = to_int(value)
    if parsed is None:
        return None
    return parsed / scale


def parse_historical_result_text(text: str, *, race_date: date) -> list[dict[str, Any]]:
    """Best-effort parser for official K text files.

    The raw text is still stored; this parser extracts rows only when the
    official fixed-width text is recognizable.
    """
    rows: list[dict[str, Any]] = []
    current_jcd: str | None = None
    current_rno: int | None = None
    for raw_line in text.splitlines():
        line = normalize_text(raw_line)
        for code, venue in VENUE_BY_CODE.items():
            if venue.name in line:
                current_jcd = code
                break
        race_match = re.match(r"^\s*(\d{1,2})R\b", line)
        if race_match:
            current_rno = int(race_match.group(1))
            continue
        if current_jcd is None or current_rno is None:
            continue
        row_match = re.match(
            r"^\s*(?P<rank>[1-6])\s+(?P<lane>[1-6])\s+"
            r"(?:(?P<racer_no>\d{4})\s+)?(?P<name>.+?)\s+"
            r"(?P<time>\d\.\d{2}\.\d|[0-9.]+)?\s*$",
            line,
        )
        if row_match:
            rows.append(
                {
                    "race_id": race_id(race_date.isoformat(), current_jcd, current_rno),
                    "lane": int(row_match.group("lane")),
                    "rank": int(row_match.group("rank")),
                    "racer_no": to_int(row_match.group("racer_no")),
                    "racer_name": row_match.group("name").strip(),
                    "race_time": row_match.group("time"),
                    "raw_text": raw_line,
                }
            )
    return rows
