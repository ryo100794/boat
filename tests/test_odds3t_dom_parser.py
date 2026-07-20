from __future__ import annotations

from boatrace_ai.ingestion.parsers import parse_odds3t_html


def _official_matrix_html() -> str:
    headers = "".join(
        f'<th class="is-boatColor{lane}">{lane}</th><th colspan="2">R{lane}</th>'
        for lane in range(1, 7)
    )
    rows = []
    for row_index in range(20):
        cells = []
        for first in range(1, 7):
            others = [lane for lane in range(1, 7) if lane != first]
            second = others[row_index // 4]
            thirds = [
                lane for lane in range(1, 7) if lane not in {first, second}
            ]
            third = thirds[row_index % 4]
            if row_index % 4 == 0:
                cells.append(
                    f'<td class="is-boatColor{second}" rowspan="4">{second}</td>'
                )
            odds = first * 100 + second * 10 + third + 0.5
            cells.append(f'<td class="is-boatColor{third}">{third}</td>')
            cells.append(f'<td class="oddsPoint">{odds}</td>')
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return (
        "<html><body><h3>3連単オッズ</h3><table>"
        f"<thead><tr>{headers}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></body></html>"
    )


def test_parse_official_six_column_matrix_uses_only_odds_cells() -> None:
    parsed = parse_odds3t_html(_official_matrix_html())

    assert parsed["parsed_count"] == 120
    assert parsed["parser_version"] == "odds3t_dom_v2"
    assert parsed["odds"]["1-2-3"] == 123.5
    assert parsed["odds"]["2-1-3"] == 213.5
    assert parsed["odds"]["6-5-4"] == 654.5
    assert sum(value in {1, 2, 3, 4, 5, 6} for value in parsed["odds"].values()) == 0
