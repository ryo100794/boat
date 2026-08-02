from boatrace_ai.ingestion.parsers import parse_beforeinfo_html


def test_beforeinfo_ignores_previous_rank_between_ordered_lane_rows() -> None:
    blocks = []
    for lane in range(1, 7):
        blocks.append(
            f"<div>{lane}<br>Racer {lane}<br>{45 + lane}.0kg<br>"
            f"6.{60 + lane}<br>-0.5<br>R<br>進入<br>ST<br>着順</div>"
        )
        if lane == 3:
            blocks.append("<div>2</div>")
    html = (
        "<html><body>気温 26.0 風速 5 水温 24.0 波高 4"
        + "".join(blocks)
        + "</body></html>"
    )

    parsed = parse_beforeinfo_html(html)

    assert [row["lane"] for row in parsed["rows"]] == [1, 2, 3, 4, 5, 6]
    assert [row["tilt"] for row in parsed["rows"]] == [-0.5] * 6
    assert parsed["rows"][5]["exhibition_time"] == 6.66
