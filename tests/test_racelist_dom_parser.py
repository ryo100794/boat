from __future__ import annotations

import unittest
from datetime import date

from boatrace_ai.ingestion.parsers import parse_racelist_html


def racelist_html(*, with_series_history: bool) -> str:
    bodies = []
    for lane in range(1, 7):
        history = "<tr><td>3</td><td>1</td><td>6</td><td>2</td></tr>" if with_series_history else ""
        bodies.append(
            f"""
            <tbody class="is-fs12">
              <tr>
                <td class="is-boatColor{lane} is-fs14">{lane}</td>
                <td>500{lane} / <span class="is-fColor1">A1</span>
                  <div class="is-fs18 is-fBold"><a>選手 {lane}</a></div>
                  <span>東京/東京</span><span>30歳/52.0kg</span><span>F0</span><span>L0</span>
                  <span>0.15</span><span>6.10</span><span>40.0</span><span>60.0</span>
                  <span>5.20</span><span>30.0</span><span>50.0</span>
                  <span>{lane}</span><span>35.0</span><span>55.0</span>
                  <span>{lane + 100}</span><span>30.0</span><span>45.0</span>
                </td>
              </tr>
              {history}
            </tbody>
            """
        )
    return f"<html><body><table>{''.join(bodies)}</table></body></html>"


class RacelistDomParserTest(unittest.TestCase):
    def test_parses_six_lanes_before_series_history_accumulates(self) -> None:
        _, entries = parse_racelist_html(
            racelist_html(with_series_history=False),
            race_date=date(2026, 7, 19),
            jcd="01",
            rno=1,
            source_url="test",
        )
        self.assertEqual([row["lane"] for row in entries], [1, 2, 3, 4, 5, 6])
        self.assertEqual(entries[0]["racer_name"], "選手 1")

    def test_series_history_lane_numbers_do_not_split_entries(self) -> None:
        _, entries = parse_racelist_html(
            racelist_html(with_series_history=True),
            race_date=date(2026, 7, 19),
            jcd="01",
            rno=12,
            source_url="test",
        )
        self.assertEqual(len(entries), 6)
        self.assertEqual(entries[5]["racer_no"], 5006)
        self.assertEqual(entries[5]["boat_no"], 106)


if __name__ == "__main__":
    unittest.main()
