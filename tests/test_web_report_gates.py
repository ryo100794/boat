from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from boatrace_ai.web_dashboard import (
    _quality_gates,
    _remote_bankroll_gate_records,
    _remote_bankroll_report_summaries,
)


class WebReportGateTest(unittest.TestCase):
    def test_completed_remote_bankroll_exposes_roi_attribution_gate(self) -> None:
        remote = {
            "generated_at": "2026-07-18T00:00:00+00:00",
            "jobs": [
                {
                    "name": "remote-roi",
                    "kind": "bankroll_norm",
                    "status": "完了",
                    "result": {
                        "file": "data/models/remote-roi.json",
                        "modified_at": "2026-07-18T00:00:00+00:00",
                        "metrics": {
                            "roi": 1.05,
                            "profit_yen": 5_000,
                            "stake_yen": 100_000,
                            "evaluated_races": 1_000,
                            "max_drawdown_yen": 10_000,
                        },
                        "ticket_roi_attribution": {
                            "top_signals": [{"dimension": "estimated_ev", "roi_spread": 0.2}],
                            "fold_stability": {
                                "gate": "candidate",
                                "stable_signals": 2,
                                "signals": [{"dimension": "estimated_ev", "status": "stable"}],
                            },
                        },
                    },
                }
            ],
        }
        records = _remote_bankroll_gate_records(remote)
        self.assertEqual(records[0]["roi_attribution_gate"], "candidate")
        self.assertEqual(records[0]["stable_signals"], 2)

        report_rows = _remote_bankroll_report_summaries(remote)
        self.assertEqual(
            report_rows[0]["ticket_roi_attribution"]["fold_stability"]["gate"],
            "candidate",
        )

        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            gates = _quality_gates(Path(tmp), remote)
        by_target = {row["target"]: row for row in gates}
        self.assertEqual(by_target["M4-2 ROI帰属再現性"]["status"], "達成候補")


if __name__ == "__main__":
    unittest.main()
