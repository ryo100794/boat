from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from boatrace_ai.web.dashboard import (
    _quality_gates,
    _remote_bankroll_gate_records,
    _remote_bankroll_daily,
    _remote_bankroll_report_summaries,
)


class WebReportGateTest(unittest.TestCase):
    def test_daily_rows_do_not_depend_on_job_kind_name(self) -> None:
        remote = {
            "jobs": [
                {
                    "name": "calibrated_mlp",
                    "kind": "standardized_365d_v2_model",
                    "result": {
                        "metrics": {"roi": 0.91},
                        "daily": [
                            {
                                "race_date": "2026-07-18",
                                "tickets": 3,
                                "stake_yen": 300,
                                "return_yen": 200,
                                "profit_yen": -100,
                            }
                        ],
                    },
                }
            ]
        }

        daily = _remote_bankroll_daily(remote)
        self.assertEqual(daily["calibrated_mlp"][0]["date"], "2026-07-18")

    def test_completed_remote_bankroll_exposes_roi_attribution_gate(self) -> None:
        remote = {
            "generated_at": "2026-07-18T00:00:00+00:00",
            "jobs": [
                {
                    "name": "remote-roi",
                    "kind": "newton_listwise_bankroll",
                    "status": "完了",
                    "result": {
                        "file": "data/models/remote-roi.json",
                        "modified_at": "2026-07-18T00:00:00+00:00",
                        "model": "listwise_newton_cg_v1",
                        "daily": [
                            {
                                "race_date": "2026-07-17",
                                "roi": 1.05,
                                "profit_yen": 5000,
                                "cumulative_profit_yen": 5000,
                                "stake_yen": 100000,
                                "return_yen": 105000,
                                "tickets": 20,
                                "races_bet": 10,
                                "budget_used_fraction": 0.5,
                            }
                        ],
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
        self.assertEqual(report_rows[0]["model"], "listwise_newton_cg_v1")
        daily = _remote_bankroll_daily(remote)
        self.assertEqual(daily["remote-roi"][0]["date"], "2026-07-17")
        self.assertEqual(daily["remote-roi"][0]["cumulative_profit_yen"], 5000)

        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            gates = _quality_gates(Path(tmp), remote)
        by_target = {row["target"]: row for row in gates}
        self.assertEqual(by_target["M4-2 ROI帰属再現性"]["status"], "達成候補")


if __name__ == "__main__":
    unittest.main()
