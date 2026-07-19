from __future__ import annotations

import unittest

from boatrace_ai.adaptive_bankroll_pastlog_v7 import _allocate_adaptive_day
from boatrace_ai.bankroll_backtest import _candidate_tickets
from boatrace_ai.roi_attribution import (
    merge_roi_attribution,
    new_roi_attribution,
    summarize_fold_signal_stability,
    summarize_roi_attribution,
    update_roi_attribution,
)


class RoiAttributionTest(unittest.TestCase):
    def test_accumulates_stake_weighted_roi_by_feature_bucket(self) -> None:
        accumulator = new_roi_attribution()
        base = {
            "jcd": "01",
            "rno": 6,
            "combination": "1-2-3",
            "odds_source": "payout_model",
            "probability": 0.02,
            "estimated_odds": 60.0,
            "estimated_ev": 1.2,
            "kelly_fraction": 0.004,
            "payout_history_count": 50,
            "feature_context": {"first_racer_class": "A1"},
            "stake_yen": 100,
        }
        update_roi_attribution(accumulator, {**base, "hit": True, "return_yen": 6000})
        update_roi_attribution(accumulator, {**base, "hit": False, "return_yen": 0})

        result = summarize_roi_attribution(accumulator, min_tickets=1, min_stake_yen=1)
        dimensions = {row["dimension"]: row for row in result["dimensions"]}
        racer_bucket = dimensions["feature:first_racer_class"]["buckets"][0]
        self.assertEqual(racer_bucket["tickets"], 2)
        self.assertEqual(racer_bucket["stake_yen"], 200)
        self.assertEqual(racer_bucket["return_yen"], 6000)
        self.assertEqual(racer_bucket["roi"], 30.0)

    def test_adaptive_allocation_records_only_selected_tickets(self) -> None:
        accumulator = new_roi_attribution()
        candidates = [
            {
                "race_id": "2026-01-01-01-01",
                "race_date": "2026-01-01",
                "jcd": "01",
                "rno": 1,
                "combination": "1-2-3",
                "probability": 0.2,
                "estimated_odds": 10.0,
                "estimated_ev": 2.0,
                "payout_history_count": 20,
                "odds_source": "payout_model",
                "actual_payout_yen": 1000,
                "hit": True,
                "feature_context": {"first_motor_2_rate_rank": 1},
            }
        ]
        day = _allocate_adaptive_day(
            "2026-01-01",
            candidates,
            {"2026-01-01-01-01"},
            daily_budget_yen=10_000,
            fractional_kelly=0.25,
            max_daily_exposure_fraction=0.5,
            min_daily_exposure_fraction=0.0,
            race_cap_fraction=0.2,
            ticket_cap_fraction=0.04,
            max_daily_tickets=None,
            allocation_mode="kelly_floor",
            stake_granularity_yen=100,
            min_stake_yen=100,
            roi_attribution=accumulator,
        )
        self.assertEqual(day["tickets"], 1)
        result = summarize_roi_attribution(accumulator, min_tickets=1, min_stake_yen=1)
        dimensions = {row["dimension"] for row in result["dimensions"]}
        self.assertIn("feature:first_motor_2_rate_rank", dimensions)

    def test_merge_and_fold_stability_require_repeated_direction(self) -> None:
        left = new_roi_attribution()
        right = new_roi_attribution()
        ticket = {
            "jcd": "01", "rno": 1, "combination": "1-2-3", "odds_source": "payout_model",
            "probability": 0.02, "estimated_odds": 60.0, "estimated_ev": 1.2,
            "kelly_fraction": 0.004, "payout_history_count": 50, "stake_yen": 100,
            "hit": False, "return_yen": 0,
        }
        update_roi_attribution(left, ticket)
        update_roi_attribution(right, ticket)
        merge_roi_attribution(left, right)
        summary = summarize_roi_attribution(left, min_tickets=1, min_stake_yen=1)
        venue = next(row for row in summary["dimensions"] if row["dimension"] == "venue")
        self.assertEqual(venue["tickets"], 2)

        fold_signal = {
            "top_signals": [{
                "dimension": "estimated_ev", "family": "purchase", "roi_spread": 0.3,
                "best_bucket": {"bucket": "1.2-1.5"}, "worst_bucket": {"bucket": "1.0-1.1"},
            }]
        }
        stability = summarize_fold_signal_stability([fold_signal, fold_signal])
        self.assertEqual(stability["gate"], "candidate")
        self.assertEqual(stability["signals"][0]["status"], "stable")

    def test_fold_stability_uses_bucket_lift_against_remainder(self) -> None:
        def fold(lift: float) -> dict:
            return {
                "dimensions": [{
                    "dimension": "feature:first_motor_2_rate_rank",
                    "family": "motor",
                    "buckets": [
                        {"bucket": "1", "eligible": True, "roi_vs_rest": lift, "stake_yen": 20_000, "return_yen": 22_000},
                        {"bucket": "2-4", "eligible": True, "roi_vs_rest": -lift, "stake_yen": 20_000, "return_yen": 18_000},
                    ],
                }]
            }

        stable = summarize_fold_signal_stability([fold(0.20), fold(0.15), fold(0.12), fold(0.10), fold(-0.02)])
        target = next(row for row in stable["signals"] if row["bucket"] == "1")
        self.assertEqual(stable["gate"], "candidate")
        self.assertEqual(target["status"], "stable")
        self.assertEqual(target["direction_consistency"], 0.8)

        unstable = summarize_fold_signal_stability([fold(0.20), fold(-0.20), fold(0.15), fold(-0.15), fold(0.10)])
        self.assertEqual(unstable["gate"], "insufficient")

    def test_candidate_ticket_carries_role_specific_feature_context(self) -> None:
        rows = []
        for lane in range(1, 7):
            rows.append(
                {
                    "race_id": "2026-01-01-01-01",
                    "race_date": "2026-01-01",
                    "jcd": "01",
                    "rno": 1,
                    "lane": lane,
                    "probability": 1.0 / 6.0,
                    "diagnostic_features": {
                        "racer_class": "A1" if lane == 1 else "B1",
                        "motor_2_rate_rank": lane,
                        "race_month": "1",
                    },
                }
            )
        payout_model = {
            f"{first}-{second}-{third}": {
                "estimated_odds": 100.0,
                "estimated_payout_yen": 10_000.0,
                "history_count": 10.0,
            }
            for first in range(1, 7)
            for second in range(1, 7)
            for third in range(1, 7)
            if len({first, second, third}) == 3
        }
        candidates = _candidate_tickets(
            rows,
            actual={"combination": "1-2-3", "payout_yen": 10_000},
            payout_model=payout_model,
            ev_threshold=0.0,
        )
        target = next(row for row in candidates if row["combination"] == "1-2-3")
        self.assertEqual(target["feature_context"]["first_racer_class"], "A1")
        self.assertEqual(target["feature_context"]["second_motor_2_rate_rank"], 2)
        self.assertEqual(target["feature_context"]["race_month"], "1")


if __name__ == "__main__":
    unittest.main()
