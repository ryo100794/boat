from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "remote_best_roi_attribution_runner.py"
SPEC = importlib.util.spec_from_file_location("remote_best_roi_attribution_runner", SCRIPT_PATH)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class RemoteBestRoiRunnerTest(unittest.TestCase):
    def test_selects_highest_roi_and_reuses_policy(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            low = root / "low.json"
            high = root / "high.json"
            low.write_text(json.dumps({"roi": 0.8, "profit_yen": -20, "policy": {"ev_threshold": 1.1}}))
            high.write_text(
                json.dumps(
                    {
                        "roi": 1.05,
                        "profit_yen": 50,
                        "policy": {
                            "daily_budget_yen": 10_000,
                            "ev_threshold": 1.2,
                            "allocation_mode": "normalized_kelly",
                            "stake_granularity_yen": 100,
                            "min_stake_yen": 100,
                            "ignored": "value",
                        },
                    }
                )
            )
            rows = RUNNER.load_results([low, high])
            selected_path, selected = RUNNER.select_best(rows)
            self.assertEqual(selected_path, high)
            kwargs = RUNNER.policy_kwargs(selected["policy"])
            self.assertEqual(kwargs["ev_threshold"], 1.2)
            self.assertEqual(kwargs["allocation_mode"], "normalized_kelly")
            self.assertNotIn("ignored", kwargs)


if __name__ == "__main__":
    unittest.main()
