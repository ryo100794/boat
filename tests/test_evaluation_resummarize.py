import json

import pytest

from boatrace_ai.evaluation_queue import resummarize_completed_job


class _Cursor:
    def __init__(self, row=None):
        self._row = row

    def fetchone(self):
        return self._row


class _Connection:
    def __init__(self, row):
        self.row = row
        self.updates = []

    def execute(self, statement, parameters=()):
        if statement.lstrip().startswith("SELECT"):
            return _Cursor(self.row)
        self.updates.append((statement, parameters))
        return _Cursor()


def test_resummarize_completed_job_updates_closing_and_tail_metrics(tmp_path) -> None:
    result = tmp_path / "job.json"
    result.write_text(
        json.dumps(
            {
                "roi": 0.8,
                "closing_odds_forecast": {"closing_odds_log_mae": 0.17},
                "bankroll": {
                    "daily": [
                        {
                            "_tail_portfolio_rows": [
                                {
                                    "date": "2026-07-27",
                                    "race_id": "race-1",
                                    "odds": 120.0,
                                    "stake": 100,
                                    "return": 0,
                                }
                            ]
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    conn = _Connection(
        {"job_id": 42, "status": "completed", "result_path": str(result)}
    )

    output = resummarize_completed_job(conn, job_id=42, app_root=tmp_path)

    assert output["tail_diagnostics"] is True
    assert "closing_odds_log_mae" in output["summary_keys"]
    summary = json.loads(conn.updates[0][1][0])
    assert summary["closing_odds_log_mae"] == 0.17
    assert summary["tail_portfolio_diagnostics"]["tail"]["tickets"] == 1
    persisted = json.loads(result.read_text(encoding="utf-8"))
    assert "_tail_portfolio_rows" not in persisted["bankroll"]["daily"][0]


def test_resummarize_rejects_result_outside_app_root(tmp_path) -> None:
    outside = tmp_path.parent / "outside-evaluation.json"
    outside.write_text("{}", encoding="utf-8")
    conn = _Connection(
        {"job_id": 42, "status": "completed", "result_path": str(outside)}
    )

    with pytest.raises(ValueError, match="inside app_root"):
        resummarize_completed_job(conn, job_id=42, app_root=tmp_path)

    outside.unlink()
