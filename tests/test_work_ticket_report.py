from __future__ import annotations

import json
from pathlib import Path

from boatrace_ai.web import dashboard


class _Rows:
    def fetchall(self):
        return [
            {
                "ticket_key": "MODEL-OPT-001",
                "title": "model",
                "area": "model",
                "description": "iterate",
                "acceptance_criteria": "gate pass",
                "owner": "codex",
                "priority": 100,
                "status": "in_progress",
                "progress": 55,
                "related_job_id": 1,
                "source": "user",
                "created_at": "2026-07-23",
                "updated_at": "2026-07-23",
                "completed_at": None,
            }
        ]


class _Connection:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, statement):
        assert "FROM work_tickets" in statement
        return _Rows()


def test_roadmap_reads_work_tickets_from_database(monkeypatch) -> None:
    monkeypatch.setattr(dashboard, "connect", lambda _path: _Connection())

    tickets = dashboard._roadmap_work_tickets(Path("ignored"))

    assert tickets[0]["ticket_key"] == "MODEL-OPT-001"
    assert tickets[0]["progress"] == 55


def test_roadmap_page_has_database_ticket_table() -> None:
    html = Path("src/boatrace_ai/templates/roadmap_report.html").read_text(
        encoding="utf-8"
    )

    assert 'id="ticketRows"' in html
    assert "DB作業チケット" in html
    assert "data.tickets" in html


def test_roadmap_remote_evaluations_are_bounded_flat_summaries() -> None:
    remote = {
        "generated_at": "2026-07-30T12:00:00Z",
        "status": "取得済み",
        "jobs": [
            {
                "pid": 123,
                "name": "large-search",
                "milestone": "M4",
                "kind": "feature_search",
                "status": "完了",
                "running": False,
                "process": {"elapsed": "01:02:03", "cmd": "large command"},
                "result": {
                    "metrics": {
                        "roi": 1.02,
                        "profit_yen": 200,
                        "daily": [{"blob": "x" * 100_000}],
                        "search_results": [{"blob": "y" * 100_000}],
                    },
                    "daily": [{"blob": "z" * 100_000}],
                    "search_results": [{"blob": "w" * 100_000}],
                },
            }
        ],
    }

    summary = dashboard._roadmap_remote_evaluation_summary(remote)
    encoded = json.dumps(summary)

    assert summary["generated_at"] == remote["generated_at"]
    assert summary["status"] == "取得済み"
    assert summary["jobs"][0]["pid"] == 123
    assert summary["jobs"][0]["elapsed"] == "01:02:03"
    assert summary["jobs"][0]["roi"] == 1.02
    assert summary["jobs"][0]["profit_yen"] == 200
    assert "result" not in summary["jobs"][0]
    assert "process" not in summary["jobs"][0]
    assert "daily" not in encoded
    assert "search_results" not in encoded
    assert len(encoded) < 10_000


def test_roadmap_page_renders_flat_remote_evaluation_fields() -> None:
    html = Path("src/boatrace_ai/templates/roadmap_report.html").read_text(
        encoding="utf-8"
    )

    assert "r.elapsed||'-'" in html
    assert "['ROI',r.roi]" in html
    assert "r.process&&r.process.elapsed" not in html
    assert "r.result&&r.result.metrics" not in html
