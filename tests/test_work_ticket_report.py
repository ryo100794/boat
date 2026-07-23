from __future__ import annotations

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
