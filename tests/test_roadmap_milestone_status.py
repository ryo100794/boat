from pathlib import Path

from boatrace_ai.web import dashboard


def test_roadmap_page_loads_milestones_independently() -> None:
    html = Path("src/boatrace_ai/templates/roadmap_report.html").read_text(
        encoding="utf-8"
    )

    assert "/api/reports/roadmap-milestones" in html
    assert "function renderMilestones(rows)" in html
    assert "マイルストーンデータなし" in html
    assert "progressCell(r.progress)" in html


def test_roadmap_milestone_status_is_bounded(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(dashboard, "query_race_date", lambda *_args: "2026-07-30")
    monkeypatch.setattr(dashboard, "progress_active_fast", lambda *_args: {})
    monkeypatch.setattr(dashboard, "_shadow_roadmap_status", lambda *_args: {})
    monkeypatch.setattr(dashboard, "_read_remote_eval_status", lambda *_args: {})
    monkeypatch.setattr(
        dashboard, "_merge_standardized_v2_status", lambda remote, _bundle: remote
    )
    monkeypatch.setattr(dashboard, "_load_standardized_v2_bundle", lambda *_args: {})
    monkeypatch.setattr(dashboard, "_process_snapshots", lambda: [])
    monkeypatch.setattr(dashboard, "teleboat_status", lambda *_args: {})
    monkeypatch.setattr(
        dashboard,
        "_roadmap_milestones",
        lambda *_args: [{"id": "M0", "progress": 100}],
    )
    dashboard._ROADMAP_MILESTONE_CACHE.clear()

    payload = dashboard.roadmap_milestones_status(tmp_path / "races.sqlite", {})

    assert payload["milestones"] == [{"id": "M0", "progress": 100}]
    assert set(payload) == {
        "generated_at",
        "date",
        "milestones",
        "model_audit",
    }
    assert payload["model_audit"]["status"] == "評価未登録"
    assert payload["model_audit"]["audit_ready"] is False
    assert len(payload["model_audit"]["audit_snapshot_id"]) == 64
