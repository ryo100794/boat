from __future__ import annotations

from boatrace_ai.web.dashboard import MODEL_REPORT_HTML, _model_track_summaries


def test_model_selector_groups_all_sources() -> None:
    assert '<optgroup label="' in MODEL_REPORT_HTML
    for label in (
        "モデル系統",
        "予測評価",
        "資金運用",
        "実行ジョブ",
        "特徴量診断",
        "候補スイープ",
    ):
        assert label in MODEL_REPORT_HTML
    assert "seenLabels" in MODEL_REPORT_HTML


def test_model_tracks_include_listwise_search_and_newton(tmp_path) -> None:
    remote = {
        "jobs": [
            {
                "kind": "feature_teacher_search",
                "status": "実行中",
                "result": None,
            },
            {
                "kind": "newton_listwise_bankroll",
                "status": "待機中",
                "result": None,
            },
        ]
    }
    rows = _model_track_summaries(tmp_path, [], remote)
    by_id = {row["id"]: row for row in rows}
    assert by_id["feature_teacher_search"]["status"] == "実行中"
    assert by_id["newton_listwise_bankroll"]["status"] == "待機中"
    assert "Plackett-Luce" in by_id["feature_teacher_search"]["teacher"]
    assert "Newton-CG" in by_id["newton_listwise_bankroll"]["training"]
