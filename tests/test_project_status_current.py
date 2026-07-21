from pathlib import Path


STATUS = Path("docs/PROJECT_STATUS.md").read_text(encoding="utf-8")


def test_project_status_uses_current_evaluation_state() -> None:
    assert "2026-07-21" in STATUS
    assert "629 / 1,000R" in STATUS
    assert "標準365日v2は7モデル" in STATUS
    assert "標準365日最高ROI 0.8246" in STATUS
    assert "実odds暫定もROI 0.8400" in STATUS
    assert "M6 資金運用モデル | 未完了/収益ゲート未達" in STATUS
    assert "M7 ソース整理 | 完了/運用監視" in STATUS
    assert "PID 196980" not in STATUS
    assert "順番待ち" not in STATUS


def test_project_status_records_current_collection_incident() -> None:
    assert "7月21日開催は12場144R" in STATUS
    assert "出走表・120通りodds・結果は144/144" in STATUS
    assert "7月21日の予測は0/144R" in STATUS
    assert "M2-4 PostgreSQL収集常駐の予測オプション脱落" in STATUS
    assert "--predict" in STATUS


def test_project_status_records_final_race_semantics() -> None:
    assert "M2-3: 公式結果URL" in STATUS
    assert "7月21日144Rも結果残0" in STATUS
    assert "trifecta_evaluable=false" in STATUS
    assert "races.status=final" in STATUS
