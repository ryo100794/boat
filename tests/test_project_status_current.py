from pathlib import Path


STATUS = Path("docs/PROJECT_STATUS.md").read_text(encoding="utf-8")


def test_project_status_uses_current_evaluation_state() -> None:
    assert "完了16件、差替済み2件、実行中0件" in STATUS
    assert "350 / 1,000R" in STATUS
    assert "M6 資金運用モデル | 未完了/収益ゲート未達" in STATUS
    assert "PID 196980" not in STATUS
    assert "順番待ち" not in STATUS


def test_project_status_records_final_race_semantics() -> None:
    assert "全192Rが確定し、結果残0、監視中0場" in STATUS
    assert "trifecta_evaluable=false" in STATUS
    assert "着順行数だけでなく" in STATUS
