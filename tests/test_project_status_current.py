from pathlib import Path


STATUS = Path("docs/PROJECT_STATUS.md").read_text(encoding="utf-8")


def test_project_status_uses_current_evaluation_state() -> None:
    assert "2026-07-22 11:10 UTC" in STATUS
    assert "厳格T-5品質基準 | 260 / 450R" in STATUS
    assert "標準365日v2は7モデル" in STATUS
    assert "Newton市場残差の7月22日開発診断は113R" in STATUS
    assert "95%CI `[-0.05858, +0.00212]`" in STATUS
    assert "M6 資金運用モデル | 未完了/収益ゲート未達" in STATUS
    assert "M7 ソース整理 | 完了/運用監視" in STATUS
    assert "M6-11: T-10→T-5モメンタム" in STATUS
    assert "7月23日まで構造探索を凍結" in STATUS
    assert "PID 196980" not in STATUS
    assert "順番待ち" not in STATUS


def test_project_status_records_prediction_recovery() -> None:
    assert "7月22日開催は12場144R" in STATUS
    assert "出走表・120通りodds・予測は144/144" in STATUS
    assert "7月22日の締切前予測は144/144R" in STATUS
    assert "M2-4: PostgreSQL常駐予測" in STATUS
    assert "--predict" in STATUS


def test_project_status_records_final_race_semantics() -> None:
    assert "M2-3: 公式結果URL" in STATUS
    assert "7月21日144Rも結果残0" in STATUS
    assert "trifecta_evaluable=false" in STATUS
    assert "races.status=final" in STATUS
