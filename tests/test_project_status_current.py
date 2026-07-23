from pathlib import Path
import re


STATUS = Path("docs/PROJECT_STATUS.md").read_text(encoding="utf-8")


def test_project_status_uses_current_evaluation_state() -> None:
    assert re.search(r"更新日時: 20\d{2}-\d{2}-\d{2} \d{2}:\d{2} UTC", STATUS)
    assert "厳格T-5較正適格 / 正式評価 | 401R / 0R（開始7月24日）" in STATUS
    assert "標準365日v2は7モデル" in STATUS
    assert "7月22日136RでLogLoss 3.85700" in STATUS
    assert "LogLoss 3.85700（市場3.87201）" in STATUS
    assert "状態: v19稼働中" in STATUS
    assert "レース単位と日clusterの両方" in STATUS
    assert "較正401R、開発136R、正式0Rを分離" in STATUS
    assert "レースcluster 95%下限で補正" in STATUS
    assert "正式開始日を成果物へ記録" in STATUS
    assert "M6 資金運用モデル | 未完了/正式評価待ち" in STATUS
    assert "M7 ソース整理 | 完了/運用監視" in STATUS
    assert "M6-11: T-10→T-5モメンタム" in STATUS
    assert "T-10価格推移は過去日の価格holdoutで1%以上改善した場合だけ終値予測へ採用" in STATUS
    assert "PID 196980" not in STATUS
    assert "順番待ち" not in STATUS


def test_project_status_records_prediction_recovery() -> None:
    assert "7月22日開催は12場144R" in STATUS
    assert "出走表・予測・結果は144/144" in STATUS
    assert "7月22日の締切前予測は144/144R" in STATUS
    assert "M2-4: PostgreSQL常駐予測" in STATUS
    assert "--predict" in STATUS


def test_project_status_records_final_race_semantics() -> None:
    assert "M2-3: 公式結果URL" in STATUS
    assert "7月21日144Rも結果残0" in STATUS
    assert "trifecta_evaluable=false" in STATUS
    assert "races.status=final" in STATUS
