# boatrace-ai

日本国内24場のBOAT RACEデータを収集し、過去データ・当日出走表・直前情報・3連単オッズの時系列から予測を作るためのPythonパイプラインです。

## できること

- 過去10年分の番組表・競走成績LZHを日次URL規則でバックフィル
- レーサー期別成績LZHの取得
- 当日開催レースの出走表、直前情報、3連単オッズをスナップショット保存
- オッズ推移を特徴量化し、モデルへ投入
- 1着確率モデルから3連単120通りの確率と期待値を出力
- レース結果取得後の増分学習

## 注意

このリポジトリは予測システムであり、的中や利益を保証しません。BOAT RACE公式サイトのサイトポリシーでは、大量アクセスなどサイト運営に支障を与える行為が禁止されています。実運用では取得間隔、同時接続、再取得を抑え、公開情報の利用条件を確認してください。舟券の購入は日本国内では20歳以上が対象です。

## ドキュメント

このREADMEはプロジェクトの粗い入口です。実装・運用の中核詳細層は次のcanonical文書です。

- [運用ワークフロー](docs/WORKFLOW.md): データ収集、学習、監視、Webサーバー、開発・リリース運用
- [アーキテクチャ](docs/ARCHITECTURE.md): package責務、安定した実行入口、ソース構成
- [プロジェクト状況](docs/PROJECT_STATUS.md): 現在状態、評価結果、未完了項目、マイルストーン

標準ガバナンスは [コントリビューションガイド](CONTRIBUTING.md) にまとめています。DB ticket、GitHub Issue、commit、検証、deploy、証拠、rollback、文書管理はこの規約に従います。

特化した既存仕様・監査記録は [GPU workspace](docs/GPU_WORKSPACE.md)、[model feature research](docs/MODEL_FEATURE_RESEARCH.md)、[Teleboat agent audit](docs/TELEBOAT_AGENT_AUDIT.md)、[投票API](docs/TELEBOAT_API.md) です。詳細層と内容が重複する文書は作らず、変更は担当するcanonical文書へ統合します。

## 文書管理

標準ガバナンスファイルを除き、新しい話題別Markdownを追加しません。少なくともリリース前と月1回、リンク切れ、重複、陳腐化、孤立文書を確認するrepository hygiene auditを実施します。詳細な判定基準と記録方法は [CONTRIBUTING.md](CONTRIBUTING.md) に従います。

## 投票API

監査済み内部投票APIの入力形式、安全ゲート、全勝式、BOX/フォーメーション、ジャーナル仕様は [docs/TELEBOAT_API.md](docs/TELEBOAT_API.md) を参照してください。既定はdry-runで、実投票は明示的な多重ゲートなしには動作しません。

## セットアップ

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev,lzh]"
```

LZH展開は、Pythonの `lhafile`、またはOS上の `lha` / `7z` / `bsdtar` のいずれかを使います。展開できない環境でもLZHファイル自体は保存されます。

## 基本コマンド

```bash
# DB初期化
boat-ai init-db --db data/boatrace.sqlite

# 過去10年分の番組表・競走成績を取得
boat-ai backfill --db data/boatrace.sqlite --years 10 --kind both --sleep 1.5

# レーサー期別成績を取得
boat-ai fetch-racer-stats --db data/boatrace.sqlite --from-year 2016 --to-year 2026

# 当日開催分を1回取得
boat-ai collect-live-once --db data/boatrace.sqlite --date 2026-07-18

# オッズ推移を監視しながら予測を更新
boat-ai monitor --db data/boatrace.sqlite --model data/models/win_model.joblib --date 2026-07-18 --interval 120

# 学習
boat-ai train --db data/boatrace.sqlite --model data/models/win_model.joblib

# 予測
boat-ai predict --db data/boatrace.sqlite --model data/models/win_model.joblib --date 2026-07-18 --jcd 01 --rno 1
```

## データソース

- BOAT RACEオフィシャルサイトの「ダウンロード・他」から、全国24場の競走成績・番組表とレーサー期別成績を取得できます。
- 当日出走表、オッズ、直前情報、結果は `boatrace.jp/owpc/pc/race/...` の各公開ページから取得します。
- オッズは表示画面の更新操作に対応した更新時刻がページに表示されるため、このシステムでは取得時刻と公式表示の更新時刻を両方保存します。

## スキーマ概要

- `races`: レース単位の基本情報
- `entries`: 枠番ごとの選手・モーター・ボート情報
- `odds_snapshots`: 取得時点ごとのオッズ生データ
- `odds_trifecta`: 3連単120通りのオッズ
- `beforeinfo`: 展示、チルト、気象などの直前情報
- `race_results`: 着順、進入、STなどの結果
- `predictions`: 予測確率と期待値
- `raw_files` / `raw_pages`: 生LZH、TXT、HTMLの保管台帳
