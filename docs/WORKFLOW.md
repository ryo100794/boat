# 運用ワークフロー

## 1. 初段: 過去データだけでモデルを作る

初期段階ではリアルタイムオッズを使わず、公式の過去番組表・競走成績から取れる情報だけで学習とバックテストを行います。

```bash
cd /root/boat
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev,lzh]"

boat-ai init-db --db data/boatrace.sqlite
boat-ai backfill --db data/boatrace.sqlite --years 10 --kind both --sleep 1.5
boat-ai fetch-racer-stats --db data/boatrace.sqlite --from-year 2016 --to-year 2026 --sleep 1.5

boat-ai backtest --db data/boatrace.sqlite --output data/models/backtest.json --min-train-races 500
boat-ai train --db data/boatrace.sqlite --model data/models/win_model.joblib
```

この段階の特徴量は、枠番、場、レース番号、距離、選手級別、年齢、体重、平均ST、全国・当地勝率、モーター2連対率・3連対率、ボート2連対率・3連対率などです。

## 2. 次段: リアルタイムデータを蓄積する

当日開催分は、出走表、直前情報、3連単オッズ、結果を取得します。オッズは取得時刻と公式表示の更新時刻を両方保存し、スナップショット単位で時系列化します。

```bash
boat-ai collect-live-once --db data/boatrace.sqlite --date 2026-07-18
```

単一レースだけ取る場合:

```bash
boat-ai collect-live-once --db data/boatrace.sqlite --date 2026-07-18 --jcd 01 --rno 1
```

## 3. リアルタイム予測と段階的なモデル更新

初期モデルが存在する状態で監視を開始します。`--retrain-every` を指定すると、指定ループごとにリアルタイムで蓄積された出走表、結果、オッズ時系列特徴量も含めて再学習します。

```bash
boat-ai monitor \
  --db data/boatrace.sqlite \
  --model data/models/win_model.joblib \
  --date 2026-07-18 \
  --interval 120 \
  --retrain-every 10 \
  --include-odds
```

この構成では、過去データのみのベースラインから開始し、レース結果が確定してDBに入った後にリアルタイム特徴量込みのモデルへ徐々に更新します。

## 4. Webサーバー

この環境では、Webサーバーをポート `10001` で起動します。

```bash
cd /root/boat/src
python3 -m boatrace_ai.web_dashboard \
  --db ../data/boatrace.sqlite \
  --host 0.0.0.0 \
  --port 10001 \
  --backtest ../data/models/backtest.json
```

ブラウザでは次を開きます。

```text
http://127.0.0.1:10001
```

## 5. 重要な運用制約

公式サイトへの大量アクセスは避けてください。10年分のバックフィルは日次ファイル単位でも数千リクエストになります。`--sleep` を大きめにし、途中停止しても既取得ファイルをスキップして再開できる設計にしています。

予測値は統計モデルの推定値であり、的中や利益を保証しません。オッズを使う場合、期待値は控除率、市場歪み、締切直前変動、購入制限、欠場・返還を別途考慮する必要があります。
