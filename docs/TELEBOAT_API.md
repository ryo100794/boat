# Teleboat 投票API

## 目的と安全境界

teleboat-agent は、投票要求の検証、100円単位の金額計算、BOX/フォーメーションの展開、公式Teleboat画面との照合を行う内部APIです。

- 既定動作は **dry_run** で、公式サイトへログインせず投票もしません。
- 実投票は環境変数、リクエストヘッダー、確認secret、冪等性キーの全条件が揃った場合だけ有効です。
- 公式確認画面では、場、レース、勝式、全組番、展開点数、合計金額、「投票未完了」、最終ボタンの有効状態を照合します。
- 最終ボタン押下後に結果確認不能となった要求は再試行禁止です。同じ冪等性キーは保持されます。
- APIは内部用です。インターネットへ直接公開せず、localhostまたは認証済み内部ネットワークで使用してください。

## 起動

~~~bash
. .venv/bin/activate
pip install -e ".[teleboat]"
export TELEBOAT_AGENT_API_APPLICATION_TOKEN='十分に長いランダム値'
teleboat-agent --host 127.0.0.1 --port 9999
~~~

実投票を有効にする場合だけ、次も設定します。

~~~bash
export TELEBOAT_ENABLE_LIVE_VOTE=true
export TELEBOAT_LIVE_CONFIRMATION_SECRET='APIトークンとは異なるランダム値'
export TELEBOAT_MEMBER_NUMBER='加入者番号'
export TELEBOAT_PIN='暗証番号'
export TELEBOAT_AUTHORIZATION_NUMBER_OF_MOBILE='認証番号'
~~~

資格情報はログへ出力しません。秘密ファイルを併用する場合も権限0600を維持してください。

## エンドポイント

**POST /api/internal/v1/tickets/votes**

共通ヘッダー:

~~~http
Authorization: Bearer <application-token>
Content-Type: application/json
~~~

実投票時だけ必要なヘッダー:

~~~http
X-Teleboat-Live: true
X-Teleboat-Live-Confirmation: <live-confirmation-secret>
Idempotency-Key: <要求ごとに一意な128文字以内の値>
~~~

X-Teleboat-Live を省略した要求は常にdry-runです。

## 勝式

| bet_type | 表示 | 組番桁数 | 順序 |
|---|---:|---:|---|
| win | 単勝 | 1 | あり |
| place | 複勝 | 1 | あり |
| exacta | 2連単 | 2 | あり |
| quinella | 2連複 | 2 | なし |
| quinella_place | 拡連複 | 2 | なし |
| trifecta | 3連単 | 3 | あり |
| trio | 3連複 | 3 | なし |

日本語の勝式名も受理します。順序なしの勝式は昇順へ正規化し、同一組番の重複を拒否または展開時に統合します。

## 通常投票

quantityは100円単位です。100円は1、500円は5です。

~~~json
{
  "race": {"stadium_tel_code": 20, "number": 11},
  "bet_type": "exacta",
  "method": "regular",
  "tickets": [
    {"number": "12", "quantity": 1},
    {"number": "21", "quantity": 2}
  ]
}
~~~

旧APIとの互換性のため、bet_typeとmethodを省略し、ticketsの代わりにoddsを指定した要求は3連単・通常投票として扱います。

~~~json
{
  "race": {"stadium_tel_code": 20, "number": 11},
  "odds": [{"number": "123", "quantity": 1}]
}
~~~

## BOX

単勝・複勝では使用できません。選択艇から有効な組番を展開した後に点数と金額上限を検証します。

~~~json
{
  "race": {"stadium_tel_code": 20, "number": 11},
  "bet_type": "trifecta",
  "method": "box",
  "selections": [1, 2, 3],
  "quantity": 1
}
~~~

この例は6点、合計600円です。3艇BOXの展開点数は、2連単6点、2連複3点、拡連複3点、3連単6点、3連複1点です。

## フォーメーション

各配列は着順位置です。艇の重複で成立しない組合せを除外し、順序なし勝式は同一組番を統合します。

~~~json
{
  "race": {"stadium_tel_code": 20, "number": 11},
  "bet_type": "trifecta",
  "method": "formation",
  "formation": [[1], [2, 3], [2, 3, 4]],
  "quantity": 1
}
~~~

この例は1-2-3、1-2-4、1-3-2、1-3-4の4点、合計400円です。

## dry-run応答

~~~json
{
  "success": true,
  "mode": "dry_run",
  "request_id": "一意な要求ID",
  "stadium_tel_code": "20",
  "race_number": 11,
  "bet_type": "trifecta",
  "bet_type_label": "3連単",
  "method": "formation",
  "tickets": 4,
  "total_stake_yen": 400,
  "batches": [
    {
      "batch": 1,
      "tickets": 4,
      "stake_yen": 400,
      "selections": [
        {"number": "123", "quantity": 1, "stake_yen": 100}
      ],
      "codes": []
    }
  ]
}
~~~

codesは旧3連単簡易投票コードとの互換表示です。投票実行は名前付き公式フォームを使用し、このコードへ依存しません。

## 実投票時のベリファイ

1. JSON型、場コード、レース番号、勝式、方式を検証。
2. 艇番範囲、同一組番、100円単位数量を検証。
3. BOX/フォーメーションを展開し、0点、点数超過、合計金額超過を拒否。
4. Bearer、live有効化、確認secret、冪等性キーを検証。
5. ローカルHeadless Chromeで公式ホストだけを開き、ログイン完了要素を確認。
6. 場、レース、勝式、方式のradioを設定後に値を再確認。
7. 各組番・数量を入力後にinput_valueを再確認。
8. BOX/フォーメーションは公式表示とローカルの展開点数を照合。
9. ベットリストで場、レース、勝式、全組番、点数、購入金額、「投票未完了」を照合。
10. 公式hidden合計金額とローカル合計を照合。
11. live要求だけ、最終購入金額を入力・再確認して最終ボタンを押下。
12. 受付完了表示を確認し、ログアウトを試行。

最終ボタン押下前の不一致は投票せず失敗します。押下後に受付結果を確認できない場合はsubmission_unknownとし、自動再試行しません。

## HTTP応答

| 状態 | HTTP | 意味 |
|---|---:|---|
| 正常dry-run | 200 | 検証と展開のみ完了 |
| 実投票・受付確認済み | 200 | success=true、submitted_verified |
| 実投票・結果不明 | 200 | success=false、requires_manual_confirmation=true |
| 入力不正 | 400 | 艇番、数量、方式、上限などの検証失敗 |
| 認証・liveゲート不正 | 403 | Bearerまたはlive確認に失敗 |
| 冪等性キー再利用 | 409 | 同一要求の再送を拒否 |
| 公式操作失敗 | 502 | pre_submission_failedまたはsubmission_state_unknown |

~~~json
{
  "errors": [
    {
      "message": "live vote execution failed",
      "code": "submission_state_unknown",
      "retry_allowed": false
    }
  ]
}
~~~

retry_allowed=falseの場合は、Teleboat公式の投票履歴を人手で確認するまで別キーで再送しないでください。

## 上限

- TELEBOAT_MAX_TICKETS_PER_REQUEST: 展開後の最大点数。既定30。
- TELEBOAT_MAX_TOTAL_STAKE_YEN: 一要求の最大合計金額。既定10,000円。
- TELEBOAT_BATCH_SIZE: 通常投票フォーム一回あたりの行数。1から10、既定10。
- 1点のquantity: 1から999。

上限判定はBOX/フォーメーション展開後に行います。


## 投票ジャーナル

既定保存先は **data/teleboat_vote_journal.jsonl** です。TELEBOAT_JOURNAL_PATHで変更できます。

各行にはrequest ID、冪等性キーのSHA-256、場/R、勝式/方式、入力選択、展開後全組番、数量、金額、処理段階、ベリファイ結果、最終ボタン押下有無、受付状態、ログアウト結果を記録します。加入者番号、暗証番号、認証番号、Bearer、確認secretは記録せず、該当キーは再帰的にREDACTEDへ置換します。

ファイルは0600、追記後fsync、各レコードは直前hashを含むSHA-256チェーンです。次のコマンドでJSON構文、previous hash、全record hashを検証できます。

~~~bash
teleboat-journal-verify --path data/teleboat_vote_journal.jsonl
~~~

live実行前のlive_authorizedレコードが書けない場合は投票を開始せず503を返します。最終送信後の結果レコード書込に失敗した場合は冪等性キーを維持し、write_failed_after_executionとして人手確認を要求します。
