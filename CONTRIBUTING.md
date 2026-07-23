# コントリビューションガイド

この文書は、開発作業を再現可能にし、実装、評価、デプロイの状態を混同しないための標準ガバナンスです。利用者向けのデータ収集・学習手順は [docs/WORKFLOW.md](docs/WORKFLOW.md) を参照してください。

## 1. 記録の責務

- PostgreSQLの `work_tickets` が実行進捗の一次記録です。作業開始前にticketを作成し、状態、進捗、担当、受入条件を最新に保ちます。
- `work_ticket_events` は追記型の実行履歴です。検証、deploy、health check、rollbackなどの節目を、再実行可能な証拠とともに記録します。
- GitHub Issueはレビュー、議論、外部からの追跡に使います。DBとIssueで状態が不一致ならDBを正とし、Issueを同期します。
- 実装を伴う作業は原則としてDB ticketとGitHub Issueの両方を持ちます。緊急対応では先にDB eventを残し、Issueを直後に補完します。

`ticket_key` はリポジトリ全体で一意かつ不変にします。Issueタイトルは `[MODEL-OPT-001] 説明` の形式にし、本文の `ticket_key` 欄にも同じ値を必ず記載します。DBの `source` またはevent noteにIssue URLを、Issue本文に `ticket_key` とDB環境の非秘密識別子を記載し、双方から相手へ到達できるようにします。接続文字列は記載しません。

ticketの操作例:

```bash
python -m boatrace_ai.work_tickets --db "$WORK_TICKET_DB" add \
  --key MODEL-OPT-001 \
  --title "モデル評価の再現性を改善" \
  --area model \
  --description "変更範囲と目的" \
  --acceptance "観測可能な受入条件" \
  --status queued \
  --progress 0 \
  --source "https://github.com/OWNER/REPO/issues/123"

python -m boatrace_ai.work_tickets --db "$WORK_TICKET_DB" update \
  --key MODEL-OPT-001 \
  --status in_progress \
  --progress 10 \
  --note 'issue=https://github.com/OWNER/REPO/issues/123; action=started'
```

`$WORK_TICKET_DB` の値や認証情報はshell history、ログ、Issue、commitへ残さないでください。

## 2. 作業単位とcommit

一つの作業単位は、一つの受入条件を満たし、単独で検証・deploy・revertできる大きさにします。その作業に必要なcode、test、config、schema/seed、docsは、該当するものを意味的に一つのcommitへ含めます。後続commitがなければ動かない実装や、実装とtestだけを理由なく別commitにする構成は不可です。docs-onlyなど該当しない種類の空ファイルは作りません。

commit subjectには `ticket_key` を含めます。

```text
MODEL-OPT-001 Make evaluation runs reproducible
```

共有branchの履歴を書き換えず、force pushをしません。レビュー修正は追加commitとし、統合時に意味的な作業単位が保たれる方法を選びます。

## 3. 標準フロー

順序は次の通りです。失敗した段階を飛ばして先へ進めません。

1. DB ticketを作成または確認し、GitHub Issueとの相互参照、受入条件、再現手順、対象期間、評価policyを確定する。
2. ticketを `in_progress` に更新し、変更対象と非対象をevent noteへ残す。
3. code、test、config、schema/seed、docsを同じ作業単位として実装する。
4. 対象testを実行し、必要に応じて全testを実行する。正確なcommandとpassed/failed/skipped/total件数を保存する。
5. staged diffに秘密や大規模ファイルがないことを確認し、`ticket_key` 付きでcommitする。
6. branchをpushし、commit SHAを確定する。
7. リモートの `/workspace` で対象branchをfetchし、`git merge --ff-only` 相当でfast-forwardする。force、resetによる反映は禁止する。
8. `/workspace` でruntime SHAを取得し、commit SHAとの一致を確認してhealth checkを実行する。
9. DB eventを先に追記し、同じ証拠をGitHub Issue commentへ転記する。
10. 受入条件を確認し、該当する完了状態だけを更新する。

代表的な確認command:

```bash
python -m pytest
git diff --cached --check
git diff --cached --stat
git commit -m "MODEL-OPT-001 Make evaluation runs reproducible"
git push origin HEAD

# リモートで実行。ホスト名や認証情報は記録しない。
cd /workspace
git fetch origin
git merge --ff-only origin/<branch>
git rev-parse HEAD
<health-check-command>
```

リモートのdefault branchへ直接反映する場合も、deploy対象SHAを明記し、現在のSHAからfast-forward可能であることを確認します。

## 4. 証拠の形式

DB eventとIssue commentには、最低限次の情報を同じ意味で記録します。値が存在しない場合は省略せず `not-applicable` と理由を記載します。

```text
ticket_key: MODEL-OPT-001
issue: https://github.com/OWNER/REPO/issues/123
commit_sha: <40-character SHA>
runtime_sha: <40-character SHA>
commands:
  - python -m pytest tests/test_example.py -q
tests: passed=<n>, failed=0, skipped=<n>, total=<n>
data_universe_hash: sha256:<hex> | not-applicable (<reason>)
period: 2025-01-01..2025-12-31 | not-applicable (<reason>)
policy: <policy name/version or immutable path> | not-applicable (<reason>)
artifact_path: /workspace/artifacts/<ticket_key>/<run-id> | not-applicable (<reason>)
deployment: remote=/workspace; method=fast-forward; result=success
health_check: command=<command>; result=pass; observed_at=<UTC ISO-8601>
evaluation: not-run | running | completed | failed
performance_gate: not-evaluated | passed | failed
```

- `commit_sha` はpush済みcommit、`runtime_sha` はhealth check対象プロセスまたはcheckoutの実体を示します。両者が異なる場合は完了にしません。
- `commands` は秘密を展開せず、そのまま再実行できる形にします。必要な秘密は環境変数名だけを示します。
- test件数はテストランナーのsummaryから記録します。途中で中断したrunを成功として数えません。
- `data_universe_hash` は入力対象IDを安定順に並べ、期間・除外条件を含むmanifestをSHA-256でhash化した値とします。hashだけでなくmanifestのartifact pathも残します。
- `period` は両端の包含・除外とtimezoneが曖昧な場合、policyまたはartifactで明示します。
- `policy` は閾値、比較演算子、指標、holdout条件を含む変更不能な版を指します。結果を見た後にpolicyを書き換えた場合は別評価です。
- `artifact_path` はリポジトリ内の大規模生成物ではなく、`/workspace` 以下など管理された保存先を指します。

## 5. データと秘密情報

大規模なdata、学習済みmodel、cache、DB dump、生成artifact、secretはGitへcommitしません。追加前に少なくとも `git diff --cached --stat` と `git status --short` を確認します。小さなschema、migration、seed、fixtureは再現に必要なsourceとしてcommit対象です。

- ローカルの作業領域は合計2GB以内に保ちます。大規模処理と永続artifactはリモートの `/workspace` で扱います。
- PostgreSQLのDSN、password、証明書、rclone config、token、remote名に埋め込まれた資格情報を、Git、Issue、PR、DB event、test出力へ掲載しません。
- 秘密は承認された環境変数またはsecret storeから注入します。証拠には変数名と成否だけを残し、値は伏せます。
- データを外部へ転送する場合は、承認済み保存先、暗号化、保持期間を確認し、転送commandから秘密を除いた形だけを記録します。

## 6. 三つの完了判定

次の判定は独立しています。

1. **作業完了**: code、test、config、schema/seed、docsの必要な変更が揃い、push、remote fast-forward、health check、証拠記録、受入条件の確認まで完了した状態です。この時点でのみ `work_tickets.status=completed`、`progress=100` にできます。
2. **評価完了**: 指定したdata universe、period、policyで評価runが正常終了し、結果とartifactが保存された状態です。`model_evaluation_job_runs.status=completed` など評価側の記録で管理し、作業完了や性能gate通過を意味しません。
3. **性能gate通過**: 事前に固定したpolicyの全条件を評価結果が満たした状態です。評価が完了してもgateがfailedなら通過ではありません。gate通過だけでもdeployとhealth checkが未完了なら作業完了ではありません。

性能gateが受入条件に含まれるticketは、評価完了かつgate通過の証拠が揃うまで `completed` にしません。性能改善を要求しない修正では、評価とgateを `not-applicable` とし理由を残せます。

## 7. rollback

不具合時は、履歴を保持するrevertまたは検証済みprevious SHAへの切替で戻します。

```bash
git revert <bad-commit-sha>
python -m pytest <affected-tests>
git push origin HEAD
```

revert commitも同じ標準フローでremoteへfast-forwardし、health checkを行います。previous SHAへ切り替える運用基盤では、`from_runtime_sha`、`to_runtime_sha`、理由、command、test結果、health checkをDB eventとIssueへ残します。`git reset --hard`、force push、未記録の手作業によるrollbackは禁止です。

schema変更はデータを破壊しないforward-compatibleなmigrationを優先します。codeのrevertでDBデータまで自動的に戻るとは仮定せず、データ復旧が必要なら別ticket、承認、backup、復旧証拠を用意します。

## 8. 文書ガバナンス

READMEは概要と導線だけを持つ粗い入口です。詳細は次のcanonical文書へ置き、同じ事実や手順を複数箇所で管理しません。

| 文書 | canonicalな責務 |
|---|---|
| [README.md](README.md) | プロジェクト概要、最小セットアップ、詳細文書への入口 |
| [docs/WORKFLOW.md](docs/WORKFLOW.md) | 利用者向け収集・学習手順と開発・リリース運用 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | package責務、システム境界、安定した実行入口 |
| [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) | 現在状態、評価結果、未完了項目、マイルストーン |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 開発、証拠、レビュー、文書管理のガバナンス |
| [docs/GPU_WORKSPACE.md](docs/GPU_WORKSPACE.md) | GPU remote workspaceの既存運用仕様 |
| [docs/MODEL_FEATURE_RESEARCH.md](docs/MODEL_FEATURE_RESEARCH.md) | model feature調査の既存根拠 |
| [docs/TELEBOAT_AGENT_AUDIT.md](docs/TELEBOAT_AGENT_AUDIT.md) | Teleboat agentの既存監査記録 |
| [docs/TELEBOAT_API.md](docs/TELEBOAT_API.md) | Teleboat APIの安全仕様とjournal契約 |

標準ガバナンスファイル以外の新しい話題別Markdownは禁止します。新しい要件、設計、runbook、調査結果、進捗は、上表で責務を持つ既存文書へ統合します。重複を見つけた場合は一方をcanonical文書へ集約し、他方は削除または短い参照に置換します。READMEへ詳細手順や変動する状態を複製しません。

少なくとも月1回と各リリース前にrepository hygiene auditを実施します。auditでは全Markdownを棚卸しし、次を確認します。

- READMEから詳細層へ到達でき、相対リンクが存在する。
- 文書の内容と所有責務が上表に一致し、重複・矛盾・孤立文書がない。
- command、path、schema、状態、日付が現行実装と一致し、陳腐化した手順がない。
- 新規の話題別Markdown、秘密、大規模artifact、生成レポートがGitへ混入していない。

auditはDB ticketを作成して実施し、対象commit SHA、実行command、確認文書、findings、修正先、実施日時をDB eventとIssue commentへ記録します。問題なしの場合も `findings=none` を残します。

## 9. レビュー条件

PRを出す前に、IssueとPRのtemplateをすべて確認します。reviewerは、受入条件とdiffの対応、再現手順、commitの意味的一体性、test件数、秘密・大規模ファイルの不在、remote反映方法、三つの完了判定、rollback手順、文書の責務を確認します。不足があるPRはmergeせず、DB ticketを実態に合わせて更新します。
