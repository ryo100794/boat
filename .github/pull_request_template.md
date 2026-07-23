## Ticket

- ticket_key: <!-- 必須。例: MODEL-OPT-001 -->
- DB ticket: <!-- 秘密を含まない環境識別子と相互参照 -->
- Issue: <!-- 必須。Issue本文とタイトルにも同じticket_keyを記載 -->

## Change

<!-- 変更内容、変更対象外、受入条件との対応を簡潔に記載 -->

## Reproduction

```bash
# reviewerが再実行できるcommand
```

## Evidence

- commit SHA:
- commands:
- tests: passed=, failed=, skipped=, total=
- data universe hash:
- period:
- policy:
- artifact path:
- deployment target: `/workspace`
- runtime SHA:
- health check:
- evaluation: `not-run / running / completed / failed / not-applicable`
- performance gate: `not-evaluated / passed / failed / not-applicable`

## Checklist

- [ ] PR、Issueタイトル、Issue本文に同じ `ticket_key` があり、DBとIssueを相互参照できる
- [ ] code、test、config、schema/seed、docsの必要な変更が意味的に一つのcommitへ含まれている
- [ ] 受入条件を満たすtestを実行し、commandとpassed/failed/skipped/total件数を記録した
- [ ] staged diffに大規模data/model/cache、DB dump、生成artifact、secretがない
- [ ] ローカル作業領域は2GB以内で、大規模処理・artifactは `/workspace` を使用する
- [ ] PostgreSQL DSN、password、証明書、rclone config/tokenを掲載していない
- [ ] `test -> commit -> push -> remote fast-forward -> health check -> DB event / Issue comment` の順序を守る
- [ ] deployはfast-forwardで行い、commit SHAとruntime SHAの一致を確認する
- [ ] DB eventとIssue commentへcommands、test count、data universe hash、period、policy、artifact path、runtime SHAを記録する
- [ ] 作業完了、評価完了、性能gate通過を別々に判定した
- [ ] rollbackはrevertまたは検証済みprevious SHAで、手順とhealth checkが明記されている
- [ ] READMEは粗い入口に留め、新しい話題別Markdownやcanonical文書との重複を追加していない
