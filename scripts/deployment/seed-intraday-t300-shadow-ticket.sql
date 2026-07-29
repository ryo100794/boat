INSERT INTO work_tickets(
  ticket_key, title, area, description, acceptance_criteria,
  owner, priority, status, progress, source
) VALUES (
  'MODEL-INTRADAY-T300-SHADOW-001',
  '全場T300役割モデルの当日シャドー記録',
  'モデル運用',
  'PostgreSQLの既存レース・オッズから全場のT300判断を一度だけ固定し、実投票せず後刻決済する',
  '全場で120通り、対象時刻前90秒以内、配信元更新停滞120秒以内を満たす判断または理由付きno-betを保存し、再起動時に重複せず、結果確定後は判断行を変更せず決済を追記できる',
  'codex', 98, 'in_progress', 85, 'codex'
)
ON CONFLICT (ticket_key) DO UPDATE SET
  title = EXCLUDED.title,
  area = EXCLUDED.area,
  description = EXCLUDED.description,
  acceptance_criteria = EXCLUDED.acceptance_criteria,
  owner = EXCLUDED.owner,
  priority = EXCLUDED.priority,
  source = EXCLUDED.source,
  updated_at = CURRENT_TIMESTAMP;
