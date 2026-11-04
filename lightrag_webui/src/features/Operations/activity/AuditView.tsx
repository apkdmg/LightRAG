import { FormEvent, useId, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { DownloadIcon } from 'lucide-react'
import Input from '@/components/ui/Input'
import { cn } from '@/lib/utils'
import { AuditRow, opsApi } from '../api'
import AuditExportDialog from '../components/AuditExportDialog'
import KeysetPager from '../components/KeysetPager'
import { Cursor, FIRST_PAGE } from '../keyset'
import { Panel } from '../components/Panel'
import { StatusChip } from '../components/StatusChip'
import { useOpsNav, useOpsShell } from '../context'
import { fmtStamp, shortId, utcTitle } from '../format'
import { useOpsPoll } from '../poll'
import { AuditDraft, EMPTY_AUDIT_DRAFT, actionSuggestions, auditActor, auditFilters, auditParams, auditRowState, draftInvalid } from './audit'

/** §3.10: Activity refreshes every 60 s. */
const ACTIVITY_SECONDS = 60

/** The audit table (§2.8): time, actor, auth kind, action, target, reason, outcome and request ID. */
export function AuditTable({ rows, now }: { rows: AuditRow[]; now: number | null }) {
  const { t } = useTranslation()
  const nav = useOpsNav()
  const col = (key: string) => t(`operations.audit.col.${key}`)
  return (
    <div className="overflow-x-auto">
      <table className="ops-stack w-full min-w-[980px] text-[13px]">
        <thead>
          <tr className="text-muted-foreground border-b text-xs">
            <th className="p-2 text-left font-medium">{col('time')}</th>
            <th className="p-2 text-left font-medium">{col('actor')}</th>
            <th className="p-2 text-left font-medium">{col('action')}</th>
            <th className="p-2 text-left font-medium">{col('target')}</th>
            <th className="p-2 text-left font-medium">{col('reason')}</th>
            <th className="p-2 text-left font-medium">{col('outcome')}</th>
            <th className="p-2 text-left font-medium">{col('ref')}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const state = auditRowState(r)
            return (
              <tr key={r.id} className="border-b align-top last:border-0">
                <td className="p-2 font-mono text-xs whitespace-nowrap" data-label={col('time')} title={utcTitle(r.ts)}>
                  {fmtStamp(r.ts, now ?? undefined)}
                </td>
                <td className="p-2 break-all" data-label={col('actor')}>
                  <span className="flex flex-col items-end gap-0.5 md:items-start">
                    <span>{auditActor(r)}</span>
                    {r.auth_kind ? <span className="text-muted-foreground text-xs">{t(`operations.audit.auth.${r.auth_kind}`, { defaultValue: r.auth_kind })}</span> : null}
                  </span>
                </td>
                <td className="p-2 font-mono text-xs break-all" data-label={col('action')}>
                  {r.action}
                </td>
                <td className="p-2 break-all" data-label={col('target')}>
                  {r.workspace ? (
                    <button
                      type="button"
                      className="text-left underline-offset-2 hover:underline"
                      onClick={() => nav.openDrawer({ kind: 'workspace', id: r.workspace as string })}
                    >
                      {r.workspace}
                    </button>
                  ) : null}
                  {r.target && r.target !== r.workspace ? <span className="text-muted-foreground block font-mono text-xs">{r.target}</span> : null}
                  {!r.workspace && !r.target ? '–' : null}
                </td>
                <td className="text-muted-foreground max-w-[280px] p-2 break-words" data-label={col('reason')}>
                  {r.reason || '–'}
                </td>
                <td className="p-2" data-label={col('outcome')}>
                  <StatusChip state={state}>
                    {r.phase === 'intent'
                      ? t('operations.audit.intent')
                      : r.outcome
                        ? t(`operations.audit.outcome.${r.outcome}`, { defaultValue: r.outcome.replace(/_/g, ' ') })
                        : '–'}
                  </StatusChip>
                </td>
                <td className="p-2 font-mono text-xs" data-label={col('ref')} title={r.request_id ?? undefined}>
                  {r.request_id ? shortId(r.request_id, 10) : '–'}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

/**
 * The Audit section of Activity (§2.8, FR-75): every console action, reveal,
 * export and log search, plus the existing admin changes (impersonation,
 * workspace deletion, owner changes, server-provider changes and
 * cross-workspace document changes), each as an intent row and a result
 * row. Filters by actor, action and days; CSV export with a reason.
 */
export default function AuditView() {
  const { t } = useTranslation()
  const { status } = useOpsShell()
  const listId = useId()
  const [draft, setDraft] = useState<AuditDraft>(EMPTY_AUDIT_DRAFT)
  const [applied, setApplied] = useState<AuditDraft>(EMPTY_AUDIT_DRAFT)
  const [cursor, setCursor] = useState<Cursor>(FIRST_PAGE)
  const [exporting, setExporting] = useState(false)
  const poll = useOpsPoll((signal) => opsApi.audit(auditParams(applied, cursor.before), signal), [applied, cursor.before], ACTIVITY_SECONDS)
  const invalid = draftInvalid(draft)

  const apply = (e: FormEvent) => {
    e.preventDefault()
    if (invalid) return
    setApplied({ ...draft, actor: draft.actor.trim(), action: draft.action.trim() })
    setCursor(FIRST_PAGE)
  }
  const clear = () => {
    setDraft(EMPTY_AUDIT_DRAFT)
    setApplied(EMPTY_AUDIT_DRAFT)
    setCursor(FIRST_PAGE)
  }
  const active = Object.entries(auditFilters(applied))
  const summary = active.length
    ? active.map(([k, v]) => `${t(`operations.audit.filter.${k}`)}=${k === 'from' || k === 'to' ? applied[k as 'from' | 'to'] : v}`).join(' · ')
    : t('operations.audit.allRows')
  const suggestions = actionSuggestions(poll.data?.items ?? [])
  const field = 'flex min-w-0 flex-col gap-1 text-xs'

  return (
    <div className="flex flex-col gap-3">
      <form onSubmit={apply} className="bg-card flex flex-wrap items-end gap-3 rounded-[10px] border p-4" aria-label={t('operations.audit.filtersLabel')}>
        <label className={cn(field, 'w-full sm:w-44')}>
          <span className="text-muted-foreground font-medium">{t('operations.audit.filter.actor')}</span>
          <Input value={draft.actor} onChange={(e) => setDraft({ ...draft, actor: e.target.value })} placeholder={t('operations.audit.actorPlaceholder')} />
        </label>
        <label className={cn(field, 'w-full sm:w-52')}>
          <span className="text-muted-foreground font-medium">{t('operations.audit.filter.action')}</span>
          <Input value={draft.action} onChange={(e) => setDraft({ ...draft, action: e.target.value })} list={listId} placeholder={t('operations.audit.actionPlaceholder')} />
          <datalist id={listId}>
            {suggestions.map((a) => (
              <option key={a} value={a} />
            ))}
          </datalist>
        </label>
        <label className={cn(field, 'w-full sm:w-40')}>
          <span className="text-muted-foreground font-medium">{t('operations.audit.filter.from')}</span>
          <Input type="date" value={draft.from} onChange={(e) => setDraft({ ...draft, from: e.target.value })} />
        </label>
        <label className={cn(field, 'w-full sm:w-40')}>
          <span className="text-muted-foreground font-medium">{t('operations.audit.filter.to')}</span>
          <Input type="date" value={draft.to} onChange={(e) => setDraft({ ...draft, to: e.target.value })} aria-invalid={invalid || undefined} />
        </label>
        <div className="flex flex-wrap items-center gap-2">
          <button type="submit" disabled={invalid} className="bg-primary text-primary-foreground min-h-9 rounded-lg px-3 text-[13px] hover:opacity-90 disabled:opacity-50">
            {t('operations.audit.apply')}
          </button>
          <button type="button" onClick={clear} className="hover:bg-muted min-h-9 rounded-lg border px-3 text-[13px]">
            {t('operations.audit.clear')}
          </button>
        </div>
        {invalid ? <p className="m-0 basis-full text-xs text-[color:var(--ops-warn)]">{t('operations.audit.rangeInvalid')}</p> : null}
      </form>
      <Panel
        poll={poll}
        freshness={ACTIVITY_SECONDS}
        title={t('operations.audit.title')}
        sub={t('operations.audit.scope')}
        actions={
          // The CSV export is operator-only on the server (D8): hidden for other admins.
          status?.operator ? (
            <button type="button" className="hover:bg-muted inline-flex min-h-8 items-center gap-1.5 rounded-lg border px-2.5 text-[13px]" onClick={() => setExporting(true)}>
              <DownloadIcon className="size-3.5" aria-hidden="true" />
              {t('operations.audit.exportButton')}
            </button>
          ) : null
        }
      >
        {(data) => (
          <div className={cn('flex flex-col gap-3', poll.loading && 'opacity-60')}>
            {data.items.length ? (
              <AuditTable rows={data.items} now={poll.updatedAt} />
            ) : (
              <p className="text-muted-foreground m-0 text-sm">{active.length ? t('operations.audit.emptyFiltered') : t('operations.audit.empty')}</p>
            )}
            <KeysetPager next={data.next_before} cursor={cursor} onCursor={setCursor} />
            <p className="text-muted-foreground m-0 text-xs">{t('operations.audit.note')}</p>
          </div>
        )}
      </Panel>
      {status?.operator ? <AuditExportDialog open={exporting} onOpenChange={setExporting} filters={auditFilters(applied)} summary={summary} /> : null}
    </div>
  )
}
